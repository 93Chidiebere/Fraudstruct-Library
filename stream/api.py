"""FastAPI adapter for the incremental Fraudstruct detector.

Default thresholds are demonstration values in the dataset's amount units.
Configure them for the deployment currency and institution before use.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraudstruct.detect.flow_structuring import StructuringConfig
from fraudstruct.detect.gcn import train_gcn
from fraudstruct.stream.engine import StreamingEngine


def _config_from_environment() -> StructuringConfig:
    target = os.getenv("FRAUDSTRUCT_TARGET_TOTAL_AMOUNT")
    return StructuringConfig(
        window_seconds=int(os.getenv("FRAUDSTRUCT_WINDOW_SECONDS", "3600")),
        min_mules=int(os.getenv("FRAUDSTRUCT_MIN_MULES", "3")),
        minimum_total_amount=float(os.getenv("FRAUDSTRUCT_MIN_TOTAL_AMOUNT", "300000")),
        max_leg_amount=(
            None
            if os.getenv("FRAUDSTRUCT_MAX_LEG_AMOUNT", "100000").lower() in {"none", "null", ""}
            else float(os.getenv("FRAUDSTRUCT_MAX_LEG_AMOUNT", "100000"))
        ),
        flow_conservation_tolerance=float(
            os.getenv("FRAUDSTRUCT_FLOW_TOLERANCE", "0.10")
        ),
        target_total_amount=None if target is None else float(target),
        target_amount_tolerance=float(os.getenv("FRAUDSTRUCT_TARGET_TOLERANCE", "0.05")),
        max_tracked_pairs=int(os.getenv("FRAUDSTRUCT_MAX_TRACKED_PAIRS", "100000")),
        max_active_events=int(os.getenv("FRAUDSTRUCT_MAX_ACTIVE_EVENTS", "500000")),
        max_cached_accounts=int(os.getenv("FRAUDSTRUCT_MAX_CACHED_ACCOUNTS", "500000")),
    )


app = FastAPI(title="Fraudstruct Transaction Monitoring API", version="1.0.1")
structuring_config = _config_from_environment()
engine = StreamingEngine(structuring_config=structuring_config)
RISK_REVIEW_THRESHOLD = float(os.getenv("FRAUDSTRUCT_RISK_REVIEW_THRESHOLD", "0.80"))
if not 0.0 <= RISK_REVIEW_THRESHOLD <= 1.0:
    raise ValueError("FRAUDSTRUCT_RISK_REVIEW_THRESHOLD must be between 0 and 1")


class TransactionEvent(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    source_account: str = Field(min_length=1, max_length=256)
    destination_account: str = Field(min_length=1, max_length=256)
    amount: float = Field(gt=0, allow_inf_nan=False)
    timestamp: Optional[datetime] = None
    channel: str = Field(default="", max_length=64)


class LabelEntry(BaseModel):
    account_id: str = Field(min_length=1, max_length=256)
    is_fraud: int = Field(ge=0, le=1)


class TrainingRequest(BaseModel):
    labels: List[LabelEntry] = Field(min_items=2)
    epochs: int = Field(default=800, ge=1, le=10000)
    learning_rate: float = Field(default=0.1, gt=0, le=1)


@app.get("/health")
def health():
    cfg = engine.structuring_detector.config
    return {
        "status": "ok",
        "active_edge_pairs": engine.structuring_detector.active_pair_count,
        "cached_risk_scores": len(engine.risk_score_store),
        "detector": {
            "window_seconds": cfg.window_seconds,
            "min_mules": cfg.min_mules,
            "minimum_total_amount": cfg.minimum_total_amount,
            "target_total_amount": cfg.target_total_amount,
            "max_leg_amount": cfg.max_leg_amount,
            "flow_conservation_tolerance": cfg.flow_conservation_tolerance,
        },
    }


@app.post("/v1/evaluate")
def evaluate_transaction(event: TransactionEvent):
    start_time = time.perf_counter()
    timestamp = event.timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    event_dict = {
        "transaction_id": event.transaction_id,
        "source_account": event.source_account,
        "destination_account": event.destination_account,
        "amount": event.amount,
        "timestamp": timestamp,
        "channel": event.channel,
    }
    try:
        features = engine.ingest(event_dict)
    except BufferError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    decision = "APPROVE"
    reasons = []
    alert = features["structuring_alert"]
    score = features["cached_risk_score"]
    if alert is not None:
        decision = "FLAG"
        reasons.append("S/K/W multi-hop flow pattern matched")
    elif score is not None and score >= RISK_REVIEW_THRESHOLD:
        decision = "REVIEW"
        reasons.append("Cached account risk score exceeded review threshold")

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "transaction_id": event.transaction_id,
        "decision": decision,
        "reasons": reasons,
        "latency_ms": round(latency_ms, 3),
        "latency_scope": "in-process request handler; excludes network and deployment infrastructure",
        "features": {
            "structuring_alert": alert,
            "cached_risk_score": score,
            "risk_score_is_cached": features["risk_score_is_cached"],
            "active_pair_count": features["active_pair_count"],
            "window_seconds": features["rolling_window_seconds"],
        },
    }


@app.post("/v1/train")
def train_graph_model(req: TrainingRequest):
    """Cold-path training; inference scores are cached before event requests."""
    if len(engine.graph_engine.graph) < 2:
        raise HTTPException(status_code=400, detail="At least two graph accounts are required.")
    labels_dict = {entry.account_id: entry.is_fraud for entry in req.labels}
    if len(set(labels_dict.values())) < 2:
        raise HTTPException(status_code=422, detail="Training labels must contain both classes.")
    try:
        model, nodes = train_gcn(
            engine.graph_engine,
            labels_dict,
            epochs=req.epochs,
            lr=req.learning_rate,
        )
        engine.load_gnn_model(model, nodes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "Success",
        "message": "Cold-path model trained; account scores are cached for constant-time lookup.",
        "trained_labeled_accounts": len(labels_dict),
        "cached_account_scores": len(engine.risk_score_store),
        "epochs": req.epochs,
    }


@app.post("/v1/reset")
def reset_engine():
    engine.reset()
    return {"status": "Success", "message": "Streaming and cached state cleared."}
