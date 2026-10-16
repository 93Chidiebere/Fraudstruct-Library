"""Streaming state for S/K/W detection and cached cold-path risk scores."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from fraudstruct.detect.flow_structuring import (
    StreamingStructuringDetector,
    StructuringConfig,
)
from fraudstruct.engines.graph import GraphEngine


class StreamingEngine:
    """Incremental event ingestion; no graph feature recomputation per event."""

    def __init__(
        self,
        window_seconds: int = 3600,
        structuring_config: Optional[StructuringConfig] = None,
    ):
        config = structuring_config or StructuringConfig(window_seconds=window_seconds)
        self.structuring_detector = StreamingStructuringDetector(config)
        self.feature_store: Dict[str, dict] = OrderedDict()
        self.risk_score_store: Dict[str, float] = {}
        self.gnn_model = None
        self.gnn_nodes_mapping = []
        self.graph_engine = self._empty_graph()

    @staticmethod
    def _empty_graph():
        init_df = pd.DataFrame(
            columns=["source_account", "destination_account", "amount", "timestamp"]
        )
        return GraphEngine(init_df)

    def load_risk_scores(self, risk_scores: Dict[str, float]) -> None:
        """Atomically replace the account-score cache produced by cold-path work."""
        clean_scores = {}
        for account, score in risk_scores.items():
            value = float(score)
            if not 0.0 <= value <= 1.0:
                raise ValueError("cached risk scores must be in [0, 1]")
            clean_scores[str(account)] = value
            if len(clean_scores) > self.structuring_detector.config.max_cached_accounts:
                raise ValueError("risk-score cache exceeds max_cached_accounts")
        self.risk_score_store = clean_scores

    def load_gnn_model(self, model, nodes_mapping) -> None:
        """Score the current graph once and cache scores for the event path.

        This method is intended for cold-path use. ``ingest`` never invokes the
        model or recalculates graph features.
        """
        from fraudstruct.detect.gcn import prepare_graph_data

        features, nodes = prepare_graph_data(self.graph_engine)
        scores = model.predict_proba(features)
        self.load_risk_scores(dict(zip(nodes, scores)))
        self.gnn_model = model
        self.gnn_nodes_mapping = list(nodes_mapping)

    def ingest(self, event: dict):
        """Ingest one transaction and return its active-window features."""
        timestamp = event.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)

        normalized = dict(event)
        normalized["timestamp"] = timestamp
        source = str(normalized["source_account"])
        alert = self.structuring_detector.ingest(normalized)

        # Cached score lookup is constant-time. It may be stale until the
        # cold-path score cache is explicitly refreshed.
        cached_score = self.risk_score_store.get(source)
        features = {
            "structuring_alert": alert,
            "rolling_window_seconds": self.structuring_detector.config.window_seconds,
            "active_pair_count": self.structuring_detector.active_pair_count,
            "cached_risk_score": cached_score,
            "gnn_anomaly_score": cached_score if cached_score is not None else 0.0,
            "risk_score_is_cached": cached_score is not None,
            "last_updated": timestamp,
        }
        self.feature_store[source] = features
        self.feature_store.move_to_end(source)
        while len(self.feature_store) > self.structuring_detector.config.max_cached_accounts:
            self.feature_store.popitem(last=False)

        # Keep the graph current for later cold-path training. No graph metrics
        # or model inference are computed during this update.
        graph_row = {
            "source_account": source,
            "destination_account": str(normalized["destination_account"]),
            "amount": float(normalized["amount"]),
            "timestamp": timestamp,
        }
        self.graph_engine.update_graph(pd.DataFrame([graph_row]))
        self.graph_engine.prune_before(
            timestamp - pd.Timedelta(seconds=self.structuring_detector.config.window_seconds)
        )
        return features

    def reset(self) -> None:
        """Clear event-time state, graph state, and cached scores."""
        config = self.structuring_detector.config
        self.structuring_detector = StreamingStructuringDetector(config)
        self.feature_store.clear()
        self.risk_score_store.clear()
        self.gnn_model = None
        self.gnn_nodes_mapping = []
        self.graph_engine = self._empty_graph()
