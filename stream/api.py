import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from fraudstruct.stream.engine import StreamingEngine
from fraudstruct.detect.gcn import train_gcn

app = FastAPI(title="Fraudstruct Transaction Authorization API", version="1.0")

# Initialize Streaming Engine (1 Hour sliding window)
engine = StreamingEngine(window_seconds=3600)

class TransactionEvent(BaseModel):
    transaction_id: str
    source_account: str
    destination_account: str
    amount: float
    timestamp: Optional[str] = None
    channel: str = "NIP"

class LabelEntry(BaseModel):
    account_id: str
    is_fraud: int

class TrainingRequest(BaseModel):
    labels: List[LabelEntry]
    epochs: int = 20

@app.post("/v1/evaluate")
def evaluate_transaction(event: TransactionEvent):
    start_time = time.perf_counter()
    
    if event.timestamp is None:
        ts = datetime.utcnow()
    else:
        try:
            ts = pd.to_datetime(event.timestamp)
        except Exception:
            ts = datetime.utcnow()
            
    event_dict = {
        "source_account": event.source_account,
        "destination_account": event.destination_account,
        "amount": event.amount,
        "timestamp": ts,
        "transaction_id": event.transaction_id,
        "channel": event.channel
    }
    
    # Warm Path: Ingest into streaming engine (updates graph and computes features)
    features = engine.ingest(event_dict)
    
    # Hot Path: Evaluate rules and GNN score
    rolling_sum = features.get("rolling_sum", 0.0)
    rolling_count = features.get("rolling_count", 0)
    gnn_anomaly = features.get("gnn_anomaly_score", 0.0)
    
    # Policy Decision
    decision = "APPROVE"
    reasons = []
    
    if gnn_anomaly > 0.8:
        decision = "BLOCK"
        reasons.append(f"High GNN network anomaly score: {gnn_anomaly:.4f}")
    elif rolling_sum >= 500_000.0 and rolling_count >= 5:
        decision = "FLAG"
        reasons.append(f"Debit velocity structuring threshold breached: Sum={rolling_sum}, Count={rolling_count}")
    elif rolling_sum >= 2_000_000.0:
        decision = "BLOCK"
        reasons.append(f"Cumulative debit limit exceeded: Sum={rolling_sum}")
        
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        "transaction_id": event.transaction_id,
        "decision": decision,
        "reasons": reasons,
        "latency_ms": round(latency_ms, 3),
        "features": {
            "rolling_sum": rolling_sum,
            "rolling_count": rolling_count,
            "in_degree": features.get("in_degree", 0),
            "out_degree": features.get("out_degree", 0),
            "gnn_anomaly_score": round(gnn_anomaly, 4)
        }
    }

@app.post("/v1/train")
def train_graph_model(req: TrainingRequest):
    """
    Simulates the Cold Path retraining loop.
    Fits the GNN GCN model on the current graph and deploys it to the Hot Path (StreamingEngine).
    """
    if len(engine.graph_engine.graph) == 0:
        raise HTTPException(status_code=400, detail="Cannot train model on empty graph.")
        
    labels_dict = {entry.account_id: entry.is_fraud for entry in req.labels}
    
    try:
        model, nodes = train_gcn(engine.graph_engine, labels_dict, epochs=req.epochs)
        engine.load_gnn_model(model, nodes)
        return {
            "status": "Success",
            "message": "GNN model successfully trained and hot-deployed.",
            "trained_nodes_count": len(nodes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/v1/reset")
def reset_engine():
    global engine
    engine.buffers.clear()
    engine.feature_store.clear()
    # Reinitialize the GraphEngine
    import pandas as pd
    from fraudstruct.engines.graph import GraphEngine
    init_df = pd.DataFrame(columns=["source_account", "destination_account", "amount", "timestamp"])
    engine.graph_engine = GraphEngine(init_df)
    engine.gnn_model = None
    engine.gnn_nodes_mapping = []
    return {"status": "Success", "message": "Streaming engine buffers and graph state cleared."}
