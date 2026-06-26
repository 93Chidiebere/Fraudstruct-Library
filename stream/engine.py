from collections import defaultdict, deque
from datetime import datetime, timedelta
import pandas as pd
from fraudstruct.engines.graph import GraphEngine
from fraudstruct.detect.gcn import prepare_graph_data

class StreamingEngine:
    """
    High-performance near-real-time streaming engine that aggregates temporal 
    features and manages the transaction graph structure.
    """
    def __init__(self, window_seconds: int):
        self.window = timedelta(seconds=window_seconds)
        self.buffers = defaultdict(deque)
        
        # In-memory transaction database to initialize the GraphEngine
        init_df = pd.DataFrame(columns=["source_account", "destination_account", "amount", "timestamp"])
        self.graph_engine = GraphEngine(init_df)
        
        # Local feature store simulating Redis
        self.feature_store = {}
        self.gnn_model = None
        self.gnn_nodes_mapping = []

    def load_gnn_model(self, model, nodes_mapping):
        """Loads a trained GNN model to calculate running node embeddings/probabilities."""
        self.gnn_model = model
        self.gnn_nodes_mapping = nodes_mapping

    def ingest(self, event: dict):
        """
        Ingest a transaction event.
        event = {
            'source_account': str,
            'destination_account': str,
            'amount': float,
            'timestamp': datetime
        }
        """
        source = str(event["source_account"])
        dest = str(event["destination_account"])
        amount = float(event["amount"])
        now = event["timestamp"]
        if isinstance(now, str):
            now = pd.to_datetime(now)

        # 1. Update the warm-path graph structure
        event_df = pd.DataFrame([event])
        self.graph_engine.update_graph(event_df)

        # 2. Update sliding temporal windows for source account
        self.buffers[source].append(event)
        while self.buffers[source]:
            first_ts = self.buffers[source][0]["timestamp"]
            if isinstance(first_ts, str):
                first_ts = pd.to_datetime(first_ts)
            if now - first_ts > self.window:
                self.buffers[source].popleft()
            else:
                break

        # 3. Compute running temporal features (velocity, sum)
        source_events = list(self.buffers[source])
        rolling_count = len(source_events)
        rolling_sum = sum(e["amount"] for e in source_events)

        # 4. Fetch topological graph features
        g = self.graph_engine.graph
        in_degree = len(g.in_edges(source))
        out_degree = len(g.out_edges(source))
        
        # 5. Extract GNN score if model is loaded
        gnn_score = 0.0
        if self.gnn_model is not None and source in self.gnn_nodes_mapping:
            try:
                # Prepare graph features
                features, nodes = prepare_graph_data(self.graph_engine)
                node_idx = nodes.index(source)
                node_features = features[node_idx].reshape(1, -1)
                gnn_score = float(self.gnn_model.predict_proba(node_features)[0])
            except Exception:
                gnn_score = 0.0

        # 6. Save computed features to Redis-like feature store
        features = {
            "rolling_count": rolling_count,
            "rolling_sum": rolling_sum,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "gnn_anomaly_score": gnn_score,
            "last_updated": now
        }
        self.feature_store[source] = features

        return features
