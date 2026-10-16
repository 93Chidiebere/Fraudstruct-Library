import networkx as nx
import pandas as pd
from collections import deque
from datetime import datetime, timezone
from fraudstruct.core.engine import FraudstructEngine

class GraphEngine(FraudstructEngine):
    def __init__(self, df, source_col="source_account", dest_col="destination_account", amount_col="amount", time_col="timestamp"):
        super().__init__(df)
        self.engine_type = "graph"
        self.source_col = source_col
        self.dest_col = dest_col
        self.amount_col = amount_col
        self.time_col = time_col
        
        self.graph = nx.MultiDiGraph()
        self._edge_expiry = deque()
        self._last_timestamp = None
        self.update_graph(df)

    def update_graph(self, df):
        """Add new transaction rows to the NetworkX graph."""
        if df is None or len(df) == 0:
            return
            
        for _, row in df.iterrows():
            u = str(row[self.source_col])
            v = str(row[self.dest_col])
            amount = float(row[self.amount_col])
            ts = row[self.time_col]
            if isinstance(ts, str):
                ts = pd.to_datetime(ts, utc=True).to_pydatetime()
            elif isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if isinstance(ts, datetime):
                ts = ts.astimezone(timezone.utc)
                if self._last_timestamp is not None and ts < self._last_timestamp:
                    raise ValueError("graph events must arrive in nondecreasing timestamp order")
                self._last_timestamp = ts
                
            edge_key = self.graph.add_edge(
                u, v,
                amount=amount,
                timestamp=ts,
                is_fraud=row.get("is_fraud", False),
                is_adversarial=row.get("is_adversarial", False)
            )
            if isinstance(ts, datetime):
                self._edge_expiry.append((ts, u, v, edge_key))

    def prune_before(self, cutoff):
        """Remove expired edges using the chronological ingestion queue."""
        if isinstance(cutoff, str):
            cutoff = pd.to_datetime(cutoff, utc=True).to_pydatetime()
        elif isinstance(cutoff, pd.Timestamp):
            cutoff = cutoff.to_pydatetime()
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        cutoff = cutoff.astimezone(timezone.utc)
        removed = 0
        while self._edge_expiry and self._edge_expiry[0][0] < cutoff:
            _, source, destination, edge_key = self._edge_expiry.popleft()
            if self.graph.has_edge(source, destination, edge_key):
                self.graph.remove_edge(source, destination, edge_key)
                removed += 1
                for node in (source, destination):
                    if node in self.graph and self.graph.degree(node) == 0:
                        self.graph.remove_node(node)
        return removed

    def get_node_features(self):
        """
        Compute standard graph metrics for all nodes.
        These serve as node features for the downstream GNN models.
        """
        pagerank = nx.pagerank(self.graph) if len(self.graph) > 0 else {}
        
        features = []
        for node in self.graph.nodes():
            in_edges = self.graph.in_edges(node, data=True)
            out_edges = self.graph.out_edges(node, data=True)
            
            in_degree = len(in_edges)
            out_degree = len(out_edges)
            
            in_amounts = [d["amount"] for _, _, d in in_edges]
            out_amounts = [d["amount"] for _, _, d in out_edges]
            
            in_sum = sum(in_amounts)
            out_sum = sum(out_amounts)
            
            features.append({
                "entity_id": node,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "in_amount_sum": in_sum,
                "out_amount_sum": out_sum,
                "pagerank": pagerank.get(node, 0.0)
            })
            
        return pd.DataFrame(features)
