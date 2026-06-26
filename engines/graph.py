import networkx as nx
import pandas as pd
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
                ts = pd.to_datetime(ts)
                
            self.graph.add_edge(
                u, v,
                amount=amount,
                timestamp=ts,
                is_fraud=row.get("is_fraud", False),
                is_adversarial=row.get("is_adversarial", False)
            )

    def get_node_features(self):
        """
        Compute standard graph metrics for all nodes.
        These serve as node features for the downstream GNN models.
        """
        try:
            pagerank = nx.pagerank(self.graph) if len(self.graph) > 0 else {}
        except Exception:
            try:
                simple_g = nx.DiGraph(self.graph)
                pagerank = nx.pagerank(simple_g)
            except Exception:
                pagerank = {node: 1.0 / max(1, len(self.graph)) for node in self.graph.nodes()}
        
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
