import numpy as np
import pandas as pd
import networkx as nx

class NumPyLogisticRegression:
    """
    A pure NumPy implementation of Logistic Regression with L2 regularization.
    Eliminates dependencies on scikit-learn and compiled C++ libraries.
    """
    def __init__(self, lr=0.1, l2_reg=0.01, max_iter=500):
        self.lr = lr
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.w = None
        self.b = 0.0
        self.is_trained = False

    def _sigmoid(self, z):
        # Clip z to avoid overflow issues
        z = np.clip(z, -15, 15)
        return 1.0 / (1.0 + np.exp(-z))

    def train(self, X, y):
        N, D = X.shape
        self.w = np.zeros(D)
        self.b = 0.0
        
        for _ in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.w) + self.b
            p = self._sigmoid(z)
            
            # Gradients
            dw = (1.0 / N) * np.dot(X.T, (p - y)) + self.l2_reg * self.w
            db = (1.0 / N) * np.sum(p - y)
            
            # Updates
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
        self.is_trained = True

    def predict_proba(self, X):
        if not self.is_trained:
            return np.zeros(len(X))
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)

class NumPySGCModel:
    """
    Simplifying Graph Convolution (SGC) GNN model (Wu et al., 2019)
    implemented completely in pure NumPy.
    """
    def __init__(self, in_features):
        self.classifier = NumPyLogisticRegression(lr=0.1, l2_reg=0.01, max_iter=800)
        self.is_trained = False

    def train(self, x, y):
        self.classifier.train(x, y)
        self.is_trained = True

    def predict_proba(self, x):
        if not self.is_trained:
            return np.zeros(len(x))
        return self.classifier.predict_proba(x)

def prepare_graph_data(graph_engine, k=2):
    """
    Computes normalized adjacency matrix and propagates node features for K steps.
    Returns:
        propagated_features (ndarray): Node features of shape (N, F * (K+1))
        nodes (list): Node names in order.
    """
    g = graph_engine.graph
    nodes = list(g.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    if num_nodes == 0:
        return np.empty((0, 0)), []
        
    # 1. Adjacency matrix with self-loops
    adj = np.eye(num_nodes)
    for u, v, data in g.edges(data=True):
        u_idx = node_to_idx[str(u)]
        v_idx = node_to_idx[str(v)]
        adj[u_idx, v_idx] += 1.0
        
    # Symmetric normalization: D^-0.5 * A_tilde * D^-0.5
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    d_mat = np.diag(deg_inv_sqrt)
    norm_adj = d_mat.dot(adj).dot(d_mat)
    
    # 2. Extract Base Node Features
    node_feat_df = graph_engine.get_node_features()
    node_feat_df = node_feat_df.set_index("entity_id").loc[nodes].fillna(0)
    
    feat_cols = ["in_degree", "out_degree", "in_amount_sum", "out_amount_sum", "pagerank"]
    x = node_feat_df[feat_cols].values
    
    # Standardize features
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-6
    x_norm = (x - x_mean) / x_std
    
    # 3. Propagate features K steps
    propagated = [x_norm]
    curr_x = x_norm
    for _ in range(k):
        curr_x = norm_adj.dot(curr_x)
        propagated.append(curr_x)
        
    features = np.hstack(propagated)
    return features, nodes

def train_gcn(graph_engine, labels_dict, epochs=0, lr=0.0):
    """
    Trains the SGC model using propagated graph features.
    """
    features, nodes = prepare_graph_data(graph_engine)
    
    # Align labels with node ordering
    y = np.array([labels_dict.get(n, 0) for n in nodes], dtype=np.float32)
    
    # Filter for nodes that are labeled
    labeled_indices = [i for i, n in enumerate(nodes) if n in labels_dict]
    
    model = NumPySGCModel(in_features=features.shape[1])
    
    if len(labeled_indices) == 0:
        return model, nodes
        
    X_train = features[labeled_indices]
    y_train = y[labeled_indices]
    
    model.train(X_train, y_train)
    return model, nodes
