import numpy as np
import pandas as pd
import networkx as nx

class NumPyLogisticRegression:
    """
    A pure NumPy implementation of Logistic Regression with L2 regularization.
    Eliminates dependencies on scikit-learn and compiled C++ libraries.
    """
    def __init__(self, lr=0.1, l2_reg=0.01, max_iter=500):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be a finite positive value")
        if not np.isfinite(l2_reg) or l2_reg < 0:
            raise ValueError("l2_reg must be finite and non-negative")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
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

    def train(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be a non-empty 2D array aligned with y")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("training data must contain only finite values")
        if not np.isin(y, [0.0, 1.0]).all() or len(np.unique(y)) != 2:
            raise ValueError("training labels must contain both binary classes 0 and 1")
        N, D = X.shape
        if sample_weight is None:
            positives = max(1, int(y.sum()))
            negatives = N - positives
            sample_weight = np.where(y == 1, N / (2 * positives), N / (2 * negatives))
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(sample_weight) != N or not np.isfinite(sample_weight).all():
                raise ValueError("sample_weight must be finite and aligned with y")
            if np.any(sample_weight < 0) or sample_weight.sum() <= 0:
                raise ValueError("sample_weight must be non-negative with positive sum")
        sample_weight = sample_weight / sample_weight.mean()
        self.w = np.zeros(D)
        self.b = 0.0
        
        for _ in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.w) + self.b
            p = self._sigmoid(z)
            
            # Gradients
            err = (p - y) * sample_weight
            dw = (1.0 / N) * np.dot(X.T, err) + self.l2_reg * self.w
            db = (1.0 / N) * np.sum(err)
            
            # Updates
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
        self.is_trained = True

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("logistic model has not been trained")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.w):
            raise ValueError("prediction feature width does not match the trained model")
        if not np.isfinite(X).all():
            raise ValueError("prediction features must be finite")
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)

class NumPySGCModel:
    """
    Simplifying Graph Convolution (SGC) GNN model (Wu et al., 2019)
    implemented completely in pure NumPy.
    """
    def __init__(self, in_features, lr=0.1, max_iter=800):
        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError("in_features must be a positive integer")
        self.in_features = in_features
        self.classifier = NumPyLogisticRegression(lr=lr, l2_reg=0.01, max_iter=max_iter)
        self.is_trained = False

    def train(self, x, y):
        if np.asarray(x).ndim != 2 or np.asarray(x).shape[1] != self.in_features:
            raise ValueError("training feature width does not match model configuration")
        self.classifier.train(x, y)
        self.is_trained = True

    def predict_proba(self, x):
        if not self.is_trained:
            raise RuntimeError("SGC model has not been trained")
        if np.asarray(x).ndim != 2 or np.asarray(x).shape[1] != self.in_features:
            raise ValueError("prediction feature width does not match the trained model")
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
        
    # Sparse edge-list representation avoids allocating a dense N x N matrix.
    edge_src = []
    edge_dst = []
    for u, v in g.edges():
        edge_src.append(node_to_idx[str(u)])
        edge_dst.append(node_to_idx[str(v)])
    edge_src = np.asarray(edge_src + list(range(num_nodes)), dtype=np.int64)
    edge_dst = np.asarray(edge_dst + list(range(num_nodes)), dtype=np.int64)
    degree = np.bincount(edge_src, minlength=num_nodes).astype(float)
    edge_weight = 1.0 / np.sqrt(degree[edge_src] * degree[edge_dst])
    
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
        next_x = np.zeros_like(curr_x)
        np.add.at(next_x, edge_src, edge_weight[:, None] * curr_x[edge_dst])
        curr_x = next_x
        propagated.append(curr_x)
        
    features = np.hstack(propagated)
    return features, nodes

def train_gcn(graph_engine, labels_dict, epochs=800, lr=0.1):
    """
    Trains the SGC model using propagated graph features.
    """
    features, nodes = prepare_graph_data(graph_engine)
    
    # Align labels with node ordering
    # Filter for nodes that are labeled
    labeled_indices = [i for i, n in enumerate(nodes) if n in labels_dict]
    if not labeled_indices:
        raise ValueError("no supplied labels match graph nodes")

    X_train = features[labeled_indices]
    y_train = np.asarray([labels_dict[nodes[i]] for i in labeled_indices], dtype=np.float32)
    if not np.isin(y_train, [0.0, 1.0]).all() or len(np.unique(y_train)) != 2:
        raise ValueError("labels matching graph nodes must contain both binary classes")
    model = NumPySGCModel(in_features=features.shape[1], lr=lr, max_iter=epochs)
    model.train(X_train, y_train)
    return model, nodes
