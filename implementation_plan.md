# Combined Real-Time Streaming Feature Store & Graph Adversarial Fraud Detection (Thesis Architecture)

This plan outlines the architecture and implementation strategy for combining **Graph-based Adversarial Simulation (Option A)** and a **Real-Time Streaming Feature Engine (Option B)**, tailored for direct deployment in Nigerian banks (e.g., Zenith Bank via API integration with Finacle or T24).

---

## 1. Banking Integration & Deployment Architecture

For a Nigerian bank to adopt this system, the solution must satisfy strict Service Level Agreements (SLAs) during transaction authorization (typically `<50ms` response times for NIBSS Instant Payments or Card networks). 

We propose a **Three-Tier Lambda-style ML System**:

```
           [Core Banking / NIBSS / Switch]
                          │
         (1) Post Transaction / Sync API Call
                          ▼
             ┌─────────────────────────┐
             │    HOT PATH (<50ms)     │
             │   REST API (FastAPI)    │◄─── [Loaded Model Weights]
             └────────────┬────────────┘
                          │ (Retrieves pre-computed GNN + temporal features)
                          ▼
                  ┌──────────────┐
                  │ Redis Cache  │
                  └──────▲───────┘
                         │ (Updates features in real-time)
             ┌───────────┴─────────────┐
             │    WARM PATH (<1s)      │
             │   Bytewax Stream Engine │◄─── [CDC / Kafka Transaction Log]
             └───────────┬─────────────┘
                         │ (Aggregates historical subgraph structures)
                         ▼
             ┌─────────────────────────┐
             │    COLD PATH (Offline)  │
             │   PyTorch Geometric GNN │◄─── [Adversarial Graph Simulator]
             └─────────────────────────┘
```

1. **Hot Path (Synchronous Inference - API Endpoint)**:
   * **Mechanism**: A fast REST API (FastAPI) deployed inside the bank's intranet.
   * **Integration**: When Zenith Bank's Core Banking System (Finacle) or transaction switch (Postilion) receives a transaction, it queries this API.
   * **Latency**: `<20ms`. It retrieves pre-computed graph embeddings and temporal stats from Redis, runs a local GNN-derived tabular classifier (like a robust XGBoost or MLP model), and returns an `Approve / Flag / Decline` decision.

2. **Warm Path (Near Real-Time Feature Aggregator)**:
   * **Mechanism**: A **Bytewax** (Rust-backed Python stream processor) or **Faust** stream engine.
   * **Integration**: Consumes transaction events asynchronously from Kafka or database Change Data Capture (CDC).
   * **Function**: Computes rolling metrics (debit velocity, rolling sums) and updates the transaction subgraph topology, caching the results immediately in Redis.

3. **Cold Path (Offline GNN Training & Adversarial Simulation)**:
   * **Mechanism**: PyTorch Geometric + NetworkX.
   * **Function**: Runs graph-level adversarial simulations (multi-hop smurfing attacks). Trains a Graph Convolutional Network (GCN) to classify nodes (accounts) as compromised/mules. Extracts node embeddings and exports them to Redis.

---

## 2. Proposed Changes

We will refactor and expand the [Fraudstruct-Library](file:///C:/Users/Chidiebere/.gemini/antigravity/scratch/Fraudstruct-Library) structure:

### Component: Core & Engines
#### [MODIFY] [`core/engine.py`](file:///C:/Users/Chidiebere/.gemini/antigravity/scratch/Fraudstruct-Library/core/engine.py)
* Add support for a graph-based representation layer (`GraphEngine`).
#### [NEW] [`engines/graph.py`](file:///C:/Users/Chidiebere/.gemini/antigravity/scratch/Fraudstruct-Library/engines/graph.py)
* NetworkX wrapper to manage transaction subgraphs, compute node degrees, PageRank, and path characteristics.

### Component: Stream & Real-time Processing
#### [MODIFY] [`stream/engine.py`](file:///C:/Users/Chidiebere/.gemini/antigravity/scratch/Fraudstruct-Library/stream/engine.py)
* Integrate a real-time event loop that updates the `GraphEngine` and caches computed values.
#### [NEW] [`stream/api.py`](file:///C:/Users/Chidiebere/.gemini/antigravity/scratch/Fraudstruct-Library/stream/api.py)
* FastAPI server exposing `/v1/evaluate` to accept NIBSS-like payloads:
  ```json
  {
    "transaction_id": "TXN-1029384",
    "source_account": "1012345678",
    "destination_account": "2098765432",
    "amount": 250000.0,
    "timestamp": "2026-06-26T14:32:00Z",
    "channel": "NIP"
  }
  ```

### Component: AI Simulation & GNNs
#### [NEW] [`simulate/graph_attack.py`](file:///C:/Users/Chidiebere/.gemini/antigravity/scratch/Fraudstruct-Library/simulate/graph_attack.py)
* An adversarial simulator that takes a transaction graph and splits a target transaction of size $S$ across $K$ proxy nodes (multi-hop paths) using a graph search algorithm (Dijkstra or reinforcement learning pathfinding) to mimic realistic money laundering/structuring.
#### [NEW] [`detect/gcn.py`](file:///C:/Users/Chidiebere/.gemini/antigravity/scratch/Fraudstruct-Library/detect/gcn.py)
* A PyTorch Geometric GCN model that takes node features (rolling stats, degree) and edge lists to output node anomaly scores.

### Component: Training & Verification
#### [MODIFY] [`train/adversarial.py`](file:///C:/Users/Chidiebere/.gemini/antigravity/scratch/Fraudstruct-Library/train/adversarial.py)
* Fix the deprecated `.append()` bug and implement Graph Adversarial Training.

---

## 3. Academic Contributions to Literature

Executing this thesis will contribute three distinct innovations to the academic literature of financial machine learning and ML systems:

### Contribution A: Graph-Based Tabular Adversarial Attack Generation (Evasion Modeling)
* **Current Gap in Literature**: Most tabular adversarial attacks focus on perturbation of local features (e.g. modifying an account balance or transaction amount in isolation). Most Graph Neural Network (GNN) attacks focus on generic topology perturbation (adding/deleting random edges).
* **Thesis Contribution**: We introduce an attack algorithm that bridges graph topology and tabular domain constraints. It models structured fraud as a **constrained flow problem** (splitting a target amount $S$ across $K$ dynamic mule pathways over a temporal window $W$). This provides a new benchmark attack for evaluating financial GNN classifiers against realistic, multi-hop evasion strategies.

### Contribution B: Decoupled Real-Time Streaming Graph Inference Architecture (ML Systems / AI Infra)
* **Current Gap in Literature**: Academic papers on graph-based fraud detection evaluate GNNs in a static, offline batch setting (e.g., using Elliptic datasets). In practice, banks cannot run heavy, multi-hop graph neural network convolutions during the synchronous transaction authorization window (where SLAs are $<50\text{ms}$).
* **Thesis Contribution**: We propose and benchmark a novel **decoupled streaming architecture** (Lambda/Three-Tier) for graph representation learning. By separating the *Cold Path* (deep graph training and offline embedding generation) and *Warm Path* (real-time streaming feature aggregation into Redis) from the *Hot Path* (lightweight local inference), we prove that complex graph embeddings can be utilized in real-time banking pipelines without violating operational latency budgets.

### Contribution C: Online Graph Adversarial Hardening
* **Current Gap in Literature**: Existing GNN adversarial training protocols are static and require complete graph reconstruction, which fails in dynamic, streaming production systems.
* **Thesis Contribution**: We formulate a dynamic, online adversarial training framework for streaming transaction networks. We demonstrate how dynamically injected graph structuring attacks allow online models to build robust, adaptive decision boundaries against non-local evasion strategies as the transaction stream evolves.

---

## 4. Verification Plan

### Automated Tests
* Build mock unit tests simulating streaming payloads.
* Benchmark API response times (throughput and latency) under high concurrency to verify it fits bank SLAs.
* Verify GNN accuracy improvement when trained with the new graph adversarial attacks vs. standard baseline models.

### Manual Verification
* Deploy a mock banking pipeline: Run FastAPI, feed a synthetic transaction stream using a Python script, and test `/v1/evaluate` via curl/Postman.

