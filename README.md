# **Fraudstruct**

**Fraudstruct** is a high-performance Python framework for **real-time streaming feature stores and graph-adversarial machine learning in banking fraud detection**. It is designed to help Data Scientists and Risk Engineers simulate, detect, and harden models against **structuring (smurfing)**, **multi-hop money laundering**, and **threshold evasion** in transaction streams.

The library supports a production-ready **Three-Tier Architecture** that bridges deep offline Graph Neural Networks (GNNs) with sub-10ms real-time transaction authorization endpoints.

---

## **Why Fraudstruct Exists**

Traditional fraud models degrade rapidly because fraudsters actively adapt to rule-based thresholds (e.g., splitting a transaction into smaller amounts across proxy accounts/mules). 

Fraudstruct addresses these operational realities:
* **Coordinated Multi-Hop Fraud**: Fraudsters use network topologies to evade single-account checks.
* **Strict Latency Budgets**: Banks must authorize transactions in **<50ms**, making traditional graph neural network (GNN) convolution passes on the transaction hot-path infeasible.
* **Continuous Adaptive Evasion**: Evasion techniques evolve faster than weekly or monthly model retraining loops.

---

## **System Architecture Overview**

Fraudstruct resolves the latency-complexity trade-off using a **Three-Tier Lambda-style ML System**:

```
           [Core Banking / NIBSS / Switch]
                          │
         (1) Synchronous API Request (e.g. NIP/ISO 8583)
                          ▼
             ┌─────────────────────────┐
             │    HOT PATH (<10ms)     │
             │   REST API (FastAPI)    │◄─── [Loaded Model Weights]
             └────────────┬────────────┘
                          │ (Retrieves pre-computed GNN + temporal features)
                          ▼
                  ┌──────────────┐
                  │ Feature Cache│
                  └──────▲───────┘
                         │ (Updates features in real-time)
             ┌───────────┴─────────────┐
             │    WARM PATH (<1s)      │
             │  Streaming Event Loop   │◄─── [Transaction Log Stream]
             └───────────┬─────────────┘
                         │ (Aggregates historical subgraph structures)
                         ▼
             ┌─────────────────────────┐
             │    COLD PATH (Offline)  │
             │ NumPy SGC GNN Classifier│◄─── [Graph Attack Simulator]
             └─────────────────────────┘
```

1. **Hot Path (Synchronous Inference)**: A lightweight REST API that retrieves pre-computed graph features and rolling temporal stats, evaluates fast rules + GNN score thresholds, and responds in **<5ms** (satisfying strict banking SLAs).
2. **Warm Path (Near Real-Time Feature Store)**: A streaming engine that consumes transaction streams, updates the transaction graph structure (using `GraphEngine`), and updates rolling temporal windows in memory.
3. **Cold Path (Offline Training & Simulation)**: Generates synthetic multi-hop structuring attacks on the graph, trains a **Simplifying Graph Convolution (SGC)** GNN model, and hot-deploys updated model weights to the Hot Path.

---

## **Key Capabilities**

### **1. Graph representation & Feature Store**
* NetworkX-powered transaction graph engine (`GraphEngine`) that models accounts as nodes and transactions as edges.
* Real-time calculation of topological features (in-degree, out-degree, PageRank, running sums, and velocities).

### **2. Graph-Based Evasion Simulation**
* Multi-hop transaction-splitting simulator (`simulate_graph_splitting`). Splits a target amount $S$ from $A \rightarrow B$ across $K$ dynamic mule pathways, adding temporal spacing and amount jittering to mimic realistic money laundering.

### **3. Portability & Performance (Pure NumPy GNN)**
* Implementation of **Simplifying Graph Convolution (SGC)** (Wu et al., 2019) in pure NumPy, removing heavy compile dependencies (e.g. PyTorch Geometric or C++ bindings) and allowing seamless execution under Python 3.13.

### **4. Real-time REST API**
* FastAPI endpoint exposing `/v1/evaluate` for real-time transaction scoring and `/v1/train` to hot-deploy trained graph embeddings to the feature store.

---

## **Installation**

Install the library locally:

```bash
pip install -e .
```

### **Required Dependencies**
* `pandas`
* `numpy`
* `networkx`
* `fastapi`
* `httpx`

---

## **Quick Start**

### **1. Running the End-to-End Pipeline & Benchmark**
Fraudstruct comes with a comprehensive verification script that runs the entire warm, hot, and cold path loop and profiles API latency:

```bash
python test_pipeline.py
```

*Expected output log:*
```
==================================================
FRAUDSTRUCT PIPELINE VERIFICATION AND BENCHMARK
==================================================
[Step 1] Generating organic transaction traffic...
Generated 150 organic transactions.

[Step 2] Simulating adversarial smurfing path (A -> Mules -> B)...
Generated 12 attack path transactions.

[Step 3] Feeding organic stream to API evaluate endpoint...
Inference Latency Metric (ZENITH SLA check):
  - Average Latency: 2.447 ms
  - 95th Percentile: 3.407 ms

[Step 4] Feeding attack stream and checking for alerts...
ALERT TRIGGERED on TXN-ADV-46940! Decision: FLAG, Reasons: ['Debit velocity structuring threshold breached: Sum=600000.0, Count=6']

[Step 5] Triggering Cold Path GNN Training & Hot-Deployment...
Training response: {'status': 'Success', 'message': 'GNN model successfully trained and hot-deployed.', 'trained_nodes_count': 52}

[Step 6] Verifying active GNN blocking post-deployment...
Post-GNN deployment evaluation result for attacker:
  - Decision: BLOCK
  - Reasons: ['High GNN network anomaly score: 0.9650']
  - GNN Anomaly Score: 0.965
```

---

## **API Reference**

### **Evaluate Transaction**
Exposes a NIBSS Instant Payment (NIP) compliant payload structure:

* **Endpoint**: `POST /v1/evaluate`
* **Request Body**:
  ```json
  {
    "transaction_id": "TXN-90283",
    "source_account": "1029384756",
    "destination_account": "2093847561",
    "amount": 250000.0,
    "timestamp": "2026-06-26T17:00:00Z",
    "channel": "NIP"
  }
  ```
* **Response Body**:
  ```json
  {
    "transaction_id": "TXN-90283",
    "decision": "APPROVE",
    "reasons": [],
    "latency_ms": 2.447,
    "features": {
      "rolling_sum": 250000.0,
      "rolling_count": 1,
      "in_degree": 0,
      "out_degree": 1,
      "gnn_anomaly_score": 0.0
    }
  }
  ```

---

## **Academic Thesis Contributions**
This repository serves as the empirical codebase for an MSc thesis in Computer Science:
1. **Graph-Based Tabular Evasion Modeling**: Bridges graph topology with tabular business constraints by formalizing smurfing as a dynamic flow-splitting optimization problem.
2. **Decoupled Real-Time Streaming Graph Inference**: Proves that GNN node features can be successfully cached and served in transaction authorization switches under a `<10ms` SLA.
3. **Online Graph Adversarial Hardening**: Details a continuous learning framework that updates graph embeddings in response to dynamic topology shifts.

---

## **License**

MIT License
