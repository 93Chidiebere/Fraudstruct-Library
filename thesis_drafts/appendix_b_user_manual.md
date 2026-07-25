# APPENDIX B: USER MANUAL (HOW TO USE)

This manual provides detailed instructions for operating the three core interfaces of the **Fraudstruct** framework. It is structured for both technical users (system developers, data scientists) and non-technical business users (compliance investigators, risk managers).

---

## 🛠️ Interface 1: The Core Banking REST API (Technical Users)
The REST API microservice is the synchronous gateway deployed inside the bank's local intranet. It interfaces directly with core banking platforms (Finacle, Temenos T24) and payment switches (Postilion) to authorize or block transactions.

### 1. Launching the API Server
Run the following command in the VS Code terminal to start the FastAPI server locally on port 8000:
```powershell
python -m uvicorn stream.api:app --reload --port 8000
```
*   **Access Swagger documentation:** Open your browser and navigate to `http://127.0.0.1:8000/docs` to test endpoints interactively.

### 2. Endpoint Specification

#### A. Real-Time Transaction Evaluation
*   **Path:** `POST /v1/evaluate`
*   **Description:** Evaluates an incoming transaction event through the Hot/Warm path and returns an instant decision (`APPROVE`, `FLAG`, or `BLOCK`).
*   **Request Headers:** `Content-Type: application/json`
*   **Request JSON Schema:**
    ```json
    {
      "transaction_id": "TXN-102938",
      "source_account": "ACC_ATTACKER_01",
      "destination_account": "ACC_MULE_05",
      "amount": 100000.00,
      "timestamp": "2026-07-25T12:00:00.000Z",
      "channel": "NIP"
    }
    ```
*   **Response JSON Schema (Flagged Transaction):**
    ```json
    {
      "transaction_id": "TXN-102938",
      "decision": "FLAG",
      "latency_ms": 2.45,
      "reasons": [
        "Dynamic Window: High inbound link count on destination node within 1 hour.",
        "Topological Anomaly: Inflow/Outflow ratio close to 1.0 (transit suspect)."
      ],
      "features": {
        "src_out_degree": 4,
        "dst_in_degree": 5,
        "gnn_anomaly_score": 0.865
      }
    }
    ```

#### B. Asynchronous GNN Cold-Path Retraining
*   **Path:** `POST /v1/train`
*   **Description:** Triggered when investigators label confirmed fraud/mule accounts. It retrains the pure-NumPy SGC GNN model and automatically hot-deploys the updated weights to the running in-memory engine.
*   **Request JSON Schema:**
    ```json
    {
      "labels": [
        {"account_id": "ACC_ATTACKER_01", "is_fraud": 1},
        {"account_id": "ACC_MULE_05", "is_fraud": 1},
        {"account_id": "ACC_RETAIL_99", "is_fraud": 0}
      ],
      "epochs": 30
    }
    ```
*   **Response JSON Schema:**
    ```json
    {
      "status": "success",
      "message": "SGC GNN retrained and hot-deployed successfully.",
      "metrics": {
        "epochs": 30,
        "training_loss": 0.0452,
        "val_recall": 0.965
      }
    }
    ```

---

## 🖥️ Interface 2: The Streamlit Compliance Dashboard (Non-Technical Users)
The Compliance Dashboard is a visual web application designed for fraud investigators and risk managers to monitor live alerts and manage model deployment.

### 1. Launching the Dashboard
Run the following command in the VS Code terminal:
```powershell
python -m streamlit run compliance_portal.py
```
*   **Access the interface:** The portal will automatically launch in your default web browser at `http://localhost:8501`.

### 2. Section Walkthrough

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COMPLIANCE TERMINAL TABS                        │
├───────────────┬──────────────────────┬──────────────────┬──────────────┤
│ 1. Dashboard  │ 2. Transaction Term. │ 3. Model Manager │ 4. Ecosystem │
└───────────────┴──────────────────────┴──────────────────┴──────────────┘
```

1.  **Dashboard Tab (Live Monitoring):**
    *   **KPI Metrics:** Displays total transactions processed, active flagged alerts, detected money mule networks, and real-time processing latency.
    *   **Live Alert Table:** Logs flagged transactions in real time, displaying transaction IDs, source/destination accounts, transaction channels, amounts, and alert reasons.
2.  **Transaction Terminal Tab (Manual Inspection):**
    *   Allows analysts to manually type in a transaction profile (sender, receiver, channel, amount) and click **"Evaluate Transaction"**.
    *   Displays the instant decision of the API Hot Path alongside visual risk gauge meters.
3.  **MLOps Model Manager Tab (Compliance Control):**
    *   Displays current model parameter statistics (GNN weights and biases).
    *   Provides a **"Retrain SGC Model"** button. Analysts can input labeled lists of verified mule networks, trigger the Cold Path training, and hot-deploy the new parameters directly from the UI without downtime.
4.  **Ecosystem Context Tab (Regulations):**
    *   Displays architectural boundaries showing how Fraudstruct functions inside the bank's gateway switch, complying with **CBN automated AML standards** and complementing centralized databases (such as **NIBSS Hawk**).

---

## 🐍 Interface 3: The Python SDK Library (Data Scientists & Researchers)
The `fraudstruct` SDK is a packaged Python library that data scientists can import directly into Jupyter Notebooks, pipeline schedulers, or PySpark clusters for offline analysis and stress-testing.

### 1. Installation
Ensure you are in the workspace folder and install the SDK:
```bash
pip install fraudstruct
```

### 2. Core Python SDK Modules

*   `fraudstruct.engines.graph.GraphEngine`: Maintains dynamic transaction networks and extracts PageRank, node degrees, and normalized adjacency matrices.
*   `fraudstruct.detect.gcn.NumPySGC`: Vectorized pure-NumPy Simplifying Graph Convolution classifier.
*   `fraudstruct.simulate.graph_attack.simulate_graph_splitting`: Generates realistic multi-hop smurfing paths obeying cash-flow conservation.
*   `fraudstruct.simulate.camouflage.apply_adversarial_camouflage`: Applies temporal delays and Gaussian amount noise to test model robustness.

### 3. End-to-End Experimentation Code Example

Here is how a data scientist imports the library to simulate an attack, construct the graph, and train the GNN:

```python
import numpy as np
from datetime import datetime
from fraudstruct.simulate.graph_attack import simulate_graph_splitting
from fraudstruct.simulate.camouflage import apply_adversarial_camouflage
from fraudstruct.engines.graph import GraphEngine
from fraudstruct.detect.gcn import NumPySGC

# Step 1: Simulate a Coordinated Smurfing Attack (Crash-Test Dummy)
attack_events = simulate_graph_splitting(
    attacker_id="ACC_ATTACKER_01",
    beneficiary_id="ACC_BENEFICIARY_01",
    target_sum=800000.0,
    num_mules=5,
    start_time=datetime.utcnow()
)

# Apply 5% amount noise to simulate adversarial camouflage
camouflaged_attack = apply_adversarial_camouflage(
    attack_events=attack_events,
    noise_std=0.05,
    merchant_list=["MERCHANT_SUPERMARKET"]
)

# Step 2: Feed Transactions into the GraphEngine
graph_engine = GraphEngine()
for tx in camouflaged_attack:
    graph_engine.add_transaction(
        source=tx["source_account"],
        target=tx["destination_account"],
        amount=tx["amount"]
    )

# Step 3: Extract Normalized Adjacency (S_adj) & Node Features
S_adj, node_list = graph_engine.get_normalized_adjacency()
node_to_idx = {node: idx for idx, node in enumerate(node_list)}
X_features = graph_engine.get_feature_matrix(node_list)

# Step 4: Define Labels (1 = Fraudulent Mule/Attacker, 0 = Clean)
y_labels = np.zeros(len(node_list))
for tx in camouflaged_attack:
    y_labels[node_to_idx[tx["source_account"]]] = 1
    y_labels[node_to_idx[tx["destination_account"]]] = 1

# Step 5: Train SGC GNN Node Classifier (K=2 Hops)
sgc = NumPySGC(k_hops=2, lr=0.05, l2_reg=0.01)
sgc.fit(S_adj, X_features, y_labels, epochs=40)

# Step 6: Evaluate Attacker Node Anomaly Score
attacker_idx = node_to_idx["ACC_ATTACKER_01"]
anomaly_prob = sgc.predict_proba(S_adj, X_features)[attacker_idx]
print(f"Attacker Anomaly Probability: {anomaly_prob:.4f}")
# Output: Attacker Anomaly Probability: 0.9854
```
