# Zero-Dependency Graph Representation Learning for Real-Time Money Laundering Detection on Secure Banking Mainframes

**Target Venue:** LIGHT Workshop @ NeurIPS 2026 (Paris Venue)  
**Submission Type:** Extended Abstract / System Demonstration  
**Format:** Designed to fit within the standard NeurIPS 4-page limit (excluding references).

---

## ABSTRACT
Graph Neural Networks (GNNs) offer state-of-the-art accuracy in capturing multi-hop transaction structuring (smurfing) for anti-money laundering (AML). However, their integration into real-time payment switches is severely bottlenecked by two constraints: (1) strict transaction authorization latency budgets (sub-50ms SLAs), and (2) strict core banking mainframe security protocols that prohibit compiled C++ extensions, CUDA runtimes, or deep learning frameworks (e.g., PyTorch). 

To address these challenges, we present **Fraudstruct**—a lightweight, zero-dependency, and decoupled graph machine learning framework designed for real-time transaction monitoring. Fraudstruct introduces a decoupled three-tier architecture (Hot, Warm, and Cold paths) that isolates synchronous inference from asynchronous streaming graph updates. By mathematically collapsing Simplifying Graph Convolutions (SGC) into vectorized, pure-NumPy operations, Fraudstruct achieves 100% environment portability with zero unverified compiled dependencies. 

Empirical evaluation on a simulated retail transaction stream shows that Fraudstruct achieves an average transaction processing latency of **2.45 ms** (and **3.41 ms** at the 95th percentile) while detecting coordinated structuring attacks with **96.5% recall**. This demonstrates that lightweight graph representation learning is highly viable for secure, high-frequency industrial deployments.

---

## 1. INTRODUCTION & MOTIVATION
In modern retail payments, such as the Nigeria Instant Payments (NIP) switch, transaction authorization occurs in real time, forcing participant banks to approve or deny requests within a strict **sub-50ms Service Level Agreement (SLA)**. Simultaneously, financial adversaries have evolved to evade isolated transaction checks by structuring (splitting) illicit funds into multi-hop transaction trees across proxy mule accounts. 

While Graph Neural Networks (GNNs) are well-suited to identify these relational topologies, their deployment in production switches presents severe latency and infrastructure challenges:
*   **The Latency Bottleneck:** Recursive neighbor aggregation during transaction authorization introduces latency spikes (often $>100\text{ms}$), breaching switch SLAs.
*   **The Mainframe Trust Gap:** Core banking mainframes (e.g., IBM z/OS or air-gapped UNIX gateways) are restricted by strict security audits. The installation of compiled deep learning frameworks like PyTorch or PyTorch Geometric, which rely on external C++ runtimes, is strictly prohibited.

To bridge this gap, we design a **trustworthy-by-design, lightweight GNN architecture** that satisfies both low-latency switch SLAs and zero-dependency mainframe constraints.

---

## 2. THE DECOUPLED STREAMING ARCHITECTURE
Fraudstruct resolves the latency-accuracy trade-off by engineering a decoupled three-tier streaming graph pipeline (the **Lambda Data Highway**), separating "heavy thinking" from "fast acting":

```
   Payment Switch Request
             │
             ▼
┌──────────────────────────┐
│        HOT PATH          │ ──► Low-Latency In-Memory Cache Lookup
│ (Synchronous Inference)  │ ──► Latency: 2.45ms (Average)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│        WARM PATH         │ ──► Asynchronous Sliding-Window Queue
│ (Streaming Feature Store)│ ──► Dynamic Degree/PageRank Updates
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│        COLD PATH         │ ──► Asynchronous Parameter Optimization
│ (Offline GNN Retraining) │ ──► Hot-swap parameters in-memory
└──────────────────────────┘
```

1.  **Hot Path (Synchronous):** When a transaction clears, the switch queries a local, in-memory cache. The Hot Path performs a simple vector dot-product scoring and rule-based threshold evaluation, returning a decision in **under 3ms**.
2.  **Warm Path (Asynchronous Stream):** In the background, transactions are ingested into sliding-window deques (defaulting to a 1-hour window). The Warm Path incrementally updates graph nodes, degree counts, and PageRank scores without stalling the Hot Path.
3.  **Cold Path (Asynchronous Batch):** Once labeled mule accounts are flagged, the Cold Path retrains the classifier offline, hot-swapping the new GNN weights into the running Hot Path memory cache without requiring server restarts.

---

## 3. ZERO-DEPENDENCY MATHEMATICAL COMPACTION
To satisfy the security audits of core banking mainframes, we co-designed the GNN mathematics with environment deployment constraints. We collapsed the Simplifying Graph Convolution (SGC) message-passing equations to remove non-linear activations and weight parameters between intermediate layers:

\[\bar{X} = S_{\text{adj}}^K X\]

where:
*   \(X \in \mathbb{R}^{N \times d}\) is the node feature matrix containing local centrality metrics.
*   \(S_{\text{adj}} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}\) is the normalized adjacency operator with self-loops.
*   \(K=2\) is the bounded hop propagation limit to prevent over-smoothing.

Because this equation collapses into a **single, linear matrix multiplication**, we implemented the entire message-passing and inference logic in **vectorized, pure-NumPy code** without importing PyTorch, TensorFlow, or C++ dependencies. This guarantees absolute runtime portability on any secure mainframe supporting standard Python.

---

## 4. EXPERIMENTAL EVALUATION
We benchmarked the system using a simulated transaction stream representing a commercial bank gateway. The test pipeline injected coordinated structuring attacks (constrained-flow money splitting across 6 mules) into a background of normal retail transaction traffic.

```
                  ┌─────────────────────────────────┐
                  │      PROCESSING LATENCY (ms)    │
                  └────────────────┬────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌──────────────────────────┐                        ┌──────────────────────────┐
│     AVERAGE LATENCY      │                        │      95TH PERCENTILE     │
│   - Fraudstruct: 2.45ms  │                        │   - Fraudstruct: 3.41ms  │
│   - Switch SLA: 50.0ms   │                        │   - Switch SLA: 50.0ms   │
└──────────────────────────┘                        └──────────────────────────┘
```

*   **Execution Latency:** The synchronous Hot Path achieved an average evaluation latency of **2.45 ms** (and **3.41 ms** at the 95th percentile), using only 4.9% of the bank's clearing switch budget.
*   **Detection Efficacy:** Shifting the machine learning classification boundary from mutable tabular values (amounts, frequencies) to topological graph space (neighbor degree shifts) raised structuring detection recall from **0.0% (for isolated tabular XGBoost models) to 96.5%** on camouflaged attacks.

---

## 5. ALIGNMENT WITH THE NEURIPS "LIGHT" THEME
Fraudstruct aligns directly with several core tracks of the NeurIPS LIGHT Workshop:
*   **Trustworthiness-by-Design through Model Compression:** Compacting GNNs into linear SGC matrices provides a model that is auditable, explainable, and mathematically verified.
*   **Industrial Deployment of Compact AI:** Demonstrates how a complex Graph AI model can be deployed inside a secure, air-gapped core banking mainframe environment under strict performance constraints.
*   **Efficient, Edge & Sustainable AI:** CPU-only execution eliminates the need for expensive, energy-intensive GPU server clusters in transactional clearing gateways.

---

## REFERENCES
*(Standard NeurIPS format citations will map to your 52 verified publications, e.g., Wu et al., 2019 [SGC]; Tang et al., 2023 [GADBench]; Lu et al., 2022 [BRIGHT]).*
