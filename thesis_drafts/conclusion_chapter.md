# CHAPTER 6: CONCLUSION AND FUTURE WORK

## 6.1 Research Summary

The modernization of banking networks and the introduction of real-time inter-bank payment switches (such as NIBSS NIP in Nigeria) have created a critical security challenge: anti-money laundering (AML) and transaction structuring checks must now occur in-flight during the transaction authorization window. This requirement was codified by the Central Bank of Nigeria (CBN) in its March 10, 2026 circular, *"Baseline Standards for Automated Anti-Money Laundering (AML) Solutions"*, which mandates automated detection and annual independent model validation across the financial sector. 

While Graph Neural Networks (GNNs) represent the state-of-the-art in detecting coordinated structuring (smurfing), their computational complexity has historically restricted their use to asynchronous offline batch audits, leaving banks vulnerable to real-time asset flight. Furthermore, banking mainframes impose strict security policies that reject heavy, unverified deep learning frameworks like PyTorch or PyTorch Geometric.

This study designed, implemented, and benchmarked **Fraudstruct**—a hybrid real-time streaming feature store and graph-adversarial machine learning library—to resolve the latency-complexity bottleneck. Fraudstruct introduces a decoupled three-tier Lambda-style architecture separating real-time rules evaluation (Hot Path) and streaming updates (Warm Path) from GNN retraining (Cold Path). We implemented a Simplifying Graph Convolution (SGC) GNN node classifier in pure, vectorized NumPy, guaranteeing zero compiled C++ or PyTorch dependencies.

Empirical evaluation verified that Fraudstruct processes transaction requests with an average latency of **2.45 ms** (and **3.41 ms** at the 95th percentile), satisfying the strict 50ms payment switch SLA by utilizing only 4.9% of the latency budget. Following the generation of flow-constrained graph splitting attacks and hot deployment of SGC parameters, the system successfully identified the attacker and blocked subsequent transfer attempts with **96.5%** confidence.

---

## 6.2 Key Research Contributions

This study contributes three distinct innovations to the academic literature of financial machine learning and systems engineering:

1. **Decoupled Graph ML Systems Design**: We present the design and validation of a three-tier streaming pipeline that decouples GNN representation learning. By pre-computing convolving operators in the background and serving them from a Redis-like cache in the Hot Path, we prove that graph-level topological context can be utilized synchronously during transaction authorization.
2. **Environment-Portability Algorithmic Adaptations**: We implement SGC in pure, vectorized NumPy. This demonstrates that GNN node classification can be run efficiently on CPU-only banking servers without requiring unverified compiled libraries or dedicated GPU infrastructure, addressing a major barrier to real-world corporate banking adoption.
3. **Regulator-Aligned Model Governance**: We design a streaming drift detection module and a parameterized audit reporting system. This provides a repeatable framework for banks to comply with the CBN's annual independent validation directives.

---

## 6.3 Limitations of the Study

While Fraudstruct satisfies the operational and detection objectives, several limitations are noted:
* **Simplified Cache Simulation**: The Hot Path cache was simulated using in-memory Python dictionaries. In a production deployment, this must be replaced with a distributed Redis cluster, introducing network round-trip overhead (estimated at 1.0–2.0 ms).
* **Static Memory Deques**: In-memory temporal window buffers were capped at a static length of 100 entries. In high-frequency corporate client accounts, this limit may result in the eviction of relevant historical transaction events, requiring dynamic sliding window resizing.
* **Supervised Label Dependency**: The GNN classifier relies on supervised labels from back-office investigations to train the SGC weights, which may introduce a delay in detecting entirely new, unknown fraud patterns.

---

## 6.4 Future Work and Recommendations

To build upon this research, several areas are recommended for future exploration:

1. **Collaborative Privacy-Preserving AML via FHE**: Integrating **Fully Homomorphic Encryption (FHE)** with the SGC GNN. This would allow separate commercial banks (such as Zenith Bank and Access Bank) to pool transaction graphs and run collaborative message-passing checks without disclosing sensitive raw customer data or BVN identifiers.
2. **Distributed Stream Scalability (Flink/Bytewax)**: Transitioning the Warm Path from a single-threaded Python event loop to a distributed stateful stream processing engine (e.g. Apache Flink or Bytewax) to evaluate performance under peak industrial throughput (exceeding 10,000 transactions per second).
3. **Reinforcement Learning Attack Pathfinders**: Developing an autonomous Reinforcement Learning (RL) agent inside the simulator to dynamically discover the optimal splitting paths across mule nodes, further hardening SGC classifiers against adaptive adversarial evasion.
