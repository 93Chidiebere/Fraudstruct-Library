# THESIS TITLE PAGE

# Fraudstruct: A Low-Latency Decoupled Streaming Framework for Real-Time Transaction Structuring Detection

A thesis submitted in partial fulfillment of the requirements for the degree of  
**Master of Science (MSc) in Computer Science**

* **Author:** Chidiebere V. Christopher
* **Email:** vchidiebere.vc@gmail.com
* **Project Repository:** Fraudstruct-Library
* **Date:** July 2026

---

# ABSTRACT

The rapid growth of real-time retail payment infrastructures, such as the Nigeria Instant Payments (NIP) switch, has dramatically reduced transaction times, while simultaneously enabling sophisticated financial crimes. Among these, transaction structuring (smurfing)—where a large, reportable asset sum is split into multiple sub-threshold transactions across a parallel network of proxy accounts (mules)—remains a significant threat to banking integrity. 

Traditional rule-based compliance systems struggle to detect these coordinated multi-hop patterns, as they evaluate accounts in isolation. While Graph Neural Networks (GNNs) offer state-of-the-art capability in capturing neighborhood topological anomalies, their computational footprint and latency (>100ms) prevent their integration into the synchronous payment authorization hot-path, which operates under strict Service Level Agreements (SLAs) of sub-50ms. Furthermore, banking mainframes frequently restrict the installation of heavy deep learning runtimes (such as PyTorch or compiled C++ binaries) due to strict security policies.

To address these challenges, this study designs, implements, and benchmarks **Fraudstruct**—a hybrid real-time streaming feature store and graph-adversarial machine learning library for banking fraud detection. Fraudstruct introduces a **Three-Tier Lambda-Style Architecture** that decouples near-real-time graph updates (Warm Path) and offline GNN retraining (Cold Path) from synchronous transaction scoring (Hot Path). We implement a **Simplifying Graph Convolution (SGC)** GNN node classifier using only vectorized NumPy arithmetic, ensuring zero external dependencies and absolute portability.

Empirical evaluation on a simulated transaction stream demonstrates that Fraudstruct achieves an average transaction evaluation latency of **2.45 ms** (and **3.41 ms** at the 95th percentile), satisfying real-time banking SLAs by a factor of 20. Following the offline generation of multi-hop splitting topologies and automated hot-deployment of SGC weights, the classifier successfully blocked subsequent transaction attempts from coordinated attackers with **96.5%** confidence. 

These results prove that complex graph representations can be utilized in high-frequency payment switches without violating operational latency budgets, providing a scalable blueprint for real-time financial crime prevention.

---
---

# CHAPTER 1: INTRODUCTION

## 1.1 Background of the Study
The modernization of financial technology has shifted global banking toward instant, high-frequency settlement networks. In Nigeria, the introduction of the Nigeria Instant Payments (NIP) system by the Nigeria Inter-Bank Settlement System (NIBSS) has positioned the country as a leader in real-time retail payment volume. While these systems provide seamless liquidity, they have also dramatically compressed the execution window for fraud detection. Commercial banks must evaluate, authorize, or deny transaction requests in real-time—often with a strict latency budget of less than 50 milliseconds.

Historically, Anti-Money Laundering (AML) and fraud detection systems operated offline in a batch processing mode. Transactions were aggregated daily, weekly, or monthly, and audited retrospectively. While retrospectively valuable for forensic audits, this cold-path processing model fails to prevent funds leakage; once illicit funds are cleared and transferred out of the institution, recovery is virtually impossible. Modern banking defense requires real-time, in-flight transaction authorization controls.

This necessity has recently transitioned from an operational preference to a strict regulatory mandate. On March 10, 2026, the Central Bank of Nigeria (CBN) issued a landmark circular titled *"Baseline Standards for Automated Anti-Money Laundering (AML) Solutions"*. This directive mandates that all commercial banks, payment service providers, and financial technology institutions in Nigeria automate their AML, Counter-Terrorist Financing (CFT), and Counter-Proliferation Financing (CPF) monitoring systems, explicitly shifting the industry away from manual oversight. Under this framework, Deposit Money Banks are given an 18-month deadline (September 2027) and other financial institutions (Fintechs and Mobile Money Operators) are given 24 months (March 2028) to achieve full automated compliance. Furthermore, the circular requires all institutions to submit detailed implementation roadmaps (due by June 2026) and establishes strict parameters for continuous model governance—specifically mandating annual independent validation of deployed AI/ML models to monitor database drift and model bias.

However, real-time detection is severely bottlenecked by the evolution of coordinated fraud typologies. Fraudsters actively study bank thresholds—such as the regulatory Currency Transaction Report (CTR) limit of 5,000,000 NGN in Nigeria—and design evasion strategies. The most common of these is **transaction structuring (smurfing)**, where a large sum is divided into multiple sub-threshold transfers, routed through a complex topology of proxy "mule" accounts, and consolidated at a cash-out node. 

Because traditional banking detection systems rely on local, isolated entity checks (e.g. checking if Account A exceeded a rolling sum limit), they are structurally blind to multi-hop pathways. Resolving this requires modeling the transaction database as a dynamic, directed graph where neighborhood structural contexts can be analyzed collectively.

### 1.1.1 The Alternative Evasion Landscape
Coordinated transaction structuring does not exist in a vacuum; it is part of a broader, highly adaptive adversarial threat landscape. Financial criminals routinely employ several alternative evasion strategies to exploit gaps in traditional transactional monitoring systems:

1. **Rapid Round-Tripping (Looping):** Adversaries route funds through a circular sequence of accounts in milliseconds to disrupt audit trails. This exploits database locks and latency gaps in legacy banking ledgers.
2. **Layering via Digital Wallets (Neo-Fintechs):** Laundering networks exploit neobanks and mobile money operators (MMOs) that implement low-tier KYC controls, using fast API endpoints to split and transfer funds before central clearing flags are raised.
3. **Cross-Border Interoperability Exploits:** Moving transactions across international payment corridors to introduce jurisdictional delays and exploit friction in inter-bank messaging.
4. **Asset Transformation:** Instantly converting fiat transfers into high-liquidity digital assets, cryptocurrency, or retail e-vouchers at merchant checkouts to break the banking ledger trail.
5. **Time-Dispersed Velocity Evasion:** Spacing out transactions over weeks or months, ensuring the transfer frequencies stay below daily or weekly rolling alert counters.
6. **Threshold Customization (Irregular Amounts):** Transferring irregular, non-rounded amounts (e.g., ₦4,832,100 instead of a round ₦5,000,000) to blend into legitimate retail purchase patterns.
7. **Account Aging & Sleeper Accounts:** Activating "dormant" or compromised accounts that have been "warmed up" over months with tiny, legitimate retail utility payments to establish a clean behavioral baseline prior to executing a major structuring campaign.

---

## 1.2 Statement of the Problem
Deploying graph-based machine learning models inside online transaction authorization systems presents three critical challenges:

1. **The Latency-Complexity Trade-off**: Graph Neural Networks (GNNs) recursively aggregate neighborhood representations, requiring heavy matrix convolutions over multi-hop connections. Running these GNN convolving passes during the synchronous authorization phase introduces significant latency (often exceeding 100ms), violating the strict SLAs of payment switches like Postilion or Finacle.
2. **Training-Serving Skew and Feature Freshness**: Fraud networks adapt dynamically. If the features used to train the model offline do not match the real-time graph state in production, the model's accuracy degrades. Real-time graph feature computation (e.g. running PageRank or updating degrees on streams) is computationally expensive and difficult to sync with offline training models. Furthermore, to satisfy the annual independent model drift checks mandated by the CBN (2026), systems must actively monitor and log statistical drift without interrupting live production flows.
3. **Environment Portability and Localization Gaps**: Traditional GNN frameworks (such as PyTorch Geometric) rely on complex, compiled C++ extensions and CUDA libraries. Commercial banking mainframes operate under strict security guidelines that frequently block the installation of these heavy, unverified runtimes, demanding lightweight, zero-dependency deployment packages. Additionally, there is a severe shortage of localized, African-representative transactional datasets. Models built on foreign transaction patterns fail to capture the unique payment velocities (e.g., agency banking, mobile money, and instant inter-bank NIP transfers) characteristic of the Nigerian domestic market, necessitating the local generation of synthetic, representative adversarial flows.

### 1.2.1 Smurfing as the Latency Worst-Case Benchmark
To robustly validate any real-time Graph ML architecture, the system must be evaluated against the most computationally challenging topological anomaly. Among all the evasion strategies listed in Section 1.1.1, **Coordinated Transaction Structuring (Smurfing)** represents the absolute worst-case scenario for system latency and graph complexity:
* **The Neighborhood Expansion Problem:** Detecting smurfing requires evaluating a directed $N$-hop path from a source attacker through intermediate mules to a target beneficiary. Unlike single-account anomalies (which only query the node itself), smurfing forces the GNN to execute recursive, multi-hop neighborhood lookups and matrix multiplications synchronously during transaction clearing.
* **Topological Complexity:** It cannot be solved using local, isolated node statistics. The system must evaluate the global path structure and flow conservation ratios across multiple parallel pathways simultaneously.
* **SLA Stress-Test:** By benchmarking Fraudstruct specifically against smurfing, we subject the framework to the **worst-case graph-computation scenario**. If the architecture can successfully serve $N$-hop GNN representations under the strict 50ms payment switch SLA during a smurfing attack, it is mathematically and systemically guaranteed to handle all simpler, single-node transactional checks.

### 1.2.2 Indirect Mitigation of Alternative Evasion Typologies
While this study isolates smurfing as its primary benchmark task, Fraudstruct's underlying Graph ML architecture is designed to **indirectly neutralize** several other evasion strategies through the mathematical properties of spatial message passing (SGC):
* **Neutralizing Account Aging:** Even if a mule account is "warmed up" with a clean historical retail pattern (Account Aging), the moment it acts as a transit node in an active structuring flow, SGC feature propagation (\(\bar{X} = S_{\text{adj}}^K X\)) mathematically spreads the anomaly representation of the compromised source into the mule's node vector. Its "clean history" is overridden by its topological connection.
* **Neutralizing Threshold Customization:** Because Fraudstruct evaluates structural graph features (in-degree, out-degree, PageRank, and inflow/outflow ratios) rather than raw transaction amounts, fraudsters transferring irregular amounts (e.g., ₦4,832,100) are still blocked. The GNN checks the connectivity structure, which remains anomalous.
* **Neutralizing Rapid Round-Tripping:** The dynamic Warm Path maintains sliding time deques. Fast circular loops are captured immediately as path cycles inside the active streaming graph engine.

---

## 1.3 Objectives of the Study
The primary objective of this study is to design, implement, and benchmark **Fraudstruct**—a hybrid real-time streaming feature store and graph-adversarial machine learning library—to detect transaction structuring and multi-hop laundering under strict banking latency SLAs.

Specifically, the study aims to:
1. **Formulate a Graph-Based Adversarial Attack Simulator** capable of generating realistic multi-hop cash-splitting topologies that satisfy conservation-of-flow and threshold-evasion constraints.
2. **Develop a Lightweight GNN Node Classifier** using the Simplifying Graph Convolution (SGC) algorithm implemented in pure, vectorized NumPy to eliminate compiled C++ and PyTorch dependencies.
3. **Design a Decoupled Three-Tier (Lambda) Architecture** separating real-time rule lookups (Hot Path) and streaming graph updates (Warm Path) from offline GNN model training (Cold Path).
4. **Implement a Unified Interface** that seamlessly integrates Pandas (local development) and PySpark (enterprise scaling) database engines.
5. **Evaluate and Benchmark** the system's execution latency, throughput, and detection recall under a simulated commercial banking transaction stream.

---

## 1.4 Scope and Delimitation
This research focuses on the detection of coordinated transaction structuring (smurfing) and velocity anomalies within retail banking networks. The study is delimited to:
* **Data Domain**: Tabular retail transaction records containing source, destination, amount, and timestamp fields.
* **Algorithmic Scope**: SGC GNNs and sliding-window temporal heuristics. It does not explore GNN explainability or deep autoencoders.
* **Infrastructure**: Local Python execution simulating Redis caching and streaming deques. It does not deploy physical Apache Kafka or Redis clusters, focusing instead on validating the architectural and algorithmic logic.
* **Fraud Typology**: Coordinated money laundering. It does not address credit card cloning, biometric fraud, or endpoint device compromises.

---

## 1.5 Significance of the Study
This study holds both theoretical and practical significance:
* **Theoretical Significance**: It contributes to the academic literature of adversarial machine learning on tabular data by formalizing structuring as a constrained-flow optimization problem, and demonstrates how SGC can be adapted for zero-dependency, linear-time inference.
* **Architectural Significance**: It provides a concrete design pattern (the Three-Tier Lambda GNN pipeline) for scaling complex graph representation learning to high-frequency, low-latency streaming systems.
* **Practical Significance**: For commercial banks (specifically in Nigeria, such as Zenith Bank), this study provides a deployable blueprint for blocking coordinated money laundering at the gateway switch level, saving millions of naira in fraud losses before funds clear the institution.

### 1.5.1 Structural Comparison with Prior Art
To establish the practical significance of this study within the banking sector, it is necessary to contrast Fraudstruct against legacy transaction compliance systems and offline forensic databases. The table below delineates the structural differences across key operational dimensions:

| Dimension | Legacy AML Heuristics (SAS, Actimize, Clari5) | Offline Graph Databases (Neo4j, Palantir Foundry) | **Fraudstruct Framework (Proposed)** |
| :--- | :--- | :--- | :--- |
| **Execution Latency** | Synchronous (<50ms) but limited to simple, isolated single-node threshold checks. | Asynchronous/Batch (Minutes to Hours); runs post-incident queries. | **Synchronous (<3ms average) via pre-calculated GNN feature caching.** |
| **Relational Awareness** | **None.** Entity-centric; unable to link multi-hop pathways. | **High.** Graph-centric; queries complex connection paths. | **High.** Maps and propagates neighborhood topological risk in real-time. |
| **Mainframe Portability** | database query dependency; requires heavy server clusters. | Heavy graph DB runtimes; requires C++ runtimes and GPUs. | **Vectorized NumPy SGC compilation; zero unverified C++ dependencies.** |
| **Adversarial Resilience** | Manual rule tuning by compliance teams; easily evaded by customized thresholds. | Static manual inspection by forensic analysts. | **Continuous AI Sparring Ring; self-hardening under simulated attacks.** |

---

## 1.6 Research Novelties and Contributions
To formally establish the scientific validity of this work, this thesis introduces five distinct, highly detailed research contributions to the fields of Distributed Systems, Algorithmic MLOps, and Adversarial Machine Learning:

### 1.6.1 Novelty 1: Graph-Based Tabular Evasion Modeling (Adversarial Simulation)
Unlike traditional machine learning models that are evaluated against generic noise distributions, Fraudstruct establishes that financial adversaries operate under strict real-world constraints. Specifically, they must route a fixed amount of stolen cash ($S$) split across a specific number of mule accounts ($K$) within a strict time limit ($W$) to avoid triggering daily velocity triggers. Fraudstruct designs a mathematically constrained flow-based **Graph Adversarial Simulator** (`simulate/graph_attack.py`). Operating as a sophisticated "digital burglar" or **"crash-test dummy"**, the simulator generates realistic adversarial patterns to evaluate if security systems can catch a smart criminal.

### 1.6.2 Novelty 2: The Decoupled Lambda Graph Inference Paradigm
Traditional Graph Neural Networks perform neighborhood aggregation at the instant of inference, creating a major latency bottleneck ($150\text{ms}$–$500\text{ms}$) that violates commercial switch SLAs ($<50\text{ms}$). Fraudstruct resolves this by introducing a **decoupled three-tier architecture** separating the synchronous Hot Path from the asynchronous streaming Warm Path and the offline Cold Path (the **Lambda Data Highway**):
* **Hot Path (Synchronous Inference):** Pre-calculates and caches instant node representations, allowing the API switch (`stream/api.py`) to evaluate and approve transactions in **under 3ms** (2.45ms average), consuming only 4.9% of the 50ms inter-bank payment switch budget.
* **Warm Path (Near-Real-Time Stream):** Constantly updates dynamic graph connectivity and PageRank metrics asynchronously in the background as transactions occur.
* **Cold Path (Asynchronous Retraining):** Handles slow, offline AI parameter optimization when the bank network is quiet, deploying updated weights without server restarts.

### 1.6.3 Novelty 3: Zero-Dependency Algorithmic MLOps Co-design
Standard deep learning frameworks (such as PyTorch and PyTorch Geometric) rely on heavy, compiled C++ libraries and CUDA binaries that fail to pass the strict security audits and air-gapped constraints of banking core systems. We address this by co-designing the GNN mathematics with operational deployment constraints, compressing the Simplifying Graph Convolution (SGC) feature propagation rules into raw, vectorized NumPy array operations:

\[\bar{X} = S_{\text{adj}}^K X\]

This mathematical compaction achieves **100% environment portability with zero compiled dependencies**, allowing complex Graph AI models to run natively on secure CPU-only mainframes.

### 1.6.4 Novelty 4: Shifting Evasion Boundaries to Topological Space
Standard fraud detection systems are entity-centric, evaluating tabular transaction fields in isolation and leaving them vulnerable to structuring evasion (smurfing) below fixed limits. Fraudstruct introduces an algorithmic shift, moving the machine learning classification boundary from **mutable tabular values** to **immutable topological graph spaces** (PageRank, degrees, flow conservation). By propagating features through the collapsed adjacency matrix, the network structural risk is mathematically blended into all participating nodes, raising detection recall from **0.0% (for isolated tabular classifiers) to 96.5%** on camouflaged attacks.

### 1.6.5 Novelty 5: Continuous Online Graph Adversarial Hardening (The Digital Sparring Ring)
Static AI models quickly become obsolete because they are trained only on historical datasets, leaving them vulnerable to new patterns (structural drift). Traditional platforms rely on manual, periodic retraining. Fraudstruct introduces an **AI digital sparring ring** by plugging the Graph Adversarial Simulator directly into the training loop. The simulator constantly invents new ways to route split transactions, and the GNN model learns from these attempts in a continuous loop. This active training process automatically hardens model defenses and immunizes Fraudstruct against new, evolving evasion tricks in real-time.
