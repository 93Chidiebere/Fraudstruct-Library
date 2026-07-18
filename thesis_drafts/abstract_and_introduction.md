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

---

## 1.2 Statement of the Problem
Deploying graph-based machine learning models inside online transaction authorization systems presents three critical challenges:

1. **The Latency-Complexity Trade-off**: Graph Neural Networks (GNNs) recursively aggregate neighborhood representations, requiring heavy matrix convolutions over multi-hop connections. Running these GNN convolving passes during the synchronous authorization phase introduces significant latency (often exceeding 100ms), violating the strict SLAs of payment switches like Postilion or Finacle.
2. **Training-Serving Skew and Feature Freshness**: Fraud networks adapt dynamically. If the features used to train the model offline do not match the real-time graph state in production, the model's accuracy degrades. Real-time graph feature computation (e.g. running PageRank or updating degrees on streams) is computationally expensive and difficult to sync with offline training models. Furthermore, to satisfy the annual independent model drift checks mandated by the CBN (2026), systems must actively monitor and log statistical drift without interrupting live production flows.
3. **Environment Portability and Localization Gaps**: Traditional GNN frameworks (such as PyTorch Geometric) rely on complex, compiled C++ extensions and CUDA libraries. Commercial banking mainframes operate under strict security guidelines that frequently block the installation of these heavy, unverified runtimes, demanding lightweight, zero-dependency deployment packages. Additionally, there is a severe shortage of localized, African-representative transactional datasets. Models built on foreign transaction patterns fail to capture the unique payment velocities (e.g., agency banking, mobile money, and instant inter-bank NIP transfers) characteristic of the Nigerian domestic market, necessitating the local generation of synthetic, representative adversarial flows.

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
