# CHAPTER 2: LITERATURE REVIEW

## 2.1 Graph Representation Learning in Financial Forensics and AML
The application of Graph Neural Networks (GNNs) to anti-money laundering (AML) and financial fraud forensics has seen substantial growth. Traditional machine learning techniques, such as gradient-boosted decision trees and isolated statistical rules, evaluate transactions as independent, identically distributed (i.i.d.) observations. As demonstrated by Weber et al. (2019) ([arXiv:1908.02591](https://arxiv.org/abs/1908.02591)) in their seminal work on the Elliptic Bitcoin dataset, this assumption fails to capture relational and structural dependencies inherent in financial network interactions. Fraudsters routinely hide illicit transactions by scattering them across dynamic chains of proxy accounts. 

To overcome this, GNNs model transaction networks by recursively propagating and aggregating neighborhood representations. Early implementations, however, faced severe limitations when applied to real-time transaction processing. Lu et al. (2022) ([arXiv:2205.13084](https://arxiv.org/abs/2205.13084)) addressed this latency challenge by introducing the **BRIGHT** framework at ACM CIKM '22. BRIGHT decouples batch-based structural entity embedding from the real-time transaction evaluation pipeline, demonstrating that structural graph context can be served at inference time without running expensive graph convolutions inside the synchronous authorization path.

The complexity of modeling financial networks is further compounded by temporal dynamics and adversarial camouflage. Transactions are directional flows of assets occurring over time. Pareja et al. (2020) ([arXiv:1902.10191](https://arxiv.org/abs/1902.10191)) introduced **EvolveGCN** at AAAI 2020, adapting GNN parameters over dynamic graphs using recurrent architectures without requiring static node embeddings. 

To combat adversarial camouflage—where fraudsters deliberately connect to reputable accounts to trick ML models—Dou et al. (2020) ([arXiv:2008.08692](https://arxiv.org/abs/2008.08692)) developed **CARE-GNN** at ACM SIGKDD 2020, utilizing label-aware similarity measures and reinforcement learning neighbor selection to filter deceptive graph edges.

Furthermore, Liu et al. (2021) ([ACM DOI: 10.1145/3442381.3450009](https://dl.acm.org/doi/10.1145/3442381.3450009)) presented **PC-GNN** at The Web Conference (WWW 2021), introducing label-balanced sub-graph sampling to resolve extreme class imbalance in fraud datasets. For explainability in corporate compliance, Rao et al. (2021) ([arXiv:2011.12193](https://arxiv.org/abs/2011.12193)) presented **xFraud** at VLDB 2021, generating human-understandable node attributions for flagged heterogeneous graphs.

Despite these advancements, multi-layer GNNs face computational bottlenecks in high-frequency production environments. When graph structures grow, recursive convolutions introduce latency exceeding 100ms. Wu et al. (2019) ([arXiv:1902.07153](https://arxiv.org/abs/1902.07153)) resolved this in their ICML 2019 landmark paper on **Simplifying Graph Convolution (SGC)**. By removing non-linear activation functions and collapsing consecutive weight matrices, SGC converts multi-layer graph convolutions into a single linear feature propagation operator (\(\bar{X} = S_{\text{adj}}^K X\)). This enables asynchronous pre-computation and allows SGC node evaluations to execute in microseconds.

---

## 2.2 Adversarial Tabular Machine Learning and Evasion Defenses
Tabular data is the predominant format for banking transaction records. Unlike images or text, tabular datasets are characterized by heterogeneous features (categorical and numerical), strict column correlations, and hard business-logic constraints. Consequently, adversarial machine learning on tabular data requires specialized threat models. 

Goodfellow et al. (2014) ([arXiv:1412.6572](https://arxiv.org/abs/1412.6572)) established the foundations of adversarial evasion attacks, proving that linear decision boundaries in high-dimensional spaces are inherently vulnerable to small feature perturbations. Applying this to structured domains, Ballet et al. (2019) ([arXiv:1911.03274](https://arxiv.org/abs/1911.03274)) formalized imperceptible adversarial attacks on tabular data, demonstrating that noise perturbations respecting feature mutability constraints achieve high fooling rates against standard classifiers.

Defensive strategies for tabular models have focused on robust tree construction and generative data augmentation. Chen et al. (2019) ([arXiv:1906.03720](https://arxiv.org/abs/1906.03720)) developed **Robust Decision Trees** at ICML 2019, mathematically optimizing split criteria during tree construction to make tree ensembles provably resistant to feature perturbations. 

For generative synthetic augmentation, Xu et al. (2019) ([arXiv:1907.00503](https://arxiv.org/abs/1907.00503)) introduced **CTGAN / TGAN** at NeurIPS 2019, utilizing conditional generative adversarial networks to synthesize complex tabular distributions while preserving discrete-continuous feature correlations. 

Finally, Grinsztajn et al. (2022) ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) conducted a comprehensive NeurIPS 2022 benchmark comparing deep learning against tree ensembles on tabular data, proving that un-regularized deep models suffer from unhelpful inductive biases on heterogeneous tabular datasets, reinforcing the need for lightweight linear or tree-based serving layers.

---

## 2.3 Real-Time Streaming Systems and Low-Latency MLOps Architecture
In high-throughput financial clearing pipelines, model accuracy depends heavily on **feature freshness**—the latency between a transaction occurring and its updated aggregates being served to the inference engine. 

Centralized feature store architectures have emerged to manage this challenge. The open-source **Feast** project (Feast, 2021) introduced a decoupled feature store architecture separating low-latency online key-value stores (e.g. Redis) from high-throughput offline analytical engines (e.g. BigQuery). This structure eliminates training-serving skew by maintaining a single feature definition repository.

To achieve real-time streaming feature computation, event-driven engines are required. Carbone et al. (2015) ([ACM DOI: 10.1145/3015086](https://dl.acm.org/doi/10.1145/3015086)) detailed the architecture of **Apache Flink**, demonstrating how stateful stream processing over sliding time windows computes running aggregates with low latency and exactly-once state guarantees. 

Similarly, Armbrust et al. (2018) ([ACM DOI: 10.1145/3183713.3190664](https://dl.acm.org/doi/10.1145/3183713.3190664)) presented **Structured Streaming** in Apache Spark at ACM SIGMOD 2018, unifying batch and stream processing over incremental relational queries.

---

## 2.4 Financial Crime Typologies and Regulatory Frameworks
Coordinated transaction structuring, or smurfing, is a primary money laundering typology. The **Financial Action Task Force (FATF)** (2023) documented that digital smurfing routinely routes illicit funds through dynamic networks of proxy accounts to evade local Currency Transaction Reporting (CTR) thresholds. 

In Nigeria, the regulatory environment was updated by the **Central Bank of Nigeria (CBN)** in its March 10, 2026 circular, *"Baseline Standards for Automated Anti-Money Laundering (AML) Solutions"*, mandating that all commercial banks and fintechs replace manual compliance auditing with automated, real-time transaction monitoring systems capable of identifying multi-hop structuring across payment lines.

---

## 2.5 Industry Initiatives and Commercial Deployments in Nigeria
While academic literature and regulatory circulars focus on theoretical standards and general mandates, practical execution in Nigeria is split between centralized clearing infrastructure and institution-level commercial deployments.

### 2.5.1 Centralized Infrastructure: NIBSS Hawk
The primary ecosystem-level defense is the **NIBSS Hawk** initiative (NIBSS, 2025). Operating as a multi-tenant centralized system connecting 161 financial institutions (as of late 2025), Hawk integrates transactional data across payment channels and connects directly with the Bank Verification Number (BVN) database, the Identity Database (NIN), and the Industry Common Anti-Money Laundering Database (ICAD). According to NIBSS performance milestones, Hawk flagged over 1.13 million suspicious cases and detected 29,058 transactions linked to invalid BVNs in 2025 alone.

However, centralized ecosystem monitoring introduces severe data privacy and transfer concerns under the **Nigerian Data Protection Act (NDPA)**. Sharing raw inter-bank transaction records can violate customer confidentiality guidelines. 

### 2.5.2 Institution-Level Commercial Deployments
To secure individual transaction gateways, Nigerian commercial banks deploy global enterprise-grade fraud engines:
* **NetGuardians Enterprise Risk Platform**: Deployed by Keystone Bank and other institutions, NetGuardians utilizes machine learning and behavioral analytics to screen transaction streams in real time.
* **SAS Fraud Framework**: Adopted by Zenith Bank, the SAS Fraud Framework replaced legacy, batch-based manual audit trails with automated real-time transaction screening.
* **Clari5 Enterprise Fraud Management (EFM)**: Integrated in partnership with local technology vendors (such as CWG PLC), Clari5 monitors cross-channel transactions (ATM, Mobile, USSD, POS) in real-time.
* **NICE Actimize**: Adopted by large tier-1 institutions (like Access Bank and UBA) for unified AML compliance, watchlist screening, and KYC onboarding.

### 2.5.3 The Technical Research Gap in Commercial Systems
Despite heavy commercial investments in NetGuardians, SAS, and Actimize, these platforms remain fundamentally **entity-centric**. Their models evaluate the risk of a transaction based on the historical behavior and demographic attributes of a *single customer account*. 

Consequently, they are structurally blind to **coordinated, multi-hop structuring (smurfing)**, where a large fraudulent sum is split into multiple sub-threshold amounts and routed through clean mule accounts. Because each mule account exhibits "normal" transaction amounts and velocity profiles locally, entity-centric behavioral models do not trigger alerts. 

This technical gap highlights the need for **Fraudstruct**, a system designed to construct dynamic transaction graphs and execute Simplifying Graph Convolutions (SGC) synchronously under a sub-3ms SLA, enabling banks to detect topological structuring at the gateway.

---

## 2.6 Synthesis and Gaps in the Literature
A critical synthesis of the literature reveals three primary gaps:

1. **The Real-Time Graph Latency Gap**: While SOTA models (such as EvolveGCN and CARE-GNN) achieve high detection recall, their execution budgets are evaluated on offline static datasets. The literature does not provide a validated architectural implementation that can serve GNN-derived features under a $<10\text{ms}$ SLA in live payment switches (like Postilion or Finacle).
2. **Environment Portability Gap**: Existing Graph ML libraries (e.g. PyG, DGL) require compiled C++ runtimes and specific CUDA configurations. In commercial banking environments, security policies frequently block these dependencies. There is a lack of lightweight, zero-dependency graph ML pipelines (e.g. pure NumPy SGC) that can run portably on CPU-only banking mainframes.
3. **Coordinated Tabular Evasion**: Most tabular adversarial attacks focus on continuous, unconstrained perturbations. There is a gap in simulating realistic graph-structured attacks that respect domain rules (like flow conservation and transaction splitting) to stress-test real-time detection systems.

**Fraudstruct** addresses these gaps directly by proposing a decoupled three-tier Lambda architecture, implementing a pure-NumPy SGC model, and building a flow-constrained graph adversarial simulator.
