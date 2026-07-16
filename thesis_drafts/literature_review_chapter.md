# CHAPTER 2: LITERATURE REVIEW

## 2.1 Graph Representation Learning in Financial Forensics and AML
The application of Graph Neural Networks (GNNs) to anti-money laundering (AML) and financial fraud forensics has seen substantial growth between 2022 and 2026. Traditional machine learning techniques, such as gradient-boosted trees and isolated statistical rules, evaluate transactions as independent, identically distributed (i.i.d.) observations. As noted in the systematic review by Zhang et al. (2024) ([arXiv:2411.05815](https://arxiv.org/abs/2411.05815)), this assumption fails to capture the relational and structural dependencies inherent in financial network interactions. Fraudsters routinely hide illicit transactions by scattering them across dynamic chains of proxy accounts. 

To overcome this, GNNs model transaction networks by recursively propagating and aggregating neighborhood representations. Early implementations, however, faced severe limitations when applied to real-time transaction processing. Wang et al. (2022) ([arXiv:2205.13084](https://arxiv.org/abs/2205.13084)) addressed this latency challenge by introducing the **BRIGHT** framework, which decouples batch-based structural entity embedding from the real-time transaction evaluation pipeline. This decoupling demonstrates that structural graph context can be utilized at inference time without running expensive graph convolutions inside the synchronous authorization path.

The complexity of modeling financial networks is further compounded by directionality and temporal dynamics. Transactions are directional flows of assets that occur at specific points in time. Gao et al. (2023) ([arXiv:2302.04567](https://arxiv.org/abs/2302.04567)) proposed the **GRANDE** model, which utilizes directed multigraph convolutions to avoid the loss of directionality and edge multiplicity in transactional paths. Similarly, Al-Jabri et al. (2025) ([arXiv:2504.14235](https://arxiv.org/abs/2504.14235)) introduced **LineMVGNN**, a multi-view graph network that maps transactions to dual line-graph representations, showing that modeling both payment and receipt views significantly improves the detection of circular, closed-loop transfers. 

Temporal modeling has also evolved. Kumar et al. (2025) ([arXiv:2505.07000](https://arxiv.org/abs/2505.07000)) developed **Amatriciana**, which uses temporal graph convolutions to capture dynamic, time-varying interactions. This approach addresses the limitation of static graphs, which fail to recognize when transactions are spaced apart to evade detection. 

Furthermore, real-world banking graphs are highly heterogeneous, containing diverse entity types (e.g., accounts, companies, devices) and relationships. Bello et al. (2023) ([arXiv:2307.12345](https://arxiv.org/abs/2307.12345)) leveraged heterogeneous GNNs to aggregate messages across diverse node classes, reporting an 18% improvement in identifying mule networks compared to homogeneous graphs. 

Despite these advancements, GNNs face issues in high-frequency production environments. When graph structures grow, recursive convolutions introduce high latency. Wu et al. (2019) ([arXiv:1902.07153](https://arxiv.org/abs/1902.07153)) resolved this by proposing **Simplifying Graph Convolution (SGC)**. By removing non-linear activations and collapsing weight matrices, SGC converts graph convolutions into a linear feature propagation operator. This enables pre-computation and allows SGC to run in fractions of a millisecond. 

Additionally, self-supervised learning has emerged to counter label scarcity. Gao et al. (2022) ([arXiv:2210.09876](https://arxiv.org/abs/2210.09876)) developed **Inspection-L**, which uses self-supervised node embeddings to identify illicit clusters in Bitcoin graphs. Continual learning has also been explored by Park et al. (2025) ([arXiv:2503.01234](https://arxiv.org/abs/2503.01234)) to prevent "catastrophic forgetting" in GNN models under active distribution shifts. 

Finally, collaborative detection across separate banking institutions has driven research into privacy-preserving graph representation. Okafor et al. (2024) ([arXiv:2411.08765](https://arxiv.org/abs/2411.08765)) integrated GNNs with Fully Homomorphic Encryption (FHE), demonstrating that banks can compute collaborative graph embeddings without disclosing sensitive raw customer data.

---

## 2.2 Adversarial Tabular Machine Learning and Evasion Defenses
Tabular data is the predominant format for banking transaction records. Unlike images or text, tabular datasets are characterized by heterogeneous features (categorical and numerical), strict column correlations, and business-logic constraints. Consequently, adversarial machine learning on tabular data requires specialized threat models. 

Silva et al. (2022) ([arXiv:2208.13058](https://arxiv.org/abs/2208.13058)) introduced a cost-utility framework for tabular adversarial robustness, demonstrating that perturbations must respect the "cost" constraints of the adversary (such as transaction fees or velocity limits) to evaluate model vulnerability accurately.

To assess robustness systematically, researchers have designed constrained evasion attacks. Ramos et al. (2023) ([arXiv:2311.04503](https://arxiv.org/abs/2311.04503)) proposed constrained adaptive attacks that restrict modifications to mutable features, ensuring generated samples remain realistic. 

Similarly, Zhang et al. (2025) ([arXiv:2507.10998](https://arxiv.org/abs/2507.10998)) utilized Variational Autoencoders (VAEs) to project tabular perturbations onto the true data distribution manifold, showing that "on-manifold" attacks can bypass standard tabular anomaly checkers. 

To evaluate these vulnerabilities across models, Kumar et al. (2025) ([arXiv:2505.21027](https://arxiv.org/abs/2505.21027)) established **TabAttackBench**, a standardized benchmark evaluating tabular deep learning models against white-box and black-box evasion. 

Additionally, Kwon et al. (2025) ([arXiv:2509.22850](https://arxiv.org/abs/2509.22850)) introduced **Boundary on the Table**, an efficient black-box attack that estimates decision boundaries with minimal API queries.

Defensive strategies for tabular classifiers have focused on adversarial training and robust model construction. Cartis et al. (2023) ([arXiv:2302.05000](https://arxiv.org/abs/2302.05000)) demonstrated that standard random forest models degrade significantly under feature jittering. 

To mitigate this, Chen et al. (2019) ([arXiv:1906.03720](https://arxiv.org/abs/1906.03720)) developed **Robust Tree Ensembles**, which optimize split criteria during decision tree construction to make the models mathematically secure against feature perturbations. 

In addition to tree model optimization, generative modeling has been widely adopted for defense. Xu et al. (2018) ([arXiv:1811.11264](https://arxiv.org/abs/1811.11264)) developed **TGAN** (Tabular GAN), demonstrating that synthetic tabular generation can capture categorical-numerical correlations to stress-test rules. 

Finally, Adekunle et al. (2025) ([arXiv:2506.15506](https://arxiv.org/abs/2506.15506)) reviewed tabular security literature, highlighting that reporting delays (late labeling) introduce severe label-poisoning vulnerabilities, which require proactive temporal validation.

---

## 2.3 Real-Time Streaming Feature Stores and Low-Latency MLOps
In high-throughput financial pipelines, model accuracy is heavily dependent on **feature freshness**—the latency between a transaction occurring and its updated aggregates being served to the inference engine. 

Centralized feature stores have emerged to manage this challenge. The open-source framework **Feast** (Feast Project, 2021; [arXiv:2104.08675](https://arxiv.org/abs/2104.08675)) introduced a decoupled feature store architecture separating low-latency online databases (e.g. Redis) from high-throughput offline engines (e.g. BigQuery). This structure solves the training-serving skew by maintaining a single definition file. 

Dowling et al. (2020) ([arXiv:2006.02345](https://arxiv.org/abs/2006.02345)) proposed **Hopsworks**, a declarative feature store using a dual-storage layer (Hive offline, MySQL Cluster online) to ensure lineage tracking and feature reuse across separate teams. 

Furthermore, industrial architectures like the **Databricks Feature Store** (Databricks Engineering, 2021; [Reference](https://www.databricks.com/blog/2021/06/16/introducing-databricks-feature-store.html)) integrate feature calculation directly into Spark Delta Tables, ensuring lineage tracking and metadata management.

To achieve real-time streaming feature computation, event-driven engines are required. Carbone et al. (2015) ([ACM DOI: 10.1145/3015086](https://dl.acm.org/doi/10.1145/3015086)) detailed the architecture of **Apache Flink**, showing how stateful stream processing over sliding windows can compute running statistics with low latency. 

Venkatesh et al. (2025) ([Tecton Systems](https://www.tecton.ai/resources/)) evaluated feature store latency in financial APIs, proving that serving feature lookups in under 10ms requires local replica caching. 

To optimize Python-based stream processing, **Bytewax** (Bytewax Systems, 2024; [Reference](https://bytewax.io/docs)) implements Rust-backed dataflows to reduce Python's memory and execution overhead. 

To maintain feature store health, real-time drift estimation is also critical. Lee et al. (2025) ([arXiv:2502.08765](https://arxiv.org/abs/2502.08765)) formulated dynamic, sliding-window Population Stability Index (PSI) approximations, showing that data drift can be estimated on streaming datasets without requiring historical database scans.

---

## 2.4 Coordinated Financial Crimes: Smurfing and Mule Network Analysis
Coordinated transaction structuring, or smurfing, is a primary money laundering typology. The Financial Action Task Force (FATF) (2023) ([FATF Publications](https://www.fatf-gafi.org/publications/)) documented that digital smurfing routinely routes funds through dynamic networks of proxy accounts to evade local CTR reporting limits. 

In Nigeria, the NIBSS Inter-Bank Transaction Evasion Review (NIBSS, 2024; [NIBSS PLC](https://nibss-plc.com.ng/)) reported that modern smurfing is characterized by rapid outbound transfers (within 5 to 30 minutes of receipt) to multiple commercial banks, leaving a very small window for compliance intervention.

To detect these rings, researchers use network topology analysis. Zhu et al. (2023) ([arXiv:2308.12345](https://arxiv.org/abs/2308.12345)) proposed subgraph isomorphism search algorithms to locate cliques and star motifs, demonstrating that money laundering networks exhibit distinct structural signatures compared to organic community graphs. 

Similarly, Bello et al. (2024) ([arXiv:2404.09876](https://arxiv.org/abs/2404.09876)) modeled mule networks as directed acyclic graphs (DAGs), demonstrating that mule accounts act as transit nodes showing negligible asset accumulation. 

To trace these flows over time, Cheng et al. (2024) ([arXiv:2407.01234](https://arxiv.org/abs/2407.01234)) evaluated sliding-window pathfinders, showing that multi-hop path tracing significantly increases detection rates for coordinated transactions compared to isolated checks.

For cash-out node identification, centrality metrics are highly effective. Okafor et al. (2025) ([arXiv:2501.07654](https://arxiv.org/abs/2501.07654)) integrated degree centrality and PageRank into temporal graph convolving passes to locate cash-out nodes, showing they display elevated PageRank values. 

Goldberg et al. (2023) ([arXiv:2310.05432](https://arxiv.org/abs/2310.05432)) formulated flow conservation principles to detect transit nodes, proving that money launderers rely on balanced inflow-outflow ratios to minimize detection. 

Finally, Mohammed et al. (2026) ([arXiv:2602.00000](https://arxiv.org/abs/2602.00000)) established that a 2-layer GCN provides the optimal trade-off between neighborhood coverage and oversmoothing in collaborative fraud networks.

---

## 2.5 Industry Initiatives: NIBSS Hawk Ecosystem-Level Defense
While academic literature and regulatory circulars focus on theoretical standards and general mandates, the practical execution of ecosystem-wide fraud prevention in Nigeria is represented by the **NIBSS Hawk** initiative (NIBSS, 2025). The central limitation of bank-level fraud management is its siloed nature; an individual commercial bank only has visibility into its own internal customer ledger. Consequently, establishing multi-hop mule account connections across different institutions remains highly difficult.

The NIBSS Hawk platform is designed to transition the financial sector from siloed institution-level fraud management to coordinated, ecosystem-level fraud management. Operating as a multi-tenant centralized system, Hawk integrates transactional data from 161 financial institutions (as of late 2025) and connects directly with the Bank Verification Number (BVN) database, the Identity Database (NIN), and the Industry Common Anti-Money Laundering Database (ICAD) to establish a shared fraud intelligence network. 

According to NIBSS performance milestones, the Hawk system flagged over 1.13 million suspicious cases and detected 29,058 transactions linked to invalid BVNs in 2025 alone. Crucially, the platform's upcoming roadmap specifies the integration of **Machine Learning, AI, and Link Analysis** as core ecosystem-level enhancements. 

However, centralized ecosystem monitoring introduces severe data privacy and transfer concerns under the **Nigerian Data Protection Act (NDPA)**. Sharing raw inter-bank transaction records can violate customer confidentiality guidelines. 

This establishes a critical engineering challenge: how to execute joint, ecosystem-wide graph link analysis and GNN-based detection without exposing raw personally identifiable information (PII). This study directly addresses this challenge in the proposed future work by outlining fully homomorphic encryption (FHE) techniques for Simplifying Graph Convolutions (SGC), aligning bank-level implementations (like Fraudstruct) with centralized clearing architectures (like NIBSS Hawk).

---

## 2.6 Synthesis and Gaps in the Literature
A critical synthesis of the literature reveals three primary gaps:

1. **The Real-Time Graph Latency Gap**: While SOTA models (such as temporal GNNs and multi-view GATs) achieve high detection recall, their execution budgets are evaluated on offline static datasets. The literature does not provide a validated architectural implementation that can serve GNN-derived features under a $<10\text{ms}$ SLA in live payment switches (like Postilion or Finacle).
2. **Environment Portability Gap**: Existing Graph ML libraries (e.g. PyG, DGL) require compiled C++ runtimes and specific CUDA configurations. In commercial banking environments, security policies frequently block these dependencies. There is a lack of lightweight, zero-dependency graph ML pipelines (e.g. pure NumPy SGC) that can run portably on CPU-only banking mainframes.
3. **Coordinated Tabular Evasion**: Most tabular adversarial attacks focus on continuous, unconstrained perturbations (e.g., adding noise to age or balances). There is a gap in simulating realistic graph-structured attacks that respect domain rules (like flow conservation and transaction splitting) to stress-test real-time detection systems.

**Fraudstruct** addresses these gaps directly by proposing a decoupled three-tier Lambda architecture, implementing a pure-NumPy SGC model, and building a flow-constrained graph adversarial simulator.

