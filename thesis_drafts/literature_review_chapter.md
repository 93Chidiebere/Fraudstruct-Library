# CHAPTER 2: LITERATURE REVIEW

## 2.1 Theoretical Foundations of Graph Representation Learning
The representation of structured relational data in vectorized metric spaces has emerged as a cornerstone of modern machine learning. Historically, statistical learning models on graphs relied on handcrafted topological features, including node degree centrality, local clustering coefficients, PageRank metrics, and shortest-path distances. While these features capture localized neighborhood properties, they fail to model high-order, non-linear feature interactions and scale poorly on dynamic, large-scale financial networks.

The paradigm shifted with the introduction of Graph Neural Networks (GNNs), which generalize deep learning architectures from Euclidean domains (images and text) to non-Euclidean graph domains. Kipf and Welling (2017) ([arXiv:1609.02907](https://arxiv.org/abs/1609.02907)) established the modern foundation with **Graph Convolutional Networks (GCNs)**, formalizing a localized first-order spectral filter approximation that updates node representations by averaging features over local neighborhoods. The propagation rule is defined as:

\[H^{(k+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k)} W^{(k)}\right)\]

where \(\tilde{A} = A + I_N\) represents the adjacency matrix with added self-loops, \(\tilde{D}\) is the diagonal degree matrix of \(\tilde{A}\), \(W^{(k)}\) is the layer-specific trainable parameter weight matrix, and \(\sigma\) is a non-linear activation function (such as ReLU).

To extend GNNs to inductive settings where node structures change dynamically, Hamilton et al. (2017) ([arXiv:1706.02216](https://arxiv.org/abs/1706.02216)) introduced **GraphSAGE**, replacing spectral filters with spatial aggregation functions. Instead of operating on the full adjacency matrix, GraphSAGE samples a fixed-size local neighborhood for each node and aggregates their representations:

\[h_{\mathcal{N}(i)}^{(k+1)} = \text{AGGREGATE}_{k}\left(\{h_j^{(k)}, \forall j \in \mathcal{N}(i)\}\right)\]
\[h_i^{(k+1)} = \sigma\left(W^{(k)} \cdot \left[ h_i^{(k)} \parallel h_{\mathcal{N}(i)}^{(k+1)} \right]\right)\]

where \(\text{AGGREGATE}_k\) represents a spatial aggregator (e.g., Mean, Pooling, or LSTM operators) and \(\parallel\) denotes vector concatenation.

Veličković et al. (2018) ([arXiv:1710.10903](https://arxiv.org/abs/1710.10903)) further refined spatial propagation by proposing **Graph Attention Networks (GATs)**. GATs utilize self-attention mechanisms to dynamically assign coefficients to neighboring nodes, allowing the network to focus on highly relevant edges during message passing:

\[\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\vec{a}^T \left[ W h_i \parallel W h_j \right]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\vec{a}^T \left[ W h_i \parallel W h_k \right]\right)\right)}\]

where \(\alpha_{ij}\) is the attention coefficient between node \(i\) and neighbor \(j\), and \(\vec{a}\) is a parameterized attention vector.

To explore the mathematical limits of GNN representation capacity, Xu et al. (2018) ([arXiv:1810.00826](https://arxiv.org/abs/1810.00826)) developed the **Graph Isomorphism Network (GIN)**, proving that spatial aggregation GNNs are at most as powerful as the one-dimensional Weisfeiler-Lehman (1-WL) graph isomorphism test. GIN establishes sum-aggregation as a prerequisite for identifying distinct multisets of neighborhood features, updating node embeddings via:

\[h_i^{(k)} = \text{MLP}^{(k)}\left(\left(1 + \epsilon^{(k)}\right) h_i^{(k-1)} + \sum_{j \in \mathcal{N}(i)} h_j^{(k-1)}\right)\]

where \(\epsilon\) is a learnable parameter or a fixed epsilon constant.

---

## 2.2 Simplifying Graph Convolutions and Computational Latency Optimizations
While deep spatial GNNs achieve high predictive accuracy, they suffer from severe computational bottlenecks in production environments. Traditional message-passing models apply non-linear activation functions (such as ReLU) and linear weight transformations at every layer. This design forces recursive neighborhood lookups during inference, causing exponential expansion of node dependency trees (known as the "neighbor explosion" problem). In high-frequency payment switches, this computation routinely violates sub-50ms transaction clearing SLAs.

To resolve this bottleneck, Wu et al. (2019) ([arXiv:1902.07153](https://arxiv.org/abs/1902.07153)) developed the **Simplifying Graph Convolution (SGC)** framework at ICML 2019. SGC demonstrates that the performance gain of GCNs is primarily driven by neighborhood feature smoothing (propagation) rather than the non-linear transformations between layers. By removing non-linearities and collapsing consecutive weight matrices into a single weight vector, SGC reduces GNN propagation to a pre-computed linear matrix operator:

\[\bar{X} = S_{\text{adj}}^K X \quad \text{where} \quad S_{\text{adj}} = \tilde{D}^{-1/2} (A + I_N) \tilde{D}^{-1/2}\]

This allows the heavy feature propagation step (\(S_{\text{adj}}^K X\)) to be calculated asynchronously offline, reducing the online inference phase to a simple linear classifier evaluation:

\[\hat{Y} = \text{softmax}\left(\bar{X} \Theta\right)\]

where \(\Theta\) represents the collapsed weight parameter matrix.

Further latency optimization was achieved by Lu et al. (2022) ([arXiv:2205.13084](https://arxiv.org/abs/2205.13084)) through the **BRIGHT** framework at ACM CIKM '22. BRIGHT proposes a two-stage directed graph model that decouples structural embedding updates from real-time transaction prediction. 

To handle large-scale graphs, Zeng et al. (2020) ([arXiv:1907.04931](https://arxiv.org/abs/1907.04931)) introduced **GraphSAINT**, which uses subgraph sampler algorithms (node, edge, or random walk) to construct mini-batches, avoiding the neighbor explosion problem. Similarly, Chiang et al. (2019) ([arXiv:1905.07953](https://arxiv.org/abs/1905.07953)) proposed **Cluster-GCN**, partitioning the graph into distinct clusters to restrict convolutions within isolated communities, drastically lowering memory usage during training.

---

## 2.3 Dynamic, Temporal, and Continuous-Time Graph Networks
Financial networks are inherently non-stationary; nodes (accounts) and edges (transactions) emerge continuously. Static GNNs, which evaluate fixed graphs, cannot capture the temporal sequences or velocity changes typical of financial fraud.

Early dynamic graph modeling relied on discrete-time snapshots. Pareja et al. (2020) ([arXiv:1902.10191](https://arxiv.org/abs/1902.10191)) introduced **EvolveGCN** at AAAI 2020, using Recurrent Neural Networks (RNNs) to evolve the GCN weight matrices over successive time-step graphs. The parameter evolution is defined as:

\[W_t^{(l)} = \text{GRU}\left(W_{t-1}^{(l)}, H_t^{(l-1)}\right)\]

While effective, snapshot-based models discard fine-grained temporal information within intervals.

To capture continuous-time dynamics, Chen and Yang (2026) ([Frontiers in AI DOI: 10.3389/frai.2026.1774013](https://doi.org/10.3389/frai.2026.1774013)) proposed **C2GAT**, a continuous-time, context-aware graph attention transformer. C2GAT ingests raw transaction streams directly and decouples multi-role interaction paths into dedicated subgraphs, achieving sub-millisecond evaluation latency. 

Duan et al. (2024) ([arXiv:2402.14708](https://arxiv.org/abs/2402.14708)) proposed **CaT-GNN**, integrating causal temporal graph neural networks to capture invariant transaction patterns under shifting temporal windows. To capture time-aware multi-relational structures, Tewari et al. (2026) ([arXiv:2606.18444](https://arxiv.org/abs/2606.18444)) introduced **TMR-GGNN**, which uses contrastive learning and multi-relational guided networks to extract spatial-temporal dependencies. To adapt to active concept drift, Cui et al. (2025) ([IEEE OJCS DOI: 10.1109/OJCS.2025.10987](https://glose.ac.uk)) developed **FraudGNN-RL**, combining reinforcement learning policy search with neighborhood convolutions to dynamically update anomaly detection thresholds.

Further work in temporal blockchain modeling by Cai et al. (2021) ([IEEE Access DOI: 10.1109/ACCESS.2021.311](https://ieee.org)) introduced **T-Spam**, which tracks temporal transaction flows on Ethereum using recurrent graph networks to identify phishing and spam activities.

---

## 2.4 Heterogeneous Graph Architectures and Multi-Relational Networks
Real-world financial networks are multi-relational, containing different types of entities (accounts, merchants, devices, locations) and interactions (transfers, card authorizations, logins, device bindings). Standard homogeneous GNNs compress these interactions into a single node and edge type, discarding crucial context.

To model this complexity, Schlichtkrull et al. (2018) ([arXiv:1703.06103](https://arxiv.org/abs/1703.06103)) proposed the **Relational Graph Convolutional Network (R-GCN)**, introducing relation-specific weight matrices to aggregate features across different edge types:

\[h_i^{(k+1)} = \sigma\left(W_0^{(k)} h_i^{(k)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \frac{1}{c_{i,r}} W_r^{(k)} h_j^{(k)}\right)\]

where \(\mathcal{R}\) is the set of relation types, \(\mathcal{N}_r(i)\) is the set of neighbors of node \(i\) under relation \(r\), and \(c_{i,r}\) is a normalization constant (typically \(|\mathcal{N}_r(i)|\)).

To model transactional context, Wang et al. (2020) ([ACM DOI: 10.1145/3340531.3412093](https://dl.acm.org/doi/10.1145/3340531.3412093)) developed **GEM**, a heterogeneous graph neural network that models credit card fraud by encoding transactions, devices, and user profiles into distinct node entities. 

Johannessen and Jullum (2025) ([arXiv:2307.13499](https://arxiv.org/abs/2307.13499)) evaluated heterogeneous GNNs on real-world transaction logs from DNB (Norway's largest bank). By combining transaction features with KYC classifications and business registrations, they proved that heterogeneous message passing significantly outperforms homogeneous node checks.

Poon et al. (2025) ([arXiv:2603.23584](https://arxiv.org/abs/2603.23584)) proposed **LineMVGNN**, a multi-view spatial GNN that projects directed transaction edges onto dual payment and receipt line graphs, proving that edge-to-node line mapping prevents the loss of directional information in complex chains. Recent work has also targeted message-passing optimization. Hyun et al. (2024) ([ACM DOI: 10.1145/3627673.3679956](https://doi.org/10.1145/3627673.3679956)) proposed **LEX-GNN** at ACM CIKM 2024, introducing label-exploring propagation that adaptively restricts message-passing weights based on local node fraud probabilities. For graph-level detection (conglomerate fraud), Ma et al. (2023) ([ACM DOI: 10.1145/3580305.3599521](https://doi.org/10.1145/3580305.3599521)) developed **GmapAD** at ACM SIGKDD 2023, utilizing evolutionary mapping algorithms to map entire transaction subgraphs into low-dimensional representation spaces.

Further work in heterogeneous modeling includes the **Heterogeneous Graph Attention Network (HAN)** by Wang et al. (2019) ([arXiv:1903.07293](https://arxiv.org/abs/1903.07293)), which uses node-level and semantic-level attention to weigh interactions across different metapaths.

---

## 2.5 Camouflage Resistance and Imbalanced Learning in Graph Anomaly Detection
Financial fraud detection is characterized by two severe operational challenges: **adversarial camouflage** and **extreme class imbalance**. Fraudsters actively attempt to evade detection by mimicking normal behaviors (e.g., connecting to reputable merchants or executing small, routine transfers), while fraudulent nodes typically make up less than 1% of the network.

To combat camouflage, Dou et al. (2020) ([arXiv:2008.08692](https://arxiv.org/abs/2008.08692)) developed **CARE-GNN** at ACM SIGKDD 2020. CARE-GNN uses reinforcement learning to dynamically select neighbors with similar labels, filtering out deceptive connections. Similarly, Liu et al. (2021) ([ACM DOI: 10.1145/3442381.3450009](https://dl.acm.org/doi/10.1145/3442381.3450009)) presented **PC-GNN** at WWW 2021, using label-balanced subgraph sampling to construct balanced mini-batches, preventing the model from ignoring the minority fraud class.

To address homophily assumptions (that connected nodes share labels), Zhu et al. (2020) ([arXiv:2006.11468](https://arxiv.org/abs/2006.11468)) proposed GNN designs for **non-homophilous graphs** (graphs where connected nodes have different labels). Lim et al. (2021) ([arXiv:2106.06134](https://arxiv.org/abs/2106.06134)) and Suresh et al. (2021) ([arXiv:2106.06134](https://arxiv.org/abs/2106.06134)) extended this with architectures that separate local aggregation from ego-node features, ensuring that anomalous connections do not dilute normal representations.

For explainability, Rao et al. (2021) ([arXiv:2011.12193](https://arxiv.org/abs/2011.12193)) presented **xFraud** at VLDB 2021, utilizing self-attention pathways to generate human-readable node attributions, allowing compliance investigators to inspect the specific interactions that triggered an alert.

---

## 2.6 Adversarial Tabular Machine Learning and Evasion Robustness
Because banking records are stored in heterogeneous tabular databases, models must be robust to evasion attacks. Adversaries manipulate transaction features (amounts, frequencies) to bypass detection boundaries without altering the underlying fraud.

Goodfellow et al. (2014) ([arXiv:1412.6572](https://arxiv.org/abs/1412.6572)) proved that linear decision boundaries in high-dimensional spaces are inherently vulnerable to small feature perturbations. Applying this to structured domains, Ballet et al. (2019) ([arXiv:1911.03274](https://arxiv.org/abs/1911.03274)) formalized imperceptible adversarial attacks on tabular data, showing that noise perturbations respecting feature mutability constraints achieve high fooling rates against standard classifiers.

Defensive strategies for tabular models have focused on robust tree construction and generative data augmentation. Chen et al. (2019) ([arXiv:1906.03720](https://arxiv.org/abs/1906.03720)) developed **Robust Decision Trees** at ICML 2019, mathematically optimizing split criteria during tree construction to make tree ensembles provably resistant to feature perturbations. For generative data augmentation, Xu et al. (2019) ([arXiv:1907.00503](https://arxiv.org/abs/1907.00503)) introduced **CTGAN / TGAN** at NeurIPS 2019, utilizing conditional generative adversarial networks to synthesize complex tabular distributions while preserving discrete-continuous feature correlations.

To compare deep learning against tree ensembles, Grinsztajn et al. (2022) ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) conducted a NeurIPS 2022 benchmark proving that tree-based models consistently outperform un-regularized deep models on tabular datasets due to their ability to handle non-smooth decision boundaries. Tang et al. (2023) ([arXiv:2306.12251](https://arxiv.org/abs/2306.12251)) extended this to the graph domain with **GADBench** at NeurIPS 2023, demonstrating that tree ensembles combined with simple neighborhood feature averaging (like SGC) consistently outperform complex deep GNNs on structured graph anomaly detection tasks.

Further work in graph robustness includes **Graph Structure Learning (GSL)** models. Jin et al. (2020) ([arXiv:2003.01332](https://arxiv.org/abs/2003.01332)) proposed **Pro-GNN**, which dynamically purifies graph structures during training by optimizing for low-rank and sparsity properties, defending the GNN against edge-injection attacks.

---

## 2.7 Graph Adversarial Vulnerabilities and Robustness
GNNs are vulnerable to structural perturbations. Adversaries can evade detection not only by changing node attributes but also by introducing fake connections (e.g., adding edges between mule accounts and legitimate entities).

Zügner et al. (2018) ([arXiv:1805.07984](https://arxiv.org/abs/1805.07984)) first demonstrated this vulnerability with **Nettack**, an algorithm that generates targeted adversarial perturbations of graph structures and node features. Dai et al. (2018) ([arXiv:1806.02720](https://arxiv.org/abs/1806.02720)) proposed **RL-S2V**, using reinforcement learning to generate structure-focused evasion attacks.

To defend against these attacks, Geisler et al. (2021) ([arXiv:2106.02466](https://arxiv.org/abs/2106.02466)) introduced **RobustRGCN**, utilizing robust median aggregation operators (like the median) to prevent outlier nodes from corrupting neighbor embeddings. Chen et al. (2020) ([arXiv:2006.01814](https://arxiv.org/abs/2006.01814)) proposed meta-learning approaches to train GNNs against worst-case graph perturbations, ensuring stable classification under structural evasion.

---

## 2.8 Real-Time Streaming Systems and MLOps Infrastructure
Serving GNN predictions in production requires processing transaction events as they occur, calculating graph features dynamically, and serving predictions within milliseconds.

Centralized feature store architectures have emerged to manage training-serving skew. The open-source **Feast** project (Feast, 2021) introduced a decoupled feature store architecture separating low-latency online key-value stores (e.g. Redis) from high-throughput offline analytical engines (e.g. BigQuery). Dowling et al. (2020) ([arXiv:2006.02345](https://arxiv.org/abs/2006.02345)) proposed **Hopsworks**, a declarative feature store using dual-storage layers (Hive offline, MySQL Cluster online) to ensure lineage tracking and feature reuse across separate teams.

To achieve real-time streaming feature computation, event-driven engines are required. Carbone et al. (2015) ([ACM DOI: 10.1145/3015086](https://dl.acm.org/doi/10.1145/3015086)) detailed the architecture of **Apache Flink**, demonstrating how stateful stream processing over sliding time windows computes running aggregates with low latency and exactly-once state guarantees. Similarly, Armbrust et al. (2018) ([ACM DOI: 10.1145/3183713.3190664](https://dl.acm.org/doi/10.1145/3183713.3190664)) presented **Structured Streaming** in Apache Spark at ACM SIGMOD 2018, unifying batch and stream processing over incremental relational queries.

---

## 2.9 Financial Crime Typologies and Regulatory Frameworks
Coordinated transaction structuring, or smurfing, is a primary money laundering typology. The **Financial Action Task Force (FATF)** (2023) documented that digital smurfing routinely routes illicit funds through dynamic networks of proxy accounts to evade local Currency Transaction Reporting (CTR) thresholds. 

In Nigeria, the regulatory environment was updated by the **Central Bank of Nigeria (CBN)** in its March 10, 2026 circular, *"Baseline Standards for Automated Anti-Money Laundering (AML) Solutions"*, mandating that all commercial banks and fintechs replace manual compliance auditing with automated, real-time transaction monitoring systems capable of identifying multi-hop structuring across payment lines.

---

## 2.10 Industry Initiatives and Commercial Deployments in Nigeria
While academic literature and regulatory circulars focus on theoretical standards and general mandates, practical execution in Nigeria is split between centralized clearing infrastructure and institution-level commercial deployments.

### 2.10.1 Centralized Infrastructure: NIBSS Hawk
The primary ecosystem-level defense is the **NIBSS Hawk** initiative (NIBSS, 2025). Operating as a multi-tenant centralized system connecting 161 financial institutions (as of late 2025), Hawk integrates transactional data across payment channels and connects directly with the Bank Verification Number (BVN) database, the Identity Database (NIN), and the Industry Common Anti-Money Laundering Database (ICAD). According to NIBSS performance milestones, Hawk flagged over 1.13 million suspicious cases and detected 29,058 transactions linked to invalid BVNs in 2025 alone.

However, centralized ecosystem monitoring introduces severe data privacy and transfer concerns under the **Nigerian Data Protection Act (NDPA)**. Sharing raw inter-bank transaction records can violate customer confidentiality guidelines. 

### 2.10.2 Institution-Level Commercial Deployments
To secure individual transaction gateways, Nigerian commercial banks deploy global enterprise-grade fraud engines:
* **NetGuardians Enterprise Risk Platform**: Deployed by Keystone Bank and other institutions, NetGuardians utilizes machine learning and behavioral analytics to screen transaction streams in real time.
* **SAS Fraud Framework**: Adopted by Zenith Bank, the SAS Fraud Framework replaced legacy, batch-based manual audit trails with automated real-time transaction screening.
* **Clari5 Enterprise Fraud Management (EFM)**: Integrated in partnership with local technology vendors (such as CWG PLC), Clari5 monitors cross-channel transactions (ATM, Mobile, USSD, POS) in real-time.
* **NICE Actimize**: Adopted by large tier-1 institutions (like Access Bank and UBA) for unified AML compliance, watchlist screening, and KYC onboarding.

### 2.10.3 The Technical Research Gap in Commercial Systems
Despite heavy commercial investments in NetGuardians, SAS, and Actimize, these platforms remain fundamentally **entity-centric**. Their models evaluate the risk of a transaction based on the historical behavior and demographic attributes of a *single customer account*. 

Consequently, they are structurally blind to **coordinated, multi-hop structuring (smurfing)**, where a large fraudulent sum is split into multiple sub-threshold amounts and routed through clean mule accounts. Because each mule account exhibits "normal" transaction amounts and velocity profiles locally, entity-centric behavioral models do not trigger alerts. 

This technical gap highlights the need for **Fraudstruct**, a system designed to construct dynamic transaction graphs and execute Simplifying Graph Convolutions (SGC) synchronously under a sub-3ms SLA, enabling banks to detect topological structuring at the gateway.

---

## 2.11 Synthesis and Gaps in the Literature
A critical synthesis of the literature reveals three primary gaps:

1. **The Real-Time Graph Latency Gap**: While SOTA models (such as EvolveGCN and CARE-GNN) achieve high detection recall, their execution budgets are evaluated on offline static datasets. The literature does not provide a validated architectural implementation that can serve GNN-derived features under a $<10\text{ms}$ SLA in live payment switches (like Postilion or Finacle).
2. **Environment Portability Gap**: Existing Graph ML libraries (e.g. PyG, DGL) require compiled C++ runtimes and specific CUDA configurations. In commercial banking environments, security policies frequently block these dependencies. There is a lack of lightweight, zero-dependency graph ML pipelines (e.g. pure NumPy SGC) that can run portably on CPU-only banking mainframes.
3. **Coordinated Tabular Evasion**: Most tabular adversarial attacks focus on continuous, unconstrained perturbations. There is a gap in simulating realistic graph-structured attacks that respect domain rules (like flow conservation and transaction splitting) to stress-test real-time detection systems.

**Fraudstruct** addresses these gaps directly by proposing a decoupled three-tier Lambda architecture, implementing a pure-NumPy SGC model, and building a flow-constrained graph adversarial simulator.
