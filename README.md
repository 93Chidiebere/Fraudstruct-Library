# **Fraudstruct**

**Fraudstruct** is a *library-first* Python framework for **adversarial machine learning in banking fraud detection**, designed to help Data Scientists detect **structuring (smurfing)**, **threshold evasion**, and **behavioral manipulation** in transaction data — even when data originates from **multiple external providers** (NIBSS, PTSPs, Fintechs).

It supports **batch analytics**, **large-scale Spark processing**, **adversarial training loops**, and **audit-ready outputs** aligned with **banking model risk requirements**.

---

## **Why Fraudstruct Exists**

In real banking environments:

- Transaction data does **not** originate from a single internal database  
- POS, Card, USSD, Mobile App transactions arrive from **different providers**  
- Customer or merchant identifiers are often **missing or inconsistent**  
- Fraudsters actively **adapt to model thresholds**

**Fraudstruct is built for this reality.**

---

## **Core Capabilities**

### 🔍 **Adversarial Fraud Detection**
- Structuring / Smurfing detection  
- Threshold evasion patterns  
- Rapid debit behavior analysis  
- Temporal aggregation across noisy identifiers  

### ⚙️ **Batch-First, Model-Builder Friendly**
- Designed for **offline fraud analysis**  
- Integrates naturally into **feature engineering pipelines**  
- Pandas for small data, Spark for large data  

### 🧠 **Adversarial Training Integration**
- Generate adversarial transaction scenarios  
- Stress-test fraud models before deployment  
- Plug into ML training loops (CV/NLP-style adversarial training)  

### 🏦 **Audit & Model Risk Alignment**
- Deterministic outputs  
- Parameter traceability  
- Monthly / quarterly / annual audit support  
- SR 11-7-style documentation readiness  

---

## **Installation**

### **Standard (Pandas / Local Analysis)**

```bash
pip install fraudstruct
```

**With Spark Support**

```bash
pip install fraudstruct[spark]
```

## **Quick Start**

**1. Load Transaction Data**

Fraudstruct assumes provider-agnostic schemas, not idealized bank schemas.

Minimum required columns:

- entity_id → proxy identifier (account hash, device ID, token, etc.)

- timestamp

- amount

```python
import pandas as pd

df = pd.read_csv("transactions.csv")
```

**2. Initialize the Processing Engine**

```python
from fraudstruct.engine import PandasEngine

engine = PandasEngine(df)
```

**3. Detect Structuring (Smurfing)**

```python
from fraudstruct.detect.structuring import detect_structuring

results = detect_structuring(
    engine,
    window="1H",
    count_threshold=5,
    sum_threshold=300_000
)

```

What this detects:

- Multiple small debits

- Within a short time window

- That collectively breach regulatory or internal thresholds


**Supported Detection Modules**

| Module              | Description                        |
| ------------------- | ---------------------------------- |
| `structuring`       | Smurfing / split-payment detection |
| `threshold_evasion` | Avoidance of rule-based limits     |
| `debit_velocity`    | Rapid-fire transaction behavior    |
| `behavioral_shift`  | Distributional behavior change     |


**Adversarial Data Simulation**

Fraudstruct can generate synthetic adversarial transactions to test fraud systems.
```python
from fraudstruct.simulate.structuring import generate_structured_transactions

adv_df = generate_structured_transactions(
    base_df=df,
    target_amount=500_000,
    n_splits=10
)

```
Use cases:

- Model stress testing

- Fraud rule validation

- Red-team simulations


**Adversarial Training Loop**

Fraudstruct integrates into ML pipelines just like adversarial training in CV/NLP:
```python
for epoch in range(n_epochs):
    model.fit(X_train, y_train)

    adv_data = fraudstruct.simulate(...)
    model.fit(adv_data.X, adv_data.y)

```
This helps models learn fraudster behavior, not just past labels.

Large-Scale / Spark Support

Fraudstruct automatically supports Spark when enabled:
```python
from fraudstruct.engine import SparkEngine

engine = SparkEngine(spark_df)

```

- Same API.

- Same logic.

- Different scale.


## **Audit & Governance Support**

Fraudstruct is designed to support:

✔ Parameter logging

✔ Deterministic re-runs

✔ Result reproducibility

✔ Evidence generation for validators

Outputs are audit-ready by design, not as an afterthought.


## **Who Should Use Fraudstruct**

- Bank Data Scientists

- Fraud Analytics Teams

- Model Risk & Validation Units

- Financial Crime & Compliance Teams

- Fintech Risk Engineers
  

## **What Fraudstruct Is NOT**

❌ A real-time transaction switch

❌ A rule engine replacement

❌ A data ingestion platform

❌ A black-box ML model

Fraudstruct is a specialized analytical library.


## **License**

MIT License


**Final Note**

Fraudstruct is built to reflect how fraud actually happens, if you work in banking fraud, risk, or model validation, this library was designed for you.





































