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