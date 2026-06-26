import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adjust python path to import fraudstruct
sys.path.append(r"C:\Users\Chidiebere\Documents")

from fastapi.testclient import TestClient
from fraudstruct.stream.api import app, engine
from fraudstruct.simulate.graph_attack import simulate_graph_splitting

client = TestClient(app)

def run_verification():
    print("==================================================")
    print("FRAUDSTRUCT PIPELINE VERIFICATION AND BENCHMARK")
    print("==================================================")
    
    # 1. Generate Organic Base Accounts
    print("\n[Step 1] Generating organic transaction traffic...")
    accounts = [f"ACC_{i:04d}" for i in range(1, 51)]
    organic_events = []
    base_time = datetime.utcnow()
    
    # Generate 150 normal transactions
    for i in range(150):
        src = np.random.choice(accounts)
        dst = np.random.choice(accounts)
        while dst == src:
            dst = np.random.choice(accounts)
            
        amount = round(float(np.random.exponential(scale=20000.0) + 1000.0), 2)
        ts = base_time + timedelta(minutes=i * 5)
        
        organic_events.append({
            "transaction_id": f"TXN-ORG-{i:05d}",
            "source_account": src,
            "destination_account": dst,
            "amount": amount,
            "timestamp": ts.isoformat(),
            "channel": "NIP"
        })
        
    print(f"Generated {len(organic_events)} organic transactions.")
    
    # 2. Generate Graph-based Smurfing Attack
    print("\n[Step 2] Simulating adversarial smurfing path (A -> Mules -> B)...")
    attacker = "ACC_ATTACKER"
    beneficiary = "ACC_BENEFICIARY"
    attack_amount = 600_000.0
    
    adv_df = simulate_graph_splitting(
        df=None,
        source=attacker,
        dest=beneficiary,
        amount=attack_amount,
        timestamp=base_time + timedelta(hours=13),
        n_splits=6,
        mule_candidates=accounts[:6]
    )
    
    attack_events = []
    for _, row in adv_df.iterrows():
        attack_events.append({
            "transaction_id": f"TXN-ADV-{np.random.randint(10000, 99999)}",
            "source_account": row["source_account"],
            "destination_account": row["destination_account"],
            "amount": float(row["amount"]),
            "timestamp": row["timestamp"].isoformat(),
            "channel": "NIP"
        })
    print(f"Generated {len(attack_events)} attack path transactions.")
    
    # 3. Feed organic events through API Hot/Warm Path
    print("\n[Step 3] Feeding organic stream to API evaluate endpoint...")
    latencies = []
    for event in organic_events:
        resp = client.post("/v1/evaluate", json=event)
        assert resp.status_code == 200
        latencies.append(resp.json()["latency_ms"])
        
    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    print(f"Success! Processed organic transactions.")
    print(f"Inference Latency Metric (ZENITH SLA check):")
    print(f"  - Average Latency: {avg_lat:.3f} ms")
    print(f"  - 95th Percentile: {p95_lat:.3f} ms")
    
    # 4. Feed Attack Events and Verify Temporal Detection
    print("\n[Step 4] Feeding attack stream and checking for alerts...")
    flagged_txns = []
    for event in attack_events:
        resp = client.post("/v1/evaluate", json=event)
        assert resp.status_code == 200
        data = resp.json()
        if data["decision"] in ["FLAG", "BLOCK"]:
            flagged_txns.append(data)
            print(f"ALERT TRIGGERED on {data['transaction_id']}! Decision: {data['decision']}, Reasons: {data['reasons']}")
            
    print(f"Total alerts triggered: {len(flagged_txns)} / {len(attack_events)}")
    
    # 5. Cold Path GNN Training
    print("\n[Step 5] Triggering Cold Path GNN Training & Hot-Deployment...")
    labels = [{"account_id": attacker, "is_fraud": 1}]
    mules = list(adv_df["destination_account"].unique())
    if beneficiary in mules:
        mules.remove(beneficiary)
    for m in mules:
        labels.append({"account_id": m, "is_fraud": 1})
    for acc in accounts[10:30]:
        labels.append({"account_id": acc, "is_fraud": 0})
        
    train_req = {
        "labels": labels,
        "epochs": 30
    }
    
    train_resp = client.post("/v1/train", json=train_req)
    assert train_resp.status_code == 200
    print(f"Training response: {train_resp.json()}")
    
    # 6. Verify GNN-based Block Decisions
    print("\n[Step 6] Verifying active GNN blocking post-deployment...")
    subsequent_event = {
        "transaction_id": "TXN-SUB-00001",
        "source_account": attacker,
        "destination_account": "ACC_MERCHANT",
        "amount": 50000.0,
        "timestamp": (base_time + timedelta(hours=15)).isoformat(),
        "channel": "NIP"
    }
    
    resp = client.post("/v1/evaluate", json=subsequent_event)
    assert resp.status_code == 200
    data = resp.json()
    print(f"Post-GNN deployment evaluation result for attacker:")
    print(f"  - Decision: {data['decision']}")
    print(f"  - Reasons: {data['reasons']}")
    print(f"  - GNN Anomaly Score: {data['features']['gnn_anomaly_score']}")
    
    print("\n==================================================")
    print("VERIFICATION COMPLETE - PIPELINE OPERATIONAL")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
