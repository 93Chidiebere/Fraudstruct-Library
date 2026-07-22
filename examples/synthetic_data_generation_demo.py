"""
================================================================================
FRAUDSTRUCT SDK TUTORIAL: SYNTHETIC DATASET GENERATION
================================================================================
Author: Chidiebere V. Christopher
Package: fraudstruct (v1.0.0)

This script demonstrates how risk engineers and data scientists can generate 
highly realistic synthetic transaction datasets combining:
1. Organic Baseline Heavy-Tailed Traffic (Log-Normal / Exponential)
2. Trust-Building Warm-up Periods (Accounts building normal history first)
3. Adversarial Camouflage (Amount noise, time delays, merchant distraction)
4. Delayed Chargeback Labeling (Simulating late fraud reporting)
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import from the published fraudstruct library!
import fraudstruct
from fraudstruct.simulate.graph_attack import simulate_graph_splitting
from fraudstruct.simulate.trust_building import generate_trust_building_history
from fraudstruct.simulate.camouflage import apply_adversarial_camouflage


def generate_realistic_banking_dataset(
    num_accounts: int = 100,
    num_organic_txns: int = 500,
    num_attack_groups: int = 3,
    start_time: datetime = None
) -> pd.DataFrame:
    """
    Generates a complete, end-to-end synthetic banking transaction dataset
    that mimics real-world retail and payment switch traffic patterns.
    """
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(days=7)

    np.random.seed(42)
    accounts = [f"ACC_{i:04d}" for i in range(1, num_accounts + 1)]
    merchants = [f"MERCHANT_{i:02d}" for i in range(1, 10)]

    print(f"Generating realistic dataset for {num_accounts} accounts over 7 days...")

    # -------------------------------------------------------------------------
    # 1. ORGANIC BASELINE TRAFFIC (HEAVY-TAILED AMOUNT DISTRIBUTION)
    # -------------------------------------------------------------------------
    records = []
    channels = ["NIP", "POS", "MOBILE", "WEB"]
    channel_weights = [0.55, 0.25, 0.15, 0.05]  # NIP instant payment dominant

    for i in range(num_organic_txns):
        src = np.random.choice(accounts)
        # 20% of transfers go to merchants, 80% to other retail accounts
        if np.random.rand() < 0.2:
            dst = np.random.choice(merchants)
        else:
            dst = np.random.choice(accounts)
            while dst == src:
                dst = np.random.choice(accounts)

        # Log-Normal distribution models realistic retail payment amounts
        # Most transactions are small (2,000 NGN), few are large (150,000 NGN)
        amount = round(float(np.random.lognormal(mean=9.0, sigma=1.0) + 500.0), 2)
        
        # Poisson-like timestamp distribution
        offset_minutes = int(np.random.uniform(0, 7 * 24 * 60))
        txn_time = start_time + timedelta(minutes=offset_minutes)

        records.append({
            "transaction_id": f"TXN-ORG-{i:06d}",
            "source_account": src,
            "destination_account": dst,
            "amount": amount,
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": np.random.choice(channels, p=channel_weights),
            "is_fraud": 0,
            "pattern_type": "ORGANIC"
        })

    df_organic = pd.DataFrame(records)
    print(f"  ✓ Created {len(df_organic)} organic baseline transactions.")

    # -------------------------------------------------------------------------
    # 2. TRUST-BUILDING WARM-UP HISTORY (ACCOUNTS ACTING NORMAL FIRST)
    # -------------------------------------------------------------------------
    # Fraudsters often warm up proxy accounts by paying bills/utilities for weeks
    trust_events = generate_trust_building_history(
        target_accounts=["ACC_ATTACKER_1", "ACC_ATTACKER_2", "ACC_MULE_01"],
        num_days=5,
        start_time=start_time
    )
    df_trust = pd.DataFrame(trust_events)
    df_trust["is_fraud"] = 0
    df_trust["pattern_type"] = "TRUST_BUILDING"
    print(f"  ✓ Created {len(df_trust)} trust-building warm-up transactions.")

    # -------------------------------------------------------------------------
    # 3. ADVERSARIAL ATTACKS WITH CAMOUFLAGE (TRANSACTION STRUCTURING)
    # -------------------------------------------------------------------------
    attack_records = []
    for g in range(num_attack_groups):
        attacker_id = f"ACC_ATTACKER_{g+1}"
        beneficiary_id = f"ACC_BENEFICIARY_{g+1}"
        target_sum = 750000.0 + (g * 250000.0)  # e.g., 750k, 1M, 1.25M NGN
        
        # Generate multi-hop splitting events
        raw_attack = simulate_graph_splitting(
            attacker_id=attacker_id,
            beneficiary_id=beneficiary_id,
            target_sum=target_sum,
            num_mules=5,
            start_time=start_time + timedelta(days=4, hours=g*2)
        )
        
        # Apply adversarial camouflage (add amount noise, time delays, merchant noise)
        camouflaged_attack = apply_adversarial_camouflage(
            attack_events=raw_attack,
            noise_std=0.05,        # 5% Gaussian noise on transfer amounts
            merchant_list=merchants
        )
        
        for ev in camouflaged_attack:
            ev["is_fraud"] = 1
            ev["pattern_type"] = "ADVERSARIAL_SMURFING"
            attack_records.append(ev)

    df_attack = pd.DataFrame(attack_records)
    print(f"  ✓ Created {len(df_attack)} camouflaged adversarial attack legs across {num_attack_groups} fraud rings.")

    # -------------------------------------------------------------------------
    # 4. MERGE & SORT COMPLETE DATASET BY TIMESTAMP
    # -------------------------------------------------------------------------
    df_full = pd.concat([df_organic, df_trust, df_attack], ignore_index=True)
    df_full["timestamp_dt"] = pd.to_datetime(df_full["timestamp"])
    df_full = df_full.sort_values(by="timestamp_dt").drop(columns=["timestamp_dt"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 5. SIMULATE DELAYED CHARGEBACK LABELING (OPERATIONAL REALITY)
    # -------------------------------------------------------------------------
    # In real banks, fraud labels are not available instantly. They arrive 30-90 days late!
    # We add a `label_delay_days` column to simulate realistic MLOps feature store conditions.
    df_full["label_delay_days"] = np.where(df_full["is_fraud"] == 1, np.random.randint(14, 60, size=len(df_full)), 0)

    return df_full


def main():
    print("=================================================================")
    print(f"  FRAUDSTRUCT SDK (v{fraudstruct.__version__}): SYNTHETIC DATASET GENERATOR  ")
    print("=================================================================\n")

    # Generate synthetic dataset
    df = generate_realistic_banking_dataset(
        num_accounts=50,
        num_organic_txns=300,
        num_attack_groups=2
    )

    print("\n-----------------------------------------------------------------")
    print("DATASET SUMMARY & STATISTICAL PROFILE:")
    print("-----------------------------------------------------------------")
    print(f"Total Transactions Generated: {len(df)}")
    print(f"Total Organic Transfers:     {len(df[df['pattern_type'] == 'ORGANIC'])}")
    print(f"Total Trust Warm-up Txns:    {len(df[df['pattern_type'] == 'TRUST_BUILDING'])}")
    print(f"Total Fraud Attack Legs:     {len(df[df['pattern_type'] == 'ADVERSARIAL_SMURFING'])}")
    print(f"Overall Fraud Ratio:         {(df['is_fraud'].mean() * 100):.2f}%\n")

    print("SAMPLE RECORD PREVIEW:")
    print(df[["transaction_id", "source_account", "destination_account", "amount", "channel", "pattern_type", "is_fraud"]].head(10).to_string(index=False))

    # Save generated dataset to CSV for experimentation
    output_filename = "synthetic_banking_transactions.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n✓ Dataset successfully saved to '{output_filename}'!")

    print("\n=================================================================")
    print("HOW TO USE THIS SYNTHETIC DATASET:")
    print("1. Stress-test your production ML model (Random Forest, XGBoost).")
    print("2. Train Fraudstruct's NumPy SGC GNN engine on realistic traffic.")
    print("3. Simulate delayed labeling to benchmark MLOps feature store drift.")
    print("=================================================================")

if __name__ == "__main__":
    main()
