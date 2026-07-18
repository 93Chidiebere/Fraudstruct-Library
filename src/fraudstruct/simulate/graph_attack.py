import numpy as np
import pandas as pd

def simulate_graph_splitting(
    df,
    source="A",
    dest="B",
    amount=500_000.0,
    timestamp=None,
    n_splits=5,
    mule_candidates=None,
    source_col="source_account",
    dest_col="destination_account",
    amount_col="amount",
    time_col="timestamp"
):
    """
    Simulates a multi-hop transaction splitting (smurfing/money laundering) attack.
    Splits a large amount from 'source' to 'dest' across 'n_splits' intermediate mule accounts.
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now()
    elif isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
        
    # If no candidate mule nodes are provided, we generate synthetic account IDs
    if mule_candidates is None or len(mule_candidates) < n_splits:
        mule_candidates = [f"MULE_{i:04d}" for i in range(1, n_splits + 1)]
    else:
        # Sample candidates
        mule_candidates = np.random.choice(mule_candidates, size=n_splits, replace=False)
        mule_candidates = [str(m) for m in mule_candidates]
        
    # Split the amount (with a small random perturbation so they are not exactly equal)
    base_split = amount / n_splits
    splits = []
    remaining = amount
    for i in range(n_splits - 1):
        val = np.round(base_split * np.random.uniform(0.9, 1.1), 2)
        splits.append(val)
        remaining -= val
    splits.append(np.round(remaining, 2))
    
    adv_rows = []
    # For each split, generate A -> Mule and then Mule -> B
    for i, s_val in enumerate(splits):
        mule = mule_candidates[i]
        
        # Step 1: A -> Mule
        t1 = timestamp + pd.Timedelta(minutes=i * 2)
        row1 = {
            source_col: source,
            dest_col: mule,
            amount_col: s_val,
            time_col: t1,
            "is_fraud": True,
            "is_adversarial": True
        }
        adv_rows.append(row1)
        
        # Step 2: Mule -> B (slightly delayed)
        t2 = t1 + pd.Timedelta(minutes=np.random.randint(5, 30))
        row2 = {
            source_col: mule,
            dest_col: dest,
            amount_col: np.round(s_val * np.random.uniform(0.95, 0.99), 2),
            time_col: t2,
            "is_fraud": True,
            "is_adversarial": True
        }
        adv_rows.append(row2)
        
    adv_df = pd.DataFrame(adv_rows)
    return adv_df
