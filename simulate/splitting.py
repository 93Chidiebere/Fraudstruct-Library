import pandas as pd
import numpy as np

def simulate_splitting(df, max_txn=10, base_amount=49_900):
    df = df.copy()
    adv_rows = []

    for _, r in df.iterrows():
        n = np.random.randint(2, max_txn)
        for i in range(n):
            adv = r.copy()
            adv["amount"] = base_amount
            adv["timestamp"] += pd.Timedelta(minutes=i)
            adv["is_adversarial"] = True
            adv_rows.append(adv)

    return pd.DataFrame(adv_rows)
