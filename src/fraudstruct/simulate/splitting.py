"""Compatibility wrapper for generating explicitly labeled flow episodes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudstruct.simulate.graph_attack import simulate_graph_splitting


def simulate_splitting(
    df,
    max_txn=10,
    base_amount=49_900.0,
    seed=None,
    window_seconds=3600,
):
    """Create positive S/K/W flow episodes from event rows.

    ``df`` must contain source_account, destination_account, amount, and
    timestamp. Each input row supplies the total scenario amount and endpoints;
    ``base_amount`` is the maximum amount of an individual transfer leg.
    Every output transaction has an explicit positive ``is_fraud`` label.
    """
    required = {"source_account", "destination_account", "amount", "timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required event columns: {sorted(missing)}")
    if not isinstance(max_txn, int) or max_txn < 2:
        raise ValueError("max_txn must be an integer of at least 2")
    if not np.isfinite(base_amount) or base_amount <= 0:
        raise ValueError("base_amount must be a finite positive value")
    rng = np.random.default_rng(seed)
    episodes = []
    for index, row in df.reset_index(drop=True).iterrows():
        amount = float(row["amount"])
        min_splits = max(2, int(np.ceil(amount / float(base_amount))))
        if min_splits > max_txn:
            raise ValueError(
                f"row {index} amount cannot be split below base_amount using max_txn"
            )
        n_splits = int(rng.integers(min_splits, max_txn + 1))
        episode = simulate_graph_splitting(
            source=str(row["source_account"]),
            dest=str(row["destination_account"]),
            amount=amount,
            timestamp=row["timestamp"],
            n_splits=n_splits,
            max_leg_amount=base_amount,
            window_seconds=window_seconds,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        episode["scenario_id"] = f"generated-{seed if seed is not None else 'unseeded'}-{index}"
        episodes.append(episode)
    if not episodes:
        return pd.DataFrame(columns=list(required) + ["is_fraud", "is_adversarial", "scenario_role", "scenario_id"])
    return pd.concat(episodes, ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
