"""Constraint-checked generator for labeled two-hop structuring scenarios."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def _split_cents(total_cents: int, n_splits: int, rng, max_leg_cents: Optional[int]):
    if total_cents < n_splits:
        raise ValueError("total amount must provide at least one cent per mule")
    cap = total_cents if max_leg_cents is None else max_leg_cents
    if total_cents > n_splits * cap:
        raise ValueError("total amount cannot be split under max_leg_amount with n_splits")
    remaining = total_cents
    result = []
    for index in range(n_splits - 1):
        slots_left = n_splits - index - 1
        low = max(1, remaining - slots_left * cap)
        high = min(cap, remaining - slots_left)
        if low > high:
            raise ValueError("cannot construct positive transfers under the leg ceiling")
        amount = int(rng.integers(low, high + 1))
        result.append(amount)
        remaining -= amount
    if not 1 <= remaining <= cap:
        raise ValueError("cannot construct final transfer under the leg ceiling")
    result.append(remaining)
    rng.shuffle(result)
    return result


def _scale_cents_by_ratio(amounts, ratio: float):
    target = int(round(sum(amounts) * ratio))
    exact = [amount * ratio for amount in amounts]
    scaled = [int(math.floor(value)) for value in exact]
    remainder = target - sum(scaled)
    order = sorted(range(len(amounts)), key=lambda i: exact[i] - scaled[i], reverse=True)
    for index in order[:remainder]:
        scaled[index] += 1
    if any(value < 1 for value in scaled):
        raise ValueError("outflow_ratio produces a transfer smaller than one cent")
    return scaled


def simulate_graph_splitting(
    df=None,
    source="A",
    dest="B",
    amount=500_000.0,
    timestamp=None,
    n_splits=5,
    mule_candidates=None,
    source_col="source_account",
    dest_col="destination_account",
    amount_col="amount",
    time_col="timestamp",
    max_leg_amount=None,
    window_seconds=3600,
    outflow_ratio=1.0,
    seed=None,
):
    """Generate labeled source→mule→beneficiary transactions.

    First-hop amounts sum to the requested total (in cents). Second-hop
    amounts preserve that total by default; ``outflow_ratio`` can introduce a
    known loss for conservation-tolerance experiments. The returned labels are
    scenario ground truth, not labels inferred by a detector.
    """
    del df  # retained only for compatibility with the earlier function API
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if not isinstance(window_seconds, int) or window_seconds <= 0:
        raise ValueError("window_seconds must be a positive integer")
    if not math.isfinite(float(amount)) or float(amount) <= 0:
        raise ValueError("amount must be a finite positive value")
    if not math.isfinite(float(outflow_ratio)) or not 0 < outflow_ratio <= 1:
        raise ValueError("outflow_ratio must be in (0, 1]")
    total_cents = int(round(float(amount) * 100))
    if total_cents <= 0:
        raise ValueError("amount must be at least one cent")
    max_cents = None
    if max_leg_amount is not None:
        if not math.isfinite(float(max_leg_amount)) or float(max_leg_amount) <= 0:
            raise ValueError("max_leg_amount must be None or a finite positive value")
        max_cents = int(math.floor(float(max_leg_amount) * 100 + 1e-9))
        if max_cents <= 0:
            raise ValueError("max_leg_amount must be at least one cent")

    rng = np.random.default_rng(seed)
    if timestamp is None:
        timestamp = pd.Timestamp.now(tz="UTC")
    else:
        timestamp = pd.Timestamp(timestamp)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    if mule_candidates is None:
        mules = [f"MULE_{i:04d}" for i in range(1, n_splits + 1)]
    else:
        candidates = list(dict.fromkeys(str(value) for value in mule_candidates))
        if len(candidates) < n_splits:
            raise ValueError("mule_candidates must contain at least n_splits distinct accounts")
        mules = rng.choice(candidates, size=n_splits, replace=False).tolist()
    if source in mules or dest in mules or source == dest:
        raise ValueError("source, beneficiary, and mule accounts must be distinct")

    first_hop = _split_cents(total_cents, n_splits, rng, max_cents)
    second_hop = _scale_cents_by_ratio(first_hop, float(outflow_ratio))
    max_first_offset = window_seconds // 2
    offsets = np.linspace(0, max_first_offset, n_splits, dtype=int).tolist()
    rows = []
    for index, mule in enumerate(mules):
        first_at = timestamp + pd.Timedelta(seconds=offsets[index])
        remaining_window = max(0, window_seconds - offsets[index])
        max_delay = min(max(1, window_seconds // 4), max(1, remaining_window))
        delay = int(rng.integers(1, max_delay + 1)) if remaining_window else 0
        forwarded_at = first_at + pd.Timedelta(seconds=delay)
        rows.append({
            source_col: str(source), dest_col: str(mule),
            amount_col: first_hop[index] / 100.0, time_col: first_at,
            "is_fraud": True, "is_adversarial": True,
            "scenario_role": "source_to_mule", "scenario_id": f"S-K-W-{seed if seed is not None else '0'}",
        })
        rows.append({
            source_col: str(mule), dest_col: str(dest),
            amount_col: second_hop[index] / 100.0, time_col: forwarded_at,
            "is_fraud": True, "is_adversarial": True,
            "scenario_role": "mule_to_beneficiary", "scenario_id": f"S-K-W-{seed if seed is not None else '0'}",
        })
    return pd.DataFrame(rows).sort_values(time_col, kind="stable").reset_index(drop=True)
