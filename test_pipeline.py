"""Local API smoke/latency example; measurements are not a production SLA."""
from __future__ import annotations

from datetime import datetime, timezone
import statistics
import time
import uuid

import requests

from fraudstruct.simulate.graph_attack import simulate_graph_splitting


BASE_URL = "http://127.0.0.1:8000"


def run_smoke_check():
    base_time = datetime.now(timezone.utc)
    attack = simulate_graph_splitting(
        source="source", dest="beneficiary", amount=600_000,
        timestamp=base_time, n_splits=6, max_leg_amount=100_000,
        window_seconds=600, seed=17,
    )
    latencies_ms = []
    decisions = []
    for row in attack.to_dict(orient="records"):
        event = {
            "transaction_id": str(uuid.uuid4()),
            "source_account": row["source_account"],
            "destination_account": row["destination_account"],
            "amount": float(row["amount"]),
            "timestamp": row["timestamp"].isoformat(),
        }
        start = time.perf_counter()
        response = requests.post(f"{BASE_URL}/v1/evaluate", json=event, timeout=5)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        latencies_ms.append(elapsed)
        decisions.append(response.json()["decision"])

    print(f"Events processed: {len(decisions)}")
    print(f"FLAG decisions: {decisions.count('FLAG')}")
    print(f"Client-observed latency median: {statistics.median(latencies_ms):.3f} ms")
    print("Latency includes local HTTP/client overhead; this is not a banking SLA result.")


if __name__ == "__main__":
    run_smoke_check()
