"""Manual API smoke checks against a locally running Fraudstruct service."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid

import requests


BASE_URL = "http://127.0.0.1:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    response.raise_for_status()
    print(response.json())


def test_evaluate_endpoint():
    event = {
        "transaction_id": str(uuid.uuid4()),
        "source_account": "smoke-source",
        "destination_account": "smoke-beneficiary",
        "amount": 25_000.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": "test",
    }
    response = requests.post(f"{BASE_URL}/v1/evaluate", json=event, timeout=5)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    test_health()
    test_evaluate_endpoint()
