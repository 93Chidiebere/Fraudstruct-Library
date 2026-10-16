# Fraudstruct

Fraudstruct is a Python library for detecting transaction structuring patterns
and exploring graph-based fraud analysis. Its streaming detector identifies
two-hop flows from a source, through multiple intermediary accounts, to a
shared beneficiary.

## Install

```bash
python -m pip install fraudstruct
```

## Quick start: detect a structuring flow

The detector evaluates events in event-time order. Configure the aggregate
source amount **S**, minimum number of distinct intermediaries **K**, time
window **W**, maximum transfer size, and permitted flow difference. This
example creates an alert after three mule accounts forward the funds.

```python
from datetime import datetime, timedelta, timezone

from fraudstruct import StreamingStructuringDetector, StructuringConfig

config = StructuringConfig(
    window_seconds=3600,                 # W
    min_mules=3,                          # K
    minimum_total_amount=500.0,           # minimum S, in the transaction currency
    max_leg_amount=250.0,                 # maximum counted transfer
    flow_conservation_tolerance=0.05,     # 5% allowed difference per mule and overall
)
detector = StreamingStructuringDetector(config)
start = datetime.now(timezone.utc)

transactions = [
    # Source splits 600 across three mule accounts.
    ("source", "mule-1", 200.00, 0),
    ("source", "mule-2", 200.00, 1),
    ("source", "mule-3", 200.00, 2),
    # Each mule forwards the funds to the same beneficiary.
    ("mule-1", "beneficiary", 200.00, 10),
    ("mule-2", "beneficiary", 200.00, 11),
    ("mule-3", "beneficiary", 200.00, 12),
]

for index, (source, destination, amount, offset) in enumerate(transactions):
    alert = detector.ingest({
        "transaction_id": f"tx-{index + 1}",
        "source_account": source,
        "destination_account": destination,
        "amount": amount,
        "timestamp": start + timedelta(seconds=offset),
    })
    if alert:
        print("Structuring pattern detected:")
        print(alert)
```

`ingest()` returns `None` until an event completes a matching pattern. An alert
contains the source, beneficiary, matched mule accounts, aggregate amounts,
flow difference, and the criteria used. Alerts identify patterns for review;
they do not determine criminal intent.

## Run on a transaction dataset

The simulator below is **synthetic data**. To analyze a real dataset with 200,
200,000, or more rows, load its transaction records, map its account, amount,
and time columns, sort by event time, and pass each row to `ingest()`. The
detector processes records one at a time; the example keeps the input table in
memory so it can sort it. Replace the example CSV column names with the names
in your dataset.

```python
import pandas as pd

from fraudstruct import StreamingStructuringDetector, StructuringConfig

# Example schema mapping. Change the right-hand column names for your dataset.
raw = pd.read_csv(
    "transactions.csv",
    dtype={"sender": "string", "receiver": "string"},
)
transactions = raw.rename(columns={
    "tx_id": "transaction_id",
    "sender": "source_account",
    "receiver": "destination_account",
    "value": "amount",
    "event_time": "timestamp",
})

# If the dataset has no transaction ID, generate a unique row ID.
if "transaction_id" not in transactions.columns:
    transactions["transaction_id"] = [
        f"row-{i}" for i in range(len(transactions))
    ]

required = [
    "transaction_id", "source_account", "destination_account", "amount", "timestamp"
]
missing = sorted(set(required) - set(transactions.columns))
if missing:
    raise ValueError(f"Dataset is missing mapped columns: {missing}")

transactions["timestamp"] = pd.to_datetime(
    transactions["timestamp"], utc=True, errors="coerce"
)
transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce")
transactions = transactions.dropna(subset=required)
transactions = transactions.sort_values("timestamp", kind="stable")

detector = StreamingStructuringDetector(StructuringConfig(
    window_seconds=3600,
    min_mules=3,
    minimum_total_amount=300000.0,
    max_leg_amount=100000.0,
    flow_conservation_tolerance=0.10,
))

for row in transactions[required].itertuples(index=False, name=None):
    transaction_id, source, destination, amount, timestamp = row
    alert = detector.ingest({
        "transaction_id": str(transaction_id),
        "source_account": str(source),
        "destination_account": str(destination),
        "amount": float(amount),
        "timestamp": timestamp.to_pydatetime(),
    })
    if alert is not None:
        print("Matched pattern:", alert)
```

This runs the rule-based structuring detector; it does not train XGBoost or use
dataset fraud labels to make detections. Keep any labels from the dataset for
the evaluation stage, and check whether they identify structuring episodes or
only individual fraudulent transactions before treating them as episode-level
ground truth. Choose S, K, W, currency units, and column mappings to match the
dataset and experiment.

## Generate a synthetic scenario

`simulate_graph_splitting` creates synthetic data, not records from a public
banking dataset. Use it for controlled development tests or explicitly labeled
simulator experiments. Generated rows include `is_fraud`, `is_adversarial`,
`scenario_role`, and `scenario_id` labels.

```python
from fraudstruct.simulate.graph_attack import simulate_graph_splitting

events = simulate_graph_splitting(
    source="source",
    dest="beneficiary",
    amount=600.00,
    timestamp="2026-01-01T00:00:00Z",
    n_splits=3,
    max_leg_amount=250.00,
    window_seconds=3600,
    outflow_ratio=1.0,
    seed=17,
)

print(events[[
    "source_account", "destination_account", "amount", "timestamp",
    "is_fraud", "is_adversarial", "scenario_role",
]])
```

`outflow_ratio` controls how much each mule forwards relative to its incoming
leg. For example, `outflow_ratio=0.9` models a 10% flow loss. Simulator output
is generated scenario data and should be described as such in reports.

## Run the streaming API

Fraudstruct includes a FastAPI service. Set thresholds in the units of your
transaction amounts, then start the server:

```bash
export FRAUDSTRUCT_WINDOW_SECONDS=3600
export FRAUDSTRUCT_MIN_MULES=3
export FRAUDSTRUCT_MIN_TOTAL_AMOUNT=300000
export FRAUDSTRUCT_MAX_LEG_AMOUNT=100000
export FRAUDSTRUCT_FLOW_TOLERANCE=0.10
uvicorn fraudstruct.stream.api:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell, set variables with `$env:FRAUDSTRUCT_WINDOW_SECONDS = "3600"`
and the corresponding names above.

Send a transaction to `POST /v1/evaluate`:

```bash
curl -X POST http://127.0.0.1:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"tx-1","source_account":"source","destination_account":"mule-1","amount":200.0,"timestamp":"2026-01-01T00:00:00Z"}'
```

The response includes a `decision` (`APPROVE`, `FLAG`, or `REVIEW`), reasons,
the matching structuring alert if present, and `latency_ms`. The service also
provides `GET /health`, `POST /v1/train` for cold-path graph model training,
and `POST /v1/reset` to clear in-memory state.

## Configuration reference

| `StructuringConfig` option | Purpose |
| --- | --- |
| `window_seconds` | Maximum event-time window **W** |
| `min_mules` | Minimum distinct intermediaries **K** |
| `minimum_total_amount` | Minimum aggregate source-to-mule amount **S** |
| `max_leg_amount` | Largest transfer included in a candidate flow; use `None` to disable the cap |
| `flow_conservation_tolerance` | Maximum relative inflow/outflow difference per mule and across the pattern |
| `target_total_amount` | Optional target aggregate amount for **S** |
| `target_amount_tolerance` | Relative range allowed around `target_total_amount` |
| `max_tracked_pairs` | Maximum active account-pair state |
| `max_active_events` | Maximum active events retained in the rolling window |
| `max_cached_accounts` | Maximum accounts retained in bounded caches |

Events must arrive in nondecreasing event-time order. Expired events are
removed as the event-time window advances. Capacity limits are configurable.

## License

MIT. Dataset licenses and terms are separate and must be checked at their
respective sources.
