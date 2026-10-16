from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudstruct.detect.flow_structuring import (  # noqa: E402
    StreamingStructuringDetector,
    StructuringConfig,
)
from fraudstruct.simulate.graph_attack import simulate_graph_splitting  # noqa: E402


def config(**overrides):
    values = {
        "window_seconds": 3600,
        "min_mules": 3,
        "minimum_total_amount": 300.0,
        "max_leg_amount": 100.0,
        "flow_conservation_tolerance": 0.10,
    }
    values.update(overrides)
    return StructuringConfig(**values)


def feed_balanced_episode(detector, outflow=95.0, start=None):
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = None
    for index in range(3):
        result = detector.ingest({
            "source_account": "source",
            "destination_account": f"mule-{index}",
            "amount": 100.0,
            "timestamp": start + timedelta(seconds=index),
        })
    for index in range(3):
        result = detector.ingest({
            "source_account": f"mule-{index}",
            "destination_account": "beneficiary",
            "amount": outflow,
            "timestamp": start + timedelta(seconds=30 + index),
        })
    return result


class FlowStructuringTests(unittest.TestCase):
    def test_detects_two_hop_skw_and_reports_flow(self):
        alert = feed_balanced_episode(StreamingStructuringDetector(config()))
        self.assertIsNotNone(alert)
        self.assertEqual(alert["source_account"], "source")
        self.assertEqual(alert["beneficiary_account"], "beneficiary")
        self.assertEqual(alert["mule_count"], 3)
        self.assertEqual(alert["source_total"], 300.0)
        self.assertEqual(alert["beneficiary_total"], 285.0)
        self.assertEqual(alert["relative_flow_difference"], 0.05)

    def test_no_alert_when_fewer_than_k_mules(self):
        detector = StreamingStructuringDetector(config(min_mules=4))
        self.assertIsNone(feed_balanced_episode(detector))

    def test_no_alert_when_flow_conservation_is_outside_tolerance(self):
        detector = StreamingStructuringDetector(config(flow_conservation_tolerance=0.05))
        self.assertIsNone(feed_balanced_episode(detector, outflow=80.0))

    def test_one_mules_flow_imbalance_cannot_be_hidden_by_another(self):
        detector = StreamingStructuringDetector(config())
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(3):
            detector.ingest({
                "source_account": "source",
                "destination_account": f"mule-{index}",
                "amount": 100.0,
                "timestamp": start + timedelta(seconds=index),
            })
        alert = None
        for index, outgoing in enumerate((100.0, 80.0, 110.0)):
            alert = detector.ingest({
                "source_account": f"mule-{index}",
                "destination_account": "beneficiary",
                "amount": outgoing,
                "timestamp": start + timedelta(seconds=30 + index),
            })
        self.assertIsNone(alert)

    def test_no_alert_when_any_leg_is_over_the_ceiling(self):
        detector = StreamingStructuringDetector(config(max_leg_amount=90.0))
        self.assertIsNone(feed_balanced_episode(detector))

    def test_outflows_before_source_funding_do_not_form_a_pattern(self):
        detector = StreamingStructuringDetector(config())
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(3):
            detector.ingest({
                "source_account": f"mule-{index}",
                "destination_account": "beneficiary",
                "amount": 95.0,
                "timestamp": start + timedelta(seconds=index),
            })
        alert = None
        for index in range(3):
            alert = detector.ingest({
                "source_account": "source",
                "destination_account": f"mule-{index}",
                "amount": 100.0,
                "timestamp": start + timedelta(seconds=30 + index),
            })
        self.assertIsNone(alert)

    def test_expired_edges_are_removed_and_late_events_rejected(self):
        detector = StreamingStructuringDetector(config(window_seconds=10))
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        detector.ingest({
            "source_account": "source",
            "destination_account": "mule",
            "amount": 10.0,
            "timestamp": start,
        })
        detector.ingest({
            "source_account": "other",
            "destination_account": "next",
            "amount": 10.0,
            "timestamp": start + timedelta(seconds=11),
        })
        self.assertEqual(detector.active_pair_count, 1)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            detector.ingest({
                "source_account": "late",
                "destination_account": "account",
                "amount": 1.0,
                "timestamp": start + timedelta(seconds=2),
            })

    def test_duplicate_transaction_ids_are_not_counted_twice(self):
        detector = StreamingStructuringDetector(config())
        event = {
            "transaction_id": "tx-1",
            "source_account": "source",
            "destination_account": "mule",
            "amount": 100.0,
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        detector.ingest(event)
        with self.assertRaisesRegex(ValueError, "duplicate transaction_id"):
            detector.ingest(event)
        self.assertEqual(detector.active_event_count, 1)

    def test_simulator_conserves_amounts_and_sets_ground_truth_labels(self):
        events = simulate_graph_splitting(
            source="source",
            dest="beneficiary",
            amount=600.0,
            timestamp="2026-01-01T00:00:00Z",
            n_splits=6,
            max_leg_amount=100.0,
            window_seconds=600,
            seed=7,
        )
        first = events.loc[events.scenario_role.eq("source_to_mule")]
        second = events.loc[events.scenario_role.eq("mule_to_beneficiary")]
        self.assertEqual(round(first.amount.sum(), 2), 600.0)
        self.assertEqual(round(second.amount.sum(), 2), 600.0)
        self.assertLessEqual(events.amount.max(), 100.0)
        self.assertTrue(events.is_fraud.all())
        self.assertTrue(events.is_adversarial.all())
        for _, group in events.groupby("scenario_role"):
            self.assertEqual(len(group), 6)
        self.assertLessEqual(
            events.timestamp.max() - events.timestamp.min(), pd.Timedelta(seconds=600)
        )


if __name__ == "__main__":
    unittest.main()

