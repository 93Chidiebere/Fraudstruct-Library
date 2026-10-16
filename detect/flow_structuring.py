"""Incremental detector for two-hop S/K/W structuring patterns.

The detector looks for a source account that sends below-ceiling transfers to
multiple mule accounts, which then forward funds to a common beneficiary. It
is a pattern detector, not a determination that an account or transaction is
criminal.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Deque, Dict, Optional, Set, Tuple


@dataclass(frozen=True)
class StructuringConfig:
    """Configurable S/K/W and flow constraints.

    ``minimum_total_amount`` is the minimum aggregate source-to-mule amount
    (S). If ``target_total_amount`` is supplied, the aggregate must instead be
    within ``target_amount_tolerance`` of that target.
    """

    window_seconds: int = 3600
    min_mules: int = 3
    minimum_total_amount: float = 300_000.0
    max_leg_amount: Optional[float] = 100_000.0
    flow_conservation_tolerance: float = 0.10
    target_total_amount: Optional[float] = None
    target_amount_tolerance: float = 0.05
    max_tracked_pairs: int = 100_000
    max_active_events: int = 500_000
    max_cached_accounts: int = 500_000

    def __post_init__(self):
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.min_mules < 2:
            raise ValueError("min_mules must be at least 2")
        if not math.isfinite(self.minimum_total_amount) or self.minimum_total_amount <= 0:
            raise ValueError("minimum_total_amount must be a finite positive value")
        if self.max_leg_amount is not None and (
            not math.isfinite(self.max_leg_amount) or self.max_leg_amount <= 0
        ):
            raise ValueError("max_leg_amount must be None or a finite positive value")
        if not 0 <= self.flow_conservation_tolerance < 1:
            raise ValueError("flow_conservation_tolerance must be in [0, 1)")
        if self.target_total_amount is not None and (
            not math.isfinite(self.target_total_amount) or self.target_total_amount <= 0
        ):
            raise ValueError("target_total_amount must be None or a finite positive value")
        if not 0 <= self.target_amount_tolerance < 1:
            raise ValueError("target_amount_tolerance must be in [0, 1)")
        if self.max_tracked_pairs <= 0:
            raise ValueError("max_tracked_pairs must be positive")
        if self.max_active_events <= 0:
            raise ValueError("max_active_events must be positive")
        if self.max_cached_accounts <= 0:
            raise ValueError("max_cached_accounts must be positive")


@dataclass
class _EdgeWindow:
    events: Deque[Tuple[int, datetime, float, bool]] = field(default_factory=deque)
    eligible_sum: float = 0.0
    eligible_count: int = 0


class StreamingStructuringDetector:
    """Maintain bounded event-time state and detect source→mule→beneficiary flows.

    Events must be supplied in nondecreasing event-time order. State expires
    incrementally as the watermark advances, so ingestion cost is proportional
    to expired events and candidate mules, not to the full graph history.
    """

    def __init__(self, config: Optional[StructuringConfig] = None):
        self.config = config or StructuringConfig()
        self._edges: Dict[Tuple[str, str], _EdgeWindow] = {}
        self._expiry: Deque[Tuple[datetime, int, Tuple[str, str]]] = deque()
        self._incoming_by_mule: Dict[str, Set[str]] = defaultdict(set)
        self._outgoing_by_mule: Dict[str, Set[str]] = defaultdict(set)
        self._mules_by_source: Dict[str, Set[str]] = defaultdict(set)
        self._sequence = 0
        self._latest_timestamp: Optional[datetime] = None
        self._seen_ids: Dict[str, datetime] = {}
        self._seen_id_expiry: Deque[Tuple[datetime, str]] = deque()

    @property
    def active_pair_count(self) -> int:
        return len(self._edges)

    @property
    def active_event_count(self) -> int:
        return len(self._expiry)

    @staticmethod
    def _utc_timestamp(value) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime):
            raise TypeError("timestamp must be a datetime or ISO-8601 string")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _unlink_pair(self, source: str, destination: str) -> None:
        sources = self._incoming_by_mule.get(destination)
        if sources is not None:
            sources.discard(source)
            if not sources:
                self._incoming_by_mule.pop(destination, None)
        beneficiaries = self._outgoing_by_mule.get(source)
        if beneficiaries is not None:
            beneficiaries.discard(destination)
            if not beneficiaries:
                self._outgoing_by_mule.pop(source, None)
        mules = self._mules_by_source.get(source)
        if mules is not None:
            mules.discard(destination)
            if not mules:
                self._mules_by_source.pop(source, None)

    def _expire(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.config.window_seconds)
        while self._seen_id_expiry and self._seen_id_expiry[0][0] < cutoff:
            timestamp, transaction_id = self._seen_id_expiry.popleft()
            if self._seen_ids.get(transaction_id) == timestamp:
                del self._seen_ids[transaction_id]
        while self._expiry and self._expiry[0][0] < cutoff:
            _, sequence, key = self._expiry.popleft()
            window = self._edges.get(key)
            if window is None or not window.events or window.events[0][0] != sequence:
                continue
            _, _, amount, eligible = window.events.popleft()
            if eligible:
                window.eligible_sum -= amount
                window.eligible_count -= 1
            if not window.events:
                del self._edges[key]
                self._unlink_pair(*key)

    def _add(self, source: str, destination: str, amount: float, timestamp: datetime) -> None:
        if len(self._expiry) >= self.config.max_active_events:
            raise BufferError("active event capacity reached; increase max_active_events")
        key = (source, destination)
        window = self._edges.get(key)
        if window is None:
            if len(self._edges) >= self.config.max_tracked_pairs:
                raise BufferError("active edge-pair capacity reached; increase max_tracked_pairs")
            window = self._edges[key] = _EdgeWindow()
            self._incoming_by_mule[destination].add(source)
            self._outgoing_by_mule[source].add(destination)
            self._mules_by_source[source].add(destination)
        eligible = self.config.max_leg_amount is None or amount <= self.config.max_leg_amount
        self._sequence += 1
        window.events.append((self._sequence, timestamp, amount, eligible))
        self._expiry.append((timestamp, self._sequence, key))
        if eligible:
            window.eligible_sum += amount
            window.eligible_count += 1

    def _evaluate(self, source: str, beneficiary: str, now: datetime):
        if source == beneficiary:
            return None
        mule_amounts = []
        for mule in self._mules_by_source.get(source, ()):
            if mule == source or mule == beneficiary:
                continue
            inbound = self._edges.get((source, mule))
            outbound = self._edges.get((mule, beneficiary))
            if not (inbound and outbound and inbound.eligible_count and outbound.eligible_count):
                continue
            inbound_events = [event for event in inbound.events if event[3]]
            outbound_events = [event for event in outbound.events if event[3]]
            first_in_sequence = inbound_events[0][0]
            last_out_sequence = outbound_events[-1][0]
            # An onward edge must follow a source-to-mule edge. Ignore later
            # inflows that occur after the last onward transfer in this episode.
            if last_out_sequence <= first_in_sequence:
                continue
            inbound_amount = sum(
                event[2] for event in inbound_events if event[0] <= last_out_sequence
            )
            outbound_amount = sum(
                event[2] for event in outbound_events if event[0] > first_in_sequence
            )
            if inbound_amount > 0 and outbound_amount > 0:
                mule_difference = abs(inbound_amount - outbound_amount) / max(
                    inbound_amount, outbound_amount
                )
                if mule_difference > self.config.flow_conservation_tolerance:
                    continue
                mule_amounts.append((mule, inbound_amount, outbound_amount))
        if len(mule_amounts) < self.config.min_mules:
            return None
        total_in = sum(item[1] for item in mule_amounts)
        total_out = sum(item[2] for item in mule_amounts)
        if total_in <= 0 or total_out <= 0:
            return None
        if self.config.target_total_amount is None:
            if total_in < self.config.minimum_total_amount:
                return None
        else:
            allowed = self.config.target_total_amount * self.config.target_amount_tolerance
            if abs(total_in - self.config.target_total_amount) > allowed:
                return None
        relative_difference = abs(total_in - total_out) / max(total_in, total_out)
        if relative_difference > self.config.flow_conservation_tolerance:
            return None
        return {
            "source_account": source,
            "beneficiary_account": beneficiary,
            "mule_accounts": sorted(item[0] for item in mule_amounts),
            "mule_count": len(mule_amounts),
            "source_total": round(total_in, 2),
            "beneficiary_total": round(total_out, 2),
            "relative_flow_difference": round(relative_difference, 6),
            "window_start": (now - timedelta(seconds=self.config.window_seconds)).isoformat(),
            "window_end": now.isoformat(),
            "criteria": {
                "minimum_total_amount": self.config.minimum_total_amount,
                "target_total_amount": self.config.target_total_amount,
                "min_mules": self.config.min_mules,
                "window_seconds": self.config.window_seconds,
                "max_leg_amount": self.config.max_leg_amount,
                "flow_conservation_tolerance": self.config.flow_conservation_tolerance,
            },
        }

    def ingest(self, event: dict):
        """Add one transaction and return a matching pattern, or ``None``."""
        try:
            transaction_id = event.get("transaction_id")
            transaction_id = str(transaction_id).strip() if transaction_id is not None else None
            source = str(event["source_account"]).strip()
            destination = str(event["destination_account"]).strip()
            amount = float(event["amount"])
            timestamp = self._utc_timestamp(event["timestamp"])
        except KeyError as exc:
            raise ValueError(f"missing required event field: {exc.args[0]}") from exc
        if not source or not destination:
            raise ValueError("source_account and destination_account must be non-empty")
        if source == destination:
            raise ValueError("self-transfer events are not valid detector input")
        if transaction_id == "":
            raise ValueError("transaction_id must be non-empty when provided")
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("amount must be a finite positive value")
        if self._latest_timestamp is not None and timestamp < self._latest_timestamp:
            raise ValueError("events must arrive in nondecreasing event-time order")
        self._expire(timestamp)
        if transaction_id is not None and transaction_id in self._seen_ids:
            self._latest_timestamp = timestamp
            raise ValueError("duplicate transaction_id within the active event-time window")
        self._add(source, destination, amount, timestamp)
        self._latest_timestamp = timestamp
        if transaction_id is not None:
            self._seen_ids[transaction_id] = timestamp
            self._seen_id_expiry.append((timestamp, transaction_id))

        # Only inspect candidates adjacent to the newly updated account pair.
        candidates = {
            (source, beneficiary)
            for beneficiary in self._outgoing_by_mule.get(destination, ())
        }
        candidates.update(
            (upstream_source, destination)
            for upstream_source in self._incoming_by_mule.get(source, ())
        )
        for candidate_source, beneficiary in sorted(candidates):
            alert = self._evaluate(candidate_source, beneficiary, timestamp)
            if alert is not None:
                return alert
        return None
