from collections import defaultdict, deque
from datetime import datetime, timedelta


class StreamingEngine:
    """
    Lightweight streaming engine for near-real-time fraud signals.
    """

    def __init__(self, window_seconds: int):
        self.window = timedelta(seconds=window_seconds)
        self.buffers = defaultdict(deque)

    def ingest(self, event: dict):
        """
        event = {
            'entity_id': str,
            'timestamp': datetime,
            'amount': float
        }
        """
        entity = event["entity_id"]
        now = event["timestamp"]

        self.buffers[entity].append(event)

        # Evict old events
        while self.buffers[entity]:
            if now - self.buffers[entity][0]["timestamp"] > self.window:
                self.buffers[entity].popleft()
            else:
                break

        return list(self.buffers[entity])
