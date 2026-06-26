from fraudstruct.stream.engine import StreamingEngine
from fraudstruct.stream.detectors import detect_streaming_structuring
from fraudstruct.core.windows import normalize_window


class FraudstructStream:
    def __init__(self, window="1H"):
        _, seconds = normalize_window(window)
        self.engine = StreamingEngine(seconds)

    def process_event(self, event: dict):
        events = self.engine.ingest(event)
        return detect_streaming_structuring(events)
