def detect_streaming_structuring(
    events,
    sum_threshold=300_000,
    count_threshold=5
):
    total = sum(e["amount"] for e in events)
    count = len(events)

    return {
        "rolling_sum": total,
        "rolling_count": count,
        "structuring_flag": (
            total >= sum_threshold and count >= count_threshold
        )
    }
