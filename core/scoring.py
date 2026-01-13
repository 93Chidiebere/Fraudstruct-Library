def risk_score(structuring, behavior, drift, integrity):
    return (
        structuring.astype(int) * 0.4 +
        behavior.astype(int) * 0.3 +
        drift.astype(int) * 0.2 +
        integrity.astype(int) * 0.1
    )
