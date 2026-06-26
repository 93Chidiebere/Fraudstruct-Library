from fraudstruct.detect.structuring import detect_structuring
from fraudstruct.detect.behavior import detect_behavioral_camouflage
from fraudstruct.detect.drift import detect_behavioral_drift


def run_fraudstruct(engine):
    structuring = detect_structuring(engine)
    behavior = detect_behavioral_camouflage(engine)
    drift = detect_behavioral_drift(engine)

    return {
        "structuring": structuring,
        "behavior": behavior,
        "drift": drift
    }
