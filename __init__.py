from fraudstruct.core.engine import FraudstructEngine
from fraudstruct.detect.structuring import detect_structuring
from fraudstruct.detect.flow_structuring import StreamingStructuringDetector, StructuringConfig
from fraudstruct.train.adversarial import adversarial_fit
from fraudstruct.audit.reports import generate_report

__version__ = "1.0.1"

def from_dataframe(df):
    return FraudstructEngine.from_dataframe(df)
