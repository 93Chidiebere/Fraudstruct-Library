from fraudstruct.core.engine import FraudstructEngine
from fraudstruct.detect.structuring import detect_structuring
from fraudstruct.train.adversarial import adversarial_fit
from fraudstruct.audit.reports import generate_report

def from_dataframe(df):
    return FraudstructEngine.from_dataframe(df)
