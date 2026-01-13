from fraudstruct.core.statistics import distribution_shift
from fraudstruct.utils.validation import validate_columns


def detect_behavioral_drift(
    engine,
    entity_col="entity_id",
    amount_col="amount",
    baseline_window="30D",
    recent_window="7D",
    psi_threshold=0.25
):
    validate_columns(engine.df, [entity_col, "timestamp", amount_col])

    drift_df = distribution_shift(
        engine,
        entity_col,
        amount_col,
        baseline_window,
        recent_window
    )

    drift_df["drift_flag"] = drift_df["psi"] >= psi_threshold
    return drift_df
