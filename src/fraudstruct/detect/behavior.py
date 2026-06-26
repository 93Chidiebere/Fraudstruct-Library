from fraudstruct.core.statistics import rolling_stats
from fraudstruct.utils.validation import validate_columns


def detect_behavioral_camouflage(
    engine,
    entity_col="entity_id",
    amount_col="amount",
    window="7D",
    z_thresh=3.0
):
    validate_columns(engine.df, [entity_col, "timestamp", amount_col])

    stats = rolling_stats(
        engine,
        entity_col=entity_col,
        value_col=amount_col,
        window=window
    )

    df = stats.assign(
        behavior_flag=stats["z_score"].abs() >= z_thresh
    )

    return df
