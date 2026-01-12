from fraudstruct.utils.validation import validate_columns
from fraudstruct.utils.time import normalize_window


def detect_structuring(
    engine,
    entity_col="entity_id",
    amount_col="amount",
    window="1H",
    count_threshold=5,
    sum_threshold=300_000
):
    """
    Detect transaction structuring (smurfing) behavior.
    """

    # 1. Validate input data
    validate_columns(
        engine.df,
        [entity_col, "timestamp", amount_col]
    )

    # 2. Normalize window for pandas / spark
    pandas_window, spark_window = normalize_window(window)

    # 3. Compute rolling aggregations
    if engine.engine_type == "pandas":
        sums = engine.rolling_sum(entity_col, amount_col, pandas_window)
        counts = engine.rolling_count(entity_col, pandas_window)
    else:
        sums = engine.rolling_sum(entity_col, amount_col, spark_window)
        counts = engine.rolling_count(entity_col, spark_window)

    # 4. Combine results
    if engine.engine_type == "pandas":
        df = sums.merge(
            counts,
            on=[entity_col, "timestamp"],
            how="inner"
        )
    else:
        df = sums.join(
            counts.select(entity_col, "timestamp", "rolling_count"),
            on=[entity_col, "timestamp"],
            how="inner"
        )

    # 5. Flag structuring behavior
    df = df.withColumn(
        "structuring_flag",
        (df["rolling_sum"] >= sum_threshold) &
        (df["rolling_count"] >= count_threshold)
    ) if engine.engine_type == "spark" else df.assign(
        structuring_flag=(
            (df["rolling_sum"] >= sum_threshold) &
            (df["rolling_count"] >= count_threshold)
        )
    )

    return df
