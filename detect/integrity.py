from fraudstruct.utils.validation import validate_columns


def detect_label_integrity_issues(
    df,
    label_col="is_fraud",
    delay_days=30
):
    validate_columns(df, ["timestamp", label_col])

    df["label_missing"] = df[label_col].isna()

    df["late_label"] = (
        df[label_col].notna() &
        ((df["label_timestamp"] - df["timestamp"]).dt.days > delay_days)
    )

    return df[["timestamp", label_col, "label_missing", "late_label"]]
