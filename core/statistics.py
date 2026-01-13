import pandas as pd
import numpy as np


def rolling_stats(engine, entity_col, value_col, window):
    df = engine.df.copy()
    df = df.sort_values("timestamp")

    grouped = df.groupby(entity_col)

    df["rolling_mean"] = grouped[value_col].transform(
        lambda x: x.rolling(window).mean()
    )
    df["rolling_std"] = grouped[value_col].transform(
        lambda x: x.rolling(window).std()
    )

    df["z_score"] = (
        (df[value_col] - df["rolling_mean"]) / df["rolling_std"]
    )

    return df


def distribution_shift(
    engine,
    entity_col,
    value_col,
    baseline_window,
    recent_window
):
    df = engine.df.copy()

    baseline = df[df["timestamp"] < df["timestamp"].max() - pd.Timedelta(recent_window)]
    recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(recent_window)]

    psi = []

    for entity, base_grp in baseline.groupby(entity_col):
        recent_grp = recent[recent[entity_col] == entity]

        if len(recent_grp) == 0:
            continue

        psi_val = np.mean(
            (base_grp[value_col].mean() - recent_grp[value_col].mean()) ** 2
        )
        psi.append((entity, psi_val))

    return pd.DataFrame(psi, columns=[entity_col, "psi"])
