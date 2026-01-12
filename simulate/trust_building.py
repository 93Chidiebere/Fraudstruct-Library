def simulate_trust_building(df, days=30):
    df = df.sort_values("timestamp").copy()
    cutoff = df["timestamp"].min() + pd.Timedelta(days=days)

    df["is_adversarial"] = False
    df.loc[df["timestamp"] > cutoff, "amount"] *= 4
    df.loc[df["timestamp"] > cutoff, "is_adversarial"] = True

    return df
