def simulate_label_poisoning(df, label_col="label", rate=0.4):
    df = df.copy()
    idx = df[df[label_col] == 1].sample(frac=rate).index
    df.loc[idx, label_col] = 0
    df.loc[idx, "label_poisoned"] = True
    return df
