import numpy as np

def simulate_camouflage(df):
    df = df.copy()
    mu = df["amount"].mean()
    sigma = df["amount"].std()

    df["amount"] = np.random.normal(mu, sigma * 0.1, len(df))
    df["is_adversarial"] = True
    return df
