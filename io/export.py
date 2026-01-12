import json
import pandas as pd

def export_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def export_dataframe(df, path):
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported file format")
