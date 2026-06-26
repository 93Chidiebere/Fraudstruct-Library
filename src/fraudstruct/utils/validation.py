def validate_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def validate_engine(engine):
    if not hasattr(engine, "engine_type"):
        raise TypeError("Invalid Fraudstruct engine instance")
