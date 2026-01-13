import pandas as pd


def normalize_window(window: str):
    """
    Normalizes time windows across batch and streaming contexts.

    Examples:
    - '1H'  -> pandas.Timedelta('1H'), streaming seconds
    - '7D'  -> pandas.Timedelta('7D'), streaming seconds
    """
    pd_window = pd.Timedelta(window)
    stream_seconds = int(pd_window.total_seconds())

    return pd_window, stream_seconds

