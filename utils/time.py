def normalize_window(window):
    """
    Normalize time window into:
    - pandas-compatible string
    - spark-compatible seconds
    """
    if isinstance(window, str):
        unit = window[-1].upper()
        value = int(window[:-1])

        if unit == "H":
            return window, value * 3600
        if unit == "D":
            return window, value * 86400
        if unit == "M":
            return window, value * 60

    if isinstance(window, int):
        return f"{window}S", window

    raise ValueError("Unsupported window format")
