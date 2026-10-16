def normalize_window(window):
    """
    Normalize time window into:
    - pandas-compatible string
    - spark-compatible seconds
    """
    if isinstance(window, str):
        unit = window[-1].upper()
        value = int(window[:-1])

        # Pandas 3 removed the legacy uppercase H/D frequency aliases; M is
        # ambiguous (month-end) and must be expressed explicitly as minutes.
        if unit == "H":
            return f"{value}h", value * 3600
        if unit == "D":
            return f"{value}d", value * 86400
        if unit == "M":
            return f"{value}min", value * 60

    if isinstance(window, int):
        return f"{window}s", window

    raise ValueError("Unsupported window format")
