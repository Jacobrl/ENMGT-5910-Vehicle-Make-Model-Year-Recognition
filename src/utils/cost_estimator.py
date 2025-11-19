def estimate_cost(damage_info):

    if not damage_info:
        return {"usd": None, "mxn": None}

    # Temporary dummy values for pipeline testing
    return {
        "usd": 500.0,      # fake number
        "mxn": 500.0 * 17  # fake number
    }
