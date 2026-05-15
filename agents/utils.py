def american_to_decimal(odds: int) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        odds: American odds (e.g., 450 for +450, -110 for -110)

    Returns:
        Decimal odds (what you get back per unit staked, including stake)
    """
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1
