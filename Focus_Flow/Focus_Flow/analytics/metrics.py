"""
Analytics metrics: Productivity score calculation.

Computes productivity_ratio = productive_time / total_time (0.0-1.0).
"""


def calculate_productivity_score(
    productive_seconds: float, total_seconds: float
) -> float:
    """
    Calculate productivity score as ratio of productive time to total time.

    Args:
        productive_seconds: Seconds counted as productive (attentive).
        total_seconds: Total session duration in seconds.

    Returns:
        Score between 0.0 and 1.0. Returns 0.0 if total_seconds is 0 or negative.
    """
    if total_seconds <= 0:
        return 0.0
    ratio = productive_seconds / total_seconds
    return min(1.0, max(0.0, ratio))


# Alias for productivity ratio
calculate_productivity_ratio = calculate_productivity_score
