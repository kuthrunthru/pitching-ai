"""
Metric scoring calibration bands.

Centralized definition of score bands for all metrics, allowing easy tuning
without modifying measurement logic.
"""

# Score bands define target ranges for each metric in raw units
# Structure: {"metric_name": {"bad": value, "ok": value, "good": value}}
# For metrics where higher is better: value increases from bad -> ok -> good
# For metrics where lower is better: value decreases from bad -> ok -> good
# The score_from_bands() function handles both cases via higher_is_better parameter

SCORE_BANDS = {
    # Stride Length (ratio) - uses custom scoring function (score_stride_length_ratio)
    # Thresholds adjusted for shoulder-to-ankle normalization: GREEN (>= 3.5), YELLOW (2.7-3.5), RED (< 2.7)
    # Note: This entry is for reference only; actual scoring is handled by score_stride_length_ratio()
    # Note: Ratio is normalized by shoulder-to-ankle height (not full body height), so values are higher
    "stride_length_ratio": {
        "bad": 2.7,    # < 2.7 is red
        "ok": 3.5,     # 2.7-3.5 is yellow, >= 3.5 is green
        "good": 3.5,   # >= 3.5 is green
    },
    # Leg Lift Angle removed - metric deleted
}


def score_from_bands(value, bands, higher_is_better=True):
    """
    Compute score (0-100) and status from raw value using score bands.
    
    Args:
        value: Raw metric value (float or None)
        bands: Dict with "bad", "ok", "good" thresholds (from SCORE_BANDS)
        higher_is_better: If True, higher values score better. If False, lower values score better.
    
    Returns:
        (score, status)
        score: Integer 0-100 (None if value is None)
        status: "good", "ok", or "bad" (None if value is None)
    """
    if value is None:
        return None, None
    
    if not isinstance(bands, dict) or not all(k in bands for k in ["bad", "ok", "good"]):
        # Invalid bands - return default
        return 50, "ok"
    
    bad_thresh = bands["bad"]
    ok_thresh = bands["ok"]
    good_thresh = bands["good"]
    
    # Ensure thresholds are in correct order
    if higher_is_better:
        # For higher-is-better: bad < ok < good
        if bad_thresh >= ok_thresh or ok_thresh >= good_thresh:
            # Invalid band order - return default
            return 50, "ok"
        
        # Score based on which band the value falls into
        if value >= good_thresh:
            # Excellent: linear interpolation from ok_thresh to good_thresh, capped at 100
            if value >= good_thresh * 1.2:  # 20% above good threshold = perfect score
                return 100, "good"
            else:
                # Linear interpolation: ok_thresh -> 75, good_thresh -> 100
                score = 75 + int(25 * (value - ok_thresh) / (good_thresh - ok_thresh))
                return min(100, max(75, score)), "good"
        elif value >= ok_thresh:
            # Fair: linear interpolation from bad_thresh to ok_thresh
            score = 50 + int(25 * (value - bad_thresh) / (ok_thresh - bad_thresh))
            return min(75, max(50, score)), "ok"
        elif value >= bad_thresh:
            # Poor: linear interpolation from 0 to bad_thresh
            score = int(50 * (value - 0) / bad_thresh) if bad_thresh > 0 else 0
            return min(50, max(0, score)), "bad"
        else:
            # Very poor: below bad threshold
            return 0, "bad"
    else:
        # For lower-is-better: good < ok < bad (inverted)
        if good_thresh >= ok_thresh or ok_thresh >= bad_thresh:
            # Invalid band order - return default
            return 50, "ok"
        
        # Score based on which band the value falls into (inverted logic)
        if value <= good_thresh:
            # Excellent: below good threshold
            if value <= good_thresh * 0.8:  # 20% below good threshold = perfect score
                return 100, "good"
            else:
                # Linear interpolation: good_thresh -> 100, ok_thresh -> 75
                score = 75 + int(25 * (ok_thresh - value) / (ok_thresh - good_thresh))
                return min(100, max(75, score)), "good"
        elif value <= ok_thresh:
            # Fair: linear interpolation from good_thresh to ok_thresh
            score = 50 + int(25 * (ok_thresh - value) / (ok_thresh - good_thresh))
            return min(75, max(50, score)), "ok"
        elif value <= bad_thresh:
            # Poor: linear interpolation from ok_thresh to bad_thresh
            score = int(50 * (bad_thresh - value) / (bad_thresh - ok_thresh))
            return min(50, max(0, score)), "bad"
        else:
            # Very poor: above bad threshold
            return 0, "bad"


def get_bands_for_metric(metric_name):
    """
    Get score bands for a specific metric.
    
    Args:
        metric_name: Name of the metric (key in SCORE_BANDS)
    
    Returns:
        Dict with "bad", "ok", "good" thresholds, or None if not found
    """
    return SCORE_BANDS.get(metric_name)

