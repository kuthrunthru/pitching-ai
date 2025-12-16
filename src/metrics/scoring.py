"""
Scoring functions for all metrics.

All scoring functions use centralized calibration bands from calibration.py.
"""

from src.metrics.calibration import score_from_bands, get_bands_for_metric


# score_leg_lift_angle_deg removed - Leg Lift Angle metric deleted
# score_lead_leg_block_ratio removed - Lead Leg Block metric deleted


def score_upper_body_lean_release_deg(lean_deg):
    """
    Score upper-body lean at release (high school only).
    
    High school score bands:
      RED < 5° (too upright)
      YELLOW 5-14° (moderate lean)
      GREEN ≥ 14° (good forward lean)
    
    Args:
        lean_deg: Upper-body lean angle in degrees (None for N/A)
    
    Returns:
        (score, status)
        score: Integer 0-100 (None if lean_deg is None)
        status: "green", "yellow", or "red" (None if lean_deg is None)
    """
    if lean_deg is None:
        return None, None
    
    # High school bands: RED < 5°, YELLOW 5-14°, GREEN ≥ 14°
    if lean_deg >= 14.0:
        return 100, "green"
    elif lean_deg >= 5.0:
        return 75, "yellow"
    else:  # < 5.0
        return 25, "red"


def score_head_behind_hip_ratio(ratio):
    """
    Score head behind hip ratio.
    
    Head Behind Hip uses custom status mapping: green/yellow/red.
    This function maps the ratio to score and standard status for consistency.
    
    Args:
        ratio: Head behind hip ratio (None for N/A)
        GREEN (>= 0.02): head clearly behind hip
        YELLOW (0.00-0.02): roughly stacked
        RED (< 0.00): head leaks past hip
    
    Returns:
        (score, status)
        score: Integer 0-100 (None if ratio is None)
        status: "good" (green), "ok" (yellow), or "bad" (red) (None if ratio is None)
    """
    if ratio is None:
        return None, None
    
    # Map green/yellow/red to good/ok/bad for consistency with other metrics
    if ratio >= 0.02:
        return 100, "good"  # GREEN
    elif ratio >= 0.00:
        return 75, "ok"      # YELLOW
    else:
        return 25, "bad"     # RED


def score_stride_length_ratio(ratio):
    """
    Score stride length ratio.
    
    Thresholds:
    GREEN (>= 0.85), YELLOW (0.65-0.84), RED (< 0.65)
    
    Args:
        ratio: Stride length ratio (None for N/A)
    
    Returns:
        (score, status)
        score: Integer 0-100 (None if ratio is None)
        status: "green", "yellow", or "red" (None if ratio is None)
    """
    if ratio is None:
        return None, None
    
    # Thresholds: GREEN (>= 0.85), YELLOW (0.65-0.84), RED (< 0.65)
    if ratio >= 0.85:
        return 100, "green"
    elif 0.65 <= ratio <= 0.84:
        return 75, "yellow"
    else:  # ratio < 0.65
        return 25, "red"

