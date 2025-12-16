"""
Metric interpretation layer.

Provides human-readable status and explanations for raw metric values.
"""

import numpy as np


# Interpretation rules for each metric
INTERPRETATION_RULES = {
    # Lower body metrics
    # Note: stride_length uses custom scoring (score_stride_length_ratio) with optimal range pattern
    # Removed from INTERPRETATION_RULES as it has custom scoring logic
    # leg_lift_angle_deg removed - metric deleted
    # lead_leg_block removed - metric deleted
}


def interpret_metric(metric_key: str, value: float | None) -> tuple[str | None, str | None]:
    """
    Interpret a metric value and return status and explanation.
    
    Args:
        metric_key: Metric key (e.g., "stride_length")
        value: Raw metric value (float or None)
    
    Returns:
        (status: str|None, explanation: str|None)
        status: "excellent", "good", "ok", "poor", or None if value is None
        explanation: Human-readable explanation string, or None if value is None
    """
    if value is None or not np.isfinite(value):
        return None, None
    
    rules = INTERPRETATION_RULES.get(metric_key)
    if not rules:
        return None, None
    
    thresholds = rules.get("thresholds", {})
    higher_is_better = rules.get("higher_is_better", True)
    explanation = rules.get("explanation", "")
    
    excellent_thresh = thresholds.get("excellent")
    good_thresh = thresholds.get("good")
    ok_thresh = thresholds.get("ok")
    
    if excellent_thresh is None or good_thresh is None or ok_thresh is None:
        return None, explanation
    
    # Determine status based on thresholds
    if higher_is_better:
        if value >= excellent_thresh:
            status = "excellent"
        elif value >= good_thresh:
            status = "good"
        elif value >= ok_thresh:
            status = "ok"
        else:
            status = "poor"
    else:  # Lower is better
        if value <= excellent_thresh:
            status = "excellent"
        elif value <= good_thresh:
            status = "good"
        elif value <= ok_thresh:
            status = "ok"
        else:
            status = "poor"
    
    return status, explanation

