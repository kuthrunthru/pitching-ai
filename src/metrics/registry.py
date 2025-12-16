"""
Metrics registry.

Central registry of all intended pitching metrics with metadata.
"""

# All expected metrics (translation- and geometry-based only)
EXPECTED_METRICS = [
    # Posture metrics
    "head_behind_hip",
    # Lower body metrics
    "stride_length",
    # Posture metrics (FFP-dependent)
    "throw_elbow_vs_shoulder_line_ffp",
    "elbow_bend_ffp_deg",
    "ball_angle_vs_shoulder_line_ffp_deg",
    "upper_body_lean_release",
    # Rotation metrics (guidance)
    "chest_closed_ffp_deg",
]

# Metadata for each metric
METRIC_META = {
    # Lower body metrics
    "stride_length": {
        "label": "Stride Length",
        "units": "ratio",
        "group": "lower_body",
    },
    # Posture metrics
    "head_behind_hip": {
        "label": "Head Behind Hip Toward Plate",
        "units": "ratio",
        "group": "posture",
    },
    "throw_elbow_vs_shoulder_line_ffp": {
        "label": "Elbow Height at Foot Strike",
        "units": "ratio",
        "group": "posture",
    },
    "elbow_bend_ffp_deg": {
        "label": "Arm Angle at Foot Strike",
        "units": "deg",
        "group": "posture",
    },
    "ball_angle_vs_shoulder_line_ffp_deg": {
        "label": "Ball Location at Foot Strike",
        "units": "deg",
        "group": "posture",
    },
    "upper_body_lean_release": {
        "label": "Upper-Body Lean at Release",
        "units": "deg",
        "group": "posture",
    },
    # Rotation metrics (guidance)
    "chest_closed_ffp_deg": {
        "label": "Chest Closed at Foot Strike",
        "units": "%",
        "group": "rotation",
    },
}


def get_metric_label(metric_key: str) -> str:
    """Get display label for a metric key."""
    return METRIC_META.get(metric_key, {}).get("label", metric_key)


def get_metric_units(metric_key: str) -> str:
    """Get units for a metric key."""
    return METRIC_META.get(metric_key, {}).get("units", "")


def get_metric_group(metric_key: str) -> str:
    """Get group for a metric key."""
    return METRIC_META.get(metric_key, {}).get("group", "unknown")

