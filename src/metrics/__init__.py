"""
Metrics package for pitching analysis.

Contains calibration bands and scoring functions.
"""

from src.metrics.calibration import SCORE_BANDS, score_from_bands, get_bands_for_metric
from src.metrics.scoring import (
    score_stride_length_ratio,
    score_head_behind_hip_ratio,
    score_upper_body_lean_release_deg,
)

__all__ = [
    "SCORE_BANDS",
    "score_from_bands",
    "get_bands_for_metric",
    "score_stride_length_ratio",
    "score_head_behind_hip_ratio",
    "score_upper_body_lean_release_deg",
]

