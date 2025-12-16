"""
Forward Trunk Tilt at Release metric computation.

Computes the forward lean angle of the trunk (shoulder-pelvis line)
at the release point.
"""

import math
import numpy as np


def compute_forward_trunk_tilt_at_release_deg(
    release_idx,
    shoulder_mid,
    pelvis_mid,
    pose_quality,
    n,
    fps,
    plate_dir_sign,
    min_q=0.35,
):
    """
    Compute forward trunk tilt at release in degrees.
    
    Forward trunk tilt is the angle between the shoulder-pelvis line
    and the vertical axis. Positive values indicate forward lean.
    
    Args:
        release_idx: Frame index of release event
        shoulder_mid: Function(frame) -> (x, y) or None
        pelvis_mid: Function(frame) -> (x, y) or None
        pose_quality: List of pose quality scores per frame
        n: Total number of frames
        fps: Frames per second
        plate_dir_sign: Forward direction sign (+1 or -1)
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        (tilt_deg, debug_dict)
        tilt_deg: Forward trunk tilt in degrees (None if computation fails)
        debug_dict: Dictionary with window_start, window_end, valid_samples, etc.
    """
    debug = {
        "window_start": None,
        "window_end": None,
        "valid_samples": 0,
    }
    
    # Guardrails
    if release_idx is None:
        return None, debug
    
    if n < 3:
        return None, debug
    
    release_idx = int(np.clip(int(release_idx), 0, n - 1))
    
    # Window around release (~50ms)
    win = max(1, int(round(0.05 * fps)))
    i0 = max(0, release_idx - win)
    i1 = min(n - 1, release_idx + win)
    
    debug["window_start"] = i0
    debug["window_end"] = i1
    
    # Collect tilt angles in the window
    tilt_angles = []
    
    for t in range(i0, i1 + 1):
        if pose_quality[t] < min_q:
            continue
        
        sh = shoulder_mid(t)
        pel = pelvis_mid(t)
        
        if sh is None or pel is None:
            continue
        
        # Compute vector from pelvis to shoulder
        dx = float(sh[0]) - float(pel[0])
        dy = float(sh[1]) - float(pel[1])
        
        # Compute angle from vertical
        # In image coordinates, y increases downward
        # Forward tilt: angle from vertical (0 deg = vertical, positive = forward lean)
        # atan2(dx, -dy) gives angle from vertical (negative dy because y increases down)
        angle_rad = math.atan2(dx, -dy)
        angle_deg = math.degrees(angle_rad)
        
        # Apply forward direction sign if needed
        # plate_dir_sign indicates the direction toward the plate
        # For forward tilt, we want positive values when leaning toward plate
        if plate_dir_sign is not None:
            angle_deg = angle_deg * float(plate_dir_sign)
        
        if np.isfinite(angle_deg):
            tilt_angles.append(float(angle_deg))
    
    if not tilt_angles:
        return None, debug
    
    # Return median tilt angle
    tilt_deg = float(np.median(np.array(tilt_angles, dtype=np.float32)))
    debug["valid_samples"] = len(tilt_angles)
    
    return tilt_deg, debug


def score_trunk_tilt_release_deg(tilt_deg):
    """
    Score forward trunk tilt at release.
    
    Uses centralized score bands from calibration.py for easy tuning.
    
    Args:
        tilt_deg: Forward trunk tilt in degrees (None for N/A)
    
    Returns:
        (score, status)
        score: Integer score 0-100 (None if tilt_deg is None)
        status: "good", "ok", or "bad" (None if tilt_deg is None)
    """
    if tilt_deg is None:
        return None, None
    
    # Use centralized score bands
    from src.metrics.calibration import score_from_bands, get_bands_for_metric
    
    bands = get_bands_for_metric("trunk_tilt_release_deg")
    if bands:
        score, status = score_from_bands(tilt_deg, bands, higher_is_better=True)
        return score, status
    else:
        # Fallback to hardcoded bands if calibration not found
        if tilt_deg >= 15.0:
            return 100, "good"
        elif tilt_deg >= 8.0:
            return 75, "ok"
        elif tilt_deg >= 0.0:
            return 50, "ok"
        else:
            return 25, "bad"

