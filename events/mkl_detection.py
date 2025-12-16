"""
Max Knee Lift (MKL) detection.

Detects the frame where the lead knee reaches its maximum height (minimum y-coordinate)
before foot strike (FFP).
"""

import numpy as np


def detect_mkl_idx(pose_cache, ffp_idx=None, throwing_side=None, n=None, min_q=0.35):
    """
    Detect Max Knee Lift (MKL) frame index.
    
    Heuristic: pick the frame BEFORE FFP where lead knee is highest (smallest y),
    using only valid frames with pose_q >= min_q.
    If FFP not available, search first half of clip.
    
    Args:
        pose_cache: Cached pose data dict with "pts" (landmark arrays) and "pose_q"
        ffp_idx: Foot strike (FFP) frame index (optional)
        throwing_side: "L" or "R" (optional, defaults to "R" if not provided)
        n: Total number of frames (optional, inferred from pose_cache if not provided)
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        (mkl_idx: int|None, debug: dict)
        debug includes: start, end, valid_count, reason
    """
    debug = {
        "start": None,
        "end": None,
        "valid_count": 0,
        "reason": None,
    }
    
    if pose_cache is None or pose_cache.get("pts") is None:
        debug["reason"] = "missing_pose_cache"
        return None, debug
    
    pts = pose_cache["pts"]
    pose_q = pose_cache.get("pose_q", [])
    
    # Infer n from pose_q if not provided
    if n is None:
        if isinstance(pose_q, np.ndarray):
            n = len(pose_q)
        elif isinstance(pose_q, (list, tuple)):
            n = len(pose_q)
        else:
            debug["reason"] = "cannot_infer_n"
            return None, debug
    
    # Determine lead knee (opposite of throwing side)
    if throwing_side is None or throwing_side.upper() not in ["L", "R"]:
        throwing_side = "R"  # Default to right-handed
    
    if throwing_side.upper() == "R":
        lead_knee_key = "LEFT_KNEE"
    else:  # L
        lead_knee_key = "RIGHT_KNEE"
    
    # Check if knee landmarks are available
    if lead_knee_key not in pts:
        debug["reason"] = "knee_landmarks_not_available"
        return None, debug
    
    knee_pts = pts[lead_knee_key]
    
    # Determine search window
    if ffp_idx is not None and ffp_idx > 0:
        search_end = ffp_idx
        search_start = max(0, search_end - 80)
    else:
        # No FFP: search first half of clip
        search_end = n // 2
        search_start = max(0, search_end - 80)
    
    debug["start"] = search_start
    debug["end"] = search_end
    
    # Find valid frames (pose_q >= min_q and knee landmark is valid)
    valid_indices = []
    knee_y_values = []
    
    for i in range(search_start, search_end):
        if i >= n:
            break
        
        # Check pose quality
        if i >= len(pose_q) or pose_q[i] < min_q:
            continue
        
        # Check if knee landmark is valid (non-zero)
        knee_pt = knee_pts[i]
        if np.any(knee_pt != 0) and np.isfinite(knee_pt[1]):
            valid_indices.append(i)
            knee_y_values.append(float(knee_pt[1]))
    
    debug["valid_count"] = len(valid_indices)
    
    if len(valid_indices) == 0:
        debug["reason"] = "no_valid_frames"
        return None, debug
    
    # Find frame with minimum y (highest knee)
    min_y_idx = np.argmin(knee_y_values)
    mkl_idx = valid_indices[min_y_idx]
    
    debug["reason"] = "ok"
    return int(mkl_idx), debug

