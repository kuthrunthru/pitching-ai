"""
Robust release frame detection using speed and extension deceleration.

Implements a 2-stage scoring system to reliably detect the release point
even with noisy pose data.
"""

import numpy as np


def detect_release_idx(
    wrist_xy=None,          # np.array shape (n,2) in pixels for THROWING wrist (if throwing_side provided)
    wrist_xy_L=None,        # np.array shape (n,2) for left wrist (if picking throwing side)
    wrist_xy_R=None,        # np.array shape (n,2) for right wrist (if picking throwing side)
    elbow_xy=None,          # np.array shape (n,2) in pixels
    shoulder_xy=None,       # np.array shape (n,2) in pixels
    pose_q=None,            # np.array shape (n,) 0..1
    fps: float=30.0,
    ffp_idx: int=None,
    n: int=None,
    throwing_side=None,     # "L" or "R" (if None, will be picked automatically)
    min_q: float = 0.35,
    search_start_off: int = 4,
    search_end_off: int = 55,
    smooth_win: int = 5,
):
    """
    Detect release frame index using combined score: wrist speed peak + deceleration + near-max extension.
    
    Uses a 3-stage scoring system:
    1. Primary: Peak wrist speed after FFP
    2. Secondary: Strongest deceleration near peak speed
    3. Tertiary: Extension near maximum (top 20%)
    
    Args:
        wrist_xy: Array of (x, y) positions for throwing wrist, shape (n, 2) (if throwing_side provided)
        wrist_xy_L: Array of (x, y) positions for left wrist, shape (n, 2) (if picking throwing side)
        wrist_xy_R: Array of (x, y) positions for right wrist, shape (n, 2) (if picking throwing side)
        elbow_xy: Array of (x, y) positions for elbow, shape (n, 2)
        shoulder_xy: Array of (x, y) positions for shoulder, shape (n, 2)
        pose_q: Array of pose quality scores per frame, shape (n,)
        fps: Frames per second
        ffp_idx: Foot strike (FFP) frame index
        n: Total number of frames
        throwing_side: "L" or "R" (if None, will be picked automatically from wrist_xy_L and wrist_xy_R)
        min_q: Minimum pose quality threshold (default 0.35)
        search_start_off: Frames after FFP to start search (default 4)
        search_end_off: Frames after FFP to end search (default 55)
        smooth_win: Smoothing window size for moving average (default 5)
    
    Returns:
        (release_idx: int|None, debug: dict)
        release_idx: Detected release frame index, or None if detection fails
        debug: Dictionary with reason, i_vmax, i_decel, vmax, amin, ext_at_cand, ext_threshold, s, e, valid_ratio, throwing_side
    """
    # Initialize debug dict with required fields
    debug = {
        "reason": None,
        "i_vmax": None,
        "i_decel": None,
        "vmax": None,
        "amin": None,
        "ext_at_cand": None,
        "ext_threshold": None,
        "s": None,
        "e": None,
        "valid_ratio": None,
        "throwing_side": None,
    }
    
    # Pick throwing side if not provided
    if throwing_side is None:
        if wrist_xy_L is None or wrist_xy_R is None or pose_q is None:
            debug["reason"] = "missing_wrist_data_for_selection"
            return None, debug
        
        from src.pose.arm_selection import pick_throwing_side
        throwing_side, side_debug = pick_throwing_side(
            wrist_xy_L=wrist_xy_L,
            wrist_xy_R=wrist_xy_R,
            pose_q=pose_q,
            ffp_idx=ffp_idx,
            release_idx_hint=None,
            min_q=min_q,
            n=n,
        )
        debug["throwing_side"] = throwing_side
        debug["throwing_side_reason"] = side_debug.get("reason", "unknown")
    
    # Select wrist based on throwing side
    if wrist_xy is None:
        if throwing_side.upper() == "L":
            wrist_xy = wrist_xy_L
        else:
            wrist_xy = wrist_xy_R
        
        if wrist_xy is None:
            debug["reason"] = "missing_wrist_data"
            return None, debug
    
    debug["throwing_side"] = throwing_side.upper()
    
    # A) Guardrails
    if ffp_idx is None or ffp_idx < 0 or ffp_idx >= n - 1:
        debug["reason"] = "bad_ffp"
        return None, debug
    
    # Compute search window
    s = max(0, ffp_idx + search_start_off)
    e = min(n - 2, ffp_idx + search_end_off)  # n-2 because we need at least 2 frames for derivatives
    
    debug["s"] = int(s)
    debug["e"] = int(e)
    
    if e - s < 8:
        debug["reason"] = "window_too_small"
        return None, debug
    
    # B) Pose validity: compute valid mask (finite coords + pose_q >= min_q)
    valid = np.ones(n, dtype=bool)
    for i in range(n):
        if pose_q[i] < min_q:
            valid[i] = False
            continue
        # Check all landmarks are finite
        if (not np.isfinite(wrist_xy[i, 0]) or not np.isfinite(wrist_xy[i, 1]) or
            not np.isfinite(elbow_xy[i, 0]) or not np.isfinite(elbow_xy[i, 1]) or
            not np.isfinite(shoulder_xy[i, 0]) or not np.isfinite(shoulder_xy[i, 1])):
            valid[i] = False
    
    # Compute valid_ratio in [s:e]
    valid_in_window = np.sum(valid[s:e+1])
    total_in_window = e - s + 1
    valid_ratio = valid_in_window / total_in_window if total_in_window > 0 else 0.0
    debug["valid_ratio"] = float(valid_ratio)
    
    if valid_ratio < 0.60:
        debug["reason"] = "low_pose_quality"
        return None, debug
    
    # 2) Compute signals inside [s:e] window (pixels -> per-second using fps)
    # Smooth wrist positions with simple moving average (window=smooth_win) before computing v/a
    wrist_smooth = np.zeros_like(wrist_xy)
    for i in range(n):
        if not valid[i]:
            wrist_smooth[i] = wrist_xy[i]  # Leave invalid as-is
            continue
        
        # Compute moving average over smooth_win, only including valid points
        win_start = max(0, i - smooth_win // 2)
        win_end = min(n, i + smooth_win // 2 + 1)
        win_valid = valid[win_start:win_end]
        win_wrist = wrist_xy[win_start:win_end]
        
        if np.any(win_valid):
            valid_wrist = win_wrist[win_valid]
            wrist_smooth[i] = np.mean(valid_wrist, axis=0)
        else:
            wrist_smooth[i] = wrist_xy[i]
    
    # Compute wrist speed: v[i] = ||wrist_xy[i+1]-wrist_xy[i]|| * fps
    v = np.zeros(n)
    for i in range(n - 1):
        if valid[i] and valid[i+1]:
            dx = wrist_smooth[i+1, 0] - wrist_smooth[i, 0]
            dy = wrist_smooth[i+1, 1] - wrist_smooth[i, 1]
            v[i] = np.hypot(dx, dy) * fps
    
    # Compute acceleration: a[i] = (v[i+1]-v[i]) * fps
    a = np.zeros(n)
    for i in range(n - 1):
        if valid[i] and valid[i+1]:
            a[i] = (v[i+1] - v[i]) * fps
    
    # Compute extension: ext[i] = ||wrist_xy[i]-shoulder_xy[i]||
    ext = np.zeros(n)
    for i in range(n):
        if valid[i]:
            dx = wrist_xy[i, 0] - shoulder_xy[i, 0]
            dy = wrist_xy[i, 1] - shoulder_xy[i, 1]
            ext[i] = np.hypot(dx, dy)
    
    # 3) Candidate selection
    # A) i_vmax = argmax(v[s:e])
    v_window = v[s:e+1]
    valid_v_window = np.array([valid[i] for i in range(s, e+1)])
    
    if not np.any(valid_v_window):
        debug["reason"] = "no_valid_speed"
        return None, debug
    
    v_valid = v_window[valid_v_window]
    v_max_idx_local = np.argmax(v_valid)
    valid_indices = np.where(valid_v_window)[0]
    i_vmax = s + valid_indices[v_max_idx_local]
    vmax = float(v[i_vmax])
    
    debug["i_vmax"] = int(i_vmax)
    debug["vmax"] = round(vmax, 2)
    
    # B) In a local neighborhood around i_vmax (±6 frames, clamped), find i_decel = argmin(a)
    decel_window_start = max(s, i_vmax - 6)
    decel_window_end = min(e, i_vmax + 6)
    
    a_window = a[decel_window_start:decel_window_end+1]
    valid_a_window = np.array([valid[i] for i in range(decel_window_start, decel_window_end+1)])
    
    i_decel = None
    amin = None
    
    if np.any(valid_a_window):
        a_valid = a_window[valid_a_window]
        a_min_idx_local = np.argmin(a_valid)  # Most negative (strongest deceleration)
        valid_a_indices = np.where(valid_a_window)[0]
        i_decel = decel_window_start + valid_a_indices[a_min_idx_local]
        amin = float(a[i_decel])
        
        debug["i_decel"] = int(i_decel)
        debug["amin"] = round(amin, 2)
    
    # C) Compute ext_threshold = percentile(ext[s:e], 80) (top 20% extension)
    ext_window = ext[s:e+1]
    ext_valid = ext_window[valid_v_window]
    
    if len(ext_valid) > 0:
        ext_threshold = float(np.percentile(ext_valid, 80))
        debug["ext_threshold"] = round(ext_threshold, 2)
        
        # D) Prefer i_decel if ext[i_decel] >= ext_threshold; otherwise use i_vmax
        if i_decel is not None and valid[i_decel]:
            ext_at_decel = ext[i_decel]
            if ext_at_decel >= ext_threshold:
                cand = i_decel
                debug["ext_at_cand"] = round(float(ext_at_decel), 2)
                debug["reason"] = "vmax+decel"
            else:
                cand = i_vmax
                ext_at_vmax = ext[i_vmax] if valid[i_vmax] else 0.0
                debug["ext_at_cand"] = round(float(ext_at_vmax), 2)
                debug["reason"] = "vmax_only"
        else:
            cand = i_vmax
            ext_at_vmax = ext[i_vmax] if valid[i_vmax] else 0.0
            debug["ext_at_cand"] = round(float(ext_at_vmax), 2)
            debug["reason"] = "vmax_only"
    else:
        # No valid extension data, use i_vmax
        cand = i_vmax
        debug["ext_at_cand"] = None
        debug["ext_threshold"] = None
        debug["reason"] = "vmax_only"
    
    # 4) Apply constraints
    # cand >= ffp_idx + 6
    if cand < ffp_idx + 6:
        cand = max(ffp_idx + 6, s)
    
    # cand within [0, n-1]
    cand = int(np.clip(cand, 0, n - 1))
    
    return cand, debug

