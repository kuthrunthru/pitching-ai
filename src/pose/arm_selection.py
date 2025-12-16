"""
Throwing arm selection utilities.

Determines which arm (L or R) is the throwing arm based on motion characteristics.
"""

import numpy as np


def pick_throwing_side(
    wrist_xy_L,
    wrist_xy_R,
    pose_q,
    ffp_idx,
    release_idx_hint=None,
    min_q=0.35,
    n=None,
):
    """
    Pick throwing side (L or R) based on motion characteristics near release window.
    
    Heuristic: The throwing wrist shows a clearer speed peak and higher forward-motion
    consistency near the release window.
    
    Args:
        wrist_xy_L: Array of (x, y) positions for left wrist, shape (n, 2)
        wrist_xy_R: Array of (x, y) positions for right wrist, shape (n, 2)
        pose_q: Array of pose quality scores per frame, shape (n,)
        ffp_idx: Foot strike (FFP) frame index
        release_idx_hint: Optional hint for release frame (for window sizing)
        min_q: Minimum pose quality threshold (default 0.35)
        n: Total number of frames (if None, inferred from arrays)
    
    Returns:
        (throwing_side: str, debug: dict)
        throwing_side: "L" or "R"
        debug: Dictionary with scores, valid_ratios, and reason
    """
    if n is None:
        n = len(pose_q) if pose_q is not None else 0
    
    if n == 0 or ffp_idx is None or ffp_idx < 0 or ffp_idx >= n:
        # Default to R if we can't determine
        return "R", {"reason": "default_fallback", "score_L": None, "score_R": None}
    
    # Window: start = ffp_idx+4, end = (release_idx_hint if valid else ffp_idx+55) clamped
    window_start = max(0, ffp_idx + 4)
    if release_idx_hint is not None and release_idx_hint > window_start:
        window_end = min(n - 1, release_idx_hint + 3)
    else:
        window_end = min(n - 1, ffp_idx + 55)
    
    if window_end <= window_start:
        return "R", {"reason": "window_too_small", "score_L": None, "score_R": None}
    
    debug = {
        "reason": None,
        "score_L": None,
        "score_R": None,
        "valid_ratio_L": None,
        "valid_ratio_R": None,
        "vmax_L": None,
        "vmax_R": None,
    }
    
    def compute_side_score(wrist_xy, side_name):
        """Compute score for one side."""
        # Build valid mask
        valid = np.ones(n, dtype=bool)
        for i in range(n):
            if pose_q[i] < min_q:
                valid[i] = False
                continue
            if (not np.isfinite(wrist_xy[i, 0]) or not np.isfinite(wrist_xy[i, 1])):
                valid[i] = False
        
        # Compute valid_ratio in window
        valid_in_window = np.sum(valid[window_start:window_end+1])
        total_in_window = window_end - window_start + 1
        valid_ratio = valid_in_window / total_in_window if total_in_window > 0 else 0.0
        
        # Smooth wrist positions (simple moving average, window=5)
        wrist_smooth = np.zeros_like(wrist_xy)
        for i in range(n):
            if not valid[i]:
                wrist_smooth[i] = wrist_xy[i]
                continue
            
            win_start = max(0, i - 2)
            win_end = min(n, i + 3)
            win_valid = valid[win_start:win_end]
            win_wrist = wrist_xy[win_start:win_end]
            
            if np.any(win_valid):
                valid_wrist = win_wrist[win_valid]
                wrist_smooth[i] = np.mean(valid_wrist, axis=0)
            else:
                wrist_smooth[i] = wrist_xy[i]
        
        # Compute speed: v[i] = ||wrist_xy[i+1]-wrist_xy[i]||
        v = np.zeros(n)
        for i in range(n - 1):
            if valid[i] and valid[i+1]:
                dx = wrist_smooth[i+1, 0] - wrist_smooth[i, 0]
                dy = wrist_smooth[i+1, 1] - wrist_smooth[i, 1]
                v[i] = np.hypot(dx, dy)
        
        # Find peak speed in window
        v_window = v[window_start:window_end+1]
        valid_v_window = np.array([valid[i] for i in range(window_start, window_end+1)])
        
        if not np.any(valid_v_window):
            return None, valid_ratio, None
        
        v_valid = v_window[valid_v_window]
        vmax = float(np.max(v_valid))
        
        # Score = vmax (simpler approach)
        # If valid_ratio is too low, penalize the score
        if valid_ratio < 0.50:
            score = vmax * 0.5  # Penalty for low validity
        else:
            score = vmax
        
        return score, valid_ratio, vmax
    
    # Compute scores for both sides
    score_L, valid_ratio_L, vmax_L = compute_side_score(wrist_xy_L, "L")
    score_R, valid_ratio_R, vmax_R = compute_side_score(wrist_xy_R, "R")
    
    debug["score_L"] = round(score_L, 2) if score_L is not None else None
    debug["score_R"] = round(score_R, 2) if score_R is not None else None
    debug["valid_ratio_L"] = round(valid_ratio_L, 3) if valid_ratio_L is not None else None
    debug["valid_ratio_R"] = round(valid_ratio_R, 3) if valid_ratio_R is not None else None
    debug["vmax_L"] = round(vmax_L, 2) if vmax_L is not None else None
    debug["vmax_R"] = round(vmax_R, 2) if vmax_R is not None else None
    
    # Choose side with higher score, but prefer side with better valid_ratio if one is much better
    if score_L is None and score_R is None:
        return "R", {**debug, "reason": "no_valid_data"}
    
    if score_L is None:
        return "R", {**debug, "reason": "left_no_data"}
    
    if score_R is None:
        return "L", {**debug, "reason": "right_no_data"}
    
    # If one side has valid_ratio < 0.50 and the other >= 0.60, choose the better-valid side
    if valid_ratio_L < 0.50 and valid_ratio_R >= 0.60:
        return "R", {**debug, "reason": "right_better_validity"}
    
    if valid_ratio_R < 0.50 and valid_ratio_L >= 0.60:
        return "L", {**debug, "reason": "left_better_validity"}
    
    # Otherwise, choose side with higher score
    if score_L > score_R:
        return "L", {**debug, "reason": "left_higher_score"}
    else:
        return "R", {**debug, "reason": "right_higher_score"}

