"""
Posture metrics computation.

Includes Head Behind Hip Toward Plate and Elbow Height at Foot Strike metrics (camera-angle robust).
"""

import math
import numpy as np
from src.metrics.utils import (
    pack_metric,
    CONF_THRESH,
    find_best_frame_in_window,
    is_joint_valid,
    pick_first_valid_forward,
    pick_valid_frame_forward,
    select_lead_knee_anchor,
)


def _compute_body_height_px(pts, frame_idx):
    """
    Compute body height in pixels at a given frame.
    
    Uses distance from mid-shoulder to mid-ankle, or fallback to max-min of key points.
    
    Args:
        pts: Dict of landmark arrays
        frame_idx: Frame index
    
    Returns:
        body_height_px: float or None if landmarks missing
    """
    if frame_idx < 0:
        return None
    
    # Try mid-shoulder to mid-ankle
    if ("LEFT_SHOULDER" in pts and "RIGHT_SHOULDER" in pts and
        "LEFT_ANKLE" in pts and "RIGHT_ANKLE" in pts):
        ls = pts["LEFT_SHOULDER"][frame_idx]
        rs = pts["RIGHT_SHOULDER"][frame_idx]
        la = pts["LEFT_ANKLE"][frame_idx]
        ra = pts["RIGHT_ANKLE"][frame_idx]
        
        if (np.any(ls != 0) and np.any(rs != 0) and 
            np.any(la != 0) and np.any(ra != 0)):
            shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            ankle_mid = ((la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2)
            dx = shoulder_mid[0] - ankle_mid[0]
            dy = shoulder_mid[1] - ankle_mid[1]
            body_h = math.sqrt(dx * dx + dy * dy)
            if np.isfinite(body_h) and body_h > 0:
                return body_h
    
    # Fallback: max-min of shoulders/hips/ankles
    key_points = []
    for name in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", 
                 "LEFT_ANKLE", "RIGHT_ANKLE"]:
        if name in pts:
            pt = pts[name][frame_idx]
            if np.any(pt != 0):
                key_points.append((pt[0], pt[1]))
    
    if len(key_points) >= 4:
        ys = [p[1] for p in key_points]
        body_h = max(ys) - min(ys)
        if np.isfinite(body_h) and body_h > 0:
            return body_h
    
    return None


def compute_head_behind_hip_middrive(pose_cache, keyframes, min_q=0.35):
    """
    Compute Head Behind Hip metric at mid-drive (before FFP).
    
    Measures whether the head stays behind the hip during delivery using a camera-angle
    robust approach. Uses forward-motion axis computed from pelvis movement in early frames to FFP.
    
    Args:
        pose_cache: Cached pose data with "pts" dict
        keyframes: Dict with ffp_idx, throwing_side
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        dict: Packed metric with value (ratio), status (green/yellow/red), and debug info
    """
    metrics = {}
    
    ffp_idx = keyframes.get("ffp_idx")
    throwing_side = keyframes.get("throwing_side", "R")
    
    # Guardrails: require FFP
    if ffp_idx is None:
        return pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason="ffp_not_set",
        )
    
    if pose_cache is None or pose_cache.get("pts") is None:
        return pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason="missing_pose_cache",
        )
    
    pts = pose_cache["pts"]
    n = len(pts.get("LEFT_HIP", [])) if "LEFT_HIP" in pts else 0
    
    # Check indices are valid
    if ffp_idx < 0 or ffp_idx >= n:
        return pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason="invalid_ffp_idx",
        )
    
    # Compute mid-drive frame (use frame at 30% before FFP)
    mid_drive_idx = max(0, int(round(ffp_idx - 0.3 * ffp_idx)))
    
    # Only search forward from mid_drive_idx (never backward)
    window_start = mid_drive_idx
    window_end = min(n - 1, mid_drive_idx + 3)
    
    # Compute forward-motion axis from pelvis midpoint early frames to FFP
    if ("LEFT_HIP" not in pts or "RIGHT_HIP" not in pts):
        return pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason="missing_pelvis_landmarks",
        )
    
    # Get pelvis midpoints at early frame (20% before FFP) and FFP
    early_frame = max(0, int(ffp_idx - 0.2 * ffp_idx))
    left_hip_early = pts["LEFT_HIP"][early_frame]
    right_hip_early = pts["RIGHT_HIP"][early_frame]
    left_hip_ffp = pts["LEFT_HIP"][ffp_idx]
    right_hip_ffp = pts["RIGHT_HIP"][ffp_idx]
    
    if (np.any(left_hip_early == 0) or np.any(right_hip_early == 0) or
        np.any(left_hip_ffp == 0) or np.any(right_hip_ffp == 0)):
        return pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason="invalid_pelvis_landmarks",
        )
    
    pelvis_mid_early = (
        (float(left_hip_early[0]) + float(right_hip_early[0])) / 2.0,
        (float(left_hip_early[1]) + float(right_hip_early[1])) / 2.0,
    )
    pelvis_mid_ffp = (
        (float(left_hip_ffp[0]) + float(right_hip_ffp[0])) / 2.0,
        (float(left_hip_ffp[1]) + float(right_hip_ffp[1])) / 2.0,
    )
    
    # Forward motion vector
    forward_vec = (
        pelvis_mid_ffp[0] - pelvis_mid_early[0],
        pelvis_mid_ffp[1] - pelvis_mid_early[1],
    )
    forward_mag = math.sqrt(forward_vec[0]**2 + forward_vec[1]**2)
    
    # Normalize forward axis
    axis_reason = "ok"
    if forward_mag < 1e-6:
        # Motion too small, fall back to screen x-axis
        forward_axis = (1.0, 0.0)  # Unit vector along x-axis
        axis_reason = "motion_too_small_using_x_axis"
    else:
        forward_axis = (forward_vec[0] / forward_mag, forward_vec[1] / forward_mag)
    
    # Determine throwing-side hip
    if throwing_side.upper() == "R":
        hip_key = "RIGHT_HIP"
    else:  # L
        hip_key = "LEFT_HIP"
    
    if hip_key not in pts:
        return pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason="missing_throwing_side_hip",
        )
    
    # Best frame selection (debug only; computation still uses all valid frames)
    required_joints = ["NOSE", hip_key]
    best_idx, best_q, per_joint_validity, per_joint_confidences = find_best_frame_in_window(
        window_start=window_start,
        window_end=window_end,
        pts=pts,
        required_joints=required_joints,
        min_conf=min_q,
        n=n,
        forward_from=mid_drive_idx,
    )
    
    # Collect ratios for valid frames in window
    ratios = []
    valid_count = 0
    
    for frame_idx in range(window_start, window_end + 1):
        if frame_idx < 0 or frame_idx >= n:
            continue
        
        # Joint validity gate (hip only; confidence optional)
        if not is_joint_valid(pts, frame_idx, hip_key):
            continue
        
        # Note: pose quality already checked above for hip, head joints checked separately below
        
        # Get head position (NOSE, fallback to mid-ears) using joint validity
        head_pos = None
        if is_joint_valid(pts, frame_idx, "NOSE"):
            nose = pts["NOSE"][frame_idx]
            head_pos = (float(nose[0]), float(nose[1]))
        elif is_joint_valid(pts, frame_idx, "LEFT_EAR") and is_joint_valid(pts, frame_idx, "RIGHT_EAR"):
            left_ear = pts["LEFT_EAR"][frame_idx]
            right_ear = pts["RIGHT_EAR"][frame_idx]
            head_pos = (
                (float(left_ear[0]) + float(right_ear[0])) / 2.0,
                (float(left_ear[1]) + float(right_ear[1])) / 2.0,
            )
        
        if head_pos is None:
            continue
        
        # Get hip position
        hip = pts[hip_key][frame_idx]
        if np.any(hip == 0) or not np.isfinite(hip[0]) or not np.isfinite(hip[1]):
            continue
        
        hip_pos = (float(hip[0]), float(hip[1]))
        
        # Compute behind_px = dot((hip - head), forward_axis)
        # Positive value means head is behind hip
        hip_to_head = (hip_pos[0] - head_pos[0], hip_pos[1] - head_pos[1])
        behind_px = hip_to_head[0] * forward_axis[0] + hip_to_head[1] * forward_axis[1]
        
        # Normalize by body height
        body_height_px = _compute_body_height_px(pts, frame_idx)
        if body_height_px is not None and body_height_px > 0:
            ratio = behind_px / body_height_px
            if np.isfinite(ratio):
                ratios.append(ratio)
                valid_count += 1
    
    if len(ratios) == 0:
        return pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason="no_frame_with_all_required_joints",
            debug={
                "window": [window_start, window_end],
                "required_joints": [hip_key, "NOSE|EARS"],
                "axis_reason": axis_reason,
            },
        )
    
    # Metric value = median ratio over valid frames
    median_ratio = float(np.median(ratios))
    
    # Determine status: GREEN (>= 0.02), YELLOW (0.00-0.02), RED (< 0.00)
    if median_ratio >= 0.02:
        status = "green"
        score = 100
    elif median_ratio >= 0.00:
        status = "yellow"
        score = 75
    else:
        status = "red"
        score = 25
    
    return pack_metric(
        name="Head Behind Hip Toward Plate",
        value=round(median_ratio, 4),
        units="ratio",
        score=score,
        status=status,
        reason="ok",
        debug={
            "window": [window_start, window_end],
            "axis_reason": axis_reason,
            "ffp_idx": int(ffp_idx),
            "mid_drive_idx": int(mid_drive_idx),
            "valid_count": int(valid_count),
            "best_idx": int(best_idx) if best_idx is not None else None,
            "best_valid_count": len([k for k, v in per_joint_validity.items() if v]),
            "per_joint_validity": per_joint_validity,
            "best_q": round(float(best_q), 3) if best_idx is not None else None,
        },
    )


def compute_throw_elbow_vs_shoulder_line_ffp(pose_cache, keyframes, min_q=0.35):
    """
    Measures throwing elbow position relative to the shoulder line at FFP.
    
    Value is normalized signed distance (ratio):
      ~0 or slight below is GOOD,
      above is BAD,
      far below is BAD.
    
    Args:
        pose_cache: Cached pose data with "pts" dict and "pose_quality" list
        keyframes: Dict with ffp_idx, throwing_side
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        dict: Packed metric with value (ratio), status (green/yellow/red), and debug info
    """
    ffp_idx = keyframes.get("ffp_idx")
    throwing_side = keyframes.get("throwing_side", "R")
    
    # Guardrails: require ffp_idx and throwing_side
    if ffp_idx is None or throwing_side is None:
        return pack_metric(
            name="Elbow Height at Foot Strike",
            value=None,
            units="ratio",
            reason="missing_keyframes",
        )
    
    if pose_cache is None or pose_cache.get("pts") is None:
        return pack_metric(
            name="Elbow Height at Foot Strike",
            value=None,
            units="ratio",
            reason="missing_pose_cache",
        )
    
    pts = pose_cache["pts"]
    n = len(pts.get("LEFT_SHOULDER", [])) if "LEFT_SHOULDER" in pts else 0
    
    # Get shoulder and elbow points
    ls_key = "LEFT_SHOULDER"
    rs_key = "RIGHT_SHOULDER"
    
    if throwing_side.upper() == "R":
        elbow_key = "RIGHT_ELBOW"
    else:  # L
        elbow_key = "LEFT_ELBOW"
    
    required_joints = [ls_key, rs_key, elbow_key]
    
    # Only search forward from FFP (never backward)
    search_start = ffp_idx
    search_end = min(n - 1, ffp_idx + 3)
    best_idx, best_q, per_joint_validity, per_joint_confidences = find_best_frame_in_window(
        window_start=search_start,
        window_end=search_end,
        pts=pts,
        required_joints=required_joints,
        min_conf=min_q,
        n=n,
        forward_from=ffp_idx,  # Enforce forward-only search
    )
    
    if best_idx is None:
        return pack_metric(
            name="Elbow Height at Foot Strike",
            value=None,
            units="ratio",
            reason="no_frame_with_all_required_joints",
            debug={
                "ffp_idx": int(ffp_idx),
                "throwing_side": throwing_side,
                "window": f"{search_start}..{search_end}",
                "required_joints": required_joints,
                "available_keys_per_frame": {
                    i: [k for k, v in pts.items() if i < len(v) and v[i] is not None]
                    for i in range(search_start, search_end + 1)
                },
                "throwing_elbow": elbow_key,
            },
        )
    
    try:
        LS = pts[ls_key][best_idx]
        RS = pts[rs_key][best_idx]
        E = pts[elbow_key][best_idx]
        
        if not all(is_joint_valid(pts, best_idx, j, n=n) for j in required_joints):
            return pack_metric(
                name="Elbow Height at Foot Strike",
                value=None,
                units="ratio",
                reason="missing_joint_at_best_frame",
                debug={
                    "ffp_idx": int(ffp_idx),
                    "best_idx": int(best_idx),
                    "window": f"{search_start}..{search_end}",
                    "required_joints": required_joints,
                    "per_joint_validity": per_joint_validity,
                },
            )
        
        # Shoulder line vector v = RS - LS
        v = (
            float(RS[0]) - float(LS[0]),
            float(RS[1]) - float(LS[1]),
        )
        
        # Unit normal n = [-v_y, v_x] / ||[-v_y, v_x]||
        # This gives the perpendicular vector pointing "above" the shoulder line
        n_vec = (-v[1], v[0])
        n_mag = math.sqrt(n_vec[0]**2 + n_vec[1]**2)
        
        if n_mag < 1e-6:
            return pack_metric(
                name="Elbow Height at Foot Strike",
                value=None,
                units="ratio",
                reason="shoulder_line_too_short",
            )
        
        n_unit = (n_vec[0] / n_mag, n_vec[1] / n_mag)
        
        # Signed distance in pixels: d_px = dot((E - LS), n_unit)
        # d_px > 0 means "above shoulder line"
        E_minus_LS = (float(E[0]) - float(LS[0]), float(E[1]) - float(LS[1]))
        d_px = E_minus_LS[0] * n_unit[0] + E_minus_LS[1] * n_unit[1]
        
        # Normalize by body height
        body_height_px = _compute_body_height_px(pts, best_idx)
        
        if body_height_px is None or body_height_px <= 0:
            return pack_metric(
                name="Elbow Height at Foot Strike",
                value=None,
                units="ratio",
                reason="missing_body_height",
                debug={
                    "ffp_idx": int(ffp_idx),
                    "d_px": round(float(d_px), 1),
                    "throwing_side": throwing_side,
                },
            )
        
        # Normalized signed distance
        d = d_px / body_height_px
        
        if not np.isfinite(d):
            return pack_metric(
                name="Elbow Height at Foot Strike",
                value=None,
                units="ratio",
                reason="invalid_distance_computation",
            )
        
        # Determine status: GREEN (-0.05 to +0.02), YELLOW (-0.10 to -0.05 or +0.02 to +0.06), RED (< -0.10 or > +0.06)
        if -0.05 <= d <= 0.02:
            status = "green"
            score = 100
        elif (-0.10 <= d < -0.05) or (0.02 < d <= 0.06):
            status = "yellow"
            score = 75
        else:  # d > 0.06 or d < -0.10
            status = "red"
            score = 25
        
        return pack_metric(
            name="Elbow Height at Foot Strike",
            value=round(float(d), 4),
            units="ratio",
            score=score,
            status=status,
            reason="ok",
            debug={
                "ffp_idx": int(ffp_idx),
                "best_idx": int(best_idx),
                "window": f"{search_start}..{search_end}",
                "d_px": round(float(d_px), 1),
                "body_height_px": round(float(body_height_px), 1),
                "throwing_side": throwing_side,
                "required_joints": required_joints,
                "best_valid_count": len([k for k, v in per_joint_validity.items() if v]),
                "per_joint_validity": per_joint_validity,
                "per_joint_confidences": per_joint_confidences,
            },
        )
    except Exception as e:
        return pack_metric(
            name="Elbow Height at Foot Strike",
            value=None,
            units="ratio",
            reason=f"error: {str(e)}",
        )


def compute_elbow_bend_angle_ffp(pose_cache, keyframes, min_q=0.35):
    """
    Elbow bend angle at FFP for the throwing arm, in degrees.
    
    Good range: 80-83 deg.
    Too bent (<80 deg) is bad.
    Too straight (>83 deg) is bad (especially >120 deg).
    
    Args:
        pose_cache: Cached pose data with "pts" dict and "pose_quality" list
        keyframes: Dict with ffp_idx, throwing_side
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        dict: Packed metric with value (degrees), status (green/yellow/red), and debug info
    """
    ffp_idx = keyframes.get("ffp_idx")
    throwing_side = keyframes.get("throwing_side", "R")
    
    # Guardrails: require ffp_idx and throwing_side
    if ffp_idx is None or throwing_side is None:
        return pack_metric(
            name="Arm Angle at Foot Strike",
            value=None,
            units="deg",
            reason="missing_keyframes",
        )
    
    if pose_cache is None or pose_cache.get("pts") is None:
        return pack_metric(
            name="Arm Angle at Foot Strike",
            value=None,
            units="deg",
            reason="missing_pose_cache",
        )
    
    pts = pose_cache["pts"]
    n = len(pts.get("LEFT_SHOULDER", [])) if "LEFT_SHOULDER" in pts else 0
    
    # Get throwing-side landmarks
    if throwing_side.upper() == "R":
        shoulder_key = "RIGHT_SHOULDER"
        elbow_key = "RIGHT_ELBOW"
        wrist_key = "RIGHT_WRIST"
    else:  # L
        shoulder_key = "LEFT_SHOULDER"
        elbow_key = "LEFT_ELBOW"
        wrist_key = "LEFT_WRIST"
    
    required_joints = [shoulder_key, elbow_key, wrist_key]
    
    # Only search forward from FFP (never backward)
    search_start = ffp_idx
    search_end = min(n - 1, ffp_idx + 3)
    best_idx, best_q, per_joint_validity, per_joint_confidences = find_best_frame_in_window(
        window_start=search_start,
        window_end=search_end,
        pts=pts,
        required_joints=required_joints,
        min_conf=min_q,
        n=n,
        forward_from=ffp_idx,  # Enforce forward-only search
    )
    
    if best_idx is None:
        return pack_metric(
            name="Arm Angle at Foot Strike",
            value=None,
            units="deg",
            reason="no_frame_with_all_required_joints",
            debug={
                "ffp_idx": int(ffp_idx),
                "throwing_side": throwing_side,
                "window": f"{search_start}..{search_end}",
                "required_joints": required_joints,
                "available_keys_per_frame": {
                    i: [k for k, v in pts.items() if i < len(v) and v[i] is not None]
                    for i in range(search_start, search_end + 1)
                },
                "throwing_shoulder": shoulder_key,
                "throwing_elbow": elbow_key,
                "throwing_wrist": wrist_key,
            },
        )
    
    try:
        S = pts[shoulder_key][best_idx]
        E = pts[elbow_key][best_idx]
        W = pts[wrist_key][best_idx]
        
        if not all(is_joint_valid(pts, best_idx, j, n=n) for j in required_joints):
            return pack_metric(
                name="Arm Angle at Foot Strike",
                value=None,
                units="deg",
                reason="missing_joint_at_best_frame",
                debug={
                    "ffp_idx": int(ffp_idx),
                    "best_idx": int(best_idx),
                    "window": f"{search_start}..{search_end}",
                    "required_joints": required_joints,
                    "per_joint_validity": per_joint_validity,
                },
            )
        
        # Compute vectors: a = S - E, b = W - E
        a = (float(S[0]) - float(E[0]), float(S[1]) - float(E[1]))
        b = (float(W[0]) - float(E[0]), float(W[1]) - float(E[1]))
        
        # Compute norms
        a_norm = math.sqrt(a[0]**2 + a[1]**2)
        b_norm = math.sqrt(b[0]**2 + b[1]**2)
        
        # Check for degenerate vectors
        if a_norm < 1e-6 or b_norm < 1e-6:
            return pack_metric(
                name="Arm Angle at Foot Strike",
                value=None,
                units="deg",
                reason="degenerate_arm_vectors",
            )
        
        # Compute angle: cos = dot(a,b)/(||a||*||b||)
        dot_product = a[0] * b[0] + a[1] * b[1]
        cos_angle = dot_product / (a_norm * b_norm)
        
        # Clamp cos to [-1, 1] to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # Compute angle in degrees
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        if not np.isfinite(angle_deg):
            return pack_metric(
                name="Arm Angle at Foot Strike",
                value=None,
                units="deg",
                reason="invalid_angle_computation",
            )
        
        # Determine status: GREEN (80-83 deg), YELLOW (70-80 deg or 83-110 deg), RED (<70 deg or >110 deg)
        if 80 <= angle_deg <= 83:
            status = "green"
            score = 100
        elif (70 <= angle_deg < 80) or (83 < angle_deg <= 110):
            status = "yellow"
            score = 75
        else:  # <70 or >110
            status = "red"
            score = 25
        
        return pack_metric(
            name="Arm Angle at Foot Strike",
            value=round(float(angle_deg), 1),
            units="deg",
            score=score,
            status=status,
            reason="ok",
            debug={
                "ffp_idx": int(ffp_idx),
                "throwing_side": throwing_side,
                "best_idx": int(best_idx),
                "window": f"{search_start}..{search_end}",
                "required_joints": required_joints,
                "best_valid_count": len([k for k, v in per_joint_validity.items() if v]),
                "per_joint_validity": per_joint_validity,
                "per_joint_confidences": per_joint_confidences,
            },
        )
    except Exception as e:
        return pack_metric(
            name="Arm Angle at Foot Strike",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )


def compute_ball_angle_vs_shoulder_line_ffp(pose_cache, keyframes, min_q=0.35):
    """
    Measures the angle (degrees) between the shoulder line and the vector from throwing shoulder to throwing wrist
    at foot strike (FFP). Interprets wrist as ball position proxy.
    
    Scoring thresholds:
    - Green: 10°–25° (inclusive) - ideal range
    - Yellow: 0°–10° (arm late) and 25°–40° (arm early)
    - Red: < 0° (below shoulder line) and > 40° (too high)
    
    Args:
        pose_cache: Cached pose data with "pts" dict and "pose_quality" list
        keyframes: Dict with ffp_idx, throwing_side
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        dict: Packed metric with value (degrees), status (green/yellow/red), and debug info
    """
    ffp_idx = keyframes.get("ffp_idx")
    throwing_side = keyframes.get("throwing_side", "R")
    
    # Guardrails: require ffp_idx and throwing_side
    if ffp_idx is None or throwing_side is None:
        return pack_metric(
            name="Ball Location at Foot Strike",
            value=None,
            units="deg",
            reason="missing_keyframes",
        )
    
    if pose_cache is None or pose_cache.get("pts") is None:
        return pack_metric(
            name="Ball Location at Foot Strike",
            value=None,
            units="deg",
            reason="missing_pose_cache",
        )
    
    pts = pose_cache["pts"]
    n = len(pts.get("LEFT_SHOULDER", [])) if "LEFT_SHOULDER" in pts else 0
    
    # Get landmarks
    ls_key = "LEFT_SHOULDER"
    rs_key = "RIGHT_SHOULDER"
    
    if throwing_side.upper() == "R":
        ts_key = "RIGHT_SHOULDER"
        wrist_key = "RIGHT_WRIST"
    else:  # L
        ts_key = "LEFT_SHOULDER"
        wrist_key = "LEFT_WRIST"
    
    required_joints = [ls_key, rs_key, ts_key, wrist_key]
    
    # Only search forward from FFP (never backward)
    search_start = ffp_idx
    search_end = min(n - 1, ffp_idx + 3)
    best_idx, best_q, per_joint_validity, per_joint_confidences = find_best_frame_in_window(
        pts=pts,
        window_start=search_start,
        window_end=search_end,
        required_joints=required_joints,
        min_conf=min_q,
        n=n,
        forward_from=ffp_idx,  # Enforce forward-only search
    )
    
    if best_idx is None:
        return pack_metric(
            name="Ball Location at Foot Strike",
            value=None,
            units="deg",
            reason="no_frame_with_all_required_joints",
            debug={
                "ffp_idx": int(ffp_idx),
                "throwing_side": throwing_side,
                "window": f"{search_start}..{search_end}",
                "required_joints": required_joints,
                "available_keys_per_frame": {
                    i: [k for k, v in pts.items() if i < len(v) and v[i] is not None]
                    for i in range(search_start, search_end + 1)
                },
                "throwing_shoulder": ts_key,
                "throwing_wrist": wrist_key,
            },
        )
    
    try:
        LS = pts[ls_key][best_idx]
        RS = pts[rs_key][best_idx]
        TS = pts[ts_key][best_idx]
        W = pts[wrist_key][best_idx]  # ball proxy
        
        if not all(is_joint_valid(pts, best_idx, j, n=n) for j in required_joints):
            return pack_metric(
                name="Ball Location at Foot Strike",
                value=None,
                units="deg",
                reason="missing_joint_at_best_frame",
                debug={
                    "ffp_idx": int(ffp_idx),
                    "best_idx": int(best_idx),
                    "window": f"{search_start}..{search_end}",
                    "required_joints": required_joints,
                    "per_joint_validity": per_joint_validity,
                },
            )
        
        # Shoulder-line unit direction: v = RS - LS
        v = (float(RS[0]) - float(LS[0]), float(RS[1]) - float(LS[1]))
        v_mag = math.sqrt(v[0]**2 + v[1]**2)
        
        if v_mag < 1e-6:
            return pack_metric(
                name="Ball Location at Foot Strike",
                value=None,
                units="deg",
                reason="degenerate_shoulder_line",
            )
        
        vhat = (v[0] / v_mag, v[1] / v_mag)
        
        # Throwing-arm (ball) vector: a = W - TS
        a = (float(W[0]) - float(TS[0]), float(W[1]) - float(TS[1]))
        a_mag = math.sqrt(a[0]**2 + a[1]**2)
        
        if a_mag < 1e-6:
            return pack_metric(
                name="Ball Location at Foot Strike",
                value=None,
                units="deg",
                reason="degenerate_arm_vector",
            )
        
        ahat = (a[0] / a_mag, a[1] / a_mag)
        
        # Angle magnitude (0..180): ang = degrees(arccos(clamp(dot(vhat, ahat), -1, 1)))
        dot_product = vhat[0] * ahat[0] + vhat[1] * ahat[1]
        dot_product = max(-1.0, min(1.0, dot_product))  # clamp to [-1, 1]
        
        ang_rad = math.acos(dot_product)
        ang = math.degrees(ang_rad)
        
        # Convert to acute angle relative to shoulder line (0..90)
        ang = min(ang, 180.0 - ang)
        
        if not np.isfinite(ang):
            return pack_metric(
                name="Ball Location at Foot Strike",
                value=None,
                units="deg",
                reason="invalid_angle_computation",
            )
        
        # Determine if wrist is above or below the shoulder line
        # Signed perpendicular distance of W from shoulder line through LS
        # n = [-v_y, v_x] normalized
        n_vec = (-v[1], v[0])
        n_mag = math.sqrt(n_vec[0]**2 + n_vec[1]**2)
        
        if n_mag < 1e-6:
            return pack_metric(
                name="Ball Location at Foot Strike",
                value=None,
                units="deg",
                reason="degenerate_normal_vector",
            )
        
        n_unit = (n_vec[0] / n_mag, n_vec[1] / n_mag)
        
        # d_px = dot((W - LS), n_unit)
        # Define "above" as d_px > 0
        W_minus_LS = (float(W[0]) - float(LS[0]), float(W[1]) - float(LS[1]))
        d_px = W_minus_LS[0] * n_unit[0] + W_minus_LS[1] * n_unit[1]
        
        # Classification based on angle thresholds:
        # GREEN: 10°–25° (inclusive)
        # YELLOW: 0°–10° (arm late) and 25°–40° (arm early)
        # RED: < 0° and > 40°
        # Note: If wrist is below shoulder line (d_px <= 0), treat as negative angle for scoring
        if d_px <= 0:
            # Wrist below shoulder line - treat as negative angle (red)
            status = "red"
            score = 25
        elif 10 <= ang <= 25:
            # Green: ideal range
            status = "green"
            score = 100
        elif (0 <= ang < 10) or (25 < ang <= 40):
            # Yellow: arm late (0-10°) or arm early (25-40°)
            status = "yellow"
            score = 75
        else:  # ang > 40 or ang < 0 (shouldn't happen for acute angle, but handle edge case)
            # Red: too high (>40°) or invalid
            status = "red"
            score = 25
        
        return pack_metric(
            name="Ball Location at Foot Strike",
            value=round(float(ang), 1),
            units="deg",
            score=score,
            status=status,
            reason="ok",
            debug={
                "ffp_idx": int(ffp_idx),
                "throwing_side": throwing_side,
                "ang": round(float(ang), 1),
                "d_px": round(float(d_px), 1),
                "best_idx": int(best_idx),
                "window": f"{search_start}..{search_end}",
                "required_joints": required_joints,
                "best_valid_count": len([k for k, v in per_joint_validity.items() if v]),
                "per_joint_validity": per_joint_validity,
                "per_joint_confidences": per_joint_confidences,
            },
        )
    except Exception as e:
        return pack_metric(
            name="Ball Location at Foot Strike",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )


def compute_upper_body_lean_at_release(clip, pose_cache, keyframes, min_q=0.35):
    """
    Measures upper-body lean angle at release (high school only).
    
    Computes the angle between vertical and the vector from mid-hip to mid-shoulder.
    Uses release_idx (or release_idx+1 if release_idx is not good).
    
    Formula:
      mid_hip = (LEFT_HIP + RIGHT_HIP) / 2
      mid_sh = (LEFT_SHOULDER + RIGHT_SHOULDER) / 2
      dx = mid_sh.x - mid_hip.x
      dy = mid_sh.y - mid_hip.y
      lean_deg = degrees(atan2(abs(dx), abs(dy)))
    
    Where 0° = upright and larger = more forward lean.
    Uses forward_dir to ensure we're measuring lean toward the plate.
    
    High school score bands:
      RED < 8° (too upright)
      YELLOW 8-14° (moderate lean)
      GREEN ≥ 14° (good forward lean)
    
    Args:
        clip: Dict with video metadata (n, fps)
        pose_cache: Cached pose data with "pts" dict
        keyframes: Dict with release_idx, throwing_side, forward_dir
        min_q: Minimum pose quality threshold (default 0.35, unused but kept for API consistency)
    
    Returns:
        dict: Packed metric with value (degrees), status (green/yellow/red), and debug info
    """
    import math
    
    release_idx = keyframes.get("release_idx")
    throwing_side = keyframes.get("throwing_side", "R")
    forward_dir = keyframes.get("forward_dir", 1.0)  # Default to 1.0 if not provided
    
    # Guardrails: require release_idx
    if release_idx is None:
        return pack_metric(
            name="Upper-Body Lean at Release",
            value=None,
            units="deg",
            reason="missing_keyframes",
            debug={
                "fail_reason": "missing_keyframes",
                "release_idx": release_idx,
                "throwing_side": throwing_side,
            },
        )
    
    if pose_cache is None or pose_cache.get("pts") is None:
        return pack_metric(
            name="Upper-Body Lean at Release",
            value=None,
            units="deg",
            reason="missing_pose_cache",
            debug={
                "fail_reason": "missing_pose_cache",
                "release_idx": int(release_idx),
                "throwing_side": throwing_side,
            },
        )
    
    pts = pose_cache["pts"]
    n = len(pts.get("LEFT_SHOULDER", [])) if "LEFT_SHOULDER" in pts else 0
    
    # Frames/time metadata
    n_frames = None
    if isinstance(clip, dict):
        try:
            n_frames = int(clip.get("n"))
        except Exception:
            n_frames = None
    
    if n_frames is None or n_frames <= 0:
        n_frames = n
    
    # Use release_idx frame first, then release_idx + 1 if release_idx is not good
    selected_idx = release_idx
    used_idx = None
    
    # Try release_idx first
    from src.metrics.utils import get_joint_xy
    
    lh_xy = get_joint_xy(selected_idx, "LEFT_HIP", pts=pts)
    rh_xy = get_joint_xy(selected_idx, "RIGHT_HIP", pts=pts)
    ls_xy = get_joint_xy(selected_idx, "LEFT_SHOULDER", pts=pts)
    rs_xy = get_joint_xy(selected_idx, "RIGHT_SHOULDER", pts=pts)
    
    if lh_xy is not None and rh_xy is not None and ls_xy is not None and rs_xy is not None:
        used_idx = selected_idx
    else:
        # If release_idx is not good, try release_idx + 1 (only one frame forward)
        next_idx = selected_idx + 1
        if next_idx < n:
            lh_xy = get_joint_xy(next_idx, "LEFT_HIP", pts=pts)
            rh_xy = get_joint_xy(next_idx, "RIGHT_HIP", pts=pts)
            ls_xy = get_joint_xy(next_idx, "LEFT_SHOULDER", pts=pts)
            rs_xy = get_joint_xy(next_idx, "RIGHT_SHOULDER", pts=pts)
            if lh_xy is not None and rh_xy is not None and ls_xy is not None and rs_xy is not None:
                used_idx = next_idx
    
    if used_idx is None or lh_xy is None or rh_xy is None or ls_xy is None or rs_xy is None:
        return pack_metric(
            name="Upper-Body Lean at Release",
            value=None,
            units="deg",
            reason="Required joints (LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER) not detected at release_idx or release_idx+1",
            debug={
                "release_idx": int(release_idx),
                "selected_idx": int(selected_idx),
                "used_idx": used_idx,
                "lh_xy": lh_xy is not None,
                "rh_xy": rh_xy is not None,
                "ls_xy": ls_xy is not None,
                "rs_xy": rs_xy is not None,
            },
        )
    
    try:
        # Compute mid-hip and mid-shoulder
        mid_hip = (
            (float(lh_xy[0]) + float(rh_xy[0])) / 2.0,
            (float(lh_xy[1]) + float(rh_xy[1])) / 2.0,
        )
        mid_sh = (
            (float(ls_xy[0]) + float(rs_xy[0])) / 2.0,
            (float(ls_xy[1]) + float(rs_xy[1])) / 2.0,
        )
        
        # Compute dx and dy
        dx = mid_sh[0] - mid_hip[0]
        dy = mid_sh[1] - mid_hip[1]
        
        # Use forward_dir to determine if we're leaning forward (toward plate)
        # If forward_dir is negative, forward is -x, so we need to flip dx sign
        # For unsigned angle, we use abs(dx) to measure "how much lean" regardless of direction
        # But we should verify it's forward lean using forward_dir
        dx_signed = dx * forward_dir  # Positive if leaning forward (toward plate)
        
        # Compute lean angle: atan2(abs(dx), abs(dy)) where 0° = upright, larger = more forward lean
        # Using abs() for both ensures we get the magnitude of lean regardless of direction
        lean_deg = math.degrees(math.atan2(abs(dx), abs(dy)))
        
        # Verify we're actually leaning forward (not backward) using signed dx
        # If dx_signed is negative, we're leaning backward, which should be penalized
        # For now, we'll use the unsigned angle but could add a check
        
        if not np.isfinite(lean_deg):
            return pack_metric(
                name="Upper-Body Lean at Release",
                value=None,
                units="deg",
                reason="invalid_angle_computation",
                debug={
                    "release_idx": int(release_idx),
                    "used_idx": int(used_idx),
                    "mid_hip": [round(float(mid_hip[0]), 1), round(float(mid_hip[1]), 1)],
                    "mid_sh": [round(float(mid_sh[0]), 1), round(float(mid_sh[1]), 1)],
                    "dx": round(float(dx), 1),
                    "dy": round(float(dy), 1),
                    "forward_dir": round(float(forward_dir), 2),
                },
            )
        
        # Score using high school bands: RED < 8°, YELLOW 8-14°, GREEN ≥ 14°
        from src.metrics.scoring import score_upper_body_lean_release_deg
        score, status = score_upper_body_lean_release_deg(lean_deg) if lean_deg is not None else (None, None)
        
        return pack_metric(
            name="Upper-Body Lean at Release",
            value=round(float(lean_deg), 1),
            units="deg",
            score=score,
            status=status,
            reason="ok",
            debug={
                "release_idx": int(release_idx),
                "used_idx": int(used_idx),
                "mid_hip": [round(float(mid_hip[0]), 1), round(float(mid_hip[1]), 1)],
                "mid_sh": [round(float(mid_sh[0]), 1), round(float(mid_sh[1]), 1)],
                "dx": round(float(dx), 1),
                "dy": round(float(dy), 1),
                "dx_signed": round(float(dx_signed), 1),
                "forward_dir": round(float(forward_dir), 2),
                "lean_deg": round(float(lean_deg), 1),
            },
        )
    except Exception as e:
        return pack_metric(
            name="Upper-Body Lean at Release",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )


def compute_posture_metrics(clip, keyframes, pose_cache):
    """
    Compute posture-related metrics.
    
    Args:
        clip: Dict with video metadata (n, fps, pose_quality, process_width, process_height)
        keyframes: Dict with ffp_idx, release_idx, throwing_side
        pose_cache: Cached pose data
    
    Returns:
        dict: Metrics dict with posture metrics
    """
    metrics = {}
    
    # Head Behind Hip metric
    try:
        head_behind_hip_result = compute_head_behind_hip_middrive(pose_cache, keyframes)
        metrics["head_behind_hip"] = head_behind_hip_result
    except Exception as e:
        metrics["head_behind_hip"] = pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason=f"error: {str(e)}",
        )
    
    # Throwing Elbow vs Shoulder Line metric
    try:
        elbow_vs_shoulder_result = compute_throw_elbow_vs_shoulder_line_ffp(pose_cache, keyframes)
        metrics["throw_elbow_vs_shoulder_line_ffp"] = elbow_vs_shoulder_result
    except Exception as e:
        metrics["throw_elbow_vs_shoulder_line_ffp"] = pack_metric(
            name="Elbow Height at Foot Strike",
            value=None,
            units="ratio",
            reason=f"error: {str(e)}",
        )
    
    # Throwing Elbow Bend Angle metric
    try:
        elbow_bend_result = compute_elbow_bend_angle_ffp(pose_cache, keyframes)
        metrics["elbow_bend_ffp_deg"] = elbow_bend_result
    except Exception as e:
        metrics["elbow_bend_ffp_deg"] = pack_metric(
            name="Arm Angle at Foot Strike",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )
    
    # Ball Angle vs Shoulder Line metric
    try:
        ball_angle_result = compute_ball_angle_vs_shoulder_line_ffp(pose_cache, keyframes)
        metrics["ball_angle_vs_shoulder_line_ffp_deg"] = ball_angle_result
    except Exception as e:
        metrics["ball_angle_vs_shoulder_line_ffp_deg"] = pack_metric(
            name="Ball Location at Foot Strike",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )
    
    # Upper-Body Lean at Release metric (high school only)
    try:
        upper_body_lean_result = compute_upper_body_lean_at_release(clip, pose_cache, keyframes, min_q=0.35)
        metrics["upper_body_lean_release"] = upper_body_lean_result
    except Exception as e:
        metrics["upper_body_lean_release"] = pack_metric(
            name="Upper-Body Lean at Release",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )
    
    return metrics

