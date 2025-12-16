"""
Lower body metrics computation.

Includes Stride Length (translation- and geometry-based metrics only).
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

# MKL_FALLBACK_MAX removed - MKL keyframe deleted


def _compute_body_height_px(pts, frame_idx):
    """
    Compute body height in pixels at a given frame.
    
    Uses distance from mid-shoulder to mid-ankle (torso height).
    Note: This is shorter than full body height, so stride length ratios will be higher
    than if normalized by full body height. Thresholds are adjusted accordingly.
    
    Args:
        pts: Dict of landmark arrays
        frame_idx: Frame index
    
    Returns:
        body_height_px: float or None if landmarks missing
    """
    if frame_idx < 0:
        return None
    
    # Try mid-shoulder to mid-ankle (torso height)
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


def compute_leg_lift_metrics(clip, keyframes, pose_cache, min_q=0.35):
    """
    DISABLED: Leg Lift Angle metric has been removed.
    
    Returns empty dict.
    """
    metrics = {}
    # Metric disabled - Leg Lift Angle has been removed
    return metrics


def compute_stride_length(pose_cache, keyframes, min_q=0.35):
    """
    Compute stride length metric at FFP keyframe.
    
    Measures the distance between lead and back ankle at foot strike, normalized by body height.
    
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
            name="Stride Length",
            value=None,
            units="ratio",
            reason="missing_keyframes",
        )
    
    if pose_cache is None or pose_cache.get("pts") is None:
        return pack_metric(
            name="Stride Length",
            value=None,
            units="ratio",
            reason="missing_pose_cache",
        )
    
    pts = pose_cache["pts"]
    # pose_quality not used for gating
    n = len(pts.get("LEFT_ANKLE", [])) if "LEFT_ANKLE" in pts else 0
    
    # Determine stride/lead side
    # Lead leg is opposite of throwing side
    if throwing_side.upper() == "R":
        lead = "LEFT"
        back = "RIGHT"
    else:  # L
        lead = "RIGHT"
        back = "LEFT"
    
    # Get ankle points near FFP (window ±3)
    lead_ankle_key = f"{lead}_ANKLE"
    back_ankle_key = f"{back}_ANKLE"
    required_joints = [lead_ankle_key, back_ankle_key]
    
    # Only search forward from FFP (never backward)
    search_start = ffp_idx
    search_end = min(n - 1, ffp_idx + 3)
    best_idx, best_avg_conf, best_valid, best_conf = find_best_frame_in_window(
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
            name="Stride Length",
            value=None,
            units="ratio",
            reason="no_frame_with_all_required_joints",
            debug={
                "ffp_idx": int(ffp_idx),
                "window": [search_start, search_end],
                "required_joints": required_joints,
            },
        )
    
    try:
        lead_ankle = pts[lead_ankle_key][best_idx]
        back_ankle = pts[back_ankle_key][best_idx]
        
        # Compute stride distance in pixels = ||lead_ankle - back_ankle||
        dx = float(lead_ankle[0]) - float(back_ankle[0])
        dy = float(lead_ankle[1]) - float(back_ankle[1])
        stride_px = math.sqrt(dx * dx + dy * dy)
        
        if not np.isfinite(stride_px):
            return pack_metric(
                name="Stride Length",
                value=None,
                units="ratio",
                reason="invalid_stride_computation",
            )
        
        # Normalize by body height pixels at best_idx
        body_height_px = _compute_body_height_px(pts, best_idx)
        
        if body_height_px is None or body_height_px <= 0:
            return pack_metric(
                name="Stride Length",
                value=None,
                units="ratio",
                reason="missing_body_height",
                debug={
                    "stride_px": round(float(stride_px), 1),
                    "ffp_idx": int(ffp_idx),
                    "used_idx": int(best_idx),
                    "lead": lead,
                    "back": back,
                    "per_joint_valid": best_valid,
                    "per_joint_conf": best_conf,
                    "valid_joint_count": int(sum(1 for v in best_valid.values() if v)),
                    "avg_conf": round(float(best_avg_conf), 3),
                },
            )
        
        # Compute stride ratio
        stride_ratio = stride_px / body_height_px
        
        if not np.isfinite(stride_ratio):
            return pack_metric(
                name="Stride Length",
                value=None,
                units="ratio",
                reason="invalid_ratio_computation",
            )
        
        # Use scoring function for consistent thresholds: GREEN (>= 0.85), YELLOW (0.65-0.84), RED (< 0.65)
        from src.metrics.scoring import score_stride_length_ratio
        score, status = score_stride_length_ratio(stride_ratio) if stride_ratio is not None else (None, None)
        
        # Ensure score and status are set (fallback if scoring function fails)
        if score is None or status is None:
            score = 25
            status = "red"
        
        return pack_metric(
                name="Stride Length",
                value=round(float(stride_ratio), 3),
                units="ratio",
                score=score,
                status=status,
                reason="ok",
                debug={
                    "ffp_idx": int(ffp_idx),
                    "used_idx": int(best_idx),
                    "lead": lead,
                    "back": back,
                    "stride_px": round(float(stride_px), 1),
                    "body_height_px": round(float(body_height_px), 1),
                    "per_joint_valid": best_valid,
                    "per_joint_conf": best_conf,
                    "valid_joint_count": int(sum(1 for v in best_valid.values() if v)),
                    "avg_conf": round(float(best_avg_conf), 3),
                },
            )
    except Exception as e:
        return pack_metric(
            name="Stride Length",
            value=None,
            units="ratio",
            reason=f"error: {str(e)}",
        )


def compute_landing_leg_bend_ffp(pose_cache, keyframes, min_q=0.35):
    """
    Measures landing/stride knee bend at FFP.
    
    Uses bend_deg = 180 - knee_angle_deg so 45 deg bend means knee_angle ~135 deg.
    Ideal bend ~45 deg. Too straight is bad. Too collapsed is also bad.
    
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
            name="Landing Leg Bend",
            value=None,
            units="deg",
            reason="missing_keyframes",
        )
    
    if pose_cache is None or pose_cache.get("pts") is None:
        return pack_metric(
            name="Landing Leg Bend",
            value=None,
            units="deg",
            reason="missing_pose_cache",
        )
    
    pts = pose_cache["pts"]
    n = len(pts.get("LEFT_HIP", [])) if "LEFT_HIP" in pts else 0
    
    # Determine landing/stride side
    # Lead leg is opposite of throwing side
    if throwing_side.upper() == "R":
        lead = "LEFT"
    else:  # L
        lead = "RIGHT"
    
    hip_key = f"{lead}_HIP"
    knee_key = f"{lead}_KNEE"
    ankle_key = f"{lead}_ANKLE"
    required_joints = [hip_key, knee_key, ankle_key]
    
    # Only search forward from FFP (never backward)
    search_start = ffp_idx
    search_end = min(n - 1, ffp_idx + 3)
    best_idx, best_avg_conf, best_valid, best_conf = find_best_frame_in_window(
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
            name="Landing Leg Bend",
            value=None,
            units="deg",
            reason="no_frame_with_all_required_joints",
            debug={
                "ffp_idx": int(ffp_idx),
                "window": [search_start, search_end],
                "required_joints": required_joints,
            },
        )
    
    try:
        # Joints are already validated by is_pose_usable
        H = pts[hip_key][best_idx]
        K = pts[knee_key][best_idx]
        A = pts[ankle_key][best_idx]
        
        # Compute knee angle at K
        # a = H - K (hip to knee vector)
        # b = A - K (ankle to knee vector)
        a = (float(H[0]) - float(K[0]), float(H[1]) - float(K[1]))
        b = (float(A[0]) - float(K[0]), float(A[1]) - float(K[1]))
        
        # Compute norms
        a_norm = math.sqrt(a[0]**2 + a[1]**2)
        b_norm = math.sqrt(b[0]**2 + b[1]**2)
        
        # Check for degenerate vectors
        if a_norm < 1e-6 or b_norm < 1e-6:
            return pack_metric(
                name="Landing Leg Bend",
                value=None,
                units="deg",
                reason="degenerate_leg_vectors",
            )
        
        # Compute angle: cos = dot(a,b)/(||a||*||b||)
        dot_product = a[0] * b[0] + a[1] * b[1]
        cos_angle = dot_product / (a_norm * b_norm)
        
        # Clamp cos to [-1, 1] to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # Compute knee angle in degrees
        # 180 deg = straight, smaller = more bent
        knee_angle = math.degrees(math.acos(cos_angle))
        
        # Convert to coaching-friendly "bend": bend_deg = 180.0 - knee_angle
        # So 45 deg bend means knee_angle ~135 deg
        bend_deg = 180.0 - knee_angle
        
        if not np.isfinite(bend_deg):
            return pack_metric(
                name="Landing Leg Bend",
                value=None,
                units="deg",
                reason="invalid_bend_computation",
            )
        
        # Determine status: GREEN (35-55 deg), YELLOW (25-35 deg or 55-70 deg), RED (<25 deg or >70 deg)
        if 35 <= bend_deg <= 55:
            status = "green"
            score = 100
        elif (25 <= bend_deg < 35) or (55 < bend_deg <= 70):
            status = "yellow"
            score = 75
        else:  # <25 or >70
            status = "red"
            score = 25
        
        return pack_metric(
            name="Landing Leg Bend",
            value=round(float(bend_deg), 1),
            units="deg",
            score=score,
            status=status,
            reason="ok",
            debug={
                "ffp_idx": int(ffp_idx),
                "used_idx": int(best_idx),
                "lead": lead,
                "knee_angle_deg": round(float(knee_angle), 1),
                "bend_deg": round(float(bend_deg), 1),
                "per_joint_valid": best_valid,
                "per_joint_conf": best_conf,
                "valid_joint_count": int(sum(1 for v in best_valid.values() if v)),
                "avg_conf": round(float(best_avg_conf), 3),
            },
        )
    except Exception as e:
        return pack_metric(
            name="Landing Leg Bend",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )


def compute_lower_body_metrics(clip, keyframes, pose_cache, min_q=0.35):
    """
    Compute lower body metrics (Stride Length).
    
    Args:
        clip: Dict with video metadata (n, fps, pose_quality, process_width, process_height)
        keyframes: Dict with ffp_idx, release_idx, throwing_side, forward_dir
        pose_cache: Cached pose data
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        dict: Metrics dict with "stride_length" entry
    """
    metrics = {}
    
    n = clip.get("n")
    fps = clip.get("fps", 30.0)
    pose_quality = clip.get("pose_quality", [])
    
    ffp_idx = keyframes.get("ffp_idx")
    throwing_side = keyframes.get("throwing_side", "R")
    
    # Guardrails
    if ffp_idx is None:
        metrics["stride_length"] = pack_metric(
            name="Stride Length",
            value=None,
            units="BL",
            reason="missing_ffp_idx",
        )
        return metrics
    
    if pose_cache is None or pose_cache.get("pts") is None:
        metrics["stride_length"] = pack_metric(
            name="Stride Length",
            value=None,
            units="BL",
            reason="missing_pose_cache",
        )
        return metrics
    
    # 1. Stride Length (at FFP)
    try:
        stride_length_result = compute_stride_length(pose_cache, keyframes, min_q=min_q)
        metrics["stride_length"] = stride_length_result
    except Exception as e:
        metrics["stride_length"] = pack_metric(
            name="Stride Length",
            value=None,
            units="ratio",
            reason=f"error: {str(e)}",
        )
    
    # Lead Leg Block removed - metric deleted
    # Lead Leg Firmness After Release removed - metric deleted
    
    return metrics

