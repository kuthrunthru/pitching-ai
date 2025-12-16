"""
Release metrics computation.

Currently includes Release Height.
"""

import math
import numpy as np
from src.metrics.utils import (
    pack_metric,
    CONF_THRESH,
    find_best_frame_in_window,
    is_joint_valid,
)


def _compute_body_height_px(pts, frame_idx):
    """
    (Deprecated helper kept for backward compatibility; currently unused.)
    """
    return None



# Head Lean at Release functions removed - metric deleted

def compute_release_metrics(clip, keyframes, pose_cache, min_q=0.35):
    """
    Compute release-related metrics (Release Height).
    
    Args:
        clip: Dict with video metadata (n, fps, pose_quality, process_width, process_height)
        keyframes: Dict with ffp_idx, release_idx, throwing_side
        pose_cache: Cached pose data
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        dict: Metrics dict with "release_height" entry
    """
    from src.metrics.scoring import score_release_height_bl
    
    metrics = {}
    
    n = clip.get("n")
    fps = clip.get("fps", 30.0)
    pose_quality = clip.get("pose_quality", [])
    
    release_idx = keyframes.get("release_idx")
    throwing_side = keyframes.get("throwing_side", "R")
    
    # Guardrails
    if release_idx is None:
        metrics["release_height"] = pack_metric(
            name="Release Height",
            value=None,
            units="BL",
            reason="missing_release_idx",
        )
        return metrics
    
    if pose_cache is None or pose_cache.get("pts") is None:
        metrics["release_height"] = pack_metric(
            name="Release Height",
            value=None,
            units="BL",
            reason="missing_pose_cache",
        )
        return metrics
    
    pts = pose_cache["pts"]
    
    # Determine throwing side landmarks
    if throwing_side.upper() == "R":
        wrist_key = "RIGHT_WRIST"
    else:  # L
        wrist_key = "LEFT_WRIST"
    
    # Find best frame near release using metric-specific joint quality
    # Required joints: wrist (release point) + ankles (height reference) + shoulders (body height)
    required_joints_release_height = [wrist_key, "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_SHOULDER", "RIGHT_SHOULDER"]

    # Only search forward from release (never backward)
    search_start = release_idx
    search_end = min(n - 1, release_idx + 3)

    best_idx, best_avg_conf, best_valid, best_conf = find_best_frame_in_window(
        window_start=search_start,
        window_end=search_end,
        pts=pts,
        required_joints=required_joints_release_height,
        min_conf=min_q,
        n=n,
        forward_from=release_idx,  # Enforce forward-only search
    )

    if best_idx is None:
        metrics["release_height"] = pack_metric(
            name="Release Height",
            value=None,
            units="BL",
            reason="no_usable_forward_frame",
            debug={
                "release_idx": int(release_idx),
                "window": [search_start, search_end],
                "required_joints": required_joints_release_height,
            },
        )
        return metrics
    
    # Assertion in debug mode to ensure forward-only selection
    debug_mode = clip.get("debug_mode", False) if isinstance(clip, dict) else False
    if debug_mode and best_idx < release_idx:
        raise AssertionError(f"Release Height picked frame {best_idx} before selected release_idx {release_idx}")

    # Compute release height at best_idx
    try:
        wrist = pts[wrist_key][best_idx]
        left_ankle = pts["LEFT_ANKLE"][best_idx]
        right_ankle = pts["RIGHT_ANKLE"][best_idx]
        
        # Release point (wrist)
        release_y = float(wrist[1])
        
        # Ankle midpoint (for height reference)
        ankle_mid_y = (float(left_ankle[1]) + float(right_ankle[1])) / 2.0
        
        # Body height for normalization
        body_height_px = _compute_body_height_px(pts, best_idx)
        
        if body_height_px is not None and body_height_px > 0:
            # Release height: vertical distance from release point to ankle midpoint (normalized)
            # In image coordinates, y increases downward, so (ankle_mid_y - release_y) gives height above ankle
            release_height_px = ankle_mid_y - release_y
            release_height_bl = release_height_px / body_height_px
            
            if np.isfinite(release_height_bl) and release_height_bl >= 0:
                # Score the metric
                score, status = score_release_height_bl(release_height_bl) if release_height_bl is not None else (None, None)
                
                metrics["release_height"] = pack_metric(
                    name="Release Height",
                    value=round(float(release_height_bl), 3),
                    units="BL",
                    score=score,
                    status=status,
                    reason="ok",
                    debug={
                        "release_height_px": round(float(release_height_px), 1),
                        "body_height_px": round(float(body_height_px), 1),
                        "release_idx": int(release_idx),
                        "used_idx": int(best_idx),  # Best joint-valid frame chosen from window
                        "avg_conf_at_used_idx": round(float(best_avg_conf), 3),
                        "window": f"{search_start}..{search_end}",
                        "required_joints": required_joints_release_height,
                        "per_joint_valid": best_valid,
                        "per_joint_conf": best_conf,
                        "valid_joint_count": int(sum(1 for v in best_valid.values() if v)),
                    },
                )
            else:
                metrics["release_height"] = pack_metric(
                    name="Release Height",
                    value=None,
                    units="BL",
                    reason="invalid_release_height_computation",
                )
        else:
            metrics["release_height"] = pack_metric(
                name="Release Height",
                value=None,
                units="BL",
                reason="could_not_compute_body_height",
            )
    except Exception as e:
        metrics["release_height"] = pack_metric(
            name="Release Height",
            value=None,
            units="BL",
            reason=f"error: {str(e)}",
        )
    
    # Head Lean at Release metric removed
    
    return metrics

