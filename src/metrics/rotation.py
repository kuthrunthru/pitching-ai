"""
Rotation metrics computation.

Includes guidance metrics for rotation timing and positioning.
"""

import math
import numpy as np
from src.metrics.utils import pack_metric, CONF_THRESH, find_best_frame_in_window, is_joint_valid


def _compute_body_height_px(pts, frame_idx):
    """
    Compute body height in pixels at a given frame using shoulders/hips/ankles.
    """
    if frame_idx < 0:
        return None

    key_points = []
    for name in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_ANKLE", "RIGHT_ANKLE"]:
        if name in pts and frame_idx < len(pts[name]):
            pt = pts[name][frame_idx]
            if pt is not None and np.all(np.isfinite(pt[:2])) and np.any(pt[:2] != 0):
                key_points.append((float(pt[0]), float(pt[1])))

    if len(key_points) < 4:
        return None

    ys = [p[1] for p in key_points]
    body_h = max(ys) - min(ys)
    if np.isfinite(body_h) and body_h > 0:
        return body_h
    return None


def compute_chest_closed_ffp(pose_cache, keyframes, min_q=0.35):
    """
    Shoulder-width rotation proxy at foot strike (FFP), feet-free.

    Computes normalized shoulder width over time, builds a baseline from frames
    immediately before FFP, then measures the percent increase at the FFP frame
    as a proxy for how open the chest is.

    Returns value as percent change (positive = more open), with:
      - ~0-5%  -> GOOD
      - ~5-20% -> OK
      - >20%   -> BAD (open early)
    """
    ffp_idx = keyframes.get("ffp_idx")

    # Guardrails: require ffp_idx and pose
    if ffp_idx is None:
        return pack_metric(
            name="Chest Closed at Foot Strike",
            value=None,
            units="%",
            reason="missing_keyframes_or_pose",
        )

    if pose_cache is None or pose_cache.get("pts") is None:
        return pack_metric(
            name="Chest Closed at Foot Strike",
            value=None,
            units="%",
            reason="missing_keyframes_or_pose",
        )

    pts = pose_cache["pts"]
    n = len(pts.get("LEFT_SHOULDER", [])) if "LEFT_SHOULDER" in pts else 0
    if n <= 0:
        return pack_metric(
            name="Chest Closed at Foot Strike",
            value=None,
            units="%",
            reason="no_frames",
        )

    # Shoulder joints
    ls_key = "LEFT_SHOULDER"
    rs_key = "RIGHT_SHOULDER"
    required_joints = [ls_key, rs_key]

    # 1) Pick evaluation frame near FFP using forward-only search
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
            name="Chest Closed at Foot Strike",
            value=None,
            units="%",
            reason="no_frame_with_all_required_joints",
            debug={
                "ffp_idx": int(ffp_idx),
                "window": f"{search_start}..{search_end}",
                "required_joints": required_joints,
                "available_keys_per_frame": {
                    i: [k for k, v in pts.items() if i < len(v) and v[i] is not None]
                    for i in range(search_start, search_end + 1)
                },
            },
        )

    # 2) Build baseline shoulder-width series from frames immediately before FFP
    pre_start = max(0, ffp_idx - 6)
    pre_end = max(0, ffp_idx - 1)

    baseline_widths = []
    baseline_frames = []

    for i in range(pre_start, pre_end + 1):
        if i < 0 or i >= n:
            continue
        ls_valid, _ = is_joint_valid(pts, i, ls_key, n=n)
        rs_valid, _ = is_joint_valid(pts, i, rs_key, n=n)
        if not (ls_valid and rs_valid):
            continue
        LS = pts[ls_key][i]
        RS = pts[rs_key][i]
        if LS is None or RS is None:
            continue
        if not (np.all(np.isfinite(LS[:2])) and np.all(np.isfinite(RS[:2]))):
            continue

        body_h = _compute_body_height_px(pts, i)
        if body_h is None or body_h <= 0:
            continue

        dx = float(RS[0]) - float(LS[0])
        dy = float(RS[1]) - float(LS[1])
        shoulder_w = math.sqrt(dx * dx + dy * dy)
        if not np.isfinite(shoulder_w):
            continue

        width_norm = shoulder_w / body_h
        baseline_widths.append(width_norm)
        baseline_frames.append(i)

    if len(baseline_widths) < 2:
        return pack_metric(
            name="Chest Closed at Foot Strike",
            value=None,
            units="%",
            reason="no_baseline_frames",
            debug={
                "ffp_idx": int(ffp_idx),
                "baseline_window": [int(pre_start), int(pre_end)],
                "baseline_frames": baseline_frames,
            },
        )

    # Simple smoothing: mean of baseline widths
    baseline_mean = float(np.mean(baseline_widths))
    if not np.isfinite(baseline_mean) or baseline_mean <= 0:
        return pack_metric(
            name="Chest Closed at Foot Strike",
            value=None,
            units="%",
            reason="invalid_baseline",
            debug={
                "ffp_idx": int(ffp_idx),
                "baseline_window": [int(pre_start), int(pre_end)],
                "baseline_frames": baseline_frames,
                "baseline_mean": baseline_mean,
            },
        )

    # 3) Evaluate shoulder width at best_idx (foot-strike proxy)
    try:
        if not all(is_joint_valid(pts, best_idx, j, n=n) for j in required_joints):
            return pack_metric(
                name="Chest Closed at Foot Strike",
                value=None,
                units="%",
                reason="missing_joint_at_best_frame",
                debug={
                    "ffp_idx": int(ffp_idx),
                    "best_idx": int(best_idx),
                    "window": f"{search_start}..{search_end}",
                    "required_joints": required_joints,
                    "per_joint_validity": per_joint_validity,
                },
            )

        LS = pts[ls_key][best_idx]
        RS = pts[rs_key][best_idx]

        if LS is None or RS is None or not (
            np.all(np.isfinite(LS[:2])) and np.all(np.isfinite(RS[:2]))
        ):
            return pack_metric(
                name="Chest Closed at Foot Strike",
                value=None,
                units="%",
                reason="missing_landmarks",
            )

        body_h_eval = _compute_body_height_px(pts, best_idx)
        if body_h_eval is None or body_h_eval <= 0:
            return pack_metric(
                name="Chest Closed at Foot Strike",
                value=None,
                units="%",
                reason="missing_body_height",
            )

        dx_e = float(RS[0]) - float(LS[0])
        dy_e = float(RS[1]) - float(LS[1])
        shoulder_w_eval = math.sqrt(dx_e * dx_e + dy_e * dy_e)
        if not np.isfinite(shoulder_w_eval):
            return pack_metric(
                name="Chest Closed at Foot Strike",
                value=None,
                units="%",
                reason="invalid_eval_width",
            )

        width_norm_eval = shoulder_w_eval / body_h_eval

        # Percent increase relative to baseline
        pct_increase = (width_norm_eval - baseline_mean) / baseline_mean * 100.0
        if not np.isfinite(pct_increase):
            return pack_metric(
                name="Chest Closed at Foot Strike",
                value=None,
                units="%",
                reason="invalid_percent_increase",
            )

        # Scoring: small change GOOD, moderate OK, large BAD (open early)
        d = float(pct_increase)
        if d <= 5.0:
            score = 100
            status = "green"
        elif d <= 20.0:
            score = 75
            status = "yellow"
        else:
            score = 25
            status = "red"

        return pack_metric(
            name="Chest Closed at Foot Strike",
            value=round(float(pct_increase), 1),
            units="%",
            score=score,
            status=status,
            reason="ok",
            debug={
                "ffp_idx": int(ffp_idx),
                "best_idx": int(best_idx),
                "window": f"{search_start}..{search_end}",
                "baseline_window": [int(pre_start), int(pre_end)],
                "baseline_frames": baseline_frames,
                "baseline_mean_width_norm": float(baseline_mean),
                "width_norm_eval": float(width_norm_eval),
                "percent_increase": round(float(pct_increase), 2),
                "required_joints": required_joints,
                "best_valid_count": len([k for k, v in per_joint_validity.items() if v]),
                "per_joint_validity": per_joint_validity,
                "per_joint_confidences": per_joint_confidences,
            },
        )
    except Exception as e:
        return pack_metric(
            name="Chest Closed at Foot Strike",
            value=None,
            units="%",
            reason=f"error: {str(e)}",
        )


def compute_rotation_metrics(clip, keyframes, pose_cache):
    """
    Compute rotation-related metrics (guidance metrics).
    
    Args:
        clip: Dict with video metadata (n, fps, pose_quality, process_width, process_height)
        keyframes: Dict with ffp_idx, release_idx, throwing_side
        pose_cache: Cached pose data
    
    Returns:
        dict: Metrics dict with rotation metrics
    """
    metrics = {}
    
    # Chest Closed at Foot Strike metric
    try:
        chest_closed_result = compute_chest_closed_ffp(pose_cache, keyframes)
        metrics["chest_closed_ffp_deg"] = chest_closed_result
    except Exception as e:
        metrics["chest_closed_ffp_deg"] = pack_metric(
            name="Chest Closed at Foot Strike",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )
    
    return metrics

