"""
Shared utility functions for metrics computation.
"""

import math
import numpy as np

# Default confidence threshold for pose quality checks
CONF_THRESH = 0.35

# Knee model moved to src/experimental/knee_model.py - not used in active pipeline


def is_joint_valid(pose, i, joint, n=None, min_q=None, frame_w=None, frame_h=None, conf_thresh=CONF_THRESH):
    """
    Check if a joint is valid at frame i.

    Rules:
    - Joint must exist and be in-bounds for the frame
    - x/y must be finite and not None
    - If frame_w/h provided, x/y must lie within [0, frame_w) / [0, frame_h)
    - Confidence: if present and not None, require conf >= conf_thresh; if missing/None, do NOT fail

    Args:
        pose: Dict of landmark arrays (e.g., pts)
        i: Frame index
        joint: Joint key string
        frame_w: Optional frame width
        frame_h: Optional frame height
        conf_thresh: Confidence threshold to enforce when confidence exists

    Returns:
        (valid: bool, conf: float or None)
    """
    if pose is None or i is None or i < 0:
        return False, None

    if joint not in pose:
        return False, None

    arr = pose[joint]
    if i >= len(arr):
        return False, None

    landmark = arr[i]
    if landmark is None or len(landmark) < 2:
        return False, None

    x, y = landmark[0], landmark[1]
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return False, None

    # Frame bounds
    if frame_w is not None and (x < 0 or x >= frame_w):
        return False, None
    if frame_h is not None and (y < 0 or y >= frame_h):
        return False, None

    conf = None
    if len(landmark) >= 4:
        conf_val = landmark[3]
        if conf_val is not None and np.isfinite(conf_val):
            conf = float(conf_val)
            if conf < conf_thresh:
                return False, conf

    # Valid if coords are good; confidence only gates when present
    return True, conf


def get_pose_quality(frame_idx, pose_quality, n=None):
    """
    Safe accessor for pose quality that clamps indices.

    If confidence is missing/None/non-finite, we skip the gate by returning 1.0
    (treat as pass) instead of failing. This ensures frames are not rejected
    solely due to absent confidence values.

    Returns:
        float: Pose quality score (defaults to 1.0 when unavailable)
    """
    if pose_quality is None or len(pose_quality) == 0:
        return 1.0

    if n is not None:
        frame_idx = max(0, min(frame_idx, n - 1))

    frame_idx = max(0, min(frame_idx, len(pose_quality) - 1))

    if frame_idx < len(pose_quality):
        q = pose_quality[frame_idx]
        if q is None or not np.isfinite(q):
            return 1.0
        return float(q)

    return 1.0


def compute_joint_quality(frame_idx, pts, required_joints, n=None, frame_w=None, frame_h=None, conf_thresh=CONF_THRESH):
    """
    Compute per-frame quality using is_joint_valid across required joints.

    Returns:
        (all_valid: bool, avg_conf_present: float, per_joint_valid: dict, per_joint_conf: dict)
    """
    per_joint_valid = {}
    per_joint_conf = {}
    valid_confs = []

    for joint in required_joints:
        valid, conf = is_joint_valid(pts, frame_idx, joint, frame_w=frame_w, frame_h=frame_h, conf_thresh=conf_thresh)
        per_joint_valid[joint] = valid
        per_joint_conf[joint] = conf
        if conf is not None and np.isfinite(conf):
            valid_confs.append(conf)

    all_valid = all(per_joint_valid.values()) if required_joints else False
    avg_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
    return all_valid, avg_conf, per_joint_valid, per_joint_conf
def pick_first_valid_forward(pts, start_idx, end_idx, required_keys, n=None,
                             frame_w=None, frame_h=None, conf_thresh=CONF_THRESH):
    """
    Scan forward-only from start_idx to end_idx and return the first frame
    where *all* required joints are valid.

    Returns:
        (used_idx, per_joint_valid, per_joint_conf, valid_count)
        - used_idx: int or None
        - per_joint_valid: dict[joint] -> bool for the chosen frame
        - per_joint_conf: dict[joint] -> confidence (or None) for the chosen frame
        - valid_count: number of joints that were valid at used_idx
    """
    if pts is None or not required_keys:
        return None, {}, {}, 0

    # Derive n from pts if not provided
    if n is None:
        lengths = [len(pts.get(k, [])) for k in required_keys if k in pts]
        n = max(lengths) if lengths else 0

    if n <= 0:
        return None, {}, {}, 0

    try:
        i_start = int(start_idx)
        i_end = int(end_idx)
    except Exception:
        return None, {}, {}, 0

    i_start = max(0, min(i_start, n - 1))
    i_end = max(0, min(i_end, n - 1))

    if i_start > i_end:
        return None, {}, {}, 0

    for i in range(i_start, i_end + 1):
        all_valid, _, per_joint_valid, per_joint_conf = compute_joint_quality(
            i, pts, required_keys, n=n, frame_w=frame_w, frame_h=frame_h, conf_thresh=conf_thresh
        )
        if all_valid:
            valid_count = sum(1 for v in per_joint_valid.values() if v)
            return i, per_joint_valid, per_joint_conf, valid_count

    return None, {}, {}, 0


def pick_valid_frame_forward(
    pts,
    knee_key,
    start_idx,
    max_forward_frames,
    n=None,
    frame_w=None,
    frame_h=None,
    conf_thresh=CONF_THRESH,
):
    """
    Shared helper for knee-dependent metrics.

    Behavior:
    - Start from start_idx and scan forward up to max_forward_frames (clamped to n-1)
      to find the first frame where the specified knee joint is present/in-bounds and
      above the confidence threshold (via is_joint_valid).
    - If such a frame is found, return it directly.
    - If none is found in the window, fall back to linear interpolation of the knee
      position from the nearest valid neighbors before and after start_idx.
    - If interpolation is not possible (no valid neighbors on one or both sides),
      return used_idx=None.

    Args:
        pts: Dict[str, np.ndarray] of landmark arrays.
        knee_key: Name of the knee joint to validate (e.g., "LEFT_KNEE").
        start_idx: Intended reference frame (e.g., FFP/Release).
        max_forward_frames: Maximum number of frames to scan forward.
        n: Optional total frame count (derived from pts if None).
        frame_w: Optional frame width for in-bounds checks.
        frame_h: Optional frame height for in-bounds checks.
        conf_thresh: Confidence threshold for is_joint_valid.

    Returns:
        (used_idx, knee_xy, interpolated, per_joint_valid, per_joint_conf, valid_count)
        - used_idx: int or None
        - knee_xy: np.ndarray shape (2,) with (x, y) of knee at used_idx, or None
        - interpolated: bool, True if knee_xy came from interpolation
        - per_joint_valid: dict[knee_key] -> bool at used_idx
        - per_joint_conf: dict[knee_key] -> confidence (or None) at used_idx
        - valid_count: number of valid joints (0 or 1 here)
    """
    if pts is None or knee_key is None:
        return None, None, False, {}, {}, 0

    if n is None:
        if knee_key in pts:
            n = len(pts.get(knee_key, []))
        else:
            lengths = [len(arr) for arr in pts.values() if arr is not None]
            n = max(lengths) if lengths else 0

    if n <= 0:
        return None, None, False, {}, {}, 0

    try:
        i_start = int(start_idx)
    except Exception:
        return None, None, False, {}, {}, 0

    i_start = max(0, min(i_start, n - 1))
    try:
        window_end = i_start + int(max_forward_frames)
    except Exception:
        window_end = i_start
    i_end = max(0, min(window_end, n - 1))

    # 1) Forward scan within [i_start, i_end] for first valid knee frame
    for i in range(i_start, i_end + 1):
        valid, conf = is_joint_valid(
            pts, i, knee_key, n=n, frame_w=frame_w, frame_h=frame_h, conf_thresh=conf_thresh
        )
        if valid:
            landmark = pts[knee_key][i]
            knee_xy = np.array([float(landmark[0]), float(landmark[1])], dtype=np.float32)
            per_joint_valid = {knee_key: True}
            per_joint_conf = {knee_key: conf}
            return i, knee_xy, False, per_joint_valid, per_joint_conf, 1

    # 2) No valid frame found in forward window; attempt interpolation from nearest neighbors
    if knee_key not in pts:
        return None, None, False, {}, {}, 0

    arr = pts[knee_key]
    if arr is None or len(arr) == 0:
        return None, None, False, {}, {}, 0

    # Find previous valid frame (<= i_start-1)
    prev_idx = None
    prev_xy = None
    prev_conf = None
    for i in range(i_start - 1, -1, -1):
        valid, conf = is_joint_valid(
            pts, i, knee_key, n=n, frame_w=frame_w, frame_h=frame_h, conf_thresh=conf_thresh
        )
        if valid:
            landmark = arr[i]
            prev_xy = np.array([float(landmark[0]), float(landmark[1])], dtype=np.float32)
            prev_idx = i
            prev_conf = conf
            break

    # Find next valid frame (>= i_end+1)
    next_idx = None
    next_xy = None
    next_conf = None
    for i in range(i_end + 1, n):
        valid, conf = is_joint_valid(
            pts, i, knee_key, n=n, frame_w=frame_w, frame_h=frame_h, conf_thresh=conf_thresh
        )
        if valid:
            landmark = arr[i]
            next_xy = np.array([float(landmark[0]), float(landmark[1])], dtype=np.float32)
            next_idx = i
            next_conf = conf
            break

    if prev_idx is None or next_idx is None or next_idx == prev_idx:
        # Cannot interpolate without neighbors on both sides
        return None, None, False, {}, {}, 0

    # Interpolate knee position at the clamped intended index (i_start)
    t = (i_start - prev_idx) / float(next_idx - prev_idx)
    t = max(0.0, min(1.0, t))
    knee_xy = (1.0 - t) * prev_xy + t * next_xy

    if not np.all(np.isfinite(knee_xy)):
        return None, None, False, {}, {}, 0

    avg_conf = None
    if prev_conf is not None or next_conf is not None:
        confs = [c for c in (prev_conf, next_conf) if c is not None]
        if confs:
            avg_conf = float(np.mean(confs))

    per_joint_valid = {knee_key: True}
    per_joint_conf = {knee_key: avg_conf}
    return i_start, knee_xy.astype(np.float32), True, per_joint_valid, per_joint_conf, 1


def select_lead_knee_anchor(
    event_idx: int,
    lead_side: str,
    pts_by_frame: dict,
    conf_by_frame=None,
    n: int = None,
    window: tuple = None,
):
    """
    Shared helper to pick a reliable lead knee anchor within a time window.

    Strategy (in priority order):
    1. **Measured knee**: pick the best measured knee frame in the window
       (valid joint, highest confidence; tie-break by proximity to event_idx).
    2. **Interpolated knee**: if no measured knee, linearly interpolate knee_xy
       from >=2 valid knee samples in the window (in time) to event_idx.
    3. **Approximate knee**: if still unavailable, approximate knee_xy as the
       midpoint of (lead_hip, lead_ankle) in the best frame where both exist.

    Returns:
        (display_idx, knee_xy, knee_source, debug)
        - display_idx: int or None - frame to display in UI overlays
        - knee_xy: np.ndarray([x, y]) or None - knee position in original coord space
        - knee_source: "measured" | "interp" | "approx" | "none"
        - debug: dict with selection details
    """
    debug = {
        "event_idx": int(event_idx) if event_idx is not None else None,
        "lead_side": lead_side,
        "knee_source": "none",
    }

    if pts_by_frame is None or lead_side is None:
        return None, None, "none", debug

    lead = (lead_side or "LEFT").upper()
    knee_key = f"{lead}_KNEE"
    hip_key = f"{lead}_HIP"
    ankle_key = f"{lead}_ANKLE"

    debug["knee_key"] = knee_key
    debug["hip_key"] = hip_key
    debug["ankle_key"] = ankle_key

    # Derive n from pts if not provided
    if n is None:
        if knee_key in pts_by_frame:
            n = len(pts_by_frame.get(knee_key, []))
        else:
            lengths = [len(arr) for arr in pts_by_frame.values() if arr is not None]
            n = max(lengths) if lengths else 0
    if n is None or n <= 0:
        debug["fail_reason"] = "no_frames"
        return None, None, "none", debug

    # Resolve window [w_start, w_end]
    if window is None:
        w_start = max(0, int(event_idx))
        w_end = w_start
    else:
        try:
            w_start, w_end = window
        except Exception:
            w_start = w_end = int(event_idx)
    try:
        w_start = max(0, min(int(w_start), n - 1))
        w_end = max(0, min(int(w_end), n - 1))
    except Exception:
        w_start = max(0, min(int(event_idx), n - 1))
        w_end = w_start

    if w_start > w_end:
        w_start, w_end = w_end, w_start

    debug["window"] = [int(w_start), int(w_end)]

    # ---------- 1) MEASURED KNEE (preferred) ----------
    best_idx = None
    best_conf = -1.0
    best_dist = None
    best_xy = None

    if knee_key in pts_by_frame:
        for i in range(w_start, w_end + 1):
            valid, conf = is_joint_valid(pts_by_frame, i, knee_key, n=n, conf_thresh=CONF_THRESH)
            if not valid:
                continue
            arr = pts_by_frame[knee_key]
            if i >= len(arr):
                continue
            lm = arr[i]
            if lm is None or len(lm) < 2:
                continue
            x, y = float(lm[0]), float(lm[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            this_conf = conf if conf is not None and np.isfinite(conf) else 0.0
            this_dist = abs(int(i) - int(event_idx))
            if (this_conf > best_conf) or (this_conf == best_conf and (best_dist is None or this_dist < best_dist)):
                best_conf = this_conf
                best_dist = this_dist
                best_idx = i
                best_xy = np.array([x, y], dtype=np.float32)

    if best_idx is not None and best_xy is not None:
        debug["knee_source"] = "measured"
        debug["measured_idx"] = int(best_idx)
        debug["measured_conf"] = float(best_conf)
        debug["knee_xy"] = [float(best_xy[0]), float(best_xy[1])]
        debug["display_idx"] = int(best_idx)
        return int(best_idx), best_xy, "measured", debug

    # ---------- 2) INTERPOLATED KNEE (window samples) ----------
    samples = []
    if knee_key in pts_by_frame:
        arr = pts_by_frame[knee_key]
        for i in range(w_start, w_end + 1):
            if i < 0 or i >= len(arr):
                continue
            lm = arr[i]
            if lm is None or len(lm) < 2:
                continue
            x, y = float(lm[0]), float(lm[1])
            if np.any(np.array([x, y]) == 0) or not (np.isfinite(x) and np.isfinite(y)):
                continue
            samples.append((int(i), np.array([x, y], dtype=np.float32)))

    debug["knee_samples_in_window"] = len(samples)

    if len(samples) >= 2:
        # Sort by time index
        samples.sort(key=lambda t: t[0])
        # Find neighbors around event_idx
        prev_s = None
        next_s = None
        for idx, xy in samples:
            if idx <= event_idx:
                prev_s = (idx, xy)
            if idx >= event_idx and next_s is None:
                next_s = (idx, xy)
        if prev_s is None:
            prev_s = samples[0]
        if next_s is None:
            next_s = samples[-1]

        i1, xy1 = prev_s
        i2, xy2 = next_s
        if i1 != i2:
            t = (float(event_idx) - float(i1)) / float(i2 - i1)
            t = max(0.0, min(1.0, t))
            knee_xy = (1.0 - t) * xy1 + t * xy2
            if np.all(np.isfinite(knee_xy)):
                interp_idx = int(max(w_start, min(w_end, int(event_idx))))
                debug["knee_source"] = "interp"
                debug["interp_indices"] = [int(i1), int(i2)]
                debug["display_idx"] = int(interp_idx)
                debug["knee_xy"] = [float(knee_xy[0]), float(knee_xy[1])]
                return int(interp_idx), knee_xy.astype(np.float32), "interp", debug

    # ---------- 3) APPROXIMATE KNEE (midpoint hip/ankle) ----------
    best_idx = None
    best_valid = -1
    best_xy = None

    for i in range(w_start, w_end + 1):
        hip_valid, _ = is_joint_valid(pts_by_frame, i, hip_key, n=n, conf_thresh=CONF_THRESH)
        ankle_valid, _ = is_joint_valid(pts_by_frame, i, ankle_key, n=n, conf_thresh=CONF_THRESH)
        if not (hip_valid and ankle_valid):
            continue
        hip_arr = pts_by_frame[hip_key]
        ankle_arr = pts_by_frame[ankle_key]
        if i >= len(hip_arr) or i >= len(ankle_arr):
            continue
        hip_lm = hip_arr[i]
        ankle_lm = ankle_arr[i]
        if hip_lm is None or ankle_lm is None or len(hip_lm) < 2 or len(ankle_lm) < 2:
            continue
        hx, hy = float(hip_lm[0]), float(hip_lm[1])
        ax, ay = float(ankle_lm[0]), float(ankle_lm[1])
        if not (np.isfinite(hx) and np.isfinite(hy) and np.isfinite(ax) and np.isfinite(ay)):
            continue
        knee_xy = np.array([(hx + ax) / 2.0, (hy + ay) / 2.0], dtype=np.float32)
        if not np.all(np.isfinite(knee_xy)):
            continue
        this_valid = 2  # hip + ankle present
        if this_valid > best_valid:
            best_valid = this_valid
            best_idx = i
            best_xy = knee_xy

    if best_idx is not None and best_xy is not None:
        debug["knee_source"] = "approx"
        debug["approx_idx"] = int(best_idx)
        debug["display_idx"] = int(best_idx)
        debug["knee_xy"] = [float(best_xy[0]), float(best_xy[1])]
        return int(best_idx), best_xy, "approx", debug

    # No usable anchor
    debug["fail_reason"] = "no_knee_anchor_found"
    debug["display_idx"] = None
    return None, None, "none", debug


def find_best_frame_in_window(window_start, window_end, pts, required_joints, min_conf=CONF_THRESH,
                              n=None, frame_w=None, frame_h=None, forward_from=None):
    """
    Find best frame in window using joint validity.

    Rules:
    - Only consider frames where ALL required joints are valid
    - Pick the frame with the highest average confidence (when present); if none, tie-break by lowest index
    - Return None if no frame has all required joints valid

    Returns:
        (best_idx: int or None, best_q: float, best_valid: dict, best_conf: dict)
    """
    if window_start < 0:
        window_start = 0

    if n is not None:
        window_start = max(0, min(window_start, n - 1))
        window_end = max(0, min(window_end, n - 1))

    # Forward-only search: optionally clamp start to forward_from (selected frame)
    if forward_from is not None:
        try:
            f = int(forward_from)
            window_start = max(window_start, f)
        except Exception:
            pass

    if window_start > window_end:
        return None, 0.0, {}, {}

    best_idx = None
    best_avg_conf = -1.0
    best_valid = {}
    best_conf = {}

    for i in range(window_start, window_end + 1):
        all_valid, avg_conf, per_joint_valid, per_joint_conf = compute_joint_quality(
            i, pts, required_joints, n=n, frame_w=frame_w, frame_h=frame_h, conf_thresh=min_conf
        )
        if not all_valid:
            continue
        # Tie-break by avg_conf, then lower index
        if avg_conf > best_avg_conf or (avg_conf == best_avg_conf and best_idx is None or i < best_idx):
            best_avg_conf = avg_conf
            best_idx = i
            best_valid = per_joint_valid
            best_conf = per_joint_conf

    if best_idx is None:
        return None, 0.0, best_valid, best_conf

    return best_idx, best_avg_conf, best_valid, best_conf


def is_pose_usable(frame_idx, pts, required_joints, pose_quality=None, min_conf=CONF_THRESH, n=None):
    """
    Unified pose quality check for metrics.
    
    Checks if a frame is usable for metric computation by verifying:
    1. Frame index is valid (within bounds)
    2. Pose quality is above threshold (if pose_quality is provided)
    3. Required joints exist in pts dictionary
    4. Required joints are valid (non-zero, finite) at that frame
    
    This ensures that if one metric passes at a frame, other metrics can also
    evaluate at the same frame (as long as their required joints are present).
    Metrics are not blocked due to isolated low-confidence joints unrelated to them.
    
    Args:
        frame_idx: Frame index to check
        pts: Dict of landmark arrays (e.g., {"LEFT_HIP": [...], "RIGHT_HIP": [...]})
        required_joints: List of joint keys required for this metric (e.g., ["LEFT_HIP", "RIGHT_HIP"])
        pose_quality: Optional list/array of pose quality scores per frame
        min_conf: Minimum confidence threshold (default CONF_THRESH = 0.35)
        n: Optional total number of frames (for bounds checking)
    
    Returns:
        (is_usable: bool, reason: str)
        - is_usable: True if frame is usable for this metric
        - reason: "ok" if usable, or reason why not usable
    """
    # Check frame index validity
    if frame_idx < 0:
        return False, "invalid_frame_idx"
    
    if n is not None and frame_idx >= n:
        return False, "frame_idx_out_of_bounds"
    
    # Check pose quality if provided (using safe accessor)
    if pose_quality is not None:
        q = get_pose_quality(frame_idx, pose_quality, n=n)
        if q is not None and q < min_conf:
            return False, "low_pose_quality"
    
    # Check required joints exist in pts
    missing_joints = [joint for joint in required_joints if joint not in pts]
    if missing_joints:
        return False, f"missing_joints: {','.join(missing_joints)}"
    
    # Check required joints are valid at this frame
    for joint in required_joints:
        joint_array = pts[joint]
        
        # Check array bounds
        if frame_idx >= len(joint_array):
            return False, f"joint_{joint}_index_out_of_bounds"
        
        joint_point = joint_array[frame_idx]
        
        # Check joint is non-zero and finite
        if np.any(joint_point == 0):
            return False, f"joint_{joint}_is_zero"
        
        if not np.isfinite(joint_point[0]) or not np.isfinite(joint_point[1]):
            return False, f"joint_{joint}_not_finite"
    
    return True, "ok"


def pack_metric(name, value, units="", score=None, status=None, reason="ok", debug=None):
    """
    Pack a metric into a standardized schema.
    
    Args:
        name: Metric name (e.g., "stride_length")
        value: Raw metric value (float or None)
        units: Unit string (e.g., "BL", "deg", "%")
        score: Score 0-100 (float or None)
        status: Status string ("good"/"ok"/"bad" or None)
        reason: Reason string ("ok" or why None/fallback used)
        debug: Debug dict (optional)
    
    Returns:
        dict with standardized metric structure
    """
    return {
        "name": name,
        "value": None if value is None else float(value),
        "units": units,
        "score": None if score is None else float(score),
        "status": status,  # e.g., "good"/"ok"/"bad" or None
        "reason": reason,  # "ok" or why None / fallback used
        "debug": debug or {},  # dict
    }


def wrap180(angle_deg):
    """Wrap angle to [-180, 180] degrees."""
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


def compute_angle_from_vertical(dx, dy):
    """
    Compute angle from vertical (0 deg = vertical, positive = forward lean).
    
    Args:
        dx: Horizontal component (x difference)
        dy: Vertical component (y difference, positive downward in image coords)
    
    Returns:
        Angle in degrees
    """
    # In image coordinates, y increases downward
    # atan2(dx, -dy) gives angle from vertical (negative dy because y increases down)
    angle_rad = math.atan2(dx, -dy)
    return math.degrees(angle_rad)


def compute_line_angle(p1, p2):
    """
    Compute angle of line from p1 to p2 in degrees.
    
    Args:
        p1: (x, y) start point
        p2: (x, y) end point
    
    Returns:
        Angle in degrees (0-360, where 0 deg = right, 90 deg = down)
    """
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def compute_forward_dir(pts, ffp_idx, throwing_side, n, pose_quality, min_q=0.35):
    """
    Compute forward direction based on actual pre-FFP motion.
    
    Uses sign of smoothed pelvis midpoint x displacement from early frames to FFP.
    Falls back to lead ankle if pelvis is not available.
    
    Args:
        pts: Dict of landmark arrays
        ffp_idx: Foot Strike (FFP) frame index
        throwing_side: "L" or "R" (for fallback to lead ankle)
        n: Total number of frames
        pose_quality: List/array of pose quality scores per frame
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        (forward_dir: int, reason: str)
        forward_dir: -1 or +1 (sign of forward direction)
        reason: "pelvis_motion" or "lead_ankle_fallback" or "default_fallback"
    """
    # Try pelvis midpoint motion first (early frames to FFP)
    if ffp_idx is not None and ffp_idx >= 0 and ffp_idx < n:
        if "LEFT_HIP" in pts and "RIGHT_HIP" in pts:
            # Collect pelvis midpoint x-coordinates from early frames to FFP
            pelvis_x_list = []
            valid_indices = []
            
            # Use first 20% of frames or up to FFP, whichever is smaller
            window_start = max(0, min(int(0.2 * n), ffp_idx - 10))
            window_end = min(n - 1, ffp_idx)
            
            for i in range(window_start, window_end + 1):
                if i >= n:
                    break
                # Use safe accessor for pose quality (i is already in global frame space)
                q = get_pose_quality(i, pose_quality, n=n)
                if q is not None and q < min_q:
                    continue
                
                lh = pts["LEFT_HIP"][i]
                rh = pts["RIGHT_HIP"][i]
                
                if np.any(lh != 0) and np.any(rh != 0):
                    pelvis_x = (lh[0] + rh[0]) / 2.0
                    if np.isfinite(pelvis_x):
                        pelvis_x_list.append(float(pelvis_x))
                        valid_indices.append(i)
            
            if len(pelvis_x_list) >= 3:
                # Smooth with EMA (Exponential Moving Average)
                alpha = 0.3
                smoothed_pelvis_x = [pelvis_x_list[0]]
                for i in range(1, len(pelvis_x_list)):
                    ema_val = alpha * pelvis_x_list[i] + (1 - alpha) * smoothed_pelvis_x[i-1]
                    smoothed_pelvis_x.append(ema_val)
                
                # Compute displacement: x at FFP - x at start
                x_at_start = smoothed_pelvis_x[0]
                x_at_ffp = smoothed_pelvis_x[-1]
                dx = x_at_ffp - x_at_start
                
                # Forward direction is sign of displacement
                # If pelvis moves left (negative x), forward_dir = -1
                # If pelvis moves right (positive x), forward_dir = +1
                if abs(dx) > 1e-6:  # Significant displacement
                    forward_dir = -1 if dx < 0 else 1
                    return forward_dir, "pelvis_motion"
    
    # Fallback to lead ankle motion
    if throwing_side and ffp_idx is not None and ffp_idx >= 0 and ffp_idx < n:
        # Determine lead ankle based on throwing side
        if throwing_side.upper() == "R":
            lead_ankle_key = "LEFT_ANKLE"
        else:  # L
            lead_ankle_key = "RIGHT_ANKLE"
        
        if lead_ankle_key in pts:
            # Use a window before FFP (e.g., last 10 frames before FFP)
            window_start = max(0, ffp_idx - 10)
            window_end = ffp_idx
            
            ankle_x_list = []
            for i in range(window_start, window_end + 1):
                if i >= n:
                    break
                # Use safe accessor for pose quality (i is already in global frame space)
                q = get_pose_quality(i, pose_quality, n=n)
                if q is not None and q < min_q:
                    continue
                
                ankle = pts[lead_ankle_key][i]
                if np.any(ankle != 0) and np.isfinite(ankle[0]):
                    ankle_x_list.append(float(ankle[0]))
            
            if len(ankle_x_list) >= 3:
                # Smooth with EMA
                alpha = 0.3
                smoothed_ankle_x = [ankle_x_list[0]]
                for i in range(1, len(ankle_x_list)):
                    ema_val = alpha * ankle_x_list[i] + (1 - alpha) * smoothed_ankle_x[i-1]
                    smoothed_ankle_x.append(ema_val)
                
                # Compute displacement
                x_start = smoothed_ankle_x[0]
                x_end = smoothed_ankle_x[-1]
                dx = x_end - x_start
                
                if abs(dx) > 1e-6:
                    forward_dir = -1 if dx < 0 else 1
                    return forward_dir, "lead_ankle_fallback"
    
    # Default fallback (should rarely happen)
    # Use throwing side as last resort
    if throwing_side:
        forward_dir = -1 if throwing_side.upper() == "R" else 1
        return forward_dir, "default_fallback"
    
    # Ultimate fallback
    return 1, "default_fallback"


def get_knee_xy(frame_idx, lead_knee_key, pts_by_frame, knee_model_xy_by_frame=None, knee_model_source_by_frame=None):
    """
    Get lead knee XY coordinates and source for a specific frame.
    
    NOTE: Knee model is decoupled - this function now only uses raw pose data.
    The knee_model_xy_by_frame and knee_model_source_by_frame parameters are
    kept for API compatibility but are ignored.
    
    Args:
        frame_idx: Frame index
        lead_knee_key: Lead knee key ("LEFT_KNEE" or "RIGHT_KNEE")
        pts_by_frame: Dict of landmark arrays (required)
        knee_model_xy_by_frame: Ignored (kept for API compatibility)
        knee_model_source_by_frame: Ignored (kept for API compatibility)
    
    Returns:
        (knee_xy, source) tuple where:
        - knee_xy: (x, y) tuple or None if not available
        - source: "pose" or None
    """
    if frame_idx is None or frame_idx < 0:
        return None, None
    
    # Use raw pose data only
    if pts_by_frame is not None and lead_knee_key is not None:
        if lead_knee_key in pts_by_frame:
            arr = pts_by_frame[lead_knee_key]
            if frame_idx < len(arr):
                landmark = arr[frame_idx]
                if landmark is not None and len(landmark) >= 2:
                    x, y = landmark[0], landmark[1]
                    if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                        # Check if joint is valid using is_joint_valid
                        valid, _ = is_joint_valid(pts_by_frame, frame_idx, lead_knee_key)
                        if valid:
                            return (float(x), float(y)), "pose"
    
    return None, None


def get_joint_xy(frame_idx, joint_key, pts=None, knee_model_xy_by_frame=None, knee_model_source_by_frame=None, lead_knee_key=None):
    """
    Get joint XY coordinates from raw pose landmarks.
    
    NOTE: Knee model is decoupled - this function now only uses raw pose data.
    The knee_model_xy_by_frame, knee_model_source_by_frame, and lead_knee_key
    parameters are kept for API compatibility but are ignored.
    
    Args:
        frame_idx: Frame index
        joint_key: Joint key (e.g., "LEFT_KNEE", "RIGHT_HIP")
        pts: Dict of landmark arrays (required)
        knee_model_xy_by_frame: Ignored (kept for API compatibility)
        knee_model_source_by_frame: Ignored (kept for API compatibility)
        lead_knee_key: Ignored (kept for API compatibility)
    
    Returns:
        (x, y) tuple or None if not available
    """
    # Use raw pose data for all joints (including knees)
    if pts is None or joint_key not in pts:
        return None
    
    if frame_idx is None or frame_idx < 0:
        return None
    
    arr = pts[joint_key]
    if frame_idx >= len(arr):
        return None
    
    landmark = arr[frame_idx]
    if landmark is None or len(landmark) < 2:
        return None
    
    x, y = landmark[0], landmark[1]
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return None
    
    return (float(x), float(y))

