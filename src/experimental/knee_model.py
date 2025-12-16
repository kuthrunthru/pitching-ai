"""
Lead knee Kalman tracking for robust knee position estimation.

EXPERIMENTAL MODULE - NOT USED IN ACTIVE PIPELINE

This module is preserved for future use but is fully decoupled from the current
analysis pipeline. It provides a single source of truth for all knee-dependent metrics by:
1. Detecting knee_start_idx (first sustained forward motion)
2. Detecting hand_low_idx_post_release (lowest wrist after release)
3. Collecting observations and filtering outliers
4. Fitting a 2D constant-velocity Kalman filter
5. Providing smooth knee position for any frame in the tracking window

To use this module in the future, import from src.experimental.knee_model.
"""

import math
import numpy as np
from src.metrics.utils import (
    is_joint_valid,
    compute_forward_dir,
    CONF_THRESH,
)

# Version constant for cache invalidation
# Increment this when knee model logic changes to force cache refresh
KNEE_MODEL_VERSION = "v1.0.0"


class KalmanFilter2D:
    """
    2D constant-velocity Kalman filter for tracking knee position.
    
    State: [x, y, vx, vy]
    Observation: [x, y]
    """
    
    def __init__(self, process_noise=0.1, measurement_noise=5.0):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise covariance (motion uncertainty)
            measurement_noise: Measurement noise covariance (observation uncertainty)
        """
        # State: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        self.covariance = np.eye(4, dtype=np.float32) * 100.0  # Initial uncertainty
        
        # Process noise (Q)
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.Q[2:, 2:] *= 0.1  # Lower noise on velocity
        
        # Measurement noise (R)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1],  # vy = vy
        ], dtype=np.float32)
        
        # Observation matrix (we observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],  # observe x
            [0, 1, 0, 0],  # observe y
        ], dtype=np.float32)
    
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
    
    def update(self, observation):
        """
        Update with observation.
        
        Args:
            observation: [x, y] array
        """
        if observation is None or len(observation) < 2:
            return
        
        z = np.array(observation[:2], dtype=np.float32)
        
        # Innovation
        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        
        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
    
    def get_position(self):
        """Get current position estimate."""
        return self.state[:2].copy()
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return self.state[2:].copy()


def detect_knee_start_idx(pts, lead_knee_key, n, fps=30.0, min_forward_frames=5, forward_threshold_px=10.0, forward_dir=None):
    """
    Detect knee_start_idx as the first frame where lead knee shows sustained forward motion.
    
    Args:
        pts: Dict of landmark arrays
        lead_knee_key: Key for lead knee (e.g., "LEFT_KNEE")
        n: Total number of frames
        fps: Frames per second
        min_forward_frames: Minimum consecutive frames with forward motion
        forward_threshold_px: Minimum forward displacement in pixels
        forward_dir: Optional forward direction (1.0 for +x, -1.0 for -x)
    
    Returns:
        knee_start_idx: int or None
    """
    if lead_knee_key not in pts:
        return None
    
    knee_arr = pts[lead_knee_key]
    if len(knee_arr) == 0:
        return None
    
    # Use provided forward_dir if available, else estimate from early motion
    if forward_dir is not None:
        forward_sign = float(forward_dir)
    else:
        # Compute forward direction from early motion (first 20% of frames)
        early_window = max(5, int(0.2 * n))
        early_x = []
        for i in range(min(early_window, n)):
            valid, _ = is_joint_valid(pts, i, lead_knee_key, n=n)
            if valid:
                early_x.append(float(knee_arr[i][0]))
        
        if len(early_x) < 3:
            return None
        
        # Simple heuristic: if most early positions are on left side, forward is +x
        # This is approximate; ideally we'd use compute_forward_dir but that needs more context
        forward_sign = 1.0  # Default: assume forward is +x
    
    # Scan for sustained forward motion
    forward_count = 0
    last_valid_x = None
    
    for i in range(n):
        valid, _ = is_joint_valid(pts, i, lead_knee_key, n=n)
        if not valid:
            forward_count = 0
            last_valid_x = None
            continue
        
        x = float(knee_arr[i][0])
        
        if last_valid_x is not None:
            dx = forward_sign * (x - last_valid_x)
            if dx >= forward_threshold_px:
                forward_count += 1
                if forward_count >= min_forward_frames:
                    # Found sustained forward motion
                    return max(0, i - min_forward_frames + 1)
            else:
                forward_count = 0
        
        last_valid_x = x
    
    return None


def detect_hand_low_idx_post_release(pts, throwing_side, release_idx, n, fps=30.0, max_window_frames=20):
    """
    Detect hand_low_idx_post_release by searching after release_idx for lowest throwing-hand wrist position.
    
    Args:
        pts: Dict of landmark arrays
        throwing_side: "R" or "L"
        release_idx: Release frame index
        n: Total number of frames
        fps: Frames per second
        max_window_frames: Maximum frames to search after release
    
    Returns:
        dict with:
            - "hand_low_idx": int or None
            - "min_wrist_y": float or None (y coordinate, higher = lower position)
            - "search_window": [start_idx, end_idx]
    """
    if throwing_side.upper() == "R":
        wrist_key = "RIGHT_WRIST"
    else:
        wrist_key = "LEFT_WRIST"
    
    if wrist_key not in pts:
        return {"hand_low_idx": None, "min_wrist_y": None, "search_window": None}
    
    wrist_arr = pts[wrist_key]
    if len(wrist_arr) == 0:
        return {"hand_low_idx": None, "min_wrist_y": None, "search_window": None}
    
    # Search window: [release_idx + 1, min(n-1, release_idx + max_window_frames)]
    search_start = release_idx + 1
    search_end = min(n - 1, release_idx + max_window_frames)
    
    if search_start >= n or search_start > search_end:
        return {"hand_low_idx": None, "min_wrist_y": None, "search_window": [search_start, search_end]}
    
    best_idx = None
    best_y = None
    
    for i in range(search_start, search_end + 1):
        valid, _ = is_joint_valid(pts, i, wrist_key, n=n)
        if not valid:
            continue
        
        y = float(wrist_arr[i][1])  # y increases downward, so higher y = lower position
        
        if best_y is None or y > best_y:
            best_y = y
            best_idx = i
    
    return {
        "hand_low_idx": best_idx,
        "min_wrist_y": best_y,
        "search_window": [search_start, search_end],
    }


def build_knee_kalman_track(
    pts,
    lead_knee_key,
    knee_start_idx,
    hand_low_idx_post_release,
    n,
    body_height_px=None,
    outlier_threshold_body_heights=0.15,
    min_q=CONF_THRESH,
):
    """
    Build Kalman-filtered knee track from observations.
    
    Args:
        pts: Dict of landmark arrays
        lead_knee_key: Key for lead knee (e.g., "LEFT_KNEE")
        knee_start_idx: Start frame for tracking
        hand_low_idx_post_release: End frame for tracking
        n: Total number of frames
        body_height_px: Body height in pixels (for outlier rejection)
        outlier_threshold_body_heights: Outlier threshold as fraction of body height
        min_q: Minimum confidence threshold
    
    Returns:
        track: dict with:
            - "positions": np.array shape (n, 2) with [x, y] for each frame
            - "raw_positions": np.array shape (n, 2) with raw observations (NaN where invalid)
            - "valid_mask": np.array shape (n,) bool indicating valid observations
            - "knee_start_idx": int
            - "hand_low_idx_post_release": int
    """
    if lead_knee_key not in pts:
        return None
    
    knee_arr = pts[lead_knee_key]
    if len(knee_arr) == 0:
        return None
    
    if knee_start_idx is None or hand_low_idx_post_release is None:
        return None
    
    if knee_start_idx >= n or hand_low_idx_post_release >= n:
        return None
    
    if knee_start_idx > hand_low_idx_post_release:
        return None
    
    # Initialize arrays
    track_end = min(n - 1, hand_low_idx_post_release)
    positions = np.full((n, 2), np.nan, dtype=np.float32)
    raw_positions = np.full((n, 2), np.nan, dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)
    
    # Collect raw observations
    observations = []
    observation_frames = []
    
    for i in range(knee_start_idx, track_end + 1):
        valid, conf = is_joint_valid(pts, i, lead_knee_key, n=n, conf_thresh=min_q)
        if valid:
            x = float(knee_arr[i][0])
            y = float(knee_arr[i][1])
            observations.append([x, y])
            observation_frames.append(i)
            raw_positions[i] = [x, y]
            valid_mask[i] = True
    
    if len(observations) < 2:
        return None
    
    observations = np.array(observations, dtype=np.float32)
    
    # Outlier rejection: reject jumps > threshold relative to body height
    if body_height_px is not None and body_height_px > 0:
        outlier_threshold_px = outlier_threshold_body_heights * body_height_px
        filtered_observations = []
        filtered_frames = []
        
        for idx, (obs, frame) in enumerate(zip(observations, observation_frames)):
            if idx == 0:
                filtered_observations.append(obs)
                filtered_frames.append(frame)
                continue
            
            # Check jump size from previous observation
            prev_obs = filtered_observations[-1]
            jump_size = np.linalg.norm(obs - prev_obs)
            
            if jump_size <= outlier_threshold_px:
                filtered_observations.append(obs)
                filtered_frames.append(frame)
            # else: reject outlier
        
        if len(filtered_observations) >= 2:
            observations = np.array(filtered_observations, dtype=np.float32)
            observation_frames = filtered_frames
    
    # Initialize Kalman filter with first observation
    process_noise = 0.1
    measurement_noise = 5.0
    kf = KalmanFilter2D(process_noise=process_noise, measurement_noise=measurement_noise)
    kf.state[:2] = observations[0]
    
    # Track measurement updates vs predictions
    update_mask = np.zeros(n, dtype=bool)
    prediction_mask = np.zeros(n, dtype=bool)
    
    # Run Kalman filter through all frames
    obs_idx = 0
    for i in range(knee_start_idx, track_end + 1):
        # Predict
        kf.predict()
        prediction_mask[i] = True
        
        # Update if we have an observation at this frame
        if obs_idx < len(observation_frames) and observation_frames[obs_idx] == i:
            kf.update(observations[obs_idx])
            update_mask[i] = True
            obs_idx += 1
        
        # Store position estimate
        positions[i] = kf.get_position()
    
    return {
        "positions": positions,
        "raw_positions": raw_positions,
        "valid_mask": valid_mask,
        "update_mask": update_mask,
        "prediction_mask": prediction_mask,
        "knee_start_idx": knee_start_idx,
        "hand_low_idx_post_release": hand_low_idx_post_release,
        "observation_frames": observation_frames,
        "kalman_params": {
            "process_noise": process_noise,
            "measurement_noise": measurement_noise,
        },
    }


def get_knee_track_for_metrics(clip, keyframes, pose_cache, min_q=CONF_THRESH):
    """
    Build complete knee Kalman track for use by all knee-dependent metrics.
    
    Args:
        clip: Dict with video metadata (n, fps)
        keyframes: Dict with ffp_idx, release_idx, throwing_side
        pose_cache: Cached pose data with "pts" dict
        min_q: Minimum confidence threshold
    
    Returns:
        track: dict from build_knee_kalman_track or None
    """
    if pose_cache is None or pose_cache.get("pts") is None:
        return None
    
    pts = pose_cache["pts"]
    n = clip.get("n")
    fps = clip.get("fps", 30.0)
    throwing_side = keyframes.get("throwing_side", "R")
    
    if n is None or n <= 0:
        return None
    
    # Determine lead knee key
    if throwing_side.upper() == "R":
        lead_knee_key = "LEFT_KNEE"
    else:
        lead_knee_key = "RIGHT_KNEE"
    
    # Detect knee_start_idx (use forward_dir if available)
    forward_dir = keyframes.get("forward_dir")
    knee_start_idx = detect_knee_start_idx(pts, lead_knee_key, n, fps=fps, forward_dir=forward_dir)
    knee_start_detection_info = {
        "forward_dir": forward_dir,
        "forward_threshold_px": 10.0,
        "min_forward_frames": 5,
    }
    if knee_start_idx is None:
        # Fallback: use frame 0
        knee_start_idx = 0
        knee_start_detection_info["detection_method"] = "fallback_to_frame_0"
    else:
        knee_start_detection_info["detection_method"] = "sustained_forward_motion"
    
    # Detect hand_low_idx_post_release
    release_idx = keyframes.get("release_idx")
    if release_idx is None:
        return None
    
    hand_low_result = detect_hand_low_idx_post_release(
        pts, throwing_side, release_idx, n, fps=fps
    )
    hand_low_idx = hand_low_result.get("hand_low_idx")
    if hand_low_idx is None:
        # Fallback: use release_idx + small window
        hand_low_idx = min(n - 1, release_idx + 10)
    
    # Compute body height for outlier rejection
    body_height_px = None
    if "LEFT_SHOULDER" in pts and "RIGHT_SHOULDER" in pts and "LEFT_ANKLE" in pts and "RIGHT_ANKLE" in pts:
        # Use mid-drive frame (between mkl and ffp) if available
        ffp_idx = keyframes.get("ffp_idx")
        if ffp_idx is not None:
            # Use FFP frame for body height estimation (MKL removed)
            mid_frame = int(ffp_idx)
            if 0 <= mid_frame < n:
                ls = pts["LEFT_SHOULDER"][mid_frame]
                rs = pts["RIGHT_SHOULDER"][mid_frame]
                la = pts["LEFT_ANKLE"][mid_frame]
                ra = pts["RIGHT_ANKLE"][mid_frame]
                if (np.any(ls != 0) and np.any(rs != 0) and np.any(la != 0) and np.any(ra != 0)):
                    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
                    ankle_mid = ((la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2)
                    dx = shoulder_mid[0] - ankle_mid[0]
                    dy = shoulder_mid[1] - ankle_mid[1]
                    body_height_px = math.sqrt(dx * dx + dy * dy)
                    if not np.isfinite(body_height_px) or body_height_px <= 0:
                        body_height_px = None
    
    # Build track
    track = build_knee_kalman_track(
        pts=pts,
        lead_knee_key=lead_knee_key,
        knee_start_idx=knee_start_idx,
        hand_low_idx_post_release=hand_low_idx,
        n=n,
        body_height_px=body_height_px,
        min_q=min_q,
    )
    
    if track is not None:
        track["lead_knee_key"] = lead_knee_key
        track["throwing_side"] = throwing_side
        # Add hand_low detection details
        track["hand_low_detection"] = hand_low_result
        # Add knee_start detection info
        track["knee_start_detection_info"] = knee_start_detection_info
    
    return track


def get_knee_position_from_track(track, frame_idx):
    """
    Get knee position from Kalman track for a specific frame.
    
    Args:
        track: Track dict from build_knee_kalman_track
        frame_idx: Frame index
    
    Returns:
        knee_xy: np.array([x, y]) or None
    """
    if track is None:
        return None
    
    positions = track.get("positions")
    if positions is None:
        return None
    
    if frame_idx < 0 or frame_idx >= len(positions):
        return None
    
    pos = positions[frame_idx]
    if np.any(np.isnan(pos)):
        return None
    
    return pos.copy()


def get_knee_source_at_frame(track, frame_idx):
    """
    Determine if knee position at frame_idx is from measurement or prediction.
    
    Args:
        track: Track dict from build_knee_kalman_track
        frame_idx: Frame index
    
    Returns:
        "measured" if update occurred, "predicted" if only prediction, or None
    """
    if track is None:
        return None
    
    update_mask = track.get("update_mask")
    if update_mask is None:
        return None
    
    if frame_idx < 0 or frame_idx >= len(update_mask):
        return None
    
    if update_mask[frame_idx]:
        return "measured"
    else:
        return "predicted"


def build_knee_model_summary(track, keyframes, knee_model_xy_by_frame, knee_model_source_by_frame, knee_start_detection_info=None, hand_low_detection=None):
    """
    Build summary debug block for knee model (printed once per run).
    
    Includes:
    - knee_start_idx and how it was detected
    - hand_low_idx_post_release computed explicitly as minimum throwing-wrist Y in post-release-only window
    - For FFP/Release/HandLow: knee (x,y) and whether it was measured or predicted
    
    Args:
        track: Track dict from build_knee_kalman_track
        keyframes: Dict with ffp_idx, release_idx
        knee_model_xy_by_frame: Array of knee model XY positions [n, 2]
        knee_model_source_by_frame: Array of knee model sources [n]
        knee_start_detection_info: Optional dict with detection details
        hand_low_detection: Optional dict with hand_low detection details
    
    Returns:
        dict with summary information
    """
    if track is None:
        return None
    
    knee_start_idx = track.get("knee_start_idx")
    hand_low_idx_post_release = track.get("hand_low_idx_post_release")
    release_idx = keyframes.get("release_idx")
    
    # Get hand_low detection details
    if hand_low_detection is None:
        hand_low_detection = track.get("hand_low_detection", {})
    
    # Get knee_start detection info
    if knee_start_detection_info is None:
        knee_start_detection_info = track.get("knee_start_detection_info", {})
    
    knee_start_info = {
        "knee_start_idx": int(knee_start_idx) if knee_start_idx is not None else None,
        "detection_method": knee_start_detection_info.get("detection_method", "sustained_forward_motion"),
        "forward_dir_used": knee_start_detection_info.get("forward_dir") if knee_start_detection_info else None,
        "forward_threshold_px": knee_start_detection_info.get("forward_threshold_px", 10.0) if knee_start_detection_info else 10.0,
        "min_forward_frames": knee_start_detection_info.get("min_forward_frames", 5) if knee_start_detection_info else 5,
    }
    
    # Hand low detection summary
    hand_low_summary = {
        "hand_low_idx_post_release": int(hand_low_idx_post_release) if hand_low_idx_post_release is not None else None,
        "search_window": hand_low_detection.get("search_window"),
        "min_wrist_y": float(hand_low_detection.get("min_wrist_y")) if hand_low_detection.get("min_wrist_y") is not None else None,
        "min_wrist_y_frame": int(hand_low_detection.get("hand_low_idx")) if hand_low_detection.get("hand_low_idx") is not None else None,
    }
    
    # Get event frame knee positions and sources from knee_model_xy_by_frame
    event_knee_info = {}
    for event_name, event_idx in [
        ("FFP", keyframes.get("ffp_idx")),
        ("Release", release_idx),
        ("HandLow", hand_low_idx_post_release),
    ]:
        if event_idx is not None and knee_model_xy_by_frame is not None:
            if 0 <= event_idx < len(knee_model_xy_by_frame):
                knee_xy = knee_model_xy_by_frame[event_idx]
                if knee_xy is not None and len(knee_xy) >= 2:
                    knee_source = None
                    if knee_model_source_by_frame is not None and 0 <= event_idx < len(knee_model_source_by_frame):
                        knee_source = knee_model_source_by_frame[event_idx]
                    
                    event_knee_info[event_name] = {
                        "frame_idx": int(event_idx),
                        "knee_xy": [round(float(knee_xy[0]), 1), round(float(knee_xy[1]), 1)],
                        "knee_source": knee_source or "unknown",
                    }
                else:
                    event_knee_info[event_name] = {
                        "frame_idx": int(event_idx),
                        "knee_xy": None,
                        "knee_source": None,
                    }
            else:
                event_knee_info[event_name] = {
                    "frame_idx": int(event_idx),
                    "knee_xy": None,
                    "knee_source": None,
                }
    
    summary = {
        "knee_start_detection": knee_start_info,
        "hand_low_detection": hand_low_summary,
        "event_knee_positions": event_knee_info,
    }
    
    return summary


def build_knee_model_debug(track, keyframes):
    """
    Build comprehensive debug object for the knee Kalman tracking model.
    
    Args:
        track: Track dict from build_knee_kalman_track
        keyframes: Dict with ffp_idx, release_idx
    
    Returns:
        dict with debug information:
            - knee_start_idx: int
            - release_idx: int
            - hand_low_idx_post_release: int
            - hand_low_search_window: [start, end]
            - min_wrist_y_frame: int
            - min_wrist_y: float
            - kalman_params: {process_noise, measurement_noise}
            - update_count: int
            - prediction_count: int
            - event_knee_positions: {FFP, Release, HandLow: {frame_idx, knee_xy, knee_source}}
    """
    if track is None:
        return None
    
    knee_start_idx = track.get("knee_start_idx")
    hand_low_idx_post_release = track.get("hand_low_idx_post_release")
    hand_low_detection = track.get("hand_low_detection", {})
    kalman_params = track.get("kalman_params", {})
    update_mask = track.get("update_mask")
    prediction_mask = track.get("prediction_mask")
    
    release_idx = keyframes.get("release_idx")
    
    # Count measurement updates vs predictions
    update_count = 0
    prediction_count = 0
    if update_mask is not None and prediction_mask is not None:
        update_count = int(np.sum(update_mask))
        # Total predictions (all frames get predicted, some also get updated)
        prediction_count = int(np.sum(prediction_mask))
    
    # Get event frame knee positions and sources
    event_knee_positions = {}
    for event_name, event_idx in [
        ("FFP", keyframes.get("ffp_idx")),
        ("Release", release_idx),
        ("HandLow", hand_low_idx_post_release),
    ]:
        if event_idx is not None:
            knee_xy = get_knee_position_from_track(track, event_idx)
            knee_source = get_knee_source_at_frame(track, event_idx)
            # Normalize source: "measured" -> "kalman", "predicted" -> "predicted"
            if knee_source == "measured":
                knee_source_display = "kalman"
            elif knee_source == "predicted":
                knee_source_display = "predicted"
            else:
                knee_source_display = knee_source
            
            event_knee_positions[event_name] = {
                "frame_idx": int(event_idx),
                "knee_xy": [float(knee_xy[0]), float(knee_xy[1])] if knee_xy is not None else None,
                "knee_source": knee_source_display,
            }
    
    debug_obj = {
        "knee_start_idx": int(knee_start_idx) if knee_start_idx is not None else None,
        "release_idx": int(release_idx) if release_idx is not None else None,
        "hand_low_idx_post_release": int(hand_low_idx_post_release) if hand_low_idx_post_release is not None else None,
        "hand_low_search_window": hand_low_detection.get("search_window"),
        "min_wrist_y_frame": int(hand_low_detection.get("hand_low_idx")) if hand_low_detection.get("hand_low_idx") is not None else None,
        "min_wrist_y": float(hand_low_detection.get("min_wrist_y")) if hand_low_detection.get("min_wrist_y") is not None else None,
        "kalman_params": {
            "process_noise": float(kalman_params.get("process_noise", 0.0)),
            "measurement_noise": float(kalman_params.get("measurement_noise", 0.0)),
        },
        "update_count": update_count,
        "prediction_count": prediction_count,
        "event_knee_positions": event_knee_positions,
    }
    
    return debug_obj

