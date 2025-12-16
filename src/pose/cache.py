"""
Pose landmark caching utilities.

Caches pre-computed landmark arrays in pixel coordinates to avoid recomputing
on every Streamlit rerun. Memory-safe: uses compact dtypes, LRU eviction, and optional downsampling.
"""

import time
import numpy as np
import streamlit as st
import mediapipe as mp
from events.landmark_helpers import extract_arm_landmarks, extract_both_wrists

# Maximum number of cached clips (LRU eviction)
MAX_CACHE_KEYS = 3


def _update_lru(cache_key: str):
    """Update LRU list: move cache_key to end (most recent)."""
    if "pose_cache_lru" not in st.session_state:
        st.session_state["pose_cache_lru"] = []
    
    lru = st.session_state["pose_cache_lru"]
    
    # Remove if exists, then append to end
    if cache_key in lru:
        lru.remove(cache_key)
    lru.append(cache_key)
    
    # Evict oldest if over limit
    while len(lru) > MAX_CACHE_KEYS:
        old_key = lru.pop(0)
        if "pose_cache" in st.session_state:
            st.session_state["pose_cache"].pop(old_key, None)


def _downsample_arrays(arrays_dict: dict, n: int, decimate_factor: int = 2):
    """
    Downsample arrays by taking every Nth frame.
    
    Args:
        arrays_dict: Dict of arrays to downsample (values are np arrays)
        n: Original number of frames
        decimate_factor: Take every Nth frame (default 2 = every 2nd frame)
    
    Returns:
        Dict with downsampled arrays
    """
    n_downsampled = (n + decimate_factor - 1) // decimate_factor
    result = {}
    
    for key, arr in arrays_dict.items():
        if arr is None:
            result[key] = None
            continue
        
        if isinstance(arr, np.ndarray):
            if arr.ndim == 1:
                # 1D array (e.g., pose_q)
                downsampled = arr[::decimate_factor]
                if len(downsampled) < n_downsampled:
                    # Pad if needed
                    padded = np.zeros(n_downsampled, dtype=arr.dtype)
                    padded[:len(downsampled)] = downsampled
                    result[key] = padded
                else:
                    result[key] = downsampled[:n_downsampled]
            elif arr.ndim == 2:
                # 2D array (e.g., (n, 2) xy coordinates)
                downsampled = arr[::decimate_factor]
                if len(downsampled) < n_downsampled:
                    # Pad if needed
                    padded = np.zeros((n_downsampled, arr.shape[1]), dtype=arr.dtype)
                    padded[:len(downsampled)] = downsampled
                    result[key] = padded
                else:
                    result[key] = downsampled[:n_downsampled]
            else:
                # Keep as-is for other shapes
                result[key] = arr
        else:
            # Non-array values (e.g., dicts, scalars)
            result[key] = arr
    
    return result


def get_or_compute_pose(cache_key: str, tmp_path: str, n: int, fps: float, 
                        process_width: int, process_height: int, lms_arr, pose_quality,
                        mode: str = "metrics"):
    """
    Get cached pose data or compute it if not cached.
    
    Memory-safe: stores only numeric arrays (float32), supports LRU eviction, optional downsampling.
    
    Returns dict with:
      - pose_q: (n,) array of pose quality scores (float32)
      - wrist_xy_L: (n, 2) left wrist positions (float32)
      - wrist_xy_R: (n, 2) right wrist positions (float32)
      - wrist_xy_R_arm: (n, 2) right wrist (for right arm) (float32)
      - elbow_xy_R: (n, 2) right elbow (float32)
      - shoulder_xy_R: (n, 2) right shoulder (float32)
      - wrist_xy_L_arm: (n, 2) left wrist (for left arm) (float32)
      - elbow_xy_L: (n, 2) left elbow (float32)
      - shoulder_xy_L: (n, 2) left shoulder (float32)
      - valid_mask_R: (n,) boolean mask for right arm
      - valid_mask_L: (n,) boolean mask for left arm
      - pts: dict of landmark arrays (only if mode="metrics", else None)
      - preview_pts: dict of downsampled landmark arrays (only if mode="preview" and n > 100, else None)
      - preview_pose_q: downsampled pose_q (only if mode="preview" and n > 100, else None)
      - computed_at: timestamp when computed
      - from_cache: bool indicating if data came from cache
    
    IMPORTANT: This function does NOT touch UI; it's pure data processing.
    Stores ONLY numeric arrays (no MediaPipe objects, frames, or images).
    
    Args:
        cache_key: Unique key for this clip (e.g., "{tmp_path}:{n}:{fps}")
        tmp_path: Path to video file (for cache key generation, not used in computation)
        n: Number of frames
        fps: Frames per second
        process_width: Video width in pixels
        process_height: Video height in pixels
        lms_arr: List/array of per-frame landmarks (MediaPipe format, normalized [0,1])
        pose_quality: List/array of pose quality scores per frame
        mode: "metrics" (full resolution) or "preview" (may downsample if n > 100)
    
    Returns:
        dict with cached pose data
    """
    # Initialize cache container if needed
    if "pose_cache" not in st.session_state:
        st.session_state["pose_cache"] = {}
    
    # Update LRU and check if already cached
    _update_lru(cache_key)
    
    if cache_key in st.session_state["pose_cache"]:
        cached = st.session_state["pose_cache"][cache_key]
        # Verify cache is valid (has required fields)
        if (cached.get("pose_q") is not None and 
            cached.get("wrist_xy_L") is not None and
            cached.get("wrist_xy_R") is not None):
            # Mark as from cache
            cached["from_cache"] = True
            return cached
    
    # Not cached - compute now
    start_time = time.time()
    
    # Convert pose_quality to numpy array (float32 for memory efficiency)
    pose_q = np.array(pose_quality, dtype=np.float32) if pose_quality else np.zeros(n, dtype=np.float32)
    
    # Extract both wrists (already returns float32 from helpers)
    wrist_xy_L, wrist_xy_R = extract_both_wrists(
        lms_arr,
        n=n,
        width=process_width,
        height=process_height,
    )
    
    # Ensure float32 (helpers should already do this, but be safe)
    if wrist_xy_L is not None:
        wrist_xy_L = wrist_xy_L.astype(np.float32)
    if wrist_xy_R is not None:
        wrist_xy_R = wrist_xy_R.astype(np.float32)
    
    # Extract right arm landmarks
    wrist_xy_R_arm, elbow_xy_R, shoulder_xy_R, valid_mask_R = extract_arm_landmarks(
        lms_arr,
        hand="R",
        n=n,
        width=process_width,
        height=process_height,
    )
    
    # Extract left arm landmarks
    wrist_xy_L_arm, elbow_xy_L, shoulder_xy_L, valid_mask_L = extract_arm_landmarks(
        lms_arr,
        hand="L",
        n=n,
        width=process_width,
        height=process_height,
    )
    
    # Ensure float32 for all arm landmarks
    if wrist_xy_R_arm is not None:
        wrist_xy_R_arm = wrist_xy_R_arm.astype(np.float32)
    if elbow_xy_R is not None:
        elbow_xy_R = elbow_xy_R.astype(np.float32)
    if shoulder_xy_R is not None:
        shoulder_xy_R = shoulder_xy_R.astype(np.float32)
    if wrist_xy_L_arm is not None:
        wrist_xy_L_arm = wrist_xy_L_arm.astype(np.float32)
    if elbow_xy_L is not None:
        elbow_xy_L = elbow_xy_L.astype(np.float32)
    if shoulder_xy_L is not None:
        shoulder_xy_L = shoulder_xy_L.astype(np.float32)
    
    # Build pts dict with key landmarks (only for metrics mode, or if preview needs full res)
    pts = None
    preview_pts = None
    preview_pose_q = None
    
    if mode == "metrics" or (mode == "preview" and n <= 100):
        # Extract all major landmarks for metrics (or small previews)
        P = mp.solutions.pose.PoseLandmark
        pts = {}
        
        landmark_names = {
            "LEFT_WRIST": P.LEFT_WRIST,
            "RIGHT_WRIST": P.RIGHT_WRIST,
            "LEFT_ELBOW": P.LEFT_ELBOW,
            "RIGHT_ELBOW": P.RIGHT_ELBOW,
            "LEFT_SHOULDER": P.LEFT_SHOULDER,
            "RIGHT_SHOULDER": P.RIGHT_SHOULDER,
            "LEFT_ANKLE": P.LEFT_ANKLE,
            "RIGHT_ANKLE": P.RIGHT_ANKLE,
            "LEFT_HIP": P.LEFT_HIP,
            "RIGHT_HIP": P.RIGHT_HIP,
            "LEFT_KNEE": P.LEFT_KNEE,
            "RIGHT_KNEE": P.RIGHT_KNEE,
            "NOSE": P.NOSE,
        }
        
        for name, idx in landmark_names.items():
            arr = np.zeros((n, 2), dtype=np.float32)
            for i in range(n):
                if lms_arr[i] is None or len(lms_arr[i]) <= idx:
                    continue
                lm = lms_arr[i][idx]
                if len(lm) >= 4 and lm[3] > 0.0 and np.isfinite(lm[0]) and np.isfinite(lm[1]):
                    if process_width is not None and process_height is not None:
                        arr[i, 0] = float(lm[0]) * process_width
                        arr[i, 1] = float(lm[1]) * process_height
                    else:
                        arr[i, 0] = float(lm[0])
                        arr[i, 1] = float(lm[1])
            pts[name] = arr
    
    elif mode == "preview" and n > 100:
        # For large previews, create downsampled version
        # First compute full res, then downsample
        P = mp.solutions.pose.PoseLandmark
        pts_full = {}
        
        landmark_names = {
            "LEFT_WRIST": P.LEFT_WRIST,
            "RIGHT_WRIST": P.RIGHT_WRIST,
            "LEFT_ELBOW": P.LEFT_ELBOW,
            "RIGHT_ELBOW": P.RIGHT_ELBOW,
            "LEFT_SHOULDER": P.LEFT_SHOULDER,
            "RIGHT_SHOULDER": P.RIGHT_SHOULDER,
            "LEFT_ANKLE": P.LEFT_ANKLE,
            "RIGHT_ANKLE": P.RIGHT_ANKLE,
            "LEFT_HIP": P.LEFT_HIP,
            "RIGHT_HIP": P.RIGHT_HIP,
            "NOSE": P.NOSE,
        }
        
        # Compute full resolution first (store x, y, z=0, conf)
        for name, idx in landmark_names.items():
            arr = np.zeros((n, 4), dtype=np.float32)  # [x, y, z, conf]
            for i in range(n):
                if lms_arr[i] is None or len(lms_arr[i]) <= idx:
                    continue
                lm = lms_arr[i][idx]
                if len(lm) >= 2 and np.isfinite(lm[0]) and np.isfinite(lm[1]):
                    conf_val = 1.0
                    if len(lm) >= 4 and lm[3] is not None and np.isfinite(lm[3]):
                        conf_val = float(lm[3])
                    if process_width is not None and process_height is not None:
                        arr[i, 0] = float(lm[0]) * process_width
                        arr[i, 1] = float(lm[1]) * process_height
                    else:
                        arr[i, 0] = float(lm[0])
                        arr[i, 1] = float(lm[1])
                    arr[i, 2] = 0.0
                    arr[i, 3] = conf_val
            pts_full[name] = arr
        
        # Downsample for preview (every 2nd frame)
        preview_pts = _downsample_arrays(pts_full, n, decimate_factor=2)
        preview_pose_q = pose_q[::2].astype(np.float32)
        
        # Don't store full res pts for preview mode
        pts = None
    
    # Build result dict (ONLY numeric arrays, no MediaPipe objects or frames)
    result = {
        "pose_q": pose_q,
        "wrist_xy_L": wrist_xy_L,
        "wrist_xy_R": wrist_xy_R,
        "wrist_xy_R_arm": wrist_xy_R_arm,
        "elbow_xy_R": elbow_xy_R,
        "shoulder_xy_R": shoulder_xy_R,
        "valid_mask_R": valid_mask_R,
        "wrist_xy_L_arm": wrist_xy_L_arm,
        "elbow_xy_L": elbow_xy_L,
        "shoulder_xy_L": shoulder_xy_L,
        "valid_mask_L": valid_mask_L,
        "pts": pts,  # Full resolution landmarks (for metrics)
        "preview_pts": preview_pts,  # Downsampled landmarks (for preview, if n > 100)
        "preview_pose_q": preview_pose_q,  # Downsampled pose_q (for preview, if n > 100)
        "computed_at": time.time(),
        "computation_time": time.time() - start_time,
        "from_cache": False,
    }
    
    # Store in cache
    st.session_state["pose_cache"][cache_key] = result
    
    return result


def clear_pose_cache():
    """Clear all pose cache entries."""
    if "pose_cache" in st.session_state:
        st.session_state["pose_cache"].clear()
    if "pose_cache_lru" in st.session_state:
        st.session_state["pose_cache_lru"].clear()
