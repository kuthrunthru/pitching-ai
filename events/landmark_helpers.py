"""
Helper functions to extract landmark positions from MediaPipe pose data.
"""

import numpy as np
import mediapipe as mp


def extract_both_wrists(lms_arr, n: int = None, width: int = None, height: int = None):
    """
    Extract both left and right wrist positions from pose data.
    
    Args:
        lms_arr: List/array of per-frame landmarks (MediaPipe format, normalized [0,1])
        n: Total number of frames (if None, inferred from lms_arr)
        width: Video width in pixels (for coordinate conversion, default None = use normalized)
        height: Video height in pixels (for coordinate conversion, default None = use normalized)
    
    Returns:
        (wrist_xy_L, wrist_xy_R)
        wrist_xy_L: np.array shape (n, 2) with (x, y) positions for left wrist
        wrist_xy_R: np.array shape (n, 2) with (x, y) positions for right wrist
    """
    P = mp.solutions.pose.PoseLandmark
    
    if n is None:
        n = len(lms_arr) if lms_arr is not None else 0
    
    if n == 0 or lms_arr is None:
        return None, None
    
    wrist_xy_L = np.zeros((n, 2), dtype=np.float32)
    wrist_xy_R = np.zeros((n, 2), dtype=np.float32)
    
    for i in range(n):
        if lms_arr[i] is None or len(lms_arr[i]) <= max(P.LEFT_WRIST, P.RIGHT_WRIST):
            continue
        
        wrist_lm_L = lms_arr[i][P.LEFT_WRIST]
        wrist_lm_R = lms_arr[i][P.RIGHT_WRIST]
        
        # Check confidence and extract left wrist
        if (len(wrist_lm_L) >= 4 and wrist_lm_L[3] > 0.0 and
            np.isfinite(wrist_lm_L[0]) and np.isfinite(wrist_lm_L[1])):
            if width is not None and height is not None:
                wrist_xy_L[i, 0] = float(wrist_lm_L[0]) * width
                wrist_xy_L[i, 1] = float(wrist_lm_L[1]) * height
            else:
                wrist_xy_L[i, 0] = float(wrist_lm_L[0])
                wrist_xy_L[i, 1] = float(wrist_lm_L[1])
        
        # Check confidence and extract right wrist
        if (len(wrist_lm_R) >= 4 and wrist_lm_R[3] > 0.0 and
            np.isfinite(wrist_lm_R[0]) and np.isfinite(wrist_lm_R[1])):
            if width is not None and height is not None:
                wrist_xy_R[i, 0] = float(wrist_lm_R[0]) * width
                wrist_xy_R[i, 1] = float(wrist_lm_R[1]) * height
            else:
                wrist_xy_R[i, 0] = float(wrist_lm_R[0])
                wrist_xy_R[i, 1] = float(wrist_lm_R[1])
    
    return wrist_xy_L, wrist_xy_R


def extract_arm_landmarks(lms_arr, hand: str = "R", n: int = None, width: int = None, height: int = None):
    """
    Extract throwing arm landmark positions (wrist, elbow, shoulder) from pose data.
    
    Args:
        lms_arr: List/array of per-frame landmarks (MediaPipe format, normalized [0,1])
        hand: "R" or "L" for throwing hand (default "R")
        n: Total number of frames (if None, inferred from lms_arr)
        width: Video width in pixels (for coordinate conversion, default None = use normalized)
        height: Video height in pixels (for coordinate conversion, default None = use normalized)
    
    Returns:
        (wrist_xy, elbow_xy, shoulder_xy, valid_mask)
        wrist_xy: np.array shape (n, 2) with (x, y) positions in pixels (or normalized if width/height not provided)
        elbow_xy: np.array shape (n, 2) with (x, y) positions in pixels (or normalized if width/height not provided)
        shoulder_xy: np.array shape (n, 2) with (x, y) positions in pixels (or normalized if width/height not provided)
        valid_mask: np.array shape (n,) boolean mask of valid frames
    """
    P = mp.solutions.pose.PoseLandmark
    
    if n is None:
        n = len(lms_arr) if lms_arr is not None else 0
    
    if n == 0 or lms_arr is None:
        return None, None, None, None
    
    # Select landmarks based on handedness
    if hand.upper() == "R":
        wrist_idx = P.RIGHT_WRIST
        elbow_idx = P.RIGHT_ELBOW
        shoulder_idx = P.RIGHT_SHOULDER
    else:
        wrist_idx = P.LEFT_WRIST
        elbow_idx = P.LEFT_ELBOW
        shoulder_idx = P.LEFT_SHOULDER
    
    wrist_xy = np.zeros((n, 2), dtype=np.float32)
    elbow_xy = np.zeros((n, 2), dtype=np.float32)
    shoulder_xy = np.zeros((n, 2), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if lms_arr[i] is None or len(lms_arr[i]) <= max(wrist_idx, elbow_idx, shoulder_idx):
            continue
        
        # Extract x, y coordinates (MediaPipe landmarks are normalized [0,1])
        # Convert to pixels if width/height provided
        
        wrist_lm = lms_arr[i][wrist_idx]
        elbow_lm = lms_arr[i][elbow_idx]
        shoulder_lm = lms_arr[i][shoulder_idx]
        
        # Check confidence (index 3 in MediaPipe landmark format: [x, y, z, visibility])
        if (len(wrist_lm) >= 4 and len(elbow_lm) >= 4 and len(shoulder_lm) >= 4 and
            wrist_lm[3] > 0.0 and elbow_lm[3] > 0.0 and shoulder_lm[3] > 0.0 and
            np.isfinite(wrist_lm[0]) and np.isfinite(wrist_lm[1]) and
            np.isfinite(elbow_lm[0]) and np.isfinite(elbow_lm[1]) and
            np.isfinite(shoulder_lm[0]) and np.isfinite(shoulder_lm[1])):
            
            # Convert normalized [0,1] to pixels if dimensions provided
            if width is not None and height is not None:
                wrist_xy[i, 0] = float(wrist_lm[0]) * width
                wrist_xy[i, 1] = float(wrist_lm[1]) * height
                elbow_xy[i, 0] = float(elbow_lm[0]) * width
                elbow_xy[i, 1] = float(elbow_lm[1]) * height
                shoulder_xy[i, 0] = float(shoulder_lm[0]) * width
                shoulder_xy[i, 1] = float(shoulder_lm[1]) * height
            else:
                # Use normalized coordinates
                wrist_xy[i, 0] = float(wrist_lm[0])
                wrist_xy[i, 1] = float(wrist_lm[1])
                elbow_xy[i, 0] = float(elbow_lm[0])
                elbow_xy[i, 1] = float(elbow_lm[1])
                shoulder_xy[i, 0] = float(shoulder_lm[0])
                shoulder_xy[i, 1] = float(shoulder_lm[1])
            valid_mask[i] = True
    
    return wrist_xy, elbow_xy, shoulder_xy, valid_mask
