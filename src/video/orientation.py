"""
Video orientation and transformation utilities.
"""

import cv2
import streamlit as st
import mediapipe as mp
import numpy as np


def auto_orientation_sideways(tmp_path, pose_obj, conf_thresh, sample_frames=12):
    """
    Auto-detect orientation by scoring uprightness for candidate rotations.
    
    Tests candidate modes ["90 deg CW", "270 deg CW", "None"] by rotating sample frames,
    running MediaPipe pose, and scoring based on uprightness criteria.
    
    Args:
        tmp_path: Path to video file
        pose_obj: MediaPipe Pose object
        conf_thresh: Confidence threshold for pose detection
        sample_frames: Number of evenly spaced frames to sample (default 12)
    
    Returns:
        chosen_mode: "90 deg CW", "270 deg CW", or "None" (best scoring mode)
    """
    P = mp.solutions.pose.PoseLandmark
    LEFT_SHOULDER = P.LEFT_SHOULDER
    RIGHT_SHOULDER = P.RIGHT_SHOULDER
    LEFT_HIP = P.LEFT_HIP
    RIGHT_HIP = P.RIGHT_HIP
    NOSE = P.NOSE
    
    # Candidate modes to test
    candidate_modes = ["90 deg CW", "270 deg CW", "None"]
    
    # Open video to get frame count
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return "None"
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_frames < sample_frames:
        cap.release()
        return "None"
    
    # Sample evenly spaced frames
    frame_indices = [int(i * n_frames / (sample_frames + 1)) for i in range(1, sample_frames + 1)]
    
    # Score each candidate mode across all frames
    mode_scores = {mode: 0 for mode in candidate_modes}
    mode_wins = {mode: 0 for mode in candidate_modes}  # Count frames where this mode wins
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        # Test each candidate mode
        frame_scores = {}
        
        for mode in candidate_modes:
            # Rotate frame according to candidate mode
            if mode == "90 deg CW":
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif mode == "270 deg CW":
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif mode == "180 deg":
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            else:  # "None"
                rotated_frame = frame
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
            results = pose_obj.process(frame_rgb)
            
            if not results.pose_landmarks:
                frame_scores[mode] = 0
                continue
            
            landmarks = results.pose_landmarks.landmark
            
            # Get shoulder, hip, and nose landmarks
            ls = landmarks[LEFT_SHOULDER]
            rs = landmarks[RIGHT_SHOULDER]
            lh = landmarks[LEFT_HIP]
            rh = landmarks[RIGHT_HIP]
            nose = landmarks[NOSE]
            
            # Check confidence
            if (ls.visibility < conf_thresh or rs.visibility < conf_thresh or
                lh.visibility < conf_thresh or rh.visibility < conf_thresh or
                nose.visibility < conf_thresh):
                frame_scores[mode] = 0
                continue
            
            # Compute midpoints
            shoulder_mid_x = (ls.x + rs.x) / 2.0
            shoulder_mid_y = (ls.y + rs.y) / 2.0
            pelvis_mid_x = (lh.x + rh.x) / 2.0
            pelvis_mid_y = (lh.y + rh.y) / 2.0
            
            # Score uprightness (0-3 points)
            score = 0
            
            # Condition 1: upright = (pelvis_mid.y > shoulder_mid.y) - hips below shoulders
            if pelvis_mid_y > shoulder_mid_y:
                score += 1
            
            # Condition 2: head_above = (nose.y < shoulder_mid.y) - nose above shoulders
            if nose.y < shoulder_mid_y:
                score += 1
            
            # Condition 3: vertical_trunk = abs((shoulder_mid.x - pelvis_mid.x)) < abs((shoulder_mid.y - pelvis_mid.y))
            dx = abs(shoulder_mid_x - pelvis_mid_x)
            dy = abs(shoulder_mid_y - pelvis_mid_y)
            if dx < dy:
                score += 1
            
            frame_scores[mode] = score
        
        # Add scores to totals
        for mode in candidate_modes:
            mode_scores[mode] += frame_scores[mode]
        
        # Track which mode wins this frame (highest score)
        if len(frame_scores) > 0:
            best_mode = max(frame_scores.keys(), key=lambda m: frame_scores[m])
            if frame_scores[best_mode] > 0:  # Only count if score > 0
                mode_wins[best_mode] += 1
    
    cap.release()
    
    # Pick the mode with highest total score
    if sum(mode_scores.values()) == 0:
        # No valid frames, fall back to default
        return "None"
    
    # Find mode with highest total score
    best_total_score = max(mode_scores.values())
    candidates_with_best_score = [mode for mode in candidate_modes if mode_scores[mode] == best_total_score]
    
    if len(candidates_with_best_score) == 1:
        return candidates_with_best_score[0]
    
    # Tie: prefer the mode that wins on more frames
    if len(candidates_with_best_score) > 1:
        best_mode = max(candidates_with_best_score, key=lambda m: mode_wins[m])
        return best_mode
    
    # Fallback
    return "None"


def apply_orientation(frame_bgr, chosen_mode: str):
    """
    Apply rotation to a frame based on orientation mode. Does REAL rotation using cv2.rotate.
    
    Args:
        frame_bgr: Input frame in BGR format
        chosen_mode: Orientation mode: "None", "90 deg CW", "180 deg", "270 deg CW"
    
    Returns:
        Rotated frame in BGR format
    """
    if chosen_mode == "90 deg CW":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if chosen_mode == "180 deg":
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if chosen_mode == "270 deg CW":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame_bgr


def apply_preview_transform(frame_bgr, chosen_mode=None, needs_flip=None):
    """
    Apply orientation rotation and optional horizontal flip to a frame.
    Reads chosen_mode and needs_flip from session state if not provided.
    
    This ensures all displayed frames match the processed landmarks.
    
    Args:
        frame_bgr: Input frame in BGR format
        chosen_mode: Optional orientation mode ("None", "90 deg CW", "180 deg", "270 deg CW").
                     If None, reads from st.session_state["preview_orientation_mode"] (SINGLE SOURCE OF TRUTH)
        needs_flip: Optional horizontal flip flag. If None, reads from st.session_state["preview_needs_flip"] (SINGLE SOURCE OF TRUTH)
    
    Returns:
        Transformed frame in BGR format
    """
    # SINGLE SOURCE OF TRUTH: Get orientation settings from parameters or session state
    if chosen_mode is None:
        chosen_mode = st.session_state["preview_orientation_mode"]
    if needs_flip is None:
        needs_flip = st.session_state["preview_needs_flip"]
    
    # Apply rotation first using apply_orientation (does REAL rotation)
    frame = apply_orientation(frame_bgr, chosen_mode)
    
    # Apply horizontal flip if needed (after rotation)
    if needs_flip:
        frame = cv2.flip(frame, 1)
    
    return frame
