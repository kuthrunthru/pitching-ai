"""
Video orientation and transformation utilities.
"""

import cv2


def apply_orientation(frame_bgr, mode: str):
    """
    Apply rotation to a frame based on orientation mode.
    
    Args:
        frame_bgr: Input frame in BGR format
        mode: Orientation mode: "None", "90 deg CW", "180 deg", "270 deg CW"
    
    Returns:
        Rotated frame in BGR format
    """
    if mode == "None":
        return frame_bgr
    elif mode == "90 deg CW":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif mode == "180 deg":
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    elif mode == "270 deg CW":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Unknown mode, return original
        return frame_bgr


def apply_preview_transform(frame_bgr, mode: str, needs_flip: bool):
    """
    Apply orientation rotation and optional horizontal flip to a frame.
    
    Rotation is applied first, then flip if needed.
    
    Args:
        frame_bgr: Input frame in BGR format
        mode: Orientation mode: "None", "90 deg CW", "180 deg", "270 deg CW"
        needs_flip: Whether to horizontally flip the frame
    
    Returns:
        Transformed frame in BGR format
    """
    # Apply rotation first
    frame = apply_orientation(frame_bgr, mode)
    
    # Apply horizontal flip if needed
    if needs_flip:
        frame = cv2.flip(frame, 1)
    
    return frame

