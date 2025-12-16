"""
Video I/O utilities for handling uploaded video files.
"""

import os
import tempfile
import hashlib


def compute_video_hash(data: bytes) -> str:
    """
    Compute MD5 hash of video file data.
    
    Args:
        data: Video file bytes
    
    Returns:
        Hexadecimal MD5 hash string
    """
    return hashlib.md5(data).hexdigest()


def save_uploaded_video_to_temp(uploaded_file_bytes: bytes, video_hash: str) -> str:
    """
    Save uploaded video to a stable temporary path.
    
    Creates the directory if needed and only writes if the file doesn't exist.
    
    Args:
        uploaded_file_bytes: Raw bytes of the uploaded video file
        video_hash: MD5 hash of the video file (for filename)
    
    Returns:
        Path to the saved temporary file
    """
    # Create stable temp directory
    tmp_dir = os.path.join(tempfile.gettempdir(), "pitch_app")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Construct stable temp path
    tmp_path = os.path.join(tmp_dir, f"clip_{video_hash}.mov")
    
    # Only write if file doesn't exist
    if not os.path.exists(tmp_path):
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file_bytes)
    
    return tmp_path

