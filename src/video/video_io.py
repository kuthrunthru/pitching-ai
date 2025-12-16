"""
Video I/O helpers: save uploaded video to a stable temp path + read basic metadata.
"""

from __future__ import annotations

import os
import hashlib
import tempfile
from dataclasses import dataclass

import cv2


@dataclass
class VideoMeta:
    path: str
    video_hash: str
    n_frames: int
    fps: float
    width: int
    height: int


def _stable_tmp_path(video_hash: str, ext: str = "mp4") -> str:
    tmp_dir = os.path.join(tempfile.gettempdir(), "pitch_app_rebuild")
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, f"clip_{video_hash}.{ext}")


def save_uploaded_bytes_to_temp(data: bytes, ext: str = "mp4") -> tuple[str, str]:
    """
    Save uploaded bytes to a stable temp path keyed by content hash.
    Returns (tmp_path, video_hash).
    """
    if not data:
        raise ValueError("No video bytes received.")

    video_hash = hashlib.md5(data).hexdigest()
    tmp_path = _stable_tmp_path(video_hash, ext=ext)

    if not os.path.exists(tmp_path):
        with open(tmp_path, "wb") as f:
            f.write(data)

    return tmp_path, video_hash


def read_video_meta(video_path: str) -> VideoMeta:
    """
    Read fps, frame count, and dimensions from the video file.
    Uses safe fallbacks when metadata is missing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    # Guardrails / fallbacks
    if fps <= 0 or fps != fps:  # NaN-safe
        fps = 30.0
    fps = max(1.0, min(120.0, fps))

    if n <= 0:
        # Try to count frames as a fallback (slower, but robust)
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        n = count

    meta = VideoMeta(
        path=video_path,
        video_hash=os.path.basename(video_path),
        n_frames=int(n),
        fps=float(fps),
        width=int(w),
        height=int(h),
    )
    return meta

