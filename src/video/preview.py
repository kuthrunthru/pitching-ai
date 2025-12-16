# src/video/preview.py
import cv2
import streamlit as st
import numpy as np
from src.video.orientation import apply_preview_transform


def scale_xy(pts_xy, sx, sy):
    """
    Scale xy coordinates by sx and sy.
    
    Args:
        pts_xy: Array of shape (n, 2) or (2,) with (x, y) coordinates
        sx: Scale factor for x
        sy: Scale factor for y
    
    Returns:
        Scaled coordinates (same shape as input)
    """
    out = pts_xy.copy()
    if out.ndim == 1:
        # Single point (2,)
        out[0] *= sx
        out[1] *= sy
    else:
        # Multiple points (n, 2)
        out[..., 0] *= sx
        out[..., 1] *= sy
    return out


def render_preview_frame(frame_bgr, chosen_mode, needs_flip, *, caption=None, channels="BGR", clamp_width=None, width=None, add_debug=False):
    """
    SINGLE PREVIEW RENDERER - All preview frames MUST go through this function.
    
    Applies orientation transform and optional flip, then renders via Streamlit.
    This is the ONLY allowed way to display video frames in the app.
    
    MODE A (Draw-before-resize): Transform happens first, then resize for display.
    Overlays should be drawn on the transformed frame BEFORE calling this function.
    
    Args:
        frame_bgr: Frame in BGR format (raw, before transform)
        chosen_mode: Orientation mode ("None", "90 deg CW", "180 deg", "270 deg CW")
        needs_flip: Whether to horizontally flip (bool)
        caption: Optional caption for the image
        channels: Input channels format ("BGR" or "RGB", default "BGR")
        clamp_width: Optional maximum width for resizing (preserves aspect ratio, applied after transform)
        width: Optional width parameter for st.image (pixel width, not resize)
        add_debug: Whether to add debug overlay showing mode/flip/shape (default False)
    
    Returns:
        None (displays image directly via st.image)
    """
    if frame_bgr is None:
        st.info("Frame not available.")
        return
    
    # Convert to BGR if needed
    if channels == "RGB":
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
    
    # Apply transform (MUST call apply_preview_transform - this is the single entry point)
    frame_transformed = apply_preview_transform(frame_bgr, chosen_mode=chosen_mode, needs_flip=needs_flip)
    
    # Add debug overlay AFTER transform (so it shows correct final shape)
    if add_debug:
        frame_transformed = add_debug_overlay(frame_transformed, chosen_mode, needs_flip)
    
    # Resize if clamp_width is provided (AFTER transform and overlays)
    # This is Mode A: draw-before-resize
    native_h, native_w = frame_transformed.shape[:2]
    display_w = native_w
    display_h = native_h
    scale_x = 1.0
    scale_y = 1.0
    
    if clamp_width is not None and clamp_width > 0:
        if native_w > clamp_width:
            scale = clamp_width / float(native_w)
            display_w = int(native_w * scale)
            display_h = int(native_h * scale)
            scale_x = scale
            scale_y = scale
            frame_transformed = cv2.resize(frame_transformed, (display_w, display_h), interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame_transformed, cv2.COLOR_BGR2RGB)
    
    # Render via Streamlit (ONLY place where st.image is called for video frames)
    if width is not None:
        st.image(frame_rgb, caption=caption, width=width)
    else:
        st.image(frame_rgb, caption=caption, use_container_width=True)
    
    # Return scale factors for coordinate mapping (if caller needs them)
    return {
        "native_w": native_w,
        "native_h": native_h,
        "display_w": display_w,
        "display_h": display_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def get_transformed_preview_frame(frame_bgr, chosen_mode, needs_flip):
    """
    Return transformed frame (rotation+flip) for any overlay drawing.
    
    This helper ensures overlays are drawn on the correctly oriented frame,
    matching what will be displayed to the user.
    
    Args:
        frame_bgr: Raw frame in BGR format (before transform)
        chosen_mode: Orientation mode ("None", "90 deg CW", "180 deg", "270 deg CW")
        needs_flip: Whether to horizontally flip (bool)
    
    Returns:
        Transformed frame in BGR format (ready for overlay drawing)
    """
    return apply_preview_transform(frame_bgr, chosen_mode=chosen_mode, needs_flip=needs_flip)


def add_debug_overlay(frame_bgr, chosen_mode, needs_flip):
    """
    Add debug overlay to frame showing orientation settings and final shape.
    
    Args:
        frame_bgr: Frame in BGR format (after rotation/flip)
        chosen_mode: Orientation mode string
        needs_flip: Whether flip was applied
    
    Returns:
        Frame with debug overlay text
    """
    # Create a copy to avoid modifying original
    img = frame_bgr.copy()
    
    # Get frame dimensions after transform
    h, w = img.shape[:2]
    shape_str = f"{h}x{w}"
    
    # Prepare text lines
    lines = [
        f"mode={chosen_mode}",
        f"flip={needs_flip}",
        f"shape={shape_str}"
    ]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 25
    
    # Background color (semi-transparent black)
    overlay = img.copy()
    
    # Draw background rectangle for text
    text_x = 10
    text_y = 30
    rect_height = len(lines) * line_height + 10
    cv2.rectangle(overlay, (text_x - 5, text_y - 20), (text_x + 200, text_y + rect_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Draw text lines
    for i, line in enumerate(lines):
        y_pos = text_y + i * line_height
        cv2.putText(img, line, (text_x, y_pos), font, font_scale, (0, 255, 0), thickness)
    
    return img


def draw_pose_overlay_basic(frame_bgr, pts_by_name, pose_q=None, min_q=0.35):
    """
    Draws a minimal skeleton overlay on frame_bgr in-place and returns it.
    Highlights the throwing arm in red (left) or blue (right), all other joints/bones in light gray.
    
    Args:
        frame_bgr: Frame in BGR format (will be modified in-place)
        pts_by_name: dict like {"LEFT_SHOULDER": (x,y), ...} for THIS frame index in pixel coords.
                     Each value should be a tuple/list/array of (x, y) coordinates.
        pose_q: Optional pose quality score for this frame (if None, all joints drawn)
        min_q: Minimum pose quality threshold (default 0.35)
    
    Returns:
        Frame with overlay drawn (same object, modified in-place)
    """
    import cv2
    import streamlit as st
    
    def ok(name):
        """Check if landmark is valid."""
        p = pts_by_name.get(name, None)
        if p is None:
            return False
        # Handle both tuple/list and numpy array
        if isinstance(p, (list, tuple, np.ndarray)):
            if len(p) < 2:
                return False
            x, y = float(p[0]), float(p[1])
        else:
            return False
        # NaN check
        if not (x == x and y == y):
            return False
        return True
    
    def pt(name):
        """Get point as integer tuple."""
        p = pts_by_name[name]
        if isinstance(p, (list, tuple, np.ndarray)):
            x, y = float(p[0]), float(p[1])
        else:
            return (0, 0)
        return (int(round(x)), int(round(y)))
    
    def get_color(name):
        """
        Get color for a joint/bone based on throwing side.
        - Throwing arm (L if throwing_side=="L", R if throwing_side=="R"): RED (0,0,255) or BLUE (255,0,0)
        - All others: LIGHT GRAY (200,200,200)
        """
        throwing_side = st.session_state.get("keyframes", {}).get("throwing_side", "R")
        
        if name.startswith("LEFT_") or name.startswith("L_"):
            # Left side
            if throwing_side == "L":
                return (0, 0, 255)  # RED for left throwing arm
            else:
                return (200, 200, 200)  # LIGHT GRAY for non-throwing arm
        elif name.startswith("RIGHT_") or name.startswith("R_"):
            # Right side
            if throwing_side == "R":
                return (255, 0, 0)  # BLUE for right throwing arm
            else:
                return (200, 200, 200)  # LIGHT GRAY for non-throwing arm
        else:
            # Other joints (e.g., NOSE, or if name doesn't match pattern)
            return (200, 200, 200)  # LIGHT GRAY
    
    def get_bone_color(name_a, name_b):
        """
        Get color for a bone connecting two joints.
        If either joint is on the throwing arm, use throwing arm color.
        Otherwise use light gray.
        """
        throwing_side = st.session_state.get("keyframes", {}).get("throwing_side", "R")
        
        # Check if either joint is on the throwing arm
        is_throwing_a = (name_a.startswith("LEFT_") or name_a.startswith("L_")) and throwing_side == "L"
        is_throwing_a = is_throwing_a or ((name_a.startswith("RIGHT_") or name_a.startswith("R_")) and throwing_side == "R")
        
        is_throwing_b = (name_b.startswith("LEFT_") or name_b.startswith("L_")) and throwing_side == "L"
        is_throwing_b = is_throwing_b or ((name_b.startswith("RIGHT_") or name_b.startswith("R_")) and throwing_side == "R")
        
        if is_throwing_a or is_throwing_b:
            # At least one joint is on throwing arm
            if throwing_side == "L":
                return (0, 0, 255)  # RED for left throwing arm
            else:
                return (255, 0, 0)  # BLUE for right throwing arm
        else:
            return (200, 200, 200)  # LIGHT GRAY for non-throwing
    
    # Check pose quality if provided
    if pose_q is not None and pose_q < min_q:
        # Pose quality too low, don't draw
        return frame_bgr
    
    # Joints to draw (small circles)
    joints = [
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_ANKLE", "RIGHT_ANKLE",
        # Note: KNEE not in cache, so skip for now
    ]
    
    # Bones to draw (lines)
    bones = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_HIP", "LEFT_ANKLE"),  # Simplified: hip to ankle (no knee)
        ("RIGHT_HIP", "RIGHT_ANKLE"),  # Simplified: hip to ankle (no knee)
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ]
    
    # Draw bones first (colored based on throwing arm)
    for a, b in bones:
        if ok(a) and ok(b):
            color = get_bone_color(a, b)
            cv2.line(frame_bgr, pt(a), pt(b), color, 2, lineType=cv2.LINE_AA)
    
    # Draw joints (colored based on throwing arm)
    for j in joints:
        if ok(j):
            color = get_color(j)
            cv2.circle(frame_bgr, pt(j), 4, color, -1, lineType=cv2.LINE_AA)

    return frame_bgr


def draw_pose_overlay(frame_bgr, landmarks_normalized, process_width, process_height, 
                      chosen_mode, needs_flip, scale_x=1.0, scale_y=1.0):
    """
    Draw pose overlay on a transformed frame.
    
    MODE A (Draw-before-resize): This function expects the frame to be already transformed
    and at native resolution. Landmarks are in normalized [0,1] coords relative to process_width/process_height.
    MediaPipe draw_landmarks expects normalized [0,1] coords relative to the frame dimensions.
    
    Since the frame may have been rotated (changing dimensions), we need to convert landmarks
    from process coords to frame coords.
    
    Args:
        frame_bgr: Frame in BGR format (already transformed, native resolution)
        landmarks_normalized: List of [x, y, z, visibility] arrays (normalized [0,1] relative to process_width/process_height)
        process_width: Width of processed video (for coordinate conversion)
        process_height: Height of processed video (for coordinate conversion)
        chosen_mode: Orientation mode (for coordinate transformation if needed)
        needs_flip: Whether flip was applied (for coordinate transformation if needed)
        scale_x: Scale factor for x (unused in Mode A, kept for compatibility)
        scale_y: Scale factor for y (unused in Mode A, kept for compatibility)
    
    Returns:
        Frame with pose overlay drawn (same resolution as input)
    """
    if landmarks_normalized is None or len(landmarks_normalized) == 0:
        return frame_bgr
    
    if not process_width or not process_height:
        # Can't convert coords without process dimensions
        return frame_bgr
    
    try:
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        # Get current frame dimensions (native, after transform)
        frame_h, frame_w = frame_bgr.shape[:2]
        
        # Convert landmarks from process coords to frame coords
        # Landmarks are normalized [0,1] relative to process_width/process_height
        # MediaPipe expects normalized [0,1] relative to frame_w/frame_h
        # Since processing applied the same transform, landmarks should map correctly
        # But we need to account for dimension changes from rotation
        
        # Create MediaPipe landmark list with converted coordinates
        class SimpleLandmark:
            def __init__(self, x, y, z, visibility):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
        
        mp_landmarks = []
        for lm_data in landmarks_normalized:
            if len(lm_data) >= 4:
                # Landmarks are in normalized [0,1] relative to process_width/process_height
                # Convert to pixel coords, then back to normalized relative to frame
                px = float(lm_data[0]) * process_width
                py = float(lm_data[1]) * process_height
                
                # Convert to normalized coords relative to frame dimensions
                # (This works because processing applied the same transform)
                x_norm = px / frame_w if frame_w > 0 else 0.0
                y_norm = py / frame_h if frame_h > 0 else 0.0
                
                # Clamp to [0,1]
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))
                
                mp_landmarks.append(SimpleLandmark(
                    x=x_norm,
                    y=y_norm,
                    z=float(lm_data[2]) if len(lm_data) > 2 else 0.0,
                    visibility=float(lm_data[3]) if len(lm_data) > 3 else 0.0
                ))
        
        if not mp_landmarks:
            return frame_bgr
        
        # Create MediaPipe landmark list object
        class LandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks
        
        landmark_list = LandmarkList(mp_landmarks)
        
        # Create a copy of the frame for drawing
        frame_with_overlay = frame_bgr.copy()
        
        # Convert to RGB for MediaPipe drawing
        frame_rgb = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
        
        # Draw landmarks using MediaPipe (it expects normalized coords relative to frame)
        mp_drawing.draw_landmarks(
            frame_rgb,
            landmark_list,
            mp_pose.POSE_CONNECTIONS
        )
        
        # Convert back to BGR
        frame_with_overlay = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_with_overlay
            
    except Exception as e:
        # If drawing fails, return original frame
        return frame_bgr


@st.cache_data(show_spinner=False)
def read_raw_frame(video_path: str, frame_idx: int, max_w: int = None):
    """
    Read a raw frame from video at native resolution.
    
    NOTE: This function should NOT resize frames. Resizing happens later
    in the pipeline after transforms and overlays are applied.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to read (0-based)
        max_w: DEPRECATED - kept for compatibility but ignored (resize happens later)
    
    Returns:
        Raw frame in BGR format at native resolution (not transformed)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return None
    # DO NOT resize here - return native resolution
    return fr

def show_preview_frame(video_path: str, frame_idx: int, max_w: int = 900, caption: str = None, width: int = None, rev: int = 0):
    """
    Read a frame, apply orientation transform, and display it.
    
    SINGLE SOURCE OF TRUTH: Reads chosen_mode and needs_flip from session state.
    No defaults, no re-deriving - must use exact session state values.
    Orientation handling is done strictly in render_preview_frame().
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to read (0-based)
        max_w: Maximum width for downscaling (default 900) - used as clamp_width
        caption: Optional caption for the image
        width: Optional width for the displayed image
        rev: Revision counter for cache invalidation (harmless cache buster, default 0)
    
    Returns:
        None (displays image directly)
    """
    # SINGLE SOURCE OF TRUTH: Read from session state (no defaults)
    chosen_mode = st.session_state["preview_orientation_mode"]
    needs_flip = st.session_state["preview_needs_flip"]
    rev = st.session_state.get("orientation_rev", 0) if rev == 0 else rev
    
    # Harmless cache buster: include rev in a no-op so Streamlit re-runs correctly
    _ = rev
    
    # Read raw frame at native resolution (no resize)
    frame = read_raw_frame(video_path, frame_idx, max_w=None)
    if frame is None:
        st.info("Frame not available.")
        return
    
    # Use centralized renderer (applies transform, resizes for display)
    render_preview_frame(frame, chosen_mode, needs_flip, caption=caption, channels="BGR", 
                        clamp_width=max_w, width=width, add_debug=False)

def show_pose_frame(video_path, lms_arr, t, caption=None, overlay=True, max_w=1000, width=None, 
                   process_width=None, process_height=None):
    """
    Display a frame with optional pose overlay. Applies orientation transform at display time.
    
    MODE A (Draw-before-resize):
    1. Read raw frame at native resolution
    2. Transform frame (rotate/flip)
    3. Draw overlays on transformed frame (native resolution)
    4. Resize final composited image for display
    
    SINGLE SOURCE OF TRUTH: Reads chosen_mode and needs_flip from session state.
    
    Args:
        video_path: Path to video file
        lms_arr: Array of landmark arrays (per frame) in normalized [0,1] coords
        t: Frame index
        caption: Optional caption for the image
        overlay: Whether to overlay pose landmarks (default True)
        max_w: Maximum width for downscaling (default 1000) - used as clamp_width
        width: Optional width for displayed image (default None for container width)
        process_width: Width of processed video (for landmark coordinate conversion)
        process_height: Height of processed video (for landmark coordinate conversion)
    
    Returns:
        None (displays image directly)
    """
    # SINGLE SOURCE OF TRUTH: Read from session state (no defaults)
    chosen_mode = st.session_state["preview_orientation_mode"]
    needs_flip = st.session_state["preview_needs_flip"]
    if t is None:
        st.info("Frame not available.")
        return

    # 1) Read raw frame at native resolution (no resize)
    fr = read_raw_frame(video_path, int(t), max_w=None)
    if fr is None:
        st.info("Frame not available.")
        return

    # 2) Transform the frame FIRST (before drawing overlays)
    # This ensures overlays are drawn on the correctly oriented frame
    img = get_transformed_preview_frame(fr, chosen_mode, needs_flip)
    
    # Get native dimensions after transform
    native_h, native_w = img.shape[:2]

    # 3) Draw overlays on the TRANSFORMED frame at NATIVE resolution (Mode A)
    if overlay:
        # Try to get cached pose data for overlay
        clip = st.session_state.get("clip", {})
        cache_key = clip.get("cache_key")
        n = clip.get("n")
        fps = clip.get("fps")
        
        if cache_key and n and fps:
            try:
                from src.pose.cache import get_or_compute_pose
                pose_cache = get_or_compute_pose(
                    cache_key=cache_key,
                    tmp_path=video_path,
                    n=n,
                    fps=fps,
                    process_width=process_width or clip.get("process_width"),
                    process_height=process_height or clip.get("process_height"),
                    lms_arr=lms_arr,
                    pose_quality=clip.get("pose_quality", []),
                    mode="metrics",  # Full resolution for overlay
                )
                
                # Extract landmarks for this frame index
                frame_idx = int(t)
                if frame_idx < n and pose_cache.get("pts") is not None:
                    pts_by_name = {}
                    for name, arr in pose_cache["pts"].items():
                        if arr is not None and frame_idx < len(arr):
                            pt_xy = arr[frame_idx].copy()
                            
                            proc_w = process_width or clip.get("process_width")
                            proc_h = process_height or clip.get("process_height")
                            
                            if proc_w and proc_h and native_w > 0 and native_h > 0:
                                scale_x = native_w / float(proc_w)
                                scale_y = native_h / float(proc_h)
                                pt_xy = scale_xy(pt_xy, scale_x, scale_y)
                            
                            pts_by_name[name] = pt_xy
                    pts_by_name["frame_idx"] = frame_idx
                    
                    # Get pose quality for this frame
                    pose_q_frame = None
                    if pose_cache.get("pose_q") is not None and frame_idx < len(pose_cache["pose_q"]):
                        pose_q_frame = pose_cache["pose_q"][frame_idx]
                    
                    # Draw basic overlay using cached landmarks (already scaled to frame coords)
                    if pts_by_name:
                        img = draw_pose_overlay_basic(img, pts_by_name, pose_q=pose_q_frame)
            except Exception:
                # If cache lookup fails, fall back to old method or skip overlay
                pass


    # 4) Display via centralized renderer (frame is already transformed and has overlays)
    # Pass chosen_mode="None" and needs_flip=False to avoid double-transforming
    # clamp_width will resize the final composited image for display
    render_preview_frame(img, chosen_mode="None", needs_flip=False, caption=caption, 
                        channels="BGR", clamp_width=max_w, width=width, add_debug=False)


def render_orientation_preview(tmp_path: str, n: int):
    """Simple view-only preview renderer. Uses centralized render_preview_frame."""
    st.subheader("Preview")
    
    # Default to middle frame
    if "preview_frame" not in st.session_state:
        st.session_state["preview_frame"] = max(0, n // 2)
    
    t = st.slider("Preview frame", 0, max(0, n - 1), int(st.session_state["preview_frame"]))
    st.session_state["preview_frame"] = int(t)

    # Use show_preview_frame (which reads from single source of truth in session state)
    show_preview_frame(
        video_path=tmp_path,
        frame_idx=int(t),
        max_w=900
    )
