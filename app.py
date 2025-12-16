"""
Pitching Mechanics Analyzer - Main Application

A Streamlit app for analyzing pitching mechanics from video.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*use_column_width.*deprecated.*",
    category=DeprecationWarning,
)

import streamlit as st
import mediapipe as mp
from core.state import init_state, reset_for_new_video
from src.video.video_io import save_uploaded_bytes_to_temp, read_video_meta
from src.video.preview import render_orientation_preview
from src.video.process import process_video_landmarks
from src.video.orientation import auto_orientation_sideways
from src.ui.set_frames import render_set_frames
from src.ui.orientation_review import render_orientation_review
from src.pitching.ui.results import render_results

# Initialize session state before any access
init_state()


def main():
    """Main application entry point."""
    
    # Initialize debug mode toggle (hidden, in sidebar)
    st.session_state.setdefault("debug_mode", False)
    
    # Sidebar debug toggle
    with st.sidebar:
        st.header("Debug")
        debug_mode = st.checkbox("Debug mode", value=st.session_state.get("debug_mode", False))
        st.session_state["debug_mode"] = debug_mode
        if debug_mode:
            st.caption("Debug mode enabled: showing cache hits/misses and keyframe sources")

    stage = st.session_state.get("stage", "upload")
    if stage == "upload":
        # Upload stage UI
        st.header("Upload")
        st.markdown("""
        - Upload pitching video (mp4/mov/avi)
        - View should be a side view (i.e., view from third base to pitcher for a right handed pitcher)
        - Upload slow motion videos for more precise results
        """)
        uploaded = st.file_uploader("", type=["mp4", "mov", "avi"])
        if uploaded is None:
            st.stop()

        data = uploaded.getvalue()
        # Use original extension if possible; fallback to mp4
        name = getattr(uploaded, "name", "") or ""
        ext = name.split(".")[-1].lower() if "." in name else "mp4"
        if ext not in ("mp4", "mov", "avi"):
            ext = "mp4"

        tmp_path, video_hash = save_uploaded_bytes_to_temp(data, ext=ext)

        # If new file, reset state
        if st.session_state.get("video_hash") != video_hash:
            reset_for_new_video(video_hash)

        # Always ensure tmp_path is present (even if same file)
        st.session_state["clip"]["tmp_path"] = tmp_path

        # Check if already processed (skip reprocessing)
        clip = st.session_state["clip"]
        if clip.get("lms_arr") is not None and clip.get("n") is not None:
            # Already processed, advance to orientation_review
            st.session_state["stage"] = "orientation_review"
            st.rerun()
            return

        # Process video landmarks if not already processed
        try:
            # Auto-detect orientation before processing
            pose_obj = mp.solutions.pose.Pose(
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            
            try:
                with st.spinner("Detecting video orientation..."):
                    chosen_mode = auto_orientation_sideways(
                        tmp_path=tmp_path,
                        pose_obj=pose_obj,
                        conf_thresh=0.4,
                        sample_frames=12,
                    )
            finally:
                pose_obj.close()
            
            # Store detected orientation - SINGLE SOURCE OF TRUTH
            st.session_state["preview_orientation_mode"] = chosen_mode
            st.session_state["preview_needs_flip"] = False  # Keep flip logic unchanged
            
            with st.spinner("Processing video with MediaPipe Pose..."):
                processed = process_video_landmarks(
                    video_path=tmp_path,
                    conf_thresh=0.4,  # Default confidence threshold
                    max_process_width=1280,  # Default max width
                    model_complexity=2,
                    smooth_landmarks=True,
                    chosen_mode=chosen_mode,
                )
            
            n = processed["n"]
            
            # Guardrails
            if n < 20:
                st.error(f"Video has too few frames ({n}). Minimum 20 frames required.")
                st.stop()
            
            # Store processed results
            st.session_state["clip"]["tmp_path"] = tmp_path
            st.session_state["clip"]["lms_arr"] = processed["lms_arr"]
            st.session_state["clip"]["n"] = processed["n"]
            st.session_state["clip"]["fps"] = processed["fps"]
            st.session_state["clip"]["pose_quality"] = processed["pose_quality"]
            st.session_state["clip"]["process_width"] = processed.get("process_width")
            st.session_state["clip"]["process_height"] = processed.get("process_height")
            # preview_orientation_mode and preview_needs_flip already set above (single source of truth)
            
            # Generate cache key and pre-compute pose arrays
            cache_key = f"{tmp_path}:{processed['n']}:{processed['fps']:.2f}"
            st.session_state["clip"]["cache_key"] = cache_key
            
            # Initialize pose cache container
            st.session_state.setdefault("pose_cache", {})
            
            # Pre-compute and cache pose arrays (full resolution for metrics)
            from src.pose.cache import get_or_compute_pose
            get_or_compute_pose(
                cache_key=cache_key,
                tmp_path=tmp_path,
                n=processed["n"],
                fps=processed["fps"],
                process_width=processed.get("process_width"),
                process_height=processed.get("process_height"),
                lms_arr=processed["lms_arr"],
                pose_quality=processed["pose_quality"],
                mode="metrics",  # Full resolution for metrics computation
            )
            
            st.success(f"Processed video: {n} frames @ {processed['fps']:.1f} fps")
            
            # Advance to orientation_review stage
            st.session_state["stage"] = "orientation_review"
            st.rerun()
            
        except RuntimeError as e:
            st.error(f"Could not process video: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error processing video: {e}")
            st.stop()
    elif stage == "orientation_review":
        render_orientation_review()
    elif stage == "set_frames":
        render_set_frames()
    elif stage == "results":
        # Hard gate: ensure all keyframes are set
        from core.keyframes import get_keyframes
        clip = st.session_state.get("clip", {})
        n = clip.get("n")
        if n is None:
            st.warning("Video not processed. Returning to upload.")
            st.session_state["stage"] = "upload"
            st.rerun()
            return
        
        keyframes = get_keyframes(n)
        events = st.session_state.get("events", {})
        
        # Check keyframes for FFP/RELEASE (MKL removed)
        # Only reset to set_frames if not in results stage or results don't exist
        if (keyframes["ffp_idx"] is None or 
            keyframes["release_idx"] is None):
            if st.session_state.get("app_stage") != "results" or not st.session_state.get("results"):
                st.warning("Please set all key frames (FFP, RELEASE) before running analysis.")
                st.session_state["stage"] = "set_frames"
                st.rerun()
                return
        
        # Show warnings if any
        for warning in keyframes["warnings"]:
            st.warning(warning)
        
        render_results()
    else:
        st.session_state["stage"] = "upload"
        st.rerun()


if __name__ == "__main__":
    main()
