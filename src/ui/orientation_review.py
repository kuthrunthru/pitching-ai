"""
Orientation Review stage UI - allows user to confirm/adjust video orientation.
"""

import streamlit as st
from src.video.preview import show_preview_frame


def render_orientation_review():
    """
    Shows a preview frame and allows user to confirm or adjust orientation.
    Only advances to set_frames when user clicks "Looks Good - Continue".
    """
    clip = st.session_state.get("clip", {})
    
    # Guard: ensure clip is fully populated
    tmp_path = clip.get("tmp_path")
    n = clip.get("n")
    
    if not tmp_path or n is None:
        st.warning("Video not processed. Returning to upload.")
        st.session_state["stage"] = "upload"
        st.rerun()
        return
    
    # Initialize orientation state - SINGLE SOURCE OF TRUTH
    st.session_state.setdefault("preview_orientation_mode", "None")
    st.session_state.setdefault("preview_needs_flip", False)
    st.session_state.setdefault("orientation_rev", 0)
    st.session_state.setdefault("orientation_click_msg", "")
    
    st.header("Orientation Review")
    st.write("**Is the pitcher upright (head at top)?**")
    
    # Use middle frame for preview
    preview_frame_idx = n // 2
    
    # Handle button clicks BEFORE showing the preview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Looks Good - Continue", use_container_width=True, type="primary", key="rot_ok"):
            # Reprocess video with final chosen_mode to ensure landmarks match orientation
            from src.video.process import process_video_landmarks
            
            with st.spinner("Reprocessing video with selected orientation..."):
                processed = process_video_landmarks(
                    video_path=tmp_path,
                    conf_thresh=0.4,
                    max_process_width=1280,
                    model_complexity=1,
                    smooth_landmarks=True,
                    chosen_mode=st.session_state["preview_orientation_mode"],
                )
                
                # Update landmarks with reprocessed data
                st.session_state["clip"]["lms_arr"] = processed["lms_arr"]
                st.session_state["clip"]["pose_quality"] = processed["pose_quality"]
            
            # Advance to set_frames (only if not in results stage or results don't exist)
            if st.session_state.get("app_stage") != "results" or not st.session_state.get("results"):
                st.session_state["stage"] = "set_frames"
                st.rerun()
    
    with col2:
        if st.button("Rotate 90 deg CW", use_container_width=True, key="rot_cw"):
            st.session_state["preview_orientation_mode"] = "90 deg CW"
            st.session_state["orientation_click_msg"] = "Orientation set to 90 deg CW"
            st.session_state["orientation_rev"] = st.session_state["orientation_rev"] + 1
            st.rerun()
    
    with col3:
        if st.button("Rotate 180 deg", use_container_width=True, key="rot_180"):
            st.session_state["preview_orientation_mode"] = "180 deg"
            st.session_state["orientation_click_msg"] = "Orientation set to 180 deg"
            st.session_state["orientation_rev"] = st.session_state["orientation_rev"] + 1
            st.rerun()
    
    with col4:
        if st.button("Rotate 270 deg CW", use_container_width=True, key="rot_270"):
            st.session_state["preview_orientation_mode"] = "270 deg CW"
            st.session_state["orientation_click_msg"] = "Orientation set to 270 deg CW"
            st.session_state["orientation_rev"] = st.session_state["orientation_rev"] + 1
            st.rerun()
    
    # Show success message right above the preview
    if st.session_state["orientation_click_msg"]:
        st.success(st.session_state["orientation_click_msg"])
    
    # Only after the click-handling block, show the preview
    show_preview_frame(
        video_path=tmp_path,
        frame_idx=preview_frame_idx,
        max_w=900,
        caption=None,
        width=None,
        rev=st.session_state["orientation_rev"]
    )

