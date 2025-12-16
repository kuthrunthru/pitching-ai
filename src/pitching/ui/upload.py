"""
Upload stage UI rendering.
"""

import streamlit as st
import os
import tempfile
import hashlib


def render_upload():
    """Render the upload stage UI."""
    st.header("Upload")
    uploaded = st.file_uploader("Upload pitching video", type=["mp4", "mov", "avi"])
    if uploaded is None:
        return

    # Read uploaded file bytes
    data = uploaded.getvalue()
    
    # Compute video hash
    video_hash = hashlib.md5(data).hexdigest()
    
    # Check if this is a new video
    if st.session_state.get("video_hash") != video_hash:
        # Create stable temp directory
        tmp_dir = os.path.join(tempfile.gettempdir(), "pitch_app")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Save file to stable temp path
        tmp_path = os.path.join(tmp_dir, f"clip_{video_hash}.mp4")
        with open(tmp_path, "wb") as f:
            f.write(data)
        
        # Store path in clip
        st.session_state["clip"]["tmp_path"] = tmp_path
        
        # Reset events
        st.session_state["events"]["FFP"] = None
        st.session_state["events"]["RELEASE"] = None
        st.session_state["events"]["TORSO_OPEN"] = None
        
        # Reset manual override flags
        st.session_state["manual_override"]["FFP"] = False
        st.session_state["manual_override"]["RELEASE"] = False
        
        # Store video hash
        st.session_state["video_hash"] = video_hash
    
    # Advance to set_frames stage (only if not in results stage or results don't exist)
    if st.session_state.get("app_stage") != "results" or not st.session_state.get("results"):
        st.session_state["stage"] = "set_frames"
        st.success("Uploaded. Moving to frame selection…")
        st.rerun()

