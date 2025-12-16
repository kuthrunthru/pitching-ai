"""
Clean rebuild entrypoint.

This is a minimal Streamlit app to prove the environment works.
It demonstrates session state initialization and basic UI rendering.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*use_column_width.*deprecated.*",
    category=DeprecationWarning,
)

import streamlit as st
from session_state import init_state
from metrics.trunk_tilt import (
    compute_forward_trunk_tilt_at_release_deg,
    score_trunk_tilt_release_deg,
)

st.set_page_config(
    layout="wide",
    page_title="Pitch Mechanics Analyzer (Rebuild)"
)

# Debug: Show which file is executing
st.error("RUNNING FILE: " + __file__)

st.title("Pitch Mechanics Analyzer (Rebuild)")

# Initialize session state
init_state()


def render_analyze():
    """Render the analyze stage UI."""
    st.header("Analyze stage")
    if st.button("Go to Set Frames"):
        st.session_state["stage"] = "set_frames"
        st.rerun()


def render_set_frames():
    """Render the set frames stage UI."""
    st.header("Set Frames stage")
    if st.button("Back to Analyze"):
        st.session_state["stage"] = "analyze"
        st.rerun()
    if st.button("Go to Results"):
        st.session_state["stage"] = "results"
        st.rerun()


def render_results():
    """Render the results stage UI."""
    st.header("Results stage")
    
    # Example usage of trunk tilt functions
    # In a real implementation, these would use actual data from session state
    clip = st.session_state.get("clip", {})
    events = st.session_state.get("events", {})
    
    # Check if we have the necessary data
    release_idx = events.get("RELEASE")
    n = clip.get("n")
    fps = clip.get("fps", 30.0)
    pose_quality = clip.get("pose_quality")
    
    if release_idx is not None and n is not None and pose_quality is not None:
        # Define helper functions for shoulder_mid and pelvis_mid
        # These would normally be computed from lms_arr
        def dummy_shoulder_mid(t):
            """Placeholder - would compute from landmarks."""
            return None
        
        def dummy_pelvis_mid(t):
            """Placeholder - would compute from landmarks."""
            return None
        
        # Call trunk tilt computation (once)
        tilt_deg, debug_dict = compute_forward_trunk_tilt_at_release_deg(
            release_idx=release_idx,
            shoulder_mid=dummy_shoulder_mid,
            pelvis_mid=dummy_pelvis_mid,
            pose_quality=pose_quality,
            n=n,
            fps=fps,
            plate_dir_sign=1.0,  # Would come from session state
            min_q=0.35,
        )
        
        # Call scoring function (once)
        score, status = score_trunk_tilt_release_deg(tilt_deg)
        
        # Display results
        if tilt_deg is not None:
            st.write(f"Forward Trunk Tilt at Release: {tilt_deg:.1f} deg")
            st.write(f"Score: {score}, Status: {status}")
            st.write(f"Debug: {debug_dict}")
        else:
            st.write("Forward Trunk Tilt: N/A")
    else:
        st.info("Upload a video and set frames to see results.")
    
    if st.button("Back to Set Frames"):
        st.session_state["stage"] = "set_frames"
        st.rerun()


# Stage router
stage = st.session_state.get("stage", "analyze")
if stage == "analyze":
    render_analyze()
elif stage == "set_frames":
    render_set_frames()
elif stage == "results":
    render_results()
else:
    # Invalid stage - reset to analyze
    st.session_state["stage"] = "analyze"
    st.rerun()

# Display session state (for debugging)
with st.expander("🔍 Session State", expanded=False):
    st.json(st.session_state)
