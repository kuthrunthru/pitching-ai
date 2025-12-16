"""
Session state management for pitching analysis app.

Provides safe initialization and reset functions for Streamlit session state.
"""

import streamlit as st


def init_state():
    """
    Initialize all session state variables safely.
    Safe to call on every rerun - only initializes if keys don't exist.
    """
    # Stage management
    if "stage" not in st.session_state:
        st.session_state["stage"] = "upload"
    
    if "run_analysis" not in st.session_state:
        st.session_state["run_analysis"] = False
    
    if "video_hash" not in st.session_state:
        st.session_state["video_hash"] = None
    
    # Clip data
    if "clip" not in st.session_state:
        st.session_state["clip"] = {
            "tmp_path": None,
            "chosen_mode": "None",
            "needs_flip": False,
            "lms_arr": None,
            "n": None,
            "fps": 30.0,
            "pose_quality": None,
        }
    
    # Events
    if "events" not in st.session_state:
        st.session_state["events"] = {
            "MKL": None,
            "FFP": None,
            "RELEASE": None,
            "TORSO_OPEN": None,
        }
    
    # Manual overrides
    if "manual_override" not in st.session_state:
        st.session_state["manual_override"] = {
            "FFP": False,
            "RELEASE": False,
        }
    
    # Debug dict
    if "debug" not in st.session_state:
        st.session_state["debug"] = {}


def reset_for_new_video(video_hash):
    """
    Reset state for a new video upload.
    
    Args:
        video_hash: Hash of the new video file
    """
    # Reset clip data
    st.session_state["clip"] = {
        "tmp_path": None,
        "chosen_mode": "None",
        "needs_flip": False,
        "lms_arr": None,
        "n": None,
        "fps": 30.0,
        "pose_quality": None,
    }
    
    # Reset events
    st.session_state["events"] = {
        "MKL": None,
        "FFP": None,
        "RELEASE": None,
        "TORSO_OPEN": None,
    }
    
    # Reset manual overrides
    st.session_state["manual_override"] = {
        "FFP": False,
        "RELEASE": False,
    }
    
    # Reset stage and analysis flag
    st.session_state["stage"] = "upload"
    st.session_state["run_analysis"] = False
    
    # Store new video hash
    st.session_state["video_hash"] = video_hash

