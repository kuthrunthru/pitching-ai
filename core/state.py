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
    
    # Preview orientation - SINGLE SOURCE OF TRUTH
    if "preview_orientation_mode" not in st.session_state:
        st.session_state["preview_orientation_mode"] = "None"
    if "preview_needs_flip" not in st.session_state:
        st.session_state["preview_needs_flip"] = False
    
    # Clip data
    if "clip" not in st.session_state:
        st.session_state["clip"] = {
            "tmp_path": None,
            "lms_arr": None,
            "n": None,
            "fps": 30.0,
            "pose_quality": None,
        }
    
    # Keyframes - SINGLE SOURCE OF TRUTH for FFP, Release, and MKL
    if "keyframes" not in st.session_state:
        st.session_state["keyframes"] = {
            "ffp_idx": None,
            "release_idx": None,
            "mkl_idx": None,
            "source": {
                "ffp": "auto",
                "release": "auto",
                "mkl": "auto",
            },
            "release_validated": True,  # Default to True (will be validated on first check)
            "throwing_side": None,  # "L" or "R" - picked once per clip
        }
    
    # Events (legacy - kept for MKL and TORSO_OPEN)
    if "events" not in st.session_state:
        st.session_state["events"] = {
            "MKL": None,
            "FFP": None,  # Deprecated - use keyframes["ffp_idx"] instead
            "RELEASE": None,  # Deprecated - use keyframes["release_idx"] instead
            "TORSO_OPEN": None,
        }
    
    # Manual overrides (legacy - kept for backward compatibility)
    if "manual_override" not in st.session_state:
        st.session_state["manual_override"] = {
            "FFP": False,
            "RELEASE": False,
        }
    
    # Auto-detection blocking tracking
    if "auto_detection_blocked" not in st.session_state:
        st.session_state["auto_detection_blocked"] = {
            "ffp_blocked": False,
            "release_blocked": False,
        }

    # Throwing side override (Auto / Right / Left)
    if "throwing_side_override" not in st.session_state:
        st.session_state["throwing_side_override"] = "Auto"
    
    # Debug dict
    if "debug" not in st.session_state:
        st.session_state["debug"] = {}


def reset_for_new_video(video_hash: str):
    """
    Reset state for a new video upload.
    
    Args:
        video_hash: Hash of the new video file
    """
    # Reset preview orientation to defaults
    st.session_state["preview_orientation_mode"] = "None"
    st.session_state["preview_needs_flip"] = False
    
    # Reset clip data
    st.session_state["clip"] = {
        "tmp_path": None,
        "lms_arr": None,
        "n": None,
        "fps": 30.0,
        "pose_quality": None,
    }
    
    # Reset keyframes to defaults
    st.session_state["keyframes"] = {
        "ffp_idx": None,
        "release_idx": None,
        "source": {
            "ffp": "auto",
            "release": "auto",
        },
        "release_validated": True,  # Default to True (will be validated on first check)
        "throwing_side": None,  # "L" or "R" - picked once per clip
    }
    
    # Clear pose cache for old video (new video will get new cache_key)
    # Note: We don't clear the entire cache, just let old entries expire naturally
    # If you want to clear a specific key, do: del st.session_state["pose_cache"][old_cache_key]
    
    # Reset events (legacy - kept for MKL and TORSO_OPEN)
    st.session_state["events"] = {
        "MKL": None,
        "FFP": None,  # Deprecated - use keyframes["ffp_idx"] instead
        "RELEASE": None,  # Deprecated - use keyframes["release_idx"] instead
        "TORSO_OPEN": None,
    }
    
    # Reset manual overrides (legacy)
    st.session_state["manual_override"] = {
        "FFP": False,
        "RELEASE": False,
    }
    
    # Reset auto-detection blocking tracking
    st.session_state["auto_detection_blocked"] = {
        "ffp_blocked": False,
        "release_blocked": False,
    }

    # Reset throwing side override
    st.session_state["throwing_side_override"] = "Auto"
    
    # Reset stage and analysis flag
    st.session_state["stage"] = "upload"
    st.session_state["run_analysis"] = False
    
    # Store new video hash
    st.session_state["video_hash"] = video_hash
