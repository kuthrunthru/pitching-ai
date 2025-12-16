"""
Auto-detection guard helpers - prevent overwriting user-set keyframes.

Example usage in auto-detection code:

    from core.auto_detection_guard import try_set_auto_ffp, try_set_auto_release
    
    # Auto-detect FFP
    detected_ffp = detect_foot_strike(...)
    if try_set_auto_ffp(detected_ffp):
        st.session_state["auto_detection_blocked"]["ffp_blocked"] = False
    else:
        st.session_state["auto_detection_blocked"]["ffp_blocked"] = True
        # Log that auto-detection was blocked
"""

import streamlit as st
from core.keyframes import try_set_auto_ffp, try_set_auto_release


def init_auto_detection_blocked():
    """Initialize tracking for auto-detection blocking."""
    if "auto_detection_blocked" not in st.session_state:
        st.session_state["auto_detection_blocked"] = {
            "ffp_blocked": False,
            "release_blocked": False,
        }


# Re-export for convenience
__all__ = ["try_set_auto_ffp", "try_set_auto_release", "init_auto_detection_blocked"]

