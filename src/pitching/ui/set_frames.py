"""
Set Frames stage UI rendering.
"""

import streamlit as st


def render_set_frames():
    """Render the set frames stage UI."""
    st.header("Set Frames (stub)")
    st.write("FFP:", st.session_state["events"]["FFP"])
    st.write("RELEASE:", st.session_state["events"]["RELEASE"])

    if st.button("Back to Upload"):
        st.session_state["stage"] = "upload"
        st.rerun()

