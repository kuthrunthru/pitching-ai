"""
Keyframe management - authoritative source for FFP and Release frame indices.

Provides helpers to get, set, and validate keyframes with user override protection.
"""

import streamlit as st
import numpy as np


def get_keyframes(n: int):
    """
    Get validated keyframe indices and sources.
    
    Args:
        n: Total number of frames in video
    
    Returns:
        dict with:
        - ffp_idx: int or None (validated, clamped to [0, n-1])
        - release_idx: int or None (validated, must be > ffp_idx + 3, clamped to [0, n-1])
        - source: dict with "ffp" and "release" keys ("auto" or "user")
        - warnings: list of warning messages if validation fails
    """
    keyframes = st.session_state.get("keyframes", {})
    ffp_idx = keyframes.get("ffp_idx")
    release_idx = keyframes.get("release_idx")
    source = keyframes.get("source", {"ffp": "auto", "release": "auto"})
    
    warnings = []
    
    # Validate and clamp FFP
    if ffp_idx is not None:
        ffp_idx = int(np.clip(int(ffp_idx), 0, n - 1))
        if ffp_idx != keyframes.get("ffp_idx"):
            warnings.append(f"FFP index clamped to {ffp_idx} (was {keyframes.get('ffp_idx')})")
    else:
        ffp_idx = None
    
    # Validate and clamp Release
    release_validated = keyframes.get("release_validated", True)
    if release_idx is not None:
        release_idx = int(np.clip(int(release_idx), 0, n - 1))
        if release_idx != keyframes.get("release_idx"):
            warnings.append(f"Release index clamped to {release_idx} (was {keyframes.get('release_idx')})")
        
        # Guard: Release must be >= FFP + 6
        if ffp_idx is not None and release_idx < ffp_idx + 6:
            release_source = source.get("release", "auto")
            if release_source == "user":
                # User override: do not change index, but mark as invalid
                release_validated = False
                warnings.append(f"Release index ({release_idx}) must be >= FFP ({ffp_idx}) + 6. Marked as invalid.")
                # Update session state
                if "keyframes" not in st.session_state:
                    st.session_state["keyframes"] = {}
                st.session_state["keyframes"]["release_validated"] = False
                if "release_debug" not in st.session_state["keyframes"]:
                    st.session_state["keyframes"]["release_debug"] = {}
                st.session_state["keyframes"]["release_debug"]["reason"] = "release_before_ffp_plus_6"
            else:
                # Auto: set to None (will be re-detected)
                warnings.append(f"Release index ({release_idx}) must be >= FFP ({ffp_idx}) + 6. Setting to None.")
                release_idx = None
        else:
            # Valid: ensure release_validated is True
            if ffp_idx is not None:
                release_validated = True
                if "keyframes" in st.session_state:
                    st.session_state["keyframes"]["release_validated"] = True
    else:
        # No release_idx: default to validated
        release_validated = True
    
    # MKL removed - no longer validated
    
    # Resolve throwing side with optional override
    throwing_side_override = st.session_state.get("throwing_side_override", "Auto")
    stored_throwing_side = keyframes.get("throwing_side")
    if throwing_side_override == "Right":
        resolved_throwing_side = "R"
        throwing_side_source = "override"
        # Persist resolved side so downstream uses the same value
        st.session_state["keyframes"]["throwing_side"] = resolved_throwing_side
    elif throwing_side_override == "Left":
        resolved_throwing_side = "L"
        throwing_side_source = "override"
        st.session_state["keyframes"]["throwing_side"] = resolved_throwing_side
    else:
        resolved_throwing_side = stored_throwing_side
        throwing_side_source = "auto" if stored_throwing_side else None

    return {
        "ffp_idx": ffp_idx,
        "release_idx": release_idx,
        "source": source,
        "warnings": warnings,
        "release_validated": release_validated,
        "throwing_side": resolved_throwing_side,
        "throwing_side_source": throwing_side_source,
    }


def set_keyframe_ffp(idx: int, source: str = "user"):
    """
    Set FFP keyframe with source tracking.
    
    Args:
        idx: Frame index for FFP
        source: "user" or "auto" (default "user")
    """
    if "keyframes" not in st.session_state:
        st.session_state["keyframes"] = {
            "ffp_idx": None,
            "release_idx": None,
            "source": {"ffp": "auto", "release": "auto"},
        }
    
    st.session_state["keyframes"]["ffp_idx"] = int(idx)
    st.session_state["keyframes"]["source"]["ffp"] = source
    
    # Mirror to legacy events for backward compatibility
    st.session_state["events"]["FFP"] = int(idx)
    st.session_state["manual_override"]["FFP"] = (source == "user")


def set_keyframe_release(idx: int, source: str = "user"):
    """
    Set Release keyframe with source tracking.
    
    Args:
        idx: Frame index for Release
        source: "user" or "auto" (default "user")
    """
    if "keyframes" not in st.session_state:
        st.session_state["keyframes"] = {
            "ffp_idx": None,
            "release_idx": None,
            "source": {"ffp": "auto", "release": "auto"},
        }
    
    st.session_state["keyframes"]["release_idx"] = int(idx)
    st.session_state["keyframes"]["source"]["release"] = source
    
    # Mirror to legacy events for backward compatibility
    st.session_state["events"]["RELEASE"] = int(idx)
    st.session_state["manual_override"]["RELEASE"] = (source == "user")


# MKL functions removed - MKL keyframe no longer used


def try_set_auto_ffp(idx: int):
    """
    Attempt to set FFP via auto-detection. Only succeeds if source is "auto".
    
    Args:
        idx: Auto-detected frame index
    
    Returns:
        bool: True if set, False if blocked by user override
    """
    keyframes = st.session_state.get("keyframes", {})
    source = keyframes.get("source", {}).get("ffp", "auto")
    
    if source == "user":
        # Track that auto-detection was blocked
        if "auto_detection_blocked" not in st.session_state:
            st.session_state["auto_detection_blocked"] = {"ffp_blocked": False, "release_blocked": False}
        st.session_state["auto_detection_blocked"]["ffp_blocked"] = True
        return False  # Blocked by user override
    
    set_keyframe_ffp(idx, source="auto")
    # Track that auto-detection succeeded
    if "auto_detection_blocked" not in st.session_state:
        st.session_state["auto_detection_blocked"] = {"ffp_blocked": False, "release_blocked": False}
    st.session_state["auto_detection_blocked"]["ffp_blocked"] = False
    return True


def try_set_auto_release(idx: int):
    """
    Attempt to set Release via auto-detection. Only succeeds if source is "auto".
    Does nothing if keyframes["source"]["release"] == "user".
    
    Args:
        idx: Auto-detected frame index
    
    Returns:
        bool: True if set, False if blocked by user override
    """
    keyframes = st.session_state.get("keyframes", {})
    source = keyframes.get("source", {}).get("release", "auto")
    
    # Do nothing if user override exists
    if source == "user":
        # Track that auto-detection was blocked
        if "auto_detection_blocked" not in st.session_state:
            st.session_state["auto_detection_blocked"] = {"ffp_blocked": False, "release_blocked": False}
        st.session_state["auto_detection_blocked"]["release_blocked"] = True
        return False  # Blocked by user override
    
    # Write the detected value
    set_keyframe_release(idx, source="auto")
    # Track that auto-detection succeeded
    if "auto_detection_blocked" not in st.session_state:
        st.session_state["auto_detection_blocked"] = {"ffp_blocked": False, "release_blocked": False}
    st.session_state["auto_detection_blocked"]["release_blocked"] = False
    return True


def revert_keyframe_to_auto(keyframe_type: str):
    """
    Revert a keyframe to auto-detection mode.
    
    Args:
        keyframe_type: "ffp" or "release"
    """
    if "keyframes" not in st.session_state:
        return
    
    if keyframe_type == "ffp":
        st.session_state["keyframes"]["ffp_idx"] = None
        st.session_state["keyframes"]["source"]["ffp"] = "auto"
        st.session_state["events"]["FFP"] = None
        st.session_state["manual_override"]["FFP"] = False
    elif keyframe_type == "release":
        st.session_state["keyframes"]["release_idx"] = None
        st.session_state["keyframes"]["source"]["release"] = "auto"
        st.session_state["events"]["RELEASE"] = None
        st.session_state["manual_override"]["RELEASE"] = False
        # Clear release_debug and reset validation
        if "release_debug" in st.session_state["keyframes"]:
            del st.session_state["keyframes"]["release_debug"]
        st.session_state["keyframes"]["release_validated"] = True  # Reset to True (will be validated on next check)

