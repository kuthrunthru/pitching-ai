"""
Set Frames stage UI - manual frame selection for FFP, RELEASE.
"""

import streamlit as st
import numpy as np
from src.video.preview import show_preview_frame
from core.keyframes import get_keyframes, set_keyframe_ffp, set_keyframe_release, revert_keyframe_to_auto, try_set_auto_release
from events.release_detection import detect_release_idx
from events.landmark_helpers import extract_arm_landmarks, extract_both_wrists


def validate_keyframes(n: int):
    """
    Validate and enforce monotonic keyframes (FFP < Release).
    
    - Clamps ffp_idx and release_idx into [0, n-1] if present
    - Enforces release_idx >= ffp_idx + 6 when both exist:
        - If source["release"]=="auto": shift release_idx up to ffp_idx+6 (clamp)
        - If source["release"]=="user": do not shift; instead set release_validated=False
          and set release_debug["reason"]="release_before_ffp"
    - Writes results back into st.session_state["keyframes"]
    """
    if "keyframes" not in st.session_state:
        return
    
    keyframes = st.session_state["keyframes"]
    
    # Clamp ffp_idx
    if keyframes.get("ffp_idx") is not None:
        ffp_idx = int(np.clip(int(keyframes["ffp_idx"]), 0, n - 1))
        keyframes["ffp_idx"] = ffp_idx
    else:
        ffp_idx = None
    
    # Clamp release_idx
    if keyframes.get("release_idx") is not None:
        release_idx = int(np.clip(int(keyframes["release_idx"]), 0, n - 1))
        keyframes["release_idx"] = release_idx
    else:
        release_idx = None
    
    # Enforce release_idx >= ffp_idx + 6 when both exist
    if ffp_idx is not None and release_idx is not None:
        source = keyframes.get("source", {}).get("release", "auto")
        
        if release_idx < ffp_idx + 6:
            if source == "auto":
                # Shift release_idx up to ffp_idx + 6 (clamp to n-1)
                new_release_idx = int(np.clip(ffp_idx + 6, 0, n - 1))
                keyframes["release_idx"] = new_release_idx
                # Update debug if exists
                if "release_debug" in keyframes:
                    keyframes["release_debug"]["reason"] = "shifted_to_ffp_plus_6"
            else:
                # User override: do not shift, mark as invalid
                keyframes["release_validated"] = False
                if "release_debug" not in keyframes:
                    keyframes["release_debug"] = {}
                keyframes["release_debug"]["reason"] = "release_before_ffp"
        else:
            # Valid: ensure release_validated is True
            keyframes["release_validated"] = True


def render_set_frames():
    """
    Requires st.session_state["clip"] to be populated.
    Lets user set FFP, RELEASE with sliders and preview frames.
    Writes to st.session_state["events"] and advances to results.
    """
    def clamp_idx(x, n):
        """Safely clamp an index to [0, n-1], returning None if input is invalid."""
        if x is None:
            return None
        try:
            xi = int(x)
        except (TypeError, ValueError):
            return None
        if n is None:
            return None
        return max(0, min(int(n) - 1, xi))
    
    def safe_slider_value(x, default, n):
        """Returns an int in [0, n-1] for use in Streamlit sliders."""
        if n is None or n <= 0:
            return 0
        if x is None:
            return max(0, min(n - 1, int(default)))
        try:
            xi = int(x)
        except (TypeError, ValueError):
            xi = int(default)
        return max(0, min(n - 1, xi))
    
    clip = st.session_state.get("clip", {})
    events = st.session_state.get("events", {})
    
    # Guard: ensure clip is fully populated
    tmp_path = clip.get("tmp_path")
    lms_arr = clip.get("lms_arr")
    n = clip.get("n")
    fps = clip.get("fps", 30.0)
    
    if not tmp_path or lms_arr is None or n is None:
        # Show error but don't return early - allow button to render
        missing_items = []
        if not tmp_path:
            missing_items.append("tmp_path")
        if lms_arr is None:
            missing_items.append("lms_arr")
        if n is None:
            missing_items.append("n")
        reason = f"missing clip data: {', '.join(missing_items)}"
        st.error(f"Missing required clip data: {reason}")
    
    # Get keyframes (validated)
    keyframes = get_keyframes(n)
    ffp_idx = keyframes["ffp_idx"]
    release_idx = keyframes["release_idx"]
    ffp_source = keyframes["source"]["ffp"]
    
    # Read sources directly from session state to check for manual overrides
    release_source = st.session_state["keyframes"].get("source", {}).get("release", "auto")
    
    # --- Determine throwing_side (auto-detect once, then respect user overrides) ---
    # Primary source of truth for the app is st.session_state["throwing_side"], which is:
    # - Set once from auto-detection on first render
    # - Updated when the user changes the radio selector
    # - NOT overwritten by auto-detection on reruns
    throwing_side = st.session_state.get("throwing_side")
    if throwing_side is None and ffp_idx is not None:
        # Get cached pose data (or compute if not cached)
        cache_key = clip.get("cache_key")
        if cache_key:
            from src.pose.cache import get_or_compute_pose
            pose_cache = get_or_compute_pose(
                cache_key=cache_key,
                tmp_path=tmp_path,
                n=n,
                fps=fps,
                process_width=clip.get("process_width"),
                process_height=clip.get("process_height"),
                lms_arr=lms_arr,
                pose_quality=clip.get("pose_quality", []),
                mode="metrics",  # Full resolution for throwing side selection
            )
            
            # Extract from cache
            pose_q = pose_cache["pose_q"]
            wrist_xy_L = pose_cache["wrist_xy_L"]
            wrist_xy_R = pose_cache["wrist_xy_R"]
        else:
            # Fallback to direct extraction (shouldn't happen if cache_key is set)
            process_width = clip.get("process_width")
            process_height = clip.get("process_height")
            pose_q = np.array(clip.get("pose_quality", [0.0] * n), dtype=np.float32)
            wrist_xy_L, wrist_xy_R = extract_both_wrists(
                lms_arr,
                n=n,
                width=process_width,
                height=process_height,
            )
        
        # Pick throwing side if not already set
        if wrist_xy_L is not None and wrist_xy_R is not None:
            from src.pose.arm_selection import pick_throwing_side
            throwing_side, side_debug = pick_throwing_side(
                wrist_xy_L=wrist_xy_L,
                wrist_xy_R=wrist_xy_R,
                pose_q=pose_q,
                ffp_idx=ffp_idx,
                release_idx_hint=None,
                min_q=0.35,
                n=n,
            )
            # Persist throwing side in session state (auto default)
            if "keyframes" not in st.session_state:
                st.session_state["keyframes"] = {}
            st.session_state["keyframes"]["throwing_side"] = throwing_side
            st.session_state["keyframes"]["throwing_side_debug"] = side_debug
            # Only initialize the user-facing throwing_side once from auto
            if "throwing_side" not in st.session_state:
                st.session_state["throwing_side"] = throwing_side
    
    # Prominent throwing hand selection (Right / Left) bound directly to st.session_state["throwing_side"]
    st.markdown("## Identify Pitching Hand")
    # Map current throwing_side ("R"/"L") to radio index
    current_side = st.session_state.get("throwing_side", "R")
    current_side = (current_side or "R").upper()
    if current_side == "L":
        default_idx = 1
    else:
        default_idx = 0
    hand_choice = st.radio(
        "",
        ["Right", "Left"],
        index=default_idx,
        horizontal=True,
    )
    # Bind user choice back into session_state["throwing_side"] ("R"/"L") and keep keyframes copy in sync
    st.session_state["throwing_side"] = "R" if hand_choice == "Right" else "L"
    if "keyframes" not in st.session_state:
        st.session_state["keyframes"] = {}
    st.session_state["keyframes"]["throwing_side"] = st.session_state["throwing_side"]
    
    # Auto-detect release ONLY if source is "auto" (respect manual override)
    if release_source != "user" and ffp_idx is not None:
        # Get cached pose data (or compute if not cached)
        cache_key = clip.get("cache_key")
        if cache_key:
            from src.pose.cache import get_or_compute_pose
            pose_cache = get_or_compute_pose(
                cache_key=cache_key,
                tmp_path=tmp_path,
                n=n,
                fps=fps,
                process_width=clip.get("process_width"),
                process_height=clip.get("process_height"),
                lms_arr=lms_arr,
                pose_quality=clip.get("pose_quality", []),
                mode="metrics",  # Full resolution for release detection
            )
            
            # Extract from cache
            pose_q = pose_cache["pose_q"]
            wrist_xy_L = pose_cache["wrist_xy_L"]
            wrist_xy_R = pose_cache["wrist_xy_R"]
        else:
            # Fallback to direct extraction (shouldn't happen if cache_key is set)
            process_width = clip.get("process_width")
            process_height = clip.get("process_height")
            pose_q = np.array(clip.get("pose_quality", [0.0] * n), dtype=np.float32)
            wrist_xy_L, wrist_xy_R = extract_both_wrists(
                lms_arr,
                n=n,
                width=process_width,
                height=process_height,
            )
        
        # Get throwing side from session state (should already be set above)
        throwing_side = st.session_state.get("keyframes", {}).get("throwing_side")
        
        # Get arm landmarks from cache
        if cache_key and "pose_cache" in st.session_state and cache_key in st.session_state["pose_cache"]:
            pose_cache = st.session_state["pose_cache"][cache_key]
            if throwing_side.upper() == "R":
                wrist_xy = pose_cache["wrist_xy_R_arm"]
                elbow_xy = pose_cache["elbow_xy_R"]
                shoulder_xy = pose_cache["shoulder_xy_R"]
            else:
                wrist_xy = pose_cache["wrist_xy_L_arm"]
                elbow_xy = pose_cache["elbow_xy_L"]
                shoulder_xy = pose_cache["shoulder_xy_L"]
        else:
            # Fallback to direct extraction
            wrist_xy, elbow_xy, shoulder_xy, valid_mask = extract_arm_landmarks(
                lms_arr, 
                hand=throwing_side, 
                n=n,
                width=clip.get("process_width"),
                height=clip.get("process_height"),
            )
        
        if wrist_xy is not None and elbow_xy is not None and shoulder_xy is not None:
            detected_release, release_debug = detect_release_idx(
                wrist_xy_L=wrist_xy_L,
                wrist_xy_R=wrist_xy_R,
                elbow_xy=elbow_xy,
                shoulder_xy=shoulder_xy,
                pose_q=pose_q,
                fps=fps,
                ffp_idx=ffp_idx,
                n=n,
                throwing_side=throwing_side,  # Pass pre-chosen side
                min_q=0.35,
            )
            
            # B) When auto-detection runs: overwrite release_debug with newest debug dict every run
            # Store debug info in keyframes (always, even if detection returns None)
            if "keyframes" not in st.session_state:
                st.session_state["keyframes"] = {}
            st.session_state["keyframes"]["release_debug"] = release_debug
            
            if detected_release is not None:
                # Try to set via auto-detection (will be blocked if user override exists)
                if try_set_auto_release(detected_release):
                    # Update release_idx from keyframes
                    keyframes = get_keyframes(n)
                    release_idx = keyframes["release_idx"]
            # If detection returns None, do NOT change an existing user release_idx
            # (debug dict already stored above)
    
    # Show warnings if any
    for warning in keyframes["warnings"]:
        st.warning(warning)
    
    # Use keyframes for FFP/RELEASE defaults, fallback to legacy events
    if ffp_idx is None:
        ffp_idx = events.get("FFP", int(0.55 * n))
    
    # Compute safe fallback for Release (guarded against None ffp_idx)
    fps_safe = fps or 30
    if release_idx is None:
        # Guard: if ffp_idx is None, use time-based fallback
        if ffp_idx is None:
            fallback_release = min(n - 1, int(0.45 * fps_safe))
        else:
            # ffp_idx is valid, use FFP-based fallback
            fallback_release = min(n - 1, ffp_idx + int(0.45 * fps_safe))
        
        release_idx = events.get("RELEASE", fallback_release)
    
    # Clamp all to [0, n-1] using safe helper
    ffp_idx = clamp_idx(ffp_idx, n)
    release_idx = clamp_idx(release_idx, n)

    # --- Initialize per-event slider state once from auto-detected keyframes ---
    # These keys drive the sliders directly so the knob always matches the current frame *on first load*.
    # On subsequent reruns, the slider (session_state) is treated as the source of truth and is not overwritten.
    # FFP default: ~0.5s into clip if unknown
    ffp_default = min(n - 1, int(0.50 * (fps or 30)))
    if "ffp_idx" not in st.session_state:
        st.session_state["ffp_idx"] = int(ffp_idx) if ffp_idx is not None else ffp_default

    # RELEASE default: based on FFP if available, else time-based
    if ffp_idx is not None:
        release_default = min(n - 1, ffp_idx + int(0.45 * (fps or 30)))
    else:
        release_default = min(n - 1, int(0.45 * (fps or 30)))
    if "release_idx" not in st.session_state:
        st.session_state["release_idx"] = int(release_idx) if release_idx is not None else release_default
    
    # FFP slider
    st.subheader("Identify the exact frame where the lead foot first hits the ground.")

    # Slider bound to 'ffp_idx' so it stays in sync with auto/user selection
    ffp = st.slider(
        "",
        0,
        n - 1,
        key="ffp_idx",
    )
    
    # Revert button (only show if user has manually set it)
    if ffp_source == "user":
        if st.button("Revert to Auto", key="ffp_revert_btn"):
            revert_keyframe_to_auto("ffp")
            st.info("FFP reverted to auto-detection")
            st.rerun()
    
    # Preview FFP frame - always read from slider state
    ffp_display = st.session_state.get("ffp_idx", ffp)
    show_preview_frame(
        video_path=tmp_path,
        frame_idx=int(ffp_display),
        max_w=500,
        caption=None,
        width=400
    )
    
    st.divider()
    
    # RELEASE slider
    st.subheader("Identify the exact frame where the ball leaves the pitcher's hand.")

    # Slider bound to 'release_idx' so it stays in sync with auto/user selection
    rel = st.slider(
        "",
        0,
        n - 1,
        key="release_idx",
    )
    
    # Revert button (only show if user has manually set it)
    if release_source == "user":
        if st.button("Revert to Auto", key="release_revert_btn"):
            revert_keyframe_to_auto("release")
            st.info("Release reverted to auto-detection")
            st.rerun()
    
    # Preview RELEASE frame - always read from slider state
    rel_display = st.session_state.get("release_idx", rel)
    show_preview_frame(
        video_path=tmp_path,
        frame_idx=int(rel_display),
        max_w=500,
        caption=None,
        width=400
    )
    
    # Update keyframes from sliders immediately (for auto mode, slider changes are authoritative)
    # For user mode, slider changes are preview-only until Apply is clicked
    if ffp_source == "auto":
        set_keyframe_ffp(int(ffp), source="auto")
    
    if release_source == "auto":
        set_keyframe_release(int(rel), source="auto")
    
    st.divider()
    
    # Apply button - immediately saves all slider values
    if st.button("Apply & Run Analysis", use_container_width=True, type="primary", key="run_analysis_btn"):
        st.session_state["analysis_requested"] = True
        st.rerun()
    
    # Handle analysis request on rerun (after flag is set)
    if st.session_state.get("analysis_requested"):
        # One-shot latch and route using app_stage
        if st.session_state.get("analysis_run_id") is None:
            import time
            st.session_state["analysis_run_id"] = time.time()
            st.session_state["app_stage"] = "results"
            st.session_state["stage"] = "results"  # Also set stage for app router
            
            # Get current slider values (these are the frames the user sees)
            ffp_current = int(st.session_state.get("ffp_idx", ffp))
            rel_current = int(st.session_state.get("release_idx", rel))
            
            # Save FFP
            set_keyframe_ffp(ffp_current, source="user")
            
            # Save Release
            set_keyframe_release(rel_current, source="user")
            if "keyframes" not in st.session_state:
                st.session_state["keyframes"] = {}
            st.session_state["keyframes"]["release_debug"] = {
                "source": "user",
                "reason": "manual_pick"
            }
            st.session_state["keyframes"]["release_validated"] = True
            
            # Update legacy events dict for backward compatibility
            events["FFP"] = ffp_current
            events["RELEASE"] = rel_current
            
            # Validate keyframes before proceeding
            validate_keyframes(n)
            
            # Get keyframes and pose cache for analysis
            keyframes = get_keyframes(n)
            
            # Get pose cache if available
            cache_key = clip.get("cache_key")
            pose_cache = None
            if cache_key and "pose_cache" in st.session_state:
                pose_cache = st.session_state["pose_cache"].get(cache_key)
            
            # Run analysis
            with st.spinner("Running analysis..."):
                from src.metrics.compute import compute_all_metrics
                st.session_state["results"] = compute_all_metrics(clip, keyframes, pose_cache=pose_cache)
            
            st.rerun()
    
    # Validate keyframes after any updates (at end of function)
    validate_keyframes(n)

