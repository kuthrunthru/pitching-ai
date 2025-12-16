"""
Metrics computation entrypoint.

Single function to compute all metrics from keyframes and pose data.
Orchestrates calls to modular metric computation modules.
"""

import time
import streamlit as st
from src.metrics.release import compute_release_metrics
from src.metrics.lower_body import compute_lower_body_metrics
from src.metrics.posture import compute_posture_metrics
from src.metrics.rotation import compute_rotation_metrics
from src.metrics.registry import EXPECTED_METRICS, METRIC_META
from src.metrics.utils import pack_metric
from src.metrics.interpretation import interpret_metric

# Runtime fingerprint for verifying edited code is executing
BUILD_ID = "CURSOR_BUILD_2025-12-14_1"


def compute_all_metrics(clip, keyframes, pose_cache=None):
    """
    Compute all metrics from keyframes and pose data.
    
    Orchestrates calls to modular metric computation modules:
    - release.py: Release Height
    - lower_body.py: Stride Length
    - posture.py: Head Behind Hip Toward Plate, Elbow Height at Foot Strike, Arm Angle at Foot Strike, Ball Location at Foot Strike
    - rotation.py: Chest Closed at Foot Strike (guidance)
    
    Args:
        clip: Dict with video metadata (tmp_path, lms_arr, n, fps, pose_quality, process_width, process_height)
        keyframes: Dict with ffp_idx, release_idx, and other keyframe data
        pose_cache: Optional cached pose data (if None, will be retrieved from session state)
    
    Returns:
        dict: {"meta": {...}, "metrics": {...}, "coverage": {...}} with standardized metric structure
    """
    metrics_dict = {}
    
    # Extract required data
    ffp_idx = keyframes.get("ffp_idx")
    release_idx = keyframes.get("release_idx")
    throwing_side = keyframes.get("throwing_side") or st.session_state.get("keyframes", {}).get("throwing_side")
    
    # Guardrails: require keyframes
    if ffp_idx is None or release_idx is None:
        # Fill in placeholders for all expected metrics
        for metric_key in EXPECTED_METRICS:
            meta = METRIC_META.get(metric_key, {})
            metrics_dict[metric_key] = pack_metric(
                name=meta.get("label", metric_key),
                value=None,
                units=meta.get("units", ""),
                reason="not_implemented",
            )
        return {
            "meta": {
                "ffp_idx": ffp_idx,
                "release_idx": release_idx,
                "throwing_side": throwing_side,
                "computed_at": time.time(),
            },
            "metrics": metrics_dict,
            "coverage": {
                "expected": EXPECTED_METRICS,
                "present": sorted(list(metrics_dict.keys())),
                "missing": [],
                "computed": [],
                "na": [],
                "not_implemented": EXPECTED_METRICS,
            },
        }
    
    # Get pose cache if not provided
    if pose_cache is None:
        base_cache_key = clip.get("cache_key", "")
        if base_cache_key and "pose_cache" in st.session_state:
            pose_cache = st.session_state["pose_cache"].get(base_cache_key)
    
    # Ensure keyframes has throwing_side for modules that need it
    if throwing_side and "throwing_side" not in keyframes:
        keyframes = keyframes.copy()
        keyframes["throwing_side"] = throwing_side
    
    # Compute forward direction from actual motion (store in meta)
    forward_dir = None
    forward_dir_reason = None
    if pose_cache and pose_cache.get("pts") is not None:
        pts = pose_cache["pts"]
        pose_quality = clip.get("pose_quality", [])
        n = clip.get("n")
        
        from src.metrics.utils import compute_forward_dir
        forward_dir, forward_dir_reason = compute_forward_dir(
            pts=pts,
            ffp_idx=ffp_idx,
            throwing_side=throwing_side,
            n=n,
            pose_quality=pose_quality,
        )
    
    # Add forward_dir to keyframes so all metrics can use it
    if forward_dir is not None:
        keyframes = keyframes.copy()
        keyframes["forward_dir"] = forward_dir
        keyframes["forward_dir_reason"] = forward_dir_reason
    
    # Make a copy of keyframes so we can modify it
    keyframes = keyframes.copy() if keyframes else {}
    
    # 1. Release metrics
    try:
        release_metrics = compute_release_metrics(clip, keyframes, pose_cache)
        metrics_dict.update(release_metrics)
    except Exception as e:
        # Release Height error handling (Head Lean removed)
        metrics_dict["release_height"] = pack_metric(
            name="Release Height",
            value=None,
            units="BL",
            reason=f"error: {str(e)}",
        )
    
    # 2. Lower body metrics (Stride Length)
    try:
        lower_body_metrics = compute_lower_body_metrics(clip, keyframes, pose_cache)
        metrics_dict.update(lower_body_metrics)
    except Exception as e:
        # Set error for all lower body metrics
        for metric_key in ["stride_length"]:
            if metric_key not in metrics_dict:
                meta = METRIC_META.get(metric_key, {})
                metrics_dict[metric_key] = pack_metric(
                    name=meta.get("label", metric_key),
                    value=None,
                    units=meta.get("units", ""),
                    reason=f"error: {str(e)}",
                )
    
    # 3. Posture metrics (Head Behind Hip Toward Plate, Elbow Height at Foot Strike, Arm Angle at Foot Strike, Ball Location at Foot Strike)
    try:
        posture_metrics = compute_posture_metrics(clip, keyframes, pose_cache)
        metrics_dict.update(posture_metrics)
    except Exception as e:
        metrics_dict["head_behind_hip"] = pack_metric(
            name="Head Behind Hip Toward Plate",
            value=None,
            units="ratio",
            reason=f"error: {str(e)}",
        )
    
    # 4. Rotation metrics (Chest Closed at Foot Strike - guidance)
    try:
        rotation_metrics = compute_rotation_metrics(clip, keyframes, pose_cache)
        metrics_dict.update(rotation_metrics)
    except Exception as e:
        metrics_dict["chest_closed_ffp_deg"] = pack_metric(
            name="Chest Closed at Foot Strike",
            value=None,
            units="deg",
            reason=f"error: {str(e)}",
        )
    
    # Fill in placeholders for expected metrics that weren't computed
    for metric_key in EXPECTED_METRICS:
        if metric_key not in metrics_dict:
            meta = METRIC_META.get(metric_key, {})
            metrics_dict[metric_key] = pack_metric(
                name=meta.get("label", metric_key),
                value=None,
                units=meta.get("units", ""),
                reason="not_implemented",
            )
    
    # Add interpretation (status + explanation) to all metrics
    for metric_key, metric_data in metrics_dict.items():
        if isinstance(metric_data, dict) and "value" in metric_data:
            value = metric_data.get("value")
            status, explanation = interpret_metric(metric_key, value)
            
            # Only add status/explanation if value is not None
            if value is not None:
                if status:
                    metric_data["interpretation_status"] = status
                if explanation:
                    metric_data["interpretation_explanation"] = explanation

        # Attach build fingerprint to each metric (top-level and debug)
        metric_data["build_id"] = BUILD_ID
        if isinstance(metric_data.get("debug"), dict):
            metric_data["debug"]["build_id"] = BUILD_ID
        else:
            metric_data["debug"] = {"build_id": BUILD_ID}
    
    # Build meta section
    metrics_meta = {
        "ffp_idx": ffp_idx,
        "release_idx": release_idx,
        "throwing_side": throwing_side,
        "forward_dir": forward_dir,
        "forward_dir_reason": forward_dir_reason,
        "computed_at": time.time(),
        "build_id": BUILD_ID,
    }
    
    # Store in session state for UI access
    st.session_state["metrics_meta"] = metrics_meta
    
    # Compute coverage summary
    coverage = {
        "expected": EXPECTED_METRICS,
        "present": sorted(list(metrics_dict.keys())),
        "computed": [
            m for m in EXPECTED_METRICS 
            if m in metrics_dict and metrics_dict[m].get("value") is not None
        ],
        "na": [
            m for m in EXPECTED_METRICS 
            if m in metrics_dict 
            and metrics_dict[m].get("value") is None 
            and metrics_dict[m].get("reason") != "not_implemented"
        ],
        "not_implemented": [
            m for m in EXPECTED_METRICS 
            if m in metrics_dict 
            and metrics_dict[m].get("reason") == "not_implemented"
        ],
        "missing": [
            m for m in metrics_dict.keys() 
            if m not in EXPECTED_METRICS
        ],
    }
    
    # Add reasons for N/A metrics
    if coverage["na"]:
        coverage["na_reasons"] = {
            m: metrics_dict[m].get("reason", "unknown") 
            for m in coverage["na"] 
            if m in metrics_dict
        }
    
    return {
        "meta": metrics_meta,
        "metrics": metrics_dict,
        "coverage": coverage,
    }
