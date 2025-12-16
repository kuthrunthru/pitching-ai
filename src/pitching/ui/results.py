"""
Results stage UI rendering.
"""

import math
import os
import streamlit as st
import cv2
import numpy as np
from core.keyframes import get_keyframes
from src.metrics.calibration import get_bands_for_metric
from src.video.preview import (
    read_raw_frame,
    get_transformed_preview_frame,
    draw_pose_overlay_basic,
)


def render_results():
    """Render the results stage UI."""
    st.header("Analysis Results")
    
    clip = st.session_state.get("clip", {})
    n = clip.get("n")
    
    if n is None:
        st.error("Video not processed. Missing clip data.")
        st.session_state["stage"] = "upload"
        st.rerun()
        return
    
    # Get pose cache early so it's available throughout the function
    cache_key = clip.get("cache_key")
    pose_cache = None
    if cache_key and "pose_cache" in st.session_state:
        pose_cache = st.session_state["pose_cache"].get(cache_key)
    
    # Get keyframes (validated)
    keyframes = get_keyframes(n)
    ffp_idx = keyframes["ffp_idx"]
    release_idx = keyframes["release_idx"]
    ffp_source = keyframes["source"]["ffp"]
    release_source = keyframes["source"]["release"]
    release_validated = keyframes.get("release_validated", True)
    release_debug = st.session_state.get("keyframes", {}).get("release_debug", {})
    
    # Get metrics data (if available)
    # Support both old and new format
    metrics_raw = st.session_state.get("metrics", {})
    if isinstance(metrics_raw, dict) and "metrics" in metrics_raw:
        # New standardized format
        metrics = metrics_raw
    else:
        # Old format - wrap it for backward compatibility
        metrics = {"metrics": metrics_raw, "meta": {}}
    
    # Auto-compute metrics if not already computed and keyframes are available
    # Check if metrics dict is empty or has no actual metric values
    metrics_dict = metrics.get("metrics", {}) if isinstance(metrics, dict) else {}
    has_computed_metrics = bool(metrics_dict and any(
        isinstance(v, dict) and v.get("value") is not None 
        for v in metrics_dict.values()
    ))
    
    if not has_computed_metrics:
        # Guardrails: ensure keyframes are valid
        if ffp_idx is None or release_idx is None:
            st.error("FFP and Release must be set before computing metrics.")
        else:
            
            # Compute metrics with full error handling
            import traceback
            from src.metrics.compute import compute_all_metrics
            with st.spinner("Running analysis..."):
                try:
                    result = compute_all_metrics(clip, keyframes, pose_cache=pose_cache)
                    st.session_state["analysis_result"] = result
                    st.session_state["metrics"] = result
                    st.session_state["app_stage"] = "results"
                    st.success("Analysis complete. Switching to results…")
                    st.rerun()
                except Exception:
                    st.error("Analysis crashed. Full traceback below:")
                    st.code(traceback.format_exc())
                    # Set empty metrics to prevent infinite loop
                    metrics = {"metrics": {}, "meta": {}}
                    raise
    
    # Show warnings if any
    if keyframes["warnings"]:
        st.warning("Keyframe validation warnings:")
        for warning in keyframes["warnings"]:
            st.write(f"- {warning}")
    
    # Check release validation and show warning if invalid
    if not release_validated:
        st.error("⚠️ Release frame must be at least 6 frames after FFP. Please re-pick Release or revert to Auto.")
    
    # Helpers for annotated metric frames (results page only)
    def get_metric_frame_idx(metric_debug: dict):
        """
        Return the primary frame index a metric actually used.
        Prefer 'used_idx' when present, then 'best_idx'. No other fallbacks here.
        """
        if not isinstance(metric_debug, dict):
            return None
        for k in ("used_idx", "best_idx"):
            v = metric_debug.get(k)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    continue
        return None

    def load_transformed_frame(tmp_path, frame_idx, orientation_mode, needs_flip):
        from src.video.preview import read_raw_frame, get_transformed_preview_frame
        fr = read_raw_frame(tmp_path, int(frame_idx), max_w=None)
        if fr is None:
            return None
        return get_transformed_preview_frame(fr, orientation_mode, needs_flip)

    def draw_metric_overlay(frame_bgr, pts, frame_idx, metric_key, metric_debug, metrics_meta=None, scale_x=1.0, scale_y=1.0):
        """
        Draw metric-specific overlays on the representative frame.
        Uses consistent styling:
          - Alignment lines: turquoise, thick
          - Direction arrows: orange, thick arrowed lines
          - Angle arcs (with degree labels): lime, thick
        Returns (annotated_frame, drawn: bool).
        
        Args:
            frame_bgr: Frame in BGR format
            pts: Dict of landmark arrays (already scaled to native frame size)
            frame_idx: Frame index
            metric_key: Metric key (e.g., "stride_length")
            metric_debug: Metric debug dict (may contain knee_xy_used)
            metrics_meta: Metrics meta dict (may contain knee_model_xy_by_frame)
            scale_x: X scaling factor from process_width to native width (default 1.0)
            scale_y: Y scaling factor from process_height to native height (default 1.0)
        """
        if frame_bgr is None or pts is None or frame_idx is None:
            return frame_bgr, False

        align_color = (0, 200, 255)   # turquoise
        arrow_color = (0, 140, 255)   # orange
        angle_color = (60, 220, 60)   # lime
        thickness = 3

        def joint_exists(name):
            return name in pts and frame_idx < len(pts[name]) and pts[name][frame_idx] is not None \
                   and len(pts[name][frame_idx]) >= 2 and np.isfinite(pts[name][frame_idx][0]) and np.isfinite(pts[name][frame_idx][1])

        def get_xy(name):
            pt = pts[name][frame_idx]
            return int(pt[0]), int(pt[1])

        def draw_line(img, a, b, color=align_color):
            cv2.line(img, get_xy(a), get_xy(b), color, thickness, cv2.LINE_AA)

        def draw_arrow(img, a, b, color=arrow_color):
            cv2.arrowedLine(img, get_xy(a), get_xy(b), color, thickness, tipLength=0.15)

        def draw_angle_arc(img, center, p1, p2, color=angle_color, label=None):
            c = np.array(get_xy(center))
            v1 = np.array(get_xy(p1)) - c
            v2 = np.array(get_xy(p2)) - c
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return
            ang = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
            r = int(min(n1, n2) * 0.6)
            start = math.degrees(math.atan2(v1[1], v1[0]))
            end = math.degrees(math.atan2(v2[1], v2[0]))
            # normalize sweep direction
            sweep = end - start
            if sweep <= -180:
                sweep += 360
            if sweep > 180:
                sweep -= 360
            cv2.ellipse(img, tuple(c), (r, r), 0, start, start + sweep, color, thickness, cv2.LINE_AA)
            if label:
                text_pos = tuple(c + np.array([r + 10, 0]))
                cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Resolve side where needed
        throwing_side = metric_debug.get("throwing_side") or metric_debug.get("throwing") or "R"
        lead = metric_debug.get("lead")
        back = metric_debug.get("back")

        joints = []
        connections = []
        arrows = []
        angle_triplets = []  # (center, p1, p2, label)

        if metric_key == "stride_length":
            lead_side = (lead or "LEFT").upper()
            back_side = (back or "RIGHT").upper()
            la = f"{lead_side}_ANKLE"
            ba = f"{back_side}_ANKLE"
            joints = [la, ba]
            connections = [(la, ba)]



        elif metric_key == "head_behind_hip":
            hip_side = "RIGHT_HIP" if throwing_side and throwing_side.upper() == "L" else "LEFT_HIP"
            joints = [hip_side, "NOSE", "LEFT_HIP", "RIGHT_HIP"]
            connections = [(hip_side, "NOSE")]
            arrows = [(hip_side, "NOSE")]

        elif metric_key == "chest_closed_ffp_deg":
            joints = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
            connections = [("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_HIP", "RIGHT_HIP")]
            # angle across shoulders vs horizontal
            if joint_exists("LEFT_SHOULDER") and joint_exists("RIGHT_SHOULDER"):
                angle_triplets.append(("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_SHOULDER"))

        elif metric_key == "upper_body_lean_release":
            joints = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
            connections = [("LEFT_HIP", "RIGHT_HIP"), ("LEFT_SHOULDER", "RIGHT_SHOULDER")]
            # No arrows - we'll draw the lean line and vertical reference in the special overlay

        elif metric_key == "throw_elbow_vs_shoulder_line_ffp":
            ts = "RIGHT" if throwing_side and throwing_side.upper() == "R" else "LEFT"
            joints = ["LEFT_SHOULDER", "RIGHT_SHOULDER", f"{ts}_ELBOW"]
            connections = [("LEFT_SHOULDER", "RIGHT_SHOULDER"), (f"{ts}_ELBOW", "LEFT_SHOULDER"), (f"{ts}_ELBOW", "RIGHT_SHOULDER")]
            # alignment only

        elif metric_key == "elbow_bend_ffp_deg":
            ts = "RIGHT" if throwing_side and throwing_side.upper() == "R" else "LEFT"
            joints = [f"{ts}_SHOULDER", f"{ts}_ELBOW", f"{ts}_WRIST"]
            connections = [(f"{ts}_SHOULDER", f"{ts}_ELBOW"), (f"{ts}_ELBOW", f"{ts}_WRIST")]
            if all(joint_exists(j) for j in joints):
                angle_triplets.append((f"{ts}_ELBOW", f"{ts}_SHOULDER", f"{ts}_WRIST"))

        elif metric_key == "ball_angle_vs_shoulder_line_ffp_deg":
            ts = "RIGHT" if throwing_side and throwing_side.upper() == "R" else "LEFT"
            joints = ["LEFT_SHOULDER", "RIGHT_SHOULDER", f"{ts}_SHOULDER", f"{ts}_WRIST"]
            connections = [("LEFT_SHOULDER", "RIGHT_SHOULDER"), (f"{ts}_SHOULDER", f"{ts}_WRIST")]
            if all(joint_exists(j) for j in [f"{ts}_SHOULDER", f"{ts}_WRIST", "LEFT_SHOULDER"]):
                angle_triplets.append((f"{ts}_SHOULDER", "RIGHT_SHOULDER", f"{ts}_WRIST"))

        else:
            return frame_bgr, False

        pts_drawn = []
        for j in joints:
            if joint_exists(j):
                pts_drawn.append(j)
        if not pts_drawn:
            return frame_bgr, False

        annotated = frame_bgr.copy()
        color = (0, 215, 255)  # amber
        # Draw alignment lines
        for a, b in connections:
            if joint_exists(a) and joint_exists(b):
                draw_line(annotated, a, b, color=align_color)

        # Draw arrows
        for a, b in arrows:
            if joint_exists(a) and joint_exists(b):
                draw_arrow(annotated, a, b, color=arrow_color)

        # Draw angle arcs
        for center, p1, p2, in angle_triplets:
            if joint_exists(center) and joint_exists(p1) and joint_exists(p2):
                draw_angle_arc(annotated, center, p1, p2, color=angle_color)

        # Draw points
        for j in pts_drawn:
            cv2.circle(annotated, get_xy(j), 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        
        # Special overlays for other metrics (leg_lift_angle_deg removed)

        
        # Upper-Body Lean at Release overlay: draw mid-hip→mid-shoulder line + vertical reference
        if metric_key == "upper_body_lean_release":
            # Get joint positions from metric_debug or compute from frame
            mid_hip = metric_debug.get("mid_hip")
            mid_sh = metric_debug.get("mid_sh")
            lean_deg = metric_debug.get("lean_deg")
            # Get status from metric_debug or try to infer from value
            status = metric_debug.get("status")
            if status is None and lean_deg is not None:
                # Infer status from lean_deg using high school bands
                if lean_deg >= 14.0:
                    status = "green"
                elif lean_deg >= 8.0:
                    status = "yellow"
                else:
                    status = "red"
            
            # If not in debug, try to get from frame
            if mid_hip is None or mid_sh is None:
                from src.metrics.utils import get_joint_xy
                lh_xy = get_joint_xy(frame_idx, "LEFT_HIP", pts=pts)
                rh_xy = get_joint_xy(frame_idx, "RIGHT_HIP", pts=pts)
                ls_xy = get_joint_xy(frame_idx, "LEFT_SHOULDER", pts=pts)
                rs_xy = get_joint_xy(frame_idx, "RIGHT_SHOULDER", pts=pts)
                
                if lh_xy and rh_xy and ls_xy and rs_xy:
                    mid_hip = [
                        (float(lh_xy[0]) + float(rh_xy[0])) / 2.0,
                        (float(lh_xy[1]) + float(rh_xy[1])) / 2.0,
                    ]
                    mid_sh = [
                        (float(ls_xy[0]) + float(rs_xy[0])) / 2.0,
                        (float(ls_xy[1]) + float(rs_xy[1])) / 2.0,
                    ]
            
            if mid_hip and mid_sh and len(mid_hip) >= 2 and len(mid_sh) >= 2:
                # Scale coordinates to native frame size
                mh_x = int(float(mid_hip[0]) * scale_x)
                mh_y = int(float(mid_hip[1]) * scale_y)
                ms_x = int(float(mid_sh[0]) * scale_x)
                ms_y = int(float(mid_sh[1]) * scale_y)
                
                h, w = annotated.shape[:2]
                
                # Check bounds
                if 0 <= mh_x < w and 0 <= mh_y < h and 0 <= ms_x < w and 0 <= ms_y < h:
                    # Draw vertical reference line at mid_hip (upward, length ~100px)
                    vert_length = 100
                    vert_end_y = max(0, mh_y - vert_length)
                    cv2.line(annotated, (mh_x, mh_y), (mh_x, vert_end_y), (128, 128, 128), 2, cv2.LINE_AA)  # Gray vertical reference
                    
                    # Draw line from mid_hip to mid_shoulder (thick, colored by status)
                    if status == "green":
                        line_color = (0, 255, 0)  # Green
                    elif status == "yellow":
                        line_color = (0, 255, 255)  # Yellow (cyan in BGR)
                    else:  # red or None
                        line_color = (0, 0, 255)  # Red
                    
                    cv2.line(annotated, (mh_x, mh_y), (ms_x, ms_y), line_color, 3, cv2.LINE_AA)
                    
                    # Draw small circles at mid_hip and mid_shoulder
                    cv2.circle(annotated, (mh_x, mh_y), 6, (255, 255, 255), -1, cv2.LINE_AA)  # White circle at hip
                    cv2.circle(annotated, (ms_x, ms_y), 6, (255, 255, 255), -1, cv2.LINE_AA)  # White circle at shoulder
                    
                    # Label with angle and status
                    if lean_deg is not None:
                        status_label = "G" if status == "green" else ("Y" if status == "yellow" else "R")
                        label_text = f"Upper-body lean: {lean_deg:.1f}deg ({status_label})"
                    else:
                        label_text = "Upper-body lean: N/A"
                    
                    # Position label above the line midpoint
                    label_x = (mh_x + ms_x) // 2
                    label_y = min(mh_y, ms_y) - 20
                    if label_y < 0:
                        label_y = max(mh_y, ms_y) + 30
                    
                    cv2.putText(
                        annotated,
                        label_text,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # White text
                        2,
                        cv2.LINE_AA,
                    )
        
        return annotated, True
    
    # Helper function to generate suggestion text for a metric
    def get_metric_suggestion(metric_name, value, status, reason, interpretation_explanation):
        """Generate suggestion text for a metric based on its value, status, and reason."""
        # If value is None, provide pose quality/visibility guidance
        if value is None:
            if reason and "error" in reason.lower():
                return "Error during computation - check diagnostics for details."
            elif reason and ("missing" in reason.lower() or "landmark" in reason.lower()):
                return "Pose quality/visibility issue - try a clearer side view or re-pick keyframes."
            elif reason and "not_implemented" in reason.lower():
                return "Metric not yet implemented."
            elif reason and "mkl_not_set" in reason.lower():
                return "Required keyframe not set."
            else:
                return "Pose quality/visibility issue - try a clearer side view or re-pick keyframes."
        
        # Metric-specific suggestions
        if metric_name == "Head Behind Hip Toward Plate" and value is not None:
            ratio = float(value)
            if ratio >= 0.02:
                return "Head stays clearly behind hip - excellent posture control."
            elif ratio >= 0.00:
                return "Head roughly stacked with hip - aim to keep head slightly behind for better balance."
            else:
                return "Head leaks past hip - work on maintaining head position behind hip during delivery."
        
        if metric_name == "Stride Length" and value is not None:
            ratio = float(value)
            # Note: Ratio is normalized by shoulder-to-ankle height, so thresholds are adjusted
            # GREEN (>= 3.5), YELLOW (2.7-3.5), RED (< 2.7)
            if ratio >= 3.5:
                return "Excellent stride length - ideal for power and balance."
            elif 2.7 <= ratio < 3.5:
                return "Stride length is moderate. Aim for above 3.5 for optimal power transfer."
            else:  # ratio < 2.7
                return "Stride is too short. This limits power transfer. Work on pushing off harder and extending your stride to above 3.5."
        
        if metric_name == "Elbow Height at Foot Strike" and value is not None:
            d = float(value)
            if -0.05 <= d <= 0.02:
                return "Good - throwing elbow is on/just below the shoulder line at foot strike."
            elif 0.02 < d <= 0.06:
                return "Elbow is slightly high at foot strike - work on keeping it closer to the shoulder line."
            elif -0.10 <= d < -0.05:
                return "Elbow is slightly low at foot strike - work on getting it up near the shoulder line."
            elif d > 0.06:
                return "Elbow is too high at foot strike - work on keeping it closer to the shoulder line."
            else:  # d < -0.10
                return "Elbow is too low at foot strike - work on getting it up near the shoulder line by foot strike."
        
        if metric_name == "Arm Angle at Foot Strike" and value is not None:
            angle = float(value)
            if 85 <= angle <= 95:
                return "Good - elbow bend is in the ideal range at foot strike."
            elif angle < 75:
                return "Elbow is too bent at foot strike - work toward ~90 deg."
            elif angle > 110:
                return "Arm is too straight at foot strike - work toward ~90 deg bend."
            elif 75 <= angle < 85:
                return "Elbow is slightly too bent at foot strike - aim for ~90 deg."
            else:  # 95 < angle <= 110
                return "Arm is slightly too straight at foot strike - aim for ~90 deg bend."
        
        if metric_name == "Ball Location at Foot Strike" and value is not None:
            # Use status from scoring function which uses new thresholds:
            # GREEN: 10°–25°, YELLOW: 0°–10° or 25°–40°, RED: < 0° or > 40°
            if status and status.lower() == "green":
                return "Good - arm is in the ideal 10-25° range at foot strike, allowing proper timing between lower and upper body."
            elif status and status.lower() == "yellow":
                angle = float(value)
                if 0 <= angle < 10:
                    return "Arm is late at foot strike (0-10°) - work on getting the arm up earlier so it's in the 10-25° range when the foot lands."
                else:  # 25 < angle <= 40
                    return "Arm is early at foot strike (25-40°) - the arm may be too high, which can cause timing issues. Aim for the 10-25° range."
            elif status and status.lower() == "red":
                angle = float(value)
                if angle < 0:
                    return "Poor - arm is below the shoulder line at foot strike. Work on getting the arm up to at least the shoulder line (0°+) and ideally into the 10-25° range."
                else:  # angle > 40
                    return "Poor - arm is too high at foot strike (>40°). This can cause the arm and body to fall out of sync. Aim for the 10-25° range."
            else:
                # Fallback based on angle value alone
                angle = float(value)
                if 10 <= angle <= 25:
                    return "Good - arm is in the ideal 10-25° range at foot strike, allowing proper timing between lower and upper body."
                elif 0 <= angle < 10:
                    return "Arm is late at foot strike (0-10°) - work on getting the arm up earlier so it's in the 10-25° range when the foot lands."
                elif 25 < angle <= 40:
                    return "Arm is early at foot strike (25-40°) - the arm may be too high, which can cause timing issues. Aim for the 10-25° range."
                else:
                    return "Poor - arm position at foot strike is outside the acceptable range. Aim for the 10-25° range for optimal timing."
        
        if metric_name == "Landing Leg Bend" and value is not None:
            bend_deg = float(value)
            if 35 <= bend_deg <= 55:
                return "Good - landing leg bend is near ideal (~45 deg) at foot strike."
            elif bend_deg < 25:
                return "Landing leg is too straight at foot strike - allow some bend (~45 deg) to absorb and brace."
            elif bend_deg > 70:
                return "Landing leg collapses too much - work on landing firmer and bracing after foot strike."
            elif 25 <= bend_deg < 35:
                return "Landing leg is slightly too straight - aim for ~45 deg bend to better absorb impact."
            else:  # 55 < bend_deg <= 70
                return "Landing leg bends a bit too much - aim for firmer landing with ~45 deg bend."
        
        
        if metric_name == "Chest Closed at Foot Strike" and value is not None:
            # Get guide_status from debug info if available, otherwise infer from angle
            abs_angle = float(value)
            if abs_angle >= 60:
                return "Good - chest stays closed at foot strike, allowing lower body to lead."
            elif abs_angle >= 40:
                return "Chest is starting to open at foot strike; staying closed a bit longer may help timing."
            else:  # < 40
                return "Chest opens early at foot strike; focus on keeping the upper body back while the legs move forward."
        
        if metric_name == "Upper-Body Lean at Release" and value is not None:
            lean_deg = float(value)
            if lean_deg >= 14.0:
                return "Good - strong forward lean at release helps transfer energy to the ball."
            elif lean_deg >= 8.0:
                return "Moderate lean - aim for more forward lean (14+ deg) to maximize power transfer."
            else:  # < 8.0
                return "Too upright at release - work on getting more forward lean (8+ deg) to drive through the pitch."
        
        # If value exists, provide status-based guidance
        if status:
            status_lower = status.lower()
            # Handle color-based status (green/yellow/red) as well as standard status
            if status_lower in ["excellent", "good", "green"]:
                if interpretation_explanation:
                    return f"✅ {interpretation_explanation}"
                else:
                    return "✅ Within optimal range."
            elif status_lower in ["ok", "yellow"]:
                if interpretation_explanation:
                    return f"⚠️ {interpretation_explanation} Consider minor adjustments."
                else:
                    return "⚠️ Within acceptable range. Consider minor adjustments."
            elif status_lower in ["poor", "bad", "red"]:
                if interpretation_explanation:
                    return f"❌ {interpretation_explanation} Focus on improving this area."
                else:
                    return "❌ Below optimal range. Focus on improving this area."
        
        # Fallback
        if interpretation_explanation:
            return f"💡 {interpretation_explanation}"
        else:
            return "See metric-specific guidance."

    # Script templates per metric (plain language, short sentences)
    METRIC_SCRIPT_TEMPLATES = {
        # leg_lift_angle_deg removed
        # lead_leg_block removed - metric deleted
        "stride_length": {
            "why": "Stride length measures how far you step toward the plate compared to your height.\nA strong stride helps you transfer momentum from your legs into the throw while staying balanced and on time. For most pitchers, landing above 85% of your height is ideal. Shorter strides can limit power transfer.",
            "tips": [
                "Push off the rubber and let your front foot travel straight toward the target.",
                "Avoid stepping across your body or cutting off your stride.",
                "Mark a spot on the ground for an ideal stride and practice landing on it with good balance.",
            ],
        },
        "head_behind_hip": {
            "why": "This metric checks if your head stays behind your front hip as you move toward the plate. This helps you stay balanced and is a key to increasing velocity. See https://www.topvelocity.net/2009/02/12/the-hip-slide-to-pitching-velocity/ .",
            "tips": [
                "Keep your eyes level and feel your head riding above your body, not diving toward the plate early.",
                "Let your legs start the move and allow your head and shoulders to follow.",
                "On video, pause halfway through your move and check that your head is even with or slightly behind your front hip.",
            ],
        },
        "chest_closed_ffp_deg": {
            "why": "This metric looks at how much your chest is turned away from the plate when your front foot lands. Staying closed longer lets your legs lead and your upper body follow. Note: this is a hard metric for the 2D model to measure. Use a simple eye test to see how well a pitcher can keep closed at foot strike.",
            "tips": [
                "Try to land with your glove-side shoulder still at least partly facing the target.",
                "Avoid turning your chest toward home before your front foot hits the ground.",
                "Use simple drills where you pause at foot strike and check that your chest is still somewhat closed.",
            ],
        },
        "upper_body_lean_release": {
            "why": "Upper-body lean at release measures how much you're leaning forward toward the plate when you release the ball. More forward lean (14+ deg) helps transfer energy from your body to the ball. Coaches may say have your chest over your knee when you throw the ball. Elite pitchers almost have a straight line from the foot, up the body, to the ball at the point of release.",
            "tips": [
                "Aim for at least 14 deg of forward lean at release to maximize power transfer.",
                "Think about driving your chest forward toward the plate as you release, not staying upright.",
                "Practice drills where you focus on getting your upper body moving forward through release.",
            ],
        },
        "throw_elbow_vs_shoulder_line_ffp": {
            "why": "Elbow height at foot strike checks if your throwing elbow is on or just below your shoulders when your front foot lands. This helps keep your arm path strong and safe. You dont want the elbow too high (injury risk) or too low (how little kids push the ball to the plate).",
            "tips": [
                "Lift your elbow so it is roughly level with or a little below your shoulder at foot strike.",
                "Avoid letting the elbow ride far above your shoulders.",
                "Use mirror or wall drills to practice the arm position at front-foot plant before you throw.",
            ],
        },
        "elbow_bend_ffp_deg": {
            "why": "Arm angle at foot strike measures how much your throwing elbow is bent when your front foot lands. Around a right angle (about 90 deg) is considered ideal. Younger pitchers tend to have a long arm at foot plant, leading to dragging the arm, which creates problems with control and velocity.",
            "tips": [
                "Feel a strong bend in your elbow, close to a right angle, as the front foot hits.",
                "Avoid reaching your arm long and straight behind you as you stride.",
                "Pause at foot strike in simple drills and check your elbow angle in a mirror or on video.",
            ],
        },
        "ball_angle_vs_shoulder_line_ffp_deg": {
            "why": "This metric measures where the throwing hand is at the instant the front foot lands, expressed as the angle of the arm relative to the shoulder. It shows whether the arm is still coming up, rising naturally, or already very high at foot strike. This moment matters because foot strike is when the lower body finishes moving forward and the upper body prepares to rotate. If the arm is too low or too high at this point, the arm and body can fall out of sync, forcing the arm to rush or the upper body to move too early.",
            "tips": [
                "Try to have the ball above your shoulders between about halfway up and straight up (about 45-90 deg) at foot strike.",
                "Avoid having the ball below your shoulders when the front foot lands.",
                "Pause at foot strike in drills and check that the ball sits clearly above your shoulder line.",
            ],
        },
        # head_lean_at_release_deg removed - metric deleted
    }

    # Simple jargon replacements (best-effort safety net)
    JARGON_MAP = {
        "pelvis": "hips",
        "Pelvis": "Hips",
        "scapula": "shoulder blade",
        "Scapula": "Shoulder blade",
        "kinematic": "movement",
        "Kinematic": "Movement",
        "torque": "twisting force",
        "Torque": "Twisting force",
        "sequencing": "timing",
        "Sequencing": "Timing",
        "trunk tilt": "upper body lean",
        "Trunk tilt": "Upper body lean",
    }

    def build_metric_script(metric_key, metric_name, metric_data, metrics_meta):
        """Return a markdown coaching script for a metric."""
        value = metric_data.get("value")
        units = metric_data.get("units", "")
        score = metric_data.get("score")
        status = metric_data.get("status")
        reason = metric_data.get("reason", "ok")
        debug = metric_data.get("debug", {}) or {}

        template = METRIC_SCRIPT_TEMPLATES.get(metric_key, {})
        why = template.get(
            "why",
            f"{metric_name} captures a key aspect of your pitching mechanics related to this part of the delivery.",
        )
        tips = template.get(
            "tips",
            [
                "Use slow-motion video to review this part of your delivery.",
                "Make small adjustments and re-check how the metric responds.",
            ],
        )

        # 1) What this measures & why it matters
        section1 = f"**1. What this measures & why it matters**  \n{why}"

        # 2) What we observed (plain-language coaching, no raw numbers)
        if value is None or not np.isfinite(value):
            # Generic plain-language message when metric is unavailable
            obs_lines = [
                "We could not clearly read this part of your motion on this clip.",
                "Most often this happens when the body is blocked from the camera or key moments are hard to see.",
                "Using a clearer side view and setting keyframes carefully will help this metric update next time.",
            ]
        else:
            # Plain language based on status only (no numeric details)
            base = f"For **{metric_name}**, your movement looked"
            if status is None:
                obs_lines = [base + " acceptable on this pitch."]
            else:
                s = status.lower()
                if s in ("green", "excellent", "good"):
                    obs_lines = [
                        base + " strong and in a healthy range.",
                        "This part of your delivery is working well and can be a foundation for other changes.",
                    ]
                elif s in ("yellow", "ok"):
                    obs_lines = [
                        base + " mostly on track but with a little room to tighten up.",
                        "Small adjustments here could make your delivery smoother and more repeatable.",
                    ]
                else:  # red / bad / poor
                    obs_lines = [
                        base + " out of the ideal range on this pitch.",
                        "Cleaning up this piece can help you move more efficiently and reduce stress on your arm.",
                    ]
        section2 = "**2. What we observed**  \n" + "  \n".join(obs_lines)

        # 3) Tips to improve
        tips_lines = [f"- {t}" for t in tips]
        section3 = "**3. Tips to improve**  \n" + "  \n".join(tips_lines)

        script = section1 + "\n\n" + section2 + "\n\n" + section3 + "\n"

        # Apply simple jargon replacements to keep language plain
        for bad, simple in JARGON_MAP.items():
            script = script.replace(bad, simple)

        return script
    # Helpers for metric frames (results page only, using standard overlay)
    def get_metric_frame_idx(metric_debug: dict):
        if not isinstance(metric_debug, dict):
            return None
        for k in ["best_idx", "used_idx"]:
            v = metric_debug.get(k)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    continue
        return None

    def select_metric_frame_idx(metric_debug, metrics_meta, n_frames):
        """Priority: best_idx -> used_idx -> ffp_idx -> release_idx -> 0."""
        candidates = []
        for k in ("best_idx", "used_idx"):
            v = (metric_debug or {}).get(k)
            if v is not None:
                candidates.append(v)
        for k in ("ffp_idx", "release_idx"):
            v = (metrics_meta or {}).get(k)
            if v is not None:
                candidates.append(v)
        candidates.append(0)
        for v in candidates:
            try:
                idx = int(v)
                if n_frames is not None:
                    idx = max(0, min(int(n_frames) - 1, idx))
                return idx
            except Exception:
                continue
        return 0

    def render_metric_frame_standard(metric_key, metric_name, clip, pose_cache, metric_debug, metrics_meta):
        """Render standard overlay frame for a metric; always returns an image (placeholder if needed)."""
        tmp_path = clip.get("tmp_path") if clip else None
        n_frames = clip.get("n") if clip else None
        frame_idx = select_metric_frame_idx(metric_debug, metrics_meta, n_frames)

        def placeholder(reason: str):
            img = np.full((360, 640, 3), 180, dtype=np.uint8)
            cv2.putText(img, metric_name, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Frame: {frame_idx}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, reason, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            if tmp_path:
                cv2.putText(img, f"tmp_path: {tmp_path}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            if n_frames is not None:
                cv2.putText(img, f"n_frames: {n_frames}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            return img

        if tmp_path is None or n_frames is None:
            return placeholder("missing clip")

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            cap.release()
            return placeholder("cannot open video")

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, fr = cap.read()
        cap.release()
        if not ok or fr is None:
            return placeholder("frame read failed")

        chosen_mode = st.session_state.get("preview_orientation_mode", "None")
        needs_flip = st.session_state.get("preview_needs_flip", False)
        img = get_transformed_preview_frame(fr, chosen_mode, needs_flip)
        if img is None:
            return placeholder("transform failed")

        pts_raw = pose_cache.get("pts") if isinstance(pose_cache, dict) else None
        if not pts_raw:
            return img

        native_h, native_w = img.shape[:2]
        proc_w = clip.get("process_width")
        proc_h = clip.get("process_height")

        pts_by_name = {}
        for name, arr in pts_raw.items():
            if arr is not None and frame_idx < len(arr):
                pt_xy = np.array(arr[frame_idx]).copy()
                if proc_w and proc_h and native_w > 0 and native_h > 0:
                    scale_x = native_w / float(proc_w)
                    scale_y = native_h / float(proc_h)
                    pt_xy[0] *= scale_x
                    pt_xy[1] *= scale_y
                pts_by_name[name] = pt_xy

        pose_q_frame = None
        if pose_cache.get("pose_q") is not None and frame_idx < len(pose_cache["pose_q"]):
            pose_q_frame = pose_cache["pose_q"][frame_idx]

        # Draw basic pose overlay first
        img_with_pose = draw_pose_overlay_basic(img, pts_by_name, pose_q=pose_q_frame)
        
        # Calculate scaling factors for knee coordinates (knee_xy_used is in process_width/process_height coords)
        scale_x = 1.0
        scale_y = 1.0
        if proc_w and proc_h and native_w > 0 and native_h > 0:
            scale_x = native_w / float(proc_w)
            scale_y = native_h / float(proc_h)
        
        # Then draw metric-specific overlays (including knee markers)
        # Pass clip info so draw_metric_overlay can scale knee coordinates correctly
        img_with_metric, drawn = draw_metric_overlay(
            img_with_pose, 
            pts_by_name, 
            frame_idx, 
            metric_key, 
            metric_debug, 
            metrics_meta,
            scale_x=scale_x,
            scale_y=scale_y
        )
        
        return img_with_metric

    # Helper function to render a metric row
    def render_metric_row(metric_key, metric_name, raw_value, unit, score, status, bands=None, reason=None, 
                          interpretation_status=None, interpretation_explanation=None, metric_debug=None,
                          clip=None, pose_cache=None, metrics_meta=None, metric_data_full=None):
        """Render a single metric row with raw value, score, status, suggestion, and frame image."""
        import numpy as np
        
        # Guard against None/NaN
        is_missing = False
        if raw_value is None:
            is_missing = True
        elif isinstance(raw_value, (float, np.floating)) and (np.isnan(raw_value) or not np.isfinite(raw_value)):
            is_missing = True
        
        # Format raw value (kept for compact line; heading itself is name-only)
        if is_missing:
            raw_str = "N/A"
        elif isinstance(raw_value, float):
            raw_str = f"{raw_value:.2f} {unit}".strip()
        else:
            raw_str = f"{raw_value} {unit}".strip()
        
        # Format score
        score_str = f"{score}/100" if (score is not None and not is_missing) else "N/A"
        
        # Format status with color - derive strictly from metric["status"] returned by scoring function
        # If score is None, show N/A with neutral badge (not EXCELLENT/POOR)
        if score is None:
            # Score is None - show neutral badge
            status_icon = "⚪"
            status_str = "N/A"
        elif status is None:
            # Score exists but status is missing - show neutral badge
            status_icon = "⚪"
            status_str = "N/A"
        elif status.lower() == "guide":
            # Guidance metric - show guide badge
            status_icon = "📋"
            status_str = "GUIDE"
        else:
            # Use status from scoring function (stored in metric["status"])
            # Map status to icon (support both standard and color-based status)
            status_colors = {
                "excellent": "🟢",
                "good": "🟢",
                "green": "🟢",
                "ok": "🟡",
                "yellow": "🟡",
                "poor": "🔴",
                "bad": "🔴",
                "red": "🔴",
            }
            status_icon = status_colors.get(status.lower(), "⚪")
            # Display status name (convert color names to standard names for display)
            status_display = status.lower()
            if status_display == "green":
                status_display = "GOOD"
            elif status_display == "yellow":
                status_display = "OK"
            elif status_display == "red":
                status_display = "BAD"
            else:
                status_display = status.upper()
            status_str = f"{status_icon} {status_display}"
        
        # Display metric heading and compact score row
        st.markdown(f"#### {metric_name}")
        st.markdown(
            f"{status_str} &nbsp;&nbsp; **Value:** {raw_str}",
            unsafe_allow_html=True,
        )

        # Two-column image row: analysis frame (left), MLB placeholder (right)
        img_col_left, img_col_right = st.columns([1, 1])
        frame = None
        frame_dims = None
        with img_col_left:
            if clip and pose_cache and clip.get("tmp_path"):
                # Resolve the primary frame index from metric debug
                label_idx = get_metric_frame_idx(metric_debug or {})
                render_idx = label_idx
                if render_idx is None:
                    # Fallback: derive from meta keyframes in a fixed order, then 0
                    meta = metrics_meta or {}
                    for v in (meta.get("ffp_idx"), meta.get("release_idx"), 0):
                        if v is None:
                            continue
                        try:
                            render_idx = int(v)
                            break
                        except Exception:
                            continue
                frame = render_metric_frame_standard(
                    metric_key, metric_name, clip, pose_cache, metric_debug or {}, metrics_meta or {}
                )
                if frame is not None:
                    st.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                        caption=f"Your clip (frame {render_idx if render_idx is not None else 'n/a'})",
                    )
        with img_col_right:
            # Load MLB example image if available for this metric
            mlb_image = None
            mlb_caption = "MLB example (coming soon)"
            
            # Get MLB examples directory - try multiple methods
            mlb_examples_dir = None
            
            # Method 1: Try relative path from current working directory (most common case)
            test_path = os.path.join("assets", "mlb_examples")
            if os.path.exists(test_path):
                mlb_examples_dir = os.path.abspath(test_path)
            
            # Method 2: Try absolute path from current working directory
            if mlb_examples_dir is None:
                cwd = os.getcwd()
                test_path = os.path.join(cwd, "assets", "mlb_examples")
                if os.path.exists(test_path):
                    mlb_examples_dir = test_path
            
            # Method 3: Try to find project root by looking for app.py relative to this file
            if mlb_examples_dir is None:
                try:
                    # Get the directory of this file (src/pitching/ui/results.py)
                    current_file = os.path.abspath(__file__)
                    current_dir = os.path.dirname(current_file)
                    # Go up from src/pitching/ui/results.py to project root (4 levels up)
                    for _ in range(5):
                        assets_path = os.path.join(current_dir, "assets", "mlb_examples")
                        if os.path.exists(assets_path):
                            mlb_examples_dir = assets_path
                            break
                        parent = os.path.dirname(current_dir)
                        if parent == current_dir:
                            break
                        current_dir = parent
                except Exception:
                    pass
            
            if mlb_examples_dir and os.path.exists(mlb_examples_dir):
                
                # Debug: Check if directory exists and list files (only in debug mode)
                debug_mode = st.session_state.get("debug_mode", False)
                if debug_mode:
                    st.write(f"DEBUG: mlb_examples_dir={mlb_examples_dir}")
                    st.write(f"DEBUG: dir exists={os.path.exists(mlb_examples_dir)}")
                    if os.path.exists(mlb_examples_dir):
                        try:
                            files = os.listdir(mlb_examples_dir)
                            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            st.write(f"DEBUG: all files in dir={files}")
                            st.write(f"DEBUG: image files found={image_files}")
                        except Exception as e:
                            st.write(f"DEBUG: error listing dir: {e}")
                
                # Load images for each metric
                if metric_key == "head_behind_hip":
                    # Try to load MLB example image for Head Behind Hip metric
                    # Try multiple formats
                    for ext in [".jpg", ".jpeg", ".png"]:
                        mlb_image_path = os.path.join(mlb_examples_dir, f"head_behind_hip{ext}")
                        if os.path.exists(mlb_image_path):
                            mlb_image = cv2.imread(mlb_image_path)
                            if mlb_image is not None and mlb_image.size > 0:
                                mlb_caption = "MLB example"
                                break
                            elif st.session_state.get("debug_mode", False):
                                st.write(f"DEBUG: cv2.imread failed for {mlb_image_path}")
                        elif st.session_state.get("debug_mode", False):
                            st.write(f"DEBUG: File does not exist: {mlb_image_path}")
                elif metric_key == "stride_length":
                    # Try to load MLB example image for Stride Length metric
                    for ext in [".jpg", ".jpeg", ".png"]:
                        mlb_image_path = os.path.join(mlb_examples_dir, f"stride_length{ext}")
                        if os.path.exists(mlb_image_path):
                            mlb_image = cv2.imread(mlb_image_path)
                            if mlb_image is not None and mlb_image.size > 0:
                                mlb_caption = "MLB example"
                                break
                            elif st.session_state.get("debug_mode", False):
                                st.write(f"DEBUG: cv2.imread failed for {mlb_image_path}")
                        elif st.session_state.get("debug_mode", False):
                            st.write(f"DEBUG: File does not exist: {mlb_image_path}")
                elif metric_key == "chest_closed_ffp_deg":
                    # Try to load MLB example image for Chest Closed at Foot Strike metric
                    for ext in [".jpg", ".jpeg", ".png"]:
                        mlb_image_path = os.path.join(mlb_examples_dir, f"chest_closed_ffp_deg{ext}")
                        if os.path.exists(mlb_image_path):
                            mlb_image = cv2.imread(mlb_image_path)
                            if mlb_image is not None and mlb_image.size > 0:
                                mlb_caption = "MLB example"
                                break
                            elif st.session_state.get("debug_mode", False):
                                st.write(f"DEBUG: cv2.imread failed for {mlb_image_path}")
                        elif st.session_state.get("debug_mode", False):
                            st.write(f"DEBUG: File does not exist: {mlb_image_path}")
                elif metric_key == "throw_elbow_vs_shoulder_line_ffp":
                    # Try to load MLB example image for Elbow Height at Foot Strike metric
                    for ext in [".jpg", ".jpeg", ".png"]:
                        mlb_image_path = os.path.join(mlb_examples_dir, f"throw_elbow_vs_shoulder_line_ffp{ext}")
                        if os.path.exists(mlb_image_path):
                            mlb_image = cv2.imread(mlb_image_path)
                            if mlb_image is not None and mlb_image.size > 0:
                                mlb_caption = "MLB example"
                                break
                            elif st.session_state.get("debug_mode", False):
                                st.write(f"DEBUG: cv2.imread failed for {mlb_image_path}")
                        elif st.session_state.get("debug_mode", False):
                            st.write(f"DEBUG: File does not exist: {mlb_image_path}")
                elif metric_key == "elbow_bend_ffp_deg":
                    # Try to load MLB example image for Arm Angle at Foot Strike metric
                    for ext in [".jpg", ".jpeg", ".png"]:
                        mlb_image_path = os.path.join(mlb_examples_dir, f"elbow_bend_ffp_deg{ext}")
                        if os.path.exists(mlb_image_path):
                            mlb_image = cv2.imread(mlb_image_path)
                            if mlb_image is not None and mlb_image.size > 0:
                                mlb_caption = "MLB example"
                                break
                            elif st.session_state.get("debug_mode", False):
                                st.write(f"DEBUG: cv2.imread failed for {mlb_image_path}")
                        elif st.session_state.get("debug_mode", False):
                            st.write(f"DEBUG: File does not exist: {mlb_image_path}")
                elif metric_key == "ball_angle_vs_shoulder_line_ffp_deg":
                    # Reuse the Arm Angle at Foot Strike MLB example image for Ball Location at Foot Strike
                    for ext in [".jpg", ".jpeg", ".png"]:
                        mlb_image_path = os.path.join(mlb_examples_dir, f"elbow_bend_ffp_deg{ext}")
                        if os.path.exists(mlb_image_path):
                            mlb_image = cv2.imread(mlb_image_path)
                            if mlb_image is not None and mlb_image.size > 0:
                                mlb_caption = "MLB example"
                                break
                            elif st.session_state.get("debug_mode", False):
                                st.write(f"DEBUG: cv2.imread failed for {mlb_image_path}")
                        elif st.session_state.get("debug_mode", False):
                            st.write(f"DEBUG: File does not exist: {mlb_image_path}")
                elif metric_key == "upper_body_lean_release":
                    # Try to load MLB example image for Upper-Body Lean at Release metric
                    for ext in [".jpg", ".jpeg", ".png"]:
                        mlb_image_path = os.path.join(mlb_examples_dir, f"upper_body_lean_release{ext}")
                        if os.path.exists(mlb_image_path):
                            mlb_image = cv2.imread(mlb_image_path)
                            if mlb_image is not None and mlb_image.size > 0:
                                mlb_caption = "MLB example"
                                break
                            elif st.session_state.get("debug_mode", False):
                                st.write(f"DEBUG: cv2.imread failed for {mlb_image_path}")
                        elif st.session_state.get("debug_mode", False):
                            st.write(f"DEBUG: File does not exist: {mlb_image_path}")
            else:
                # Debug: Show why mlb_examples_dir wasn't found
                if st.session_state.get("debug_mode", False):
                    st.write(f"DEBUG: mlb_examples_dir is None or doesn't exist")
                    st.write(f"DEBUG: cwd={os.getcwd()}")
                    st.write(f"DEBUG: assets/mlb_examples in cwd={os.path.exists(os.path.join(os.getcwd(), 'assets', 'mlb_examples'))}")
            
            # Debug: Show image loading result
            if st.session_state.get("debug_mode", False):
                st.write(f"DEBUG: mlb_image loaded={mlb_image is not None}")
                if mlb_image is not None:
                    st.write(f"DEBUG: mlb_image shape={mlb_image.shape}, size={mlb_image.size}")
            
            if mlb_image is not None:
                st.image(
                    cv2.cvtColor(mlb_image, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption=mlb_caption,
                )
            else:
                # Placeholder MLB example image (easily swappable later)
                placeholder = np.full((360, 640, 3), 210, dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "MLB example",
                    (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    placeholder,
                    "(coming soon)",
                    (40, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                st.image(
                    cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption=mlb_caption,
                )

        # Full plain-language script below images
        if metric_data_full is not None:
            script_md = build_metric_script(metric_key, metric_name, metric_data_full, metrics_meta or {})
            st.markdown(script_md)
        
    
    # Display metrics from session state
    # Check if metrics are in new standardized format
    metrics_dict = None
    if metrics:
        if isinstance(metrics, dict) and "metrics" in metrics:
            # New standardized format
            metrics_dict = metrics.get("metrics", {})
        elif isinstance(metrics, dict) and any(k in metrics for k in ["stride_length"]):
            # Old format - convert on the fly (backward compatibility)
            st.warning("Metrics are in old format. Please recompute metrics.")
            metrics_dict = {}
    
    if metrics_dict:
        # Fixed ordering/groups
        # Order: (a) head_behind_hip first, (b) FFP-dependent metrics next, (c) remaining metrics last
        groups = [
            ("POSTURE", [
                "head_behind_hip",
            ]),
            ("FOOT STRIKE (FFP)", [
                "stride_length",
                "chest_closed_ffp_deg",
                "throw_elbow_vs_shoulder_line_ffp",
                "elbow_bend_ffp_deg",
                "ball_angle_vs_shoulder_line_ffp_deg",
            ]),
            ("REMAINING METRICS", [
                "upper_body_lean_release",
            ]),
        ]

        for group_title, metric_keys in groups:
            # Main section headings removed - metrics display without group headers
            for metric_key in metric_keys:
                metric_data = metrics_dict.get(metric_key)
                if not isinstance(metric_data, dict) or "name" not in metric_data:
                    continue

                name = metric_data.get("name", metric_key)
                value = metric_data.get("value")
                units = metric_data.get("units", "")
                score = metric_data.get("score")
                status = metric_data.get("status")
                reason = metric_data.get("reason", "ok")
                interpretation_status = metric_data.get("interpretation_status")
                interpretation_explanation = metric_data.get("interpretation_explanation")

                bands = None
                try:
                    from src.metrics.calibration import get_bands_for_metric
                    bands = get_bands_for_metric(metric_key)
                except:
                    pass

                render_metric_row(
                    metric_key=metric_key,
                    metric_name=name,
                    raw_value=value,
                    unit=units,
                    score=score,
                    status=status,
                    bands=bands,
                    reason=reason,
                    interpretation_status=interpretation_status,
                    interpretation_explanation=interpretation_explanation,
                    metric_debug=metric_data.get("debug", {}),
                    clip=clip,
                    pose_cache=pose_cache,
                    metrics_meta=metrics.get("meta", {}),
                    metric_data_full=metric_data,
                )
    else:
        st.info("No metrics computed yet. Metrics will appear here after analysis runs.")
    
    # Back button
    if st.button("Back to Set Frames"):
        # Only allow going back if not in results stage or results don't exist
        if st.session_state.get("app_stage") != "results" or not st.session_state.get("results"):
            st.session_state["stage"] = "set_frames"
            st.rerun()
