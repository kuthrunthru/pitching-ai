# Smoke Test Checklist

End-to-end verification checklist to confirm the app works fully before scoring calibration.

## Prerequisites
- Upload 2-3 test clips (various orientations, quality levels)
- Note: This test does NOT verify scoring accuracy, only that the app functions correctly

## Test Steps

### 1. Upload Video
- [ ] Upload a video file (mp4/mov/avi)
- [ ] Video processes successfully (no errors)
- [ ] Success message shows: "Processed video: N frames @ X.X fps"
- [ ] App advances to Orientation Review stage automatically

### 2. Orientation Review - Rotate/Flip → Overlays Stay Aligned
- [ ] Preview frame displays correctly
- [ ] Click "Rotate 90 deg CW" → preview updates immediately
- [ ] Click "Rotate 180 deg" → preview updates immediately
- [ ] Click "Rotate 270 deg CW" → preview updates immediately
- [ ] If pose overlay is visible, skeleton stays aligned with pitcher body during rotation
- [ ] Click "Looks Good → Continue" → advances to Set Frames stage

### 3. Set FFP - Auto Then Manual Override Persists
- [ ] FFP slider appears with default/auto value
- [ ] Adjust FFP slider → preview updates
- [ ] Click "Use as FFP" → source indicator shows "🔵 User"
- [ ] Navigate away and back → FFP value and source persist
- [ ] Auto-detection does NOT overwrite user-selected FFP

### 4. Set Release - Auto Then Manual Override Persists
- [ ] Release slider appears (may have auto-detected value)
- [ ] Adjust Release slider → preview updates
- [ ] Click "Use as Release" → source indicator shows "🔵 User"
- [ ] Navigate away and back → Release value and source persist
- [ ] Auto-detection does NOT overwrite user-selected Release

### 5. Validate Release >= FFP + 6 Warning Triggers Correctly
- [ ] Set Release to a frame < FFP + 6
- [ ] Warning appears: "Release frame must be at least 6 frames after FFP"
- [ ] Status shows `release_validated=False` in Diagnostics
- [ ] Adjust Release to >= FFP + 6 → warning disappears
- [ ] Status shows `release_validated=True`

### 6. Results Page Renders All Sections Without Exceptions
- [ ] Click "Apply & Run Analysis" → advances to Results stage
- [ ] Keyframes section displays (FFP and Release with sources)
- [ ] Metrics section displays (may show N/A if metrics not computed)
- [ ] Diagnostics expander exists and can be expanded
- [ ] No Python exceptions or errors in console/logs
- [ ] All UI elements render without crashes

### 7. Release Height/Extension Not N/A on Good Clips (or Show Clear Reason)
- [ ] On a clip with good pose quality:
  - [ ] Release Height shows a numeric value (not N/A)
  - [ ] Extension shows a numeric value (not N/A)
- [ ] If N/A appears, Diagnostics shows clear reason:
  - [ ] `release_height_reason` explains why (e.g., "no_valid_landmarks_near_release")
  - [ ] `extension_reason` explains why
- [ ] Values are reasonable (e.g., Release Height 0.3-0.8 BL, Extension 0.4-1.0 BL)

### 8. Diagnostics Expander Shows release_debug + throwing_side
- [ ] Expand "Diagnostics" section
- [ ] "Release Detection Debug" section shows `release_debug` dict with:
  - [ ] `reason` field
  - [ ] `ffp_idx`, `search_start`, `search_end`
  - [ ] `valid_ratio` (if auto-detected)
  - [ ] `throwing_side` and `throwing_side_reason` (if available)
- [ ] "Throwing Side" section shows:
  - [ ] Side: "L" or "R"
  - [ ] Debug dict with selection details

### 9. Navigate Back/Forward Between Stages Without Losing State
- [ ] From Results → click "Back to Set Frames"
- [ ] FFP and Release values persist (not reset)
- [ ] Source indicators persist (User/Auto)
- [ ] From Set Frames → adjust values → click "Apply & Run Analysis"
- [ ] Results show updated values
- [ ] Navigate back to Set Frames → values still correct
- [ ] No state loss or unexpected resets

### 10. Second Run on Same Clip Uses Pose Cache (No Recomputation)
- [ ] First run: Note computation time in Diagnostics (if shown)
- [ ] Navigate back to Upload stage
- [ ] Upload the SAME video file again
- [ ] Processing completes quickly (should use cache)
- [ ] Diagnostics shows "From cache" status (if debug mode enabled)
- [ ] No MediaPipe recomputation occurs (CPU usage stays low)

## Debug Mode Verification (Optional)
- [ ] Enable "Debug mode" checkbox in sidebar
- [ ] Pose cache hit/miss indicators appear
- [ ] Keyframe sources visible in debug output
- [ ] No performance degradation with debug mode enabled

## Success Criteria
- ✅ All 10 steps pass on 2-3 different clips
- ✅ Zero crashes or unhandled exceptions
- ✅ State persists correctly across navigation
- ✅ Auto-detection respects manual overrides
- ✅ Warnings trigger correctly for invalid keyframes
- ✅ Metrics display correctly (or show clear N/A reasons)

## Known Issues / Notes
- Document any issues encountered during testing
- Note any clips that fail specific steps
- Record any performance observations

