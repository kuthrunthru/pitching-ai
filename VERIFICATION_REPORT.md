# Separation Persistence (FS → Torso-Open) - Verification Report

## A) Old References Search Results

### Searches Performed:
1. `contact_idx` - **0 matches found** ✓
2. `"FS → Contact"` / `"FS.*Contact"` - **0 matches found** ✓
3. `"to contact"` - **0 matches found** ✓
4. `"contact-proxy"` - **0 matches found** ✓
5. `"through contact"` (in Separation Persistence context) - **0 matches found** ✓

**Result:** ✅ **ZERO old references found. All references have been updated to the new definition.**

## B) UI Wiring Verification

### Label Check:
- ✅ Metric label: `"Separation Persistence (FS → Torso-Open)"` (line 2403)
- ✅ Category: `"Foot Strike"` (line 2403)
- ✅ Metric ID: `8` (line 2403)

### Unit Check:
- ✅ Raw value displays `sep_persist_pct` as percentage (line 2394-2395: `f"{sep_persistence_pct:.0f}%"`)
- ✅ Additional debug info shows `sep_fs_deg` in degrees (line 2397-2399)
- ✅ Raw value format: `"{persistence}% | {separation} deg @ FS"` or `"NA"` (line 2400)

### Scoring Check:
- ✅ Uses `score_separation_persistence_pct(sep_persistence_pct, early_drop_pct=sep_early_drop_pct)` (line 2263-2266)
- ✅ Status correctly passed to `render_metric_panel` as `status=sep_persistence_status` (line 2415)
- ✅ Score bands: GREEN (≥85%), YELLOW (70-85%), RED (<70%) (lines 564-595)

**Result:** ✅ **UI wiring is consistent and correct.**

## C) Debug Fields Verification

### Fields in Return Dict (lines 1724-1732):
- ✅ `foot_down_idx`: `int(foot_down_idx)` - Always int, safe
- ✅ `torso_open_idx`: `int(torso_open_idx) if torso_open_idx is not None else None` - **FIXED:** No longer casts None to int
- ✅ `sep_fs_deg`: `round(float(sep_at_fs), 1)` - Separation at FS in degrees
- ✅ `sep_persist_pct`: `round(float(persistence_pct), 1)` - Persistence percentage [0, 100]
- ✅ `sep_early_drop_pct`: `round(float(early_drop_pct), 1) if early_drop_pct is not None else None` - Early drop diagnostic
- ✅ `end_idx_used`: `int(end_idx)` - **ADDED:** Actual end index used (for debugging)

### Guardrails Verified:
- ✅ `torso_open_idx` is None-safe: Only cast to int if not None (line 1728)
- ✅ `sep_fs` epsilon check: Returns None if `sep_at_fs < 1e-6` (line 1675)
- ✅ UI handles None gracefully: Shows "NA" when `sep_persistence_pct is None` (line 2394)

**Result:** ✅ **Debug fields are correct and safely handled.**

## D) Common Pitfalls - Fixed

### 1. Baseline Angle Wrap:
- ✅ **VERIFIED:** `detect_torso_open_idx` uses `wrap180(float(a) - float(base))` for delta (line 1584)
- ✅ Baseline computed as median of raw angles (line 1574), which is correct since we wrap the delta
- ✅ All separation calculations use `wrap180(float(sa) - float(pa))` (lines 1669, 1691)

### 2. Direction Ambiguity:
- ✅ **VERIFIED:** Uses `opening_sign` (from `plate_dir_sign`) to determine rotation direction (line 1588)
- ✅ Sign convention: `float(delta) * float(opening_sign) >= float(deg_thresh)` ensures correct direction
- ✅ Works for both RHH and LHH via `plate_dir_sign` parameter

### 3. Window Correctness:
- ✅ **VERIFIED:** Window is inclusive: `range(foot_down_idx, end_idx + 1)` (line 1682)
- ✅ Guardrail: `if end_idx <= foot_down_idx: return None` (line 1655)
- ✅ `detect_torso_open_idx` ensures `detected_idx > ffp_idx` (line 1593)

### 4. FPS Stability:
- ✅ **VERIFIED:** Early drop window is time-based: `int(0.10 * max(fps, 1.0))` (line 1710)
- ✅ Fallback window is time-based: `int(0.35 * max(fps, 1.0))` (line 1648)
- ✅ No hardcoded frame counts in persistence calculation

**Result:** ✅ **All common pitfalls addressed and verified.**

## E) Comments and Docstrings - Updated

### Function Docstrings:
- ✅ `compute_separation_persistence_pct`: Explicitly states "Start index: foot_down_idx (FS)" and "End index: torso_open_idx" (lines 1620-1623)
- ✅ `detect_torso_open_idx`: States "Uses a 'loaded baseline' computed from pre-FS window" (line 1542)
- ✅ `score_separation_persistence_pct`: States scoring bands and early drop penalty (lines 564-575)
- ✅ `separation_persistence_language`: Docstring explains metric definition (lines 994-998)

### UI Explanation Text:
- ✅ Explicitly states: "**Start:** Foot Strike (FS)" and "**End:** Torso-Open (first meaningful torso rotation toward pitcher; fallback to Release if not detected)" (lines 2408-2409)
- ✅ No references to contact, pseudo-contact, or peak velocity

### Inline Comments:
- ✅ Comment updated: "FFP to torso_open (or release/fallback if not detected)" (line 2384)
- ✅ All comments clarify window definition and fallback logic

**Result:** ✅ **All comments and docstrings updated and consistent.**

## F) Fixes Applied

### Fix 1: Prevent None Casting to Int
**Location:** Line 1728
**Issue:** `torso_open_idx` was being cast to int even when None
**Fix:** Changed to `int(torso_open_idx) if torso_open_idx is not None else None`

### Fix 2: Added Debug Field
**Location:** Line 1731
**Issue:** Missing field for actual end_idx used (useful when fallback is used)
**Fix:** Added `"end_idx_used": int(end_idx)` to return dict

### Fix 3: Updated Docstring
**Location:** Lines 1625-1631
**Issue:** Duplicate `sep_persist_pct` entry in docstring
**Fix:** Removed duplicate, clarified all fields

### Fix 4: Updated Comment
**Location:** Line 2384
**Issue:** Comment didn't mention fallback option
**Fix:** Updated to "FFP to torso_open (or release/fallback if not detected)"

## Summary

✅ **All verification checks passed:**
- Zero old references found
- UI wiring is correct and consistent
- Debug fields are properly handled with None-safety
- All common pitfalls addressed (angle wrapping, direction, window, FPS stability)
- Comments and docstrings updated and aligned

**The Separation Persistence metric is complete, hardened, and ready for production use.**

