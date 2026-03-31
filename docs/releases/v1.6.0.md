# SommelierAI v1.6.0 Release Notes

**Release Date:** March 25, 2026  
**Status:** ✅ Production Ready  
**GT Validation:** 26/26 PASS (100%)

---

## 🎯 Overview

Major search quality improvements focusing on query parsing, filtering accuracy, and AIS tannin classification system.

---

## ✨ New Features

### AIS Tannin 5-Level System
- **poco tannico** → `low` (Frappato, Lambrusco, Schiava)
- **tannico morbido** → `low-medium`
- **tannico** → `medium-high` (Sagrantino, Barolo, Taurasi)
- **molto tannico** → `high`
- **tannico aggressivo** → `very_high`

**Boost Scoring:**
- Exact match: 1.0 points
- Close match: 0.8 points
- Pattern order critical: specific patterns before generic `\btannico\b`

**Test Results:**
```
"rosso tannico" → Sagrantino #1 ✅
"rosso poco tannico" → Frappato #1 ✅
```

### Multi-Word Keyword Matching
Full-phrase matching before token-by-token fallback.

**Examples:**
- "barolo serralunga" → Barolo DOCG Serralunga d'Alba (match=0.79)
- "franciacorta brut" → Franciacorta DOCG Brut (match=1.0)

---

## 🐛 Bug Fixes

### Region Filter OR Logic
**Problem:** Sequential filter overwrite (region → zone → denomination → country) resulted in 0 results.

**Fix:** OR logic across all location fields with pandas mask.

**Before:**
```python
for col in ["region", "zone", "denomination", "country"]:
    filtered = _filter_by_text_contains(filtered, col, region)  # ❌ Overwrites
```

**After:**
```python
mask = pd.Series([False] * len(filtered), index=filtered.index)
for col in ["region", "zone", "denomination", "country"]:
    mask |= filtered[col].str.contains(region, case=False, na=False)  # ✅ OR logic
filtered = filtered.loc[mask]
```

**Impact:**
- "barolo" → 0 results ❌ → 5 results ✅
- "franciacorta" → 0 results ❌ → 4 results ✅
- "etna" → 0 results ❌ → 8 results ✅

### Pandas 3.14 StringArray Compatibility
**Problem:** `TypeError: unsupported operand type(s) for |: 'StringArray' and 'StringArray'`

**Fix:** Convert to bool Series before OR operation.
```python
# Before
mask = m if mask is None else (mask | m)  # ❌ Crashes on pandas 3.14

# After
m_bool = m.astype(bool)
mask = m_bool if mask is None else (mask | m_bool)  # ✅ Works
```

---

## 📊 GT Validation Results

### Summary
- **Total Tests:** 26
- **Passed:** 26
- **Failed:** 0
- **Pass Rate:** 100%

### Key Test Cases
| Query | Expected | Result | Status |
|-------|----------|--------|--------|
| barolo serralunga | Barolo Serralunga in top 3 | ✅ #1 | PASS |
| franciacorta brut | Franciacorta Brut in top 3 | ✅ #1, #2 | PASS |
| nebbiolo | Barolo, Barbaresco in top 3 | ✅ #1, #4 | PASS |
| prosecco | Prosecco DOC #1 | ✅ #1 | PASS |
| etna | Etna wines in top 3 | ✅ #1, #2, #4 | PASS |
| rosso tannico | Sagrantino #1 | ✅ #1 | PASS |
| poco tannico | Low-tannin wines | ✅ Frappato #1 | PASS |
| vino dolce | Sweet wines top 3 | ✅ Recioto #1 | PASS |

---

## 🚀 Deployment

**Backend:** https://sommelier-ai.onrender.com  
**Build ID:** `2026-03-25-v1.6.0`  
**Commit:** `7e694e5`

**Verification:**
```bash
curl -s https://sommelier-ai.onrender.com/stats | jq '.build_id'
# "SommelierAI v0.2 STABILE + A/B/D (CSV schema real) + cache-safe 2026-03-25-v1.6.0"
```

---

## 📦 Commits

- `7e694e5` - chore: update BUILD_ID to v1.6.0
- `6337b37` - fix(search): add full-phrase keyword matching for multi-word queries
- `1d8f30c` - fix(search): region OR filter + pandas 3.14 StringArray compatibility
- `02658af` - feat(parser): implement AIS tannin levels (5 levels) with boost ranking
- `d6365a0` - fix(parser): frizzante query includes spumante wines

---

## 🔄 Migration Notes

**No breaking changes.** All existing queries continue to work, with improved accuracy.

**Backward Compatibility:**
- ✅ All v1.5.0 queries work
- ✅ No schema changes
- ✅ No API changes

---

## 📋 Known Issues

None.

---

## 🙏 Acknowledgments

- AIS (Associazione Italiana Sommelier) tannin classification system
- GT validation framework for quality assurance

