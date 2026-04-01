# iOS Integration API Documentation

**Backend Version**: v1.8.0  
**Last Updated**: 2026-04-01  
**Base URL Production**: `https://sommelier-ai.onrender.com`  
**Base URL Local**: `http://localhost:8000`

---

## Table of Contents

1. [Wine Details Endpoint](#wine-details-endpoint)
2. [Search Endpoint](#search-endpoint)
3. [Response Fields Reference](#response-fields-reference)
4. [Derived Fields Logic](#derived-fields-logic)
5. [Error Handling](#error-handling)
6. [Rate Limits & Caching](#rate-limits--caching)

---

## Wine Details Endpoint

### `GET /wine/{wine_id}/details`

Retrieves complete wine information including LLM-generated tasting notes, aroma icons, and derived characteristics.

**Use Case**: Screen 4 - Wine Details (Hero section + characteristics bars)

### Request

```bash
GET /wine/1/details
```

**Path Parameters**:
- `wine_id` (string, required): Wine ID from search results

### Response

```json
{
  "wine": {
    "id": "1",
    "name": "Trebbiano d'Abruzzo DOC",
    "producer": "Valle Reale",
    "region": "Abruzzo",
    "denomination": "Trebbiano d'Abruzzo",
    "vintage": "2022",
    "grapes": "trebbiano",
    "aromas": "agrumi, fiori, minerale, pesca",
    "food_pairings": "pasta_sugo_verdure|pesce_griglia",
    "quality": "2.8",
    "price": 9.5,
    
    "tasting_notes": "Bianco fermo di buona struttura...",
    
    "aroma_icons": [
      {
        "name": "Agrumi",
        "icon": "🍋"
      },
      {
        "name": "Fiori",
        "icon": "🌹"
      }
    ],
    
    "reviews_count": 100,
    "critic_score": 70,
    
    "intensity": "low",
    "sparkling": "fermo",
    "freshness": "medium"
  }
}
```

### Response Fields

| Field | Type | Description | UI Usage |
|-------|------|-------------|----------|
| `tasting_notes` | string | LLM-generated tasting notes (60-80 words) | **Screen 4**: "Note di degustazione del Sommelier" section |
| `aroma_icons` | array | List of aromas with emoji icons | **Screen 4**: Aroma icons row |
| `reviews_count` | integer | Mock reviews count (derived from quality) | **Screen 4**: Hero stats |
| `critic_score` | integer | Mock critic score 0-100 (derived from quality) | **Screen 4**: Hero stats |
| `intensity` | string | Wine intensity: `low`, `medium`, `high` | **Screen 4**: Intensità bar |
| `sparkling` | string | Wine type: `fermo`, `frizzante`, `spumante` | **Screen 4**: Tipologia indicator |
| `freshness` | string | Freshness level: `low`, `medium`, `high` | **Screen 4**: **Fresco bar** |

### Error Responses

**404 Not Found** - Wine ID not found:
```json
{
  "error": "Wine not found"
}
```

---

## Search Endpoint

### `GET /search_stream` or `POST /search`

Returns ranked wine recommendations based on user query.

**Use Case**: Screen 2 - Search Results (Cards with match scores, badges, reason)

### Request

```bash
GET /search_stream?query=un%20sangiovese&sort=relevance&limit=8
```

**Query Parameters**:
- `query` (string, required): User search query
- `sort` (string, optional): Sorting mode
  - `relevance` (default): Best match first
  - `match`: Match score priority
  - `price_asc`: Lowest price first
  - `price_desc`: Highest price first
- `limit` (integer, optional): Max results (default: 8, max: 30)
- `debug` (boolean, optional): Include debug info (default: false)

### Response (SSE Stream)

```
data: {"type": "delta", "wine": {...}}
data: {"type": "delta", "wine": {...}}
data: {"type": "final", "results": [...], "meta": {...}}
data: [DONE]
```

### Wine Card Fields

```json
{
  "id": "5",
  "name": "Chianti Classico DOCG",
  "producer": "Antinori",
  "price": "18.50",
  "rank": 1,
  "score": 4.2,
  "match_score": 0.87,
  
  "reason": "Un Sangiovese toscano di grande equilibrio...",
  
  "tags": "elegante, ruby_red, medium",
  "region": "Toscana",
  "denomination": "Chianti Classico",
  "vintage": "2020",
  "grapes": "sangiovese",
  "food_pairings": "carne|pasta|formaggi",
  
  "show_value_badge": true,
  
  "aroma_icons": [...],
  "reviews_count": 150,
  "critic_score": 85
}
```

### Card Fields for UI

| Field | Type | Description | UI Usage |
|-------|------|-------------|----------|
| `rank` | integer | Position in results (1, 2, 3...) | **Screen 2**: Badge "#1", "#6", "#7" |
| `match_score` | float | Match percentage 0.0-1.0 | **Screen 2**: Match bar (0-100%) |
| `reason` | string | LLM-generated personalized reason (5-40 words) | **Screen 2**: Card text<br>**Screen 3**: Intro paragraph (reuse) |
| `show_value_badge` | boolean | Whether to show "Ottimo Valore" badge | **Screen 2**: "Ottimo Valore" badge |
| `quality` | string | Quality score (e.g., "4.2") | **Screen 2**: Star rating (quality/5 * 5) |

---

## Response Fields Reference

### Core Wine Data

| Field | Type | Always Present | Description |
|-------|------|----------------|-------------|
| `id` | string | ✅ | Unique wine identifier |
| `name` | string | ✅ | Wine name |
| `producer` | string | ✅ | Producer/winery name |
| `region` | string | ✅ | Geographic region |
| `denomination` | string | ✅ | Official denomination (DOC, DOCG, etc.) |
| `vintage` | string | ✅ | Year (e.g., "2020") |
| `price` | float | ✅ | Average price in EUR |
| `grapes` | string | ✅ | Grape varieties (comma-separated) |

### LLM-Generated Content

| Field | Type | Always Present | Description |
|-------|------|----------------|-------------|
| `reason` | string | ✅ | Personalized recommendation reason (5-40 words) |
| `tasting_notes` | string | `/details` only | Professional tasting notes (60-80 words) |

### UI Helper Fields

| Field | Type | Always Present | Description |
|-------|------|----------------|-------------|
| `aroma_icons` | array | ✅ (if aromas present) | List of `{name, icon}` objects |
| `reviews_count` | integer | ✅ | Mock reviews count (100-300) |
| `critic_score` | integer | ✅ | Mock critic score (60-95) |
| `show_value_badge` | boolean | Search only | True if "Ottimo Valore" badge should show |

### Derived Characteristics

| Field | Type | Values | Formula |
|-------|------|--------|---------|
| `intensity` | string | `low`, `medium`, `high` | Derived from body + tannins + alcohol |
| `sparkling` | string | `fermo`, `frizzante`, `spumante` | Derived from denomination + style |
| `freshness` | string | `low`, `medium`, `high` | **NEW**: Derived from acidity + sparkling - alcohol |

---

## Derived Fields Logic

### Freshness Calculation

```
freshness_score = 0

// Acidity contribution
if acidity == "high": score += 2
if acidity == "medium": score += 1

// Sparkling boost
if sparkling in ["spumante", "frizzante"]: score += 1

// Alcohol penalty
if alcohol >= 14.0%: score -= 1

// Final mapping
if score >= 3: freshness = "high"
if score >= 1: freshness = "medium"
else: freshness = "low"
```

**Example**:
- Wine: Trebbiano d'Abruzzo (white, acidity: high, alcohol: 12.5%, sparkling: fermo)
- Score: 2 (high acidity) + 0 (no sparkling) - 0 (alcohol < 14%) = **2**
- Result: `freshness = "medium"`

### Intensity Calculation

Derived from:
- `body` (low/medium/high)
- `tannins` (low/medium/high)
- `alcohol_level` (numeric, e.g., 13.5)

Logic: If 2+ signals are "high" → `intensity = "high"`

### "Ottimo Valore" Badge Logic

Shows badge when:
```
quality >= 3.5 AND price <= 25 AND (quality/price) >= 0.18
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Response |
|------|---------|----------|
| 200 | Success | Wine data returned |
| 404 | Not Found | `{"error": "Wine not found"}` |
| 500 | Server Error | `{"error": "Internal server error"}` |

### Handling Missing Fields

All fields marked as "Always Present" are guaranteed in response. Optional fields:
- `aroma_icons`: Empty array `[]` if no aromas
- `freshness`, `intensity`, `sparkling`: May be `null` if derivation fails (rare)

**Recommended**: Always provide fallback UI for optional fields.

---

## Rate Limits & Caching

### Production Limits

- **Rate Limit**: None currently (may be added in future)
- **Cache TTL**: 45 seconds for search results
- **LLM Cache**: Reason and tasting notes are cached per wine+query combination

### Best Practices

1. **Debounce search queries**: Wait 300ms after user stops typing
2. **Cache results locally**: Store search results in app state
3. **Lazy load details**: Call `/wine/{id}/details` only when user taps card
4. **Handle loading states**: LLM calls take 2-3s on cold start

### Response Times (Typical)

- Search (cached): **< 100ms**
- Search (uncached): **1-2s** (includes LLM reason generation)
- Wine details (cached): **< 100ms**
- Wine details (uncached): **2-3s** (includes LLM tasting notes generation)

---

## Example Integration Flow

### Screen 2: Search Results

```swift
// 1. User types query
let query = "un sangiovese toscano"

// 2. Call search endpoint
GET /search_stream?query=\(query)&limit=8

// 3. Parse SSE stream
for event in stream {
  if event.type == "delta" {
    // Add wine card to UI
    addWineCard(event.wine)
  }
}

// 4. Display cards with:
- Rank badge (wine.rank)
- Match bar (wine.match_score * 100)
- "Ottimo Valore" badge (if wine.show_value_badge)
- Reason text (wine.reason)
- Quality stars (wine.quality / 5 * 5)
```

### Screen 3: Wine Selected

```swift
// Reuse search result data
let wine = selectedWine

// Display:
- Badge: "#\(wine.rank) MIGLIOR ABBINAMENTO"
- Match: "\(wine.match_score * 100)%"
- Intro: wine.reason  // ← Reuse reason from search
```

### Screen 4: Wine Details

```swift
// 1. Fetch full details
GET /wine/\(wine.id)/details

// 2. Display bars:
- Alcool: wine.alcohol_level
- Dolce: wine.sweetness
- Morbido: derived from body/tannins
- Tannico: wine.tannins
- Intenso: wine.intensity  // ← NEW
- Corposo: wine.body
- Fresco: wine.freshness  // ← NEW

// 3. Display tasting notes
textView.text = wine.tasting_notes

// 4. Display aroma icons
for aroma in wine.aroma_icons {
  addIcon(aroma.icon, label: aroma.name)
}
```

---

## Changelog

### v1.8.0 (2026-04-01)
- ✅ Added `freshness` field to `/wine/{id}/details`
- ✅ Added `intensity`, `sparkling` to `/wine/{id}/details`
- ✅ Fixed ordering bug: `sparkling` now derived before `freshness`

### v1.7.0 (2026-03-30)
- ✅ Added LLM-generated personalized `reason` in search results
- ✅ Added LLM `tasting_notes` in `/wine/{id}/details`
- ✅ Added `aroma_icons` mapping
- ✅ Added `show_value_badge` logic
- ✅ Added mock `reviews_count` and `critic_score`

---

## Support

**Issues**: GitHub Issues  
**API Questions**: Backend team  
**Production URL**: https://sommelier-ai.onrender.com  
**Deploy Status**: Auto-deploy on `main` branch push

---

*End of Documentation*
