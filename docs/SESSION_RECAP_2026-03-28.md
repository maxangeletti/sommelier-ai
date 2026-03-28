# SESSION RECAP - 2026-03-28

## OBIETTIVI INIZIALI
1. Fix query "barolo" → ritornava 8 vini misti invece di 3 Barolo
2. Fix query "poco tannico" → ritornava bianchi invece di rossi leggeri
3. Implementare paginazione 5+5+5 risultati

## COMPLETATO ✅

### 1. LLM No-Inference (Denominazioni)
**Problema:** Query "barolo" inferiva `grapes: ["nebbiolo"]`, `region: "piemonte"`, `color: "rosso"` → backend filtrava TUTTI i rossi nebbiolo del Piemonte (8 vini).

**Fix:** Modificato `backend/llm_intent_parser.py` - aggiunto 3 regole CRITICO:
- NO grape inference da denominazioni
- NO region inference da denominazioni specifiche  
- NO color inference da denominazioni

**Commit:** `be6f43c`

**Test:**
- "barolo" → grapes: [], region: None, color: None ✅
- "nebbiolo" → grapes: ["nebbiolo"] ✅
- "vino del piemonte" → region: "piemonte" ✅

### 2. Keyword Exact-Match Filter
**Problema:** Anche con LLM fix, "barolo" ritornava 8 vini (qualità domina ranking).

**Fix:** Modificato `backend/main.py` - aggiunto filtro post-LLM:
```python
# Se query matcha esattamente nome/denominazione, mostra SOLO quelli
if q_lower in name_lower or q_lower in denomination_lower:
    keyword_matches.append(idx)
if len(keyword_matches) > 0 and len(keyword_matches) < len(filtered):
    filtered = filtered.loc[keyword_matches]
```

**Commit:** `be6f43c`

**Test:**
- "barolo" → 3 Barolo ✅
- "brunello" → 1 Brunello ✅
- "franciacorta" → 2 Franciacorta ✅

### 3. Tannin → Color Rosso Implicito
**Problema:** Query "poco tannico" ritornava bianchi (Champagne, Riesling) invece di rossi leggeri (Schiava, Frappato).

**Rationale:** Secondo principi AIS, tannini nei bianchi sono "irrisori/impercettibili" → discussione tannini rilevante SOLO per rossi.

**Fix:** Modificato `backend/main.py`:
```python
# Se tannins richiesto E color non specificato → implicito color=rosso
if typology_req.get("tannin") and not color_req:
    color_req = "rosso"
```

**Commit:** `795aed5`

**Test:**
- "poco tannico" → 3 rossi (Frappato, Lambrusco, Schiava) ✅
- "rosso tannico" → Sagrantino #1 ✅

### 4. Backend total_count (Preparazione Paginazione)
**Problema:** Backend con `limit=5` ritornava `meta.count=5` invece del totale disponibile → iOS non sa se ci sono più risultati.

**Fix:** Modificato `backend/main.py`:
```python
total_count = len(scored)  # BEFORE limit
sorted_cards = _apply_sort(scored, sort, value_intent=value_intent)[:limit]
meta["total_count"] = total_count
```

**Commit:** `d995638`

**Test:**
- `limit=5` → `count: 5, total_count: 47` ✅

### 5. iOS Dark Mode Fix
**Problema:** App con testo invisibile in Dark Mode (colori statici chiari).

**Fix:** Modificato `ios-app/SommelierAI/SommelierAI/AppColors.swift`:
```swift
static let backgroundPrimary = Color(UIColor.systemBackground)
static let backgroundSecondary = Color(UIColor.secondarySystemBackground)
```

**Commit:** Già presente in git (non committato nuovamente)

**Test:** App funziona in Light e Dark Mode ✅

---

## NON COMPLETATO ❌

### Paginazione iOS (5+5+5+5)
**Motivo:** Implementazione complessa (4 file iOS), errori compilazione, regression.

**Stato:** 
- Backend pronto (`total_count` disponibile)
- iOS NON implementato
- File ripristinati a commit `354088a` (ultimo funzionante)

**Rimandato a:** Prossima sessione

---

## DEPLOY STATUS

### Backend v1.6.0
**URL:** https://sommelier-ai.onrender.com
**Commit:** `d995638`
**Features:**
- ✅ LLM no-inference (grapes/region/color da denominazioni)
- ✅ Keyword exact-match filter
- ✅ Tannin → color rosso
- ✅ total_count in meta

### iOS
**Status:** Locale funzionante
**Commit base:** `354088a`
**Features:**
- ✅ Dark Mode compatibile
- ✅ Tutti i fix v1.5 presenti
- ❌ Paginazione NON implementata

---

## COMMITS SESSIONE

```
d995638 - feat(api): add total_count to meta for pagination
795aed5 - fix(search): poco tannico implica color rosso  
be6f43c - fix(search): LLM prompt + keyword exact match filter
```

---

## LEZIONI APPRESE

### Cosa Ha Funzionato
1. **Test backend locale prima di deploy** - zero regression
2. **Modifiche incrementali backend** - un fix alla volta
3. **Web search per validare concetti** (AIS tannini)

### Cosa NON Ha Funzionato
1. **Modifiche iOS senza backup adeguati** → file corrotti
2. **Dire "funziona" guardando screenshot di fretta** → perso fiducia
3. **Modifiche simultanee a 4 file iOS** → troppe dipendenze, errori compilazione
4. **str_replace su righe lunghe** → split accidentali, syntax errors

### Miglioramenti Prossima Sessione
1. **Backup iOS verificabili PRIMA di modifiche**
2. **Test compilazione dopo OGNI singolo file modificato**
3. **NO commit/push finché utente non conferma funzionante**
4. **Leggere file REALI prima di modificarli** (no assunzioni)

---

## FILE CRITICI MODIFICATI

**Backend:**
- `backend/llm_intent_parser.py` - LLM prompt rules
- `backend/main.py` - keyword filter, tannin→color, total_count

**iOS:**
- `ios-app/SommelierAI/SommelierAI/AppColors.swift` - Dark Mode

**Git Status:**
- Local = Remote
- Working tree clean
- Untracked: backup files, docs (normale)
