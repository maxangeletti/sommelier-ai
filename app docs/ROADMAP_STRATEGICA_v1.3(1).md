# ROADMAP_STRATEGICA_v1.3 — SommelierAI (iOS + Backend)

**Data:** 2026-03-01  
**Baseline iOS:** `maxangeletti/sommelier-ai-ios` tag **v1.0.2** (scroll anchor fix match/price_value)  
**Backend:** FastAPI locale `127.0.0.1:8000` (`main.py` / `mainBCK.py`)  

> Ground rules: lavoriamo per versioni; patch minime; zero refactor non richiesto; in debug un’azione alla volta.

---

## Stato attuale (snapshot)

### iOS (SwiftUI)
- `ChatView.swift`: anchor `id("top")` + scroll-to-top su cambio `selectedSort` e `rankingMode`.
- Anchor righe vino coerenti:
  - `.id("wine:\(msg.id.uuidString):\(g.representativeIndex)")`
  - rerun scroll target: `"wine:<msgUUID>:0"`
- UI Polish:
  - Evidenza primo risultato (background tenue)
  - Micro-reason solo per #1 (`topReasonLabel`)
  - Stelle “paglierino tenue” (senza cambiare palette globale)
- Known UX:
  - banner “Ordinamento non disponibile…” appare per pochi secondi se si forza Popular/Rating → Relevance (accettato, non blocca)

### Backend (FastAPI)
- Implementata opzione **B**: debug solo se richiesto (`debug=true`)
  - POST `/search` accetta `{..., "debug": true}`
  - GET `/search_stream` accetta `?debug=true`
- Con `debug=true` vengono inclusi nei risultati:
  - `__match_score`, `__quality_score`, `__value_score`, `__final_score`
  - `match_breakdown` / `match_explanation` (se presenti)
  - in `final.meta.debug`: righe + delta_vs_top (se attivo)
- Default produzione: debug OFF (niente `__quality/__value/__final` né breakdown)

---

## Roadmap (modello 27.02) — Fase 0..6

> Percentuali: stato di completamento complessivo della fase (non “precisione scientifica”, ma tracking pratico).

### Fase 0 — Freeze & disciplina di versione (90%)
- ✅ Tag e baseline stabili (iOS v1.0.2)
- ✅ Regole operative + patch minime
- 🔶 Mancano: checklist “prima di cambiare sessione” standardizzata in doc

**Prossimo micro-step:** aggiungere checklist in doc (non cambia codice)

---

### Fase 1 — UX & qualità percepita (70%)
- ✅ Scroll anchor fix + rerun coerente
- ✅ Polish #1 (highlight + micro-reason)
- 🔶 Banner sort non disponibile: comportamento noto (accettato)

**Prossimo micro-step:** (se richiesto) rendere banner meno “flash” (solo UI)

---

### Fase 2 — Ranking Engine v2 (core) (55%)
- ✅ Match scoring funziona e osservabile
- ✅ Debug flag (B) con breakdown opzionale
- 🔶 Dataset duplicati/cleanup: parziale
- 🔶 Definizione formalizzata di “Match” in doc: da completare (vedi sezione Match)

**Prossimo micro-step:** consolidare documentazione di scoring + casi A/B/C/D

---

### Fase 3 — Osservabilità & strumenti dev (60%)
- ✅ Ranking Debugger (B) via `debug=true`
- ✅ `build_id` presente in meta SSE final (già utile per log iOS)
- 🔶 Mancano: esempi “ricette curl” in doc + interpretazione delta_vs_top

**Prossimo micro-step:** aggiungere esempi curl + legenda campi

---

### Fase 4 — Dataset & qualità risultati (35%)
- 🔶 Dedup (Franciacorta Brut ecc.): non risolto
- 🔶 Normalizzazione campi (grapes/typology): in corso
- ✅ Filtri base stabili

**Prossimo micro-step:** dedup deterministico + regole (senza cambiare UX)

---

### Fase 5 — Productization (TestFlight / stabilità) (20%)
- 🔶 Bundle TestFlight (solo se richiesto)
- 🔶 Performance/cache end-to-end (SSE + UI) da verificare

**Prossimo micro-step:** definire “release checklist” + metriche latenza

---

### Fase 6 — Growth & Marketing (10%)
- 🔶 Canali, posizionamento, pricing, eventi (Vinitaly ecc.) — pending

**Prossimo micro-step:** ipotesi 2–3 canali + KPI minimi (documento separato)

---

## Punti da verificare in futuro (bassa priorità)
- UX: banner sort non disponibile (ridurre flicker)
- iOS: mostrare breakdown solo in build debug (non in release) se si integra UI
- Backend: definire “dedup key” (producer+denom+zone+vintage+price?) senza perdere varianti utili
- Ranking: controllare edge-case con `overall` temporaneamente a 0.0 (già rientrato)

---

## Nota rapida (non cambio ora)
- Non affrontiamo ora: dedup dataset, refactor architetturale, UI overhaul banner.
- Non cambiamo ora: scaling di `score` (rimane ~0..5) vs componenti normalizzate (0..1).
