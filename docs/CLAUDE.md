# SommelierAI — Project Context v1.4

## Identità
App iOS per raccomandazione vini tramite linguaggio naturale. Motore di ranking A9v2 proprietario. "Il vino giusto, per il momento giusto."

## Stack
- **Frontend:** SwiftUI (iOS) — chat-style
- **Backend:** FastAPI (Python) — `backend/main.py` (2800+ righe)
- **Hosting:** Render.com (free tier) — https://sommelier-ai.onrender.com
- **Dataset:** `data/wines.normalized.csv` — 100 vini, 27 colonne (live su Render ✅)
- **LLM Runtime:** claude-haiku-4-5-20251001 (dual-step: parse + explain)
- **LLM Sviluppo:** claude-opus-4-6 → claude-sonnet-4-6
- **Repo:** github.com/maxangeletti/sommelier-ai
- **Monorepo:** `backend/` (Python), `ios-app/SommelierAI/SommelierAI/` (Swift), `data/` (CSV)

## Dogmi (NON NEGOZIABILI)
- Il motore A9v2 NON si tocca senza sessione dedicata e GT baseline
- Il CSV non si modifica per logiche di ranking — solo arricchimento editoriale
- L'LLM non sostituisce il ranking engine — lo arricchisce (dual-step: parse + explain)
- L'interfaccia è sempre in italiano — il codice in inglese
- Ogni modifica al motore richiede classificazione A/B/C/D e test GT-01→GT-26
- Mai esporre canonical tecnici con underscore all'utente finale
- Modificare solo ciò che è richiesto — no refactoring non richiesto
- Non eliminare o riscrivere nulla senza motivo chiaro

## Architettura LLM Dual-Step [C-experimental]
Il sistema LLM è integrato con architettura dual-step:
- **Step 1 — Parse:** LLM estrae intent semantici complessi (regioni, occasioni laterali, prestige emotivo)
- **Step 2 — Explain:** LLM genera reason personalizzata contestuale alla query utente
- **Rule-based fallback:** Sempre attivo in parallelo — il motore combina segnali LLM + rule-based
- **Region alias map:** Attiva — "etna" → "Etna", "langhe" → "Langhe", "valtellina" → "Valtellina"

## Stato (2026-03-20 post-sessione Opus)

### Completati ✅
- Blocco tecnico BE+FE ~96%
- GT v2.1: **22 PASS, 4 WARN, 0 FAIL** (matrice v2.1)
- Dataset 100 vini live su Render verificato
- Scala tannicità 5 livelli AIS
- tannin_req separato da intensity_req
- Intensity soft filter con fallback
- Intensity match graduale (distanza AIS)
- UI-A→F completati
- Deploy backend live su Render
- App iOS funzionante su device
- Claude Code operativo (Opus → Sonnet)
- **LLM Step 1 integrato** (parse semantico) [C-experimental]
- **Region alias map attiva** (etna, langhe, valtellina)
- **Fix GT-23, GT-24, GT-26** tutti PASS

### GT v2.1 — Risultati Consolidati
**22 PASS, 4 WARN, 0 FAIL**

WARN rimanenti (accettabili, non bloccanti):
- **GT-09** (multi-signal bianco+fresco+pesce+cena importante) — Champagne domina per qualità base, ma ha food_pairing pesce + occasion important_dinner → comportamento corretto
- **GT-10** (vino importante generico) — Champagne #1 per qualità, rossi prestigiosi in top 5 → accettabile
- **GT-16** (rosso elegante non troppo tannico) — Gevrey-Chambertin ha tannins=medium, non low — dataset constraint
- **GT-21** (nebbiolo) — Barbaresco #1, Barolo #2 — entrambi corretti, ordine dipende da score base

### Da fare ⏳
- LLM Step 2: explain personalizzato (reason contestuale alla query)
- Estetica app: icona, splash, background theme
- Arricchimento KNOWN_GRAPES (14 vitigni mancanti)
- Apple Developer Account per App Store

## Scala Tannicità AIS (5 livelli)
- `low` (0.10) — astringenza appena percettibile
- `medium_low` (0.30) — tannino presente ma morbido
- `medium` (0.50) — tannino evidente ma equilibrato
- `medium_plus` (0.75) — astringenza marcata, struttura importante
- `high` (1.00) — tannino molto incisivo, asciugante e persistente

Nota: tannicità ≠ intensità. `tannin_req` confronta direttamente col campo tannins CSV. `intensity_req` deriva da body+tannins+alcohol.

## File Principali
- `backend/main.py` — motore ranking A9v2 + API FastAPI
- `backend/llm_intent_parser.py` — parser semantico LLM (NON integrato)
- `data/wines.normalized.csv` — dataset 100 vini
- `docs/technical/RANKING_TEST_MATRIX_v2.0.md` — matrice GT ufficiale (26 test)
- `ios-app/SommelierAI/SommelierAI/ChatView.swift` — UI chat principale
- `ios-app/SommelierAI/SommelierAI/ChatViewModel.swift` — logica chat
- `ios-app/SommelierAI/SommelierAI/ChatTypes.swift` — tipi dominio chat
- `ios-app/SommelierAI/SommelierAI/Models.swift` — modelli dati
- `ios-app/SommelierAI/SommelierAI/APIClient.swift` — networking

## API Endpoint (per test)
```bash
curl -s -X POST "https://sommelier-ai.onrender.com/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"vino tannico e strutturato"}' | python3 -m json.tool
```

## Classificazione Modifiche
- **A** = Freeze/Foundation — non toccare
- **B** = Accepted tuning — approvato, testare con GT
- **C** = Experimental — da validare
- **D** = Rejected — scartato
