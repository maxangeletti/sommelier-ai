# SommelierAI — Project Context v1.2

## Identità
App iOS per raccomandazione vini tramite linguaggio naturale. Motore di ranking A9v2 proprietario. "Il vino giusto, per il momento giusto."

## Stack
- **Frontend:** SwiftUI (iOS) — chat-style
- **Backend:** FastAPI (Python) — `backend/main.py` (2600+ righe)
- **Hosting:** Render.com (free tier) — https://sommelier-ai.onrender.com
- **Dataset:** `data/wines.normalized.csv` — 53 vini, 27 colonne
- **LLM Runtime:** claude-haiku-4-5-20251001 (parser semantico)
- **LLM Sviluppo:** claude-opus-4-6
- **Repo:** github.com/maxangeletti/sommelier-ai
- **Monorepo:** `backend/` (Python), `ios-app/SommelierAI/SommelierAI/` (Swift), `data/` (CSV)

## Dogmi (NON NEGOZIABILI)
- Il motore A9v2 NON si tocca senza sessione dedicata e GT baseline
- Il CSV non si modifica per logiche di ranking — solo arricchimento editoriale
- L'LLM non sostituisce il ranking engine — lo arricchisce
- L'interfaccia è sempre in italiano — il codice in inglese
- Ogni modifica al motore richiede classificazione A/B/C/D e test GT-01→GT-13
- Mai esporre canonical tecnici con underscore all'utente finale
- Modificare solo ciò che è richiesto — no refactoring non richiesto
- Non eliminare o riscrivere nulla senza motivo chiaro

## Stato (2026-03-20)

### Completati ✅
- Blocco tecnico BE+FE ~95%
- GT-01 → GT-13 verificati e stabili
- LLM Layer integrato (llm_intent_parser.py)
- Deploy backend live su Render
- App iOS funzionante su device
- UI-A: Debug scores off (rankingDebugMode = false)
- UI-B: Underscore e tag tecnici filtrati
- UI-C: Card espandibile (Più/Meno dettagli)
- UI-D: Cestino in navbar + filtri solo post-risultati
- UI-E: Accordion filtri collassato di default
- UI-F: Rating/Popolari nascosti dal sort menu
- Scala tannicità 5 livelli AIS (low/medium_low/medium/medium_plus/high)

### Da fare 🔄
- Tuning peso intensity_score (I) — sessione dedicata
- LLM Explain personalizzato (Step 2)
- Estetica: icona, splash, background theme
- Pillola del giorno
- Apple Developer Account ($99/anno) per App Store

## Scala Tannicità AIS (5 livelli)
- `low` (0.10) — astringenza appena percettibile
- `medium_low` (0.30) — tannino presente ma morbido
- `medium` (0.50) — tannino evidente ma equilibrato
- `medium_plus` (0.75) — astringenza marcata, struttura importante
- `high` (1.00) — tannino molto incisivo, asciugante e persistente

## File Principali
- `backend/main.py` — motore ranking A9v2 + API FastAPI
- `backend/llm_intent_parser.py` — parser semantico LLM
- `data/wines.normalized.csv` — dataset vini
- `ios-app/SommelierAI/SommelierAI/ChatView.swift` — UI chat principale
- `ios-app/SommelierAI/SommelierAI/ChatViewModel.swift` — logica chat
- `ios-app/SommelierAI/SommelierAI/ChatTypes.swift` — tipi dominio chat
- `ios-app/SommelierAI/SommelierAI/Models.swift` — modelli dati
- `ios-app/SommelierAI/SommelierAI/APIClient.swift` — networking
- `ios-app/SommelierAI/SommelierAI/AppColors.swift` — colori app
- `ios-app/SommelierAI/SommelierAI/FavoritesStore.swift` — preferiti
- `ios-app/SommelierAI/SommelierAI/TierStore.swift` — tier free/premium

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
