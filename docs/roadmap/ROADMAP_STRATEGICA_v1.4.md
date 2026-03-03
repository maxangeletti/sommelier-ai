📘 SOMMELIERAI — ROADMAP STRATEGICA UFFICIALE

Versione: 1.4
Data: 03.03.2026
Baseline tecnica: v0.2.4
Stato progetto: Alpha avanzata – Ranking Engine stabile

🟢 FASE 0 — FONDAZIONE (100% ✅)

Obiettivo: MVP funzionante con ranking proprietario.

✔ Backend FastAPI
✔ Dataset vini iniziale
✔ Ranking A9 (v1)
✔ Match score base
✔ Sort relevance / price / rating / match
✔ SSE streaming
✔ UI chat base
✔ Favorites
✔ Tier Free
✔ Grouping annate
✔ Filtri principali

Status: COMPLETATA

🔵 FASE 1 — HARDENING & STABILITÀ (95% ✅)

Obiettivo: rendere il motore deterministic, spiegabile e stabile.

✔ Completato

A9v2 Composite Engine

Match come moltiplicatore (no double count)

Target price mode con proximity bonus (0.06)

Delta breakdown (delta_vs_top)

Flatten match_breakdown

Value intent override (relevance → relevance_v2)

Test Matrix v1.0

Freeze + versioning strutturato

Repository restructuring professionale

⏳ Residuo (5%)

Golden query set definitivo (12 query)

Snapshot ufficiali baseline (CI futura)

Status: STABILE — Production-ready logic

🟣 FASE 2 — QUALITÀ PERCEPITA (60%)

Obiettivo: rendere evidente il salto qualitativo del ranking.

🔄 In corso

Debug composito lato backend (completo)

Breakdown differenziale (completo)

🔜 Da completare

UI Explainability chiara (perché è primo?)

Badge semantici intelligenti

Visual ranking bar coerente con composite

Modalità ranking selezionabile utente

🟡 FASE 3 — DATA & INTELLIGENCE (30%)

Obiettivo: aumentare qualità reale del motore.

🔜 Previsto

Expert Ratings Integration

Campo: expert_rating_avg

Peso opzionale 0.15–0.25

Dataset cleanup

Espansione vitigni + aromi

Campo “indicato per”

Nuove caratteristiche sensoriali

🔴 FASE 4 — PERFORMANCE & SCALE (20%)

Cache risultati

Debounce intelligente

Ottimizzazione latenza

Refactor architettura modulare ranking

🟠 FASE 5 — STRATEGIA & GO-TO-MARKET (15%)

Posizionamento prodotto

Strategia pricing

Target utenti

Marketing roadmap

Eventi settore (Vinitaly ecc.)

🧠 SEZIONE UFFICIALE — MATCH ENGINE
Formula finale (A9v2)
composite01 = overall_base * match_factor
(+ target proximity bonus 0.06 se attivo)
Componenti:

Q → Quality

V → Value

F → Food alignment

O → Occasion intent

I → Intensity alignment

M → Match score (moltiplicatore 0.55–1.00)

Caratteristiche:

Deterministico

Explainable

Delta breakdown tra rank

No double counting

Target price soft (non dominante)

Status: ENGINE STABILE v1.0

📊 Stato Complessivo Progetto
Area	Stato
Ranking Core	100%
Debug & Explainability Backend	100%
UI Explainability	60%
Data Intelligence	30%
Performance	20%
Strategia	15%
🎯 Prossimo Milestone Raccomandato

👉 UI Explainability upgrade
(per rendere percepibile il salto qualitativo già raggiunto nel backend)

🔒 Freeze di riferimento

v0.2.4

ranking-v0.6-proximity-006