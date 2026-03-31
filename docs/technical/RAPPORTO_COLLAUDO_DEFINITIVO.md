# SommelierAI — Rapporto di Collaudo GT v2.0 (DEFINITIVO)

**Data:** 2026-03-20  
**Dataset:** 100 vini live su Render  
**Endpoint:** `POST https://sommelier-ai.onrender.com/search`  
**Build:** post fix GT-24/GT-26 + dataset 100 vini  
**Fix applicati:** 3 (classificazione B) — REGION_PATTERNS, filtro OR, prestige patterns

---

## Sommario

| Metrica | Risultato |
|---------|-----------|
| **PASS** | **22/26** |
| **WARN** | **3/26** |
| **FAIL** | **1/26** |
| Dataset | 100 vini |
| GT-24 (etna) | FAIL → **PASS** |
| GT-26 (voglio stupire) | FAIL → **PASS** |

---

## GT Originali (GT-01 → GT-13)

| GT | Query | Verdict | Top 3 | Note |
|----|-------|---------|-------|------|
| GT-01 | `barolo serralunga` | **PASS** | #1 Barolo Serralunga (m=1.0), #2 Barolo Monforte (m=0.85), #3 Barolo Serralunga id9 (m=1.0) | Migliorato post-fix: "barolo" in REGION_PATTERNS filtra solo Baroli |
| GT-02 | `formaggi erborinati` | **PASS** | #1 Riesling Kabinett (m=0.7), #2 Recioto (m=0.7), #3 Vin Santo (m=0.7) | Vini dolci/aromatici coerenti. Passito #4 con m=1.0 |
| GT-03 | `franciacorta brut` | **PASS** | #1 Franciacorta Brut id16 (m=1.0), #2 Franciacorta Brut id31 (m=1.0) | Migliorato: "franciacorta" in REGION_PATTERNS filtra direttamente |
| GT-04 | `vino per cena importante` | **PASS** | #1 Bolgheri (m=0.65), #2 Gevrey (m=0.65), #3 Barolo (m=0.65) | Vini prestige con occasion coerente |
| GT-05 | `rosso strutturato` | **PASS** | #1 Bolgheri (m=0.85), #2 Barolo (m=0.7), #3 Pauillac (m=0.7), #4 Sagrantino (m=1.0) | Sagrantino #4, non #3 — qualità intrinseca domina |
| GT-06 | `bianco fresco` | **WARN** | #1 Champagne (m=0.7), #2 Champagne (m=0.7), #3 Riesling (m=0.7) | Champagne tecnicamente bianco fresco — comportamento noto |
| GT-07 | `rosso elegante` | **PASS** | #1 Bolgheri (m=0.875), #2 Gevrey (m=0.875) | Gevrey in top 2 come atteso |
| GT-08 | `rosso elegante per cena importante` | **PASS** | #1 Bolgheri (m=0.925), #2 Gevrey (m=0.925) | Match > 0.85 come atteso |
| GT-09 | `bianco fresco per cena importante di pesce` | **PASS** | #1 Champagne (m=0.9), #2 Champagne (m=0.9), #3 Riesling (m=0.9) | Multi-signal coerente |
| GT-10 | `vino importante` | **PASS** | #1 Bolgheri (m=0.49), #2 Gevrey (m=0.574), #3 Barolo (m=0.49) | Prestige scoring attivo |
| GT-11 | `vino importante di pesce` | **PASS** | #1 Champagne (m=0.784), #2 Champagne (m=0.784), #3 Riesling (m=0.676) | Bollicine/bianchi prestigio con food pesce |
| GT-12 | `bottiglia importante` | **PASS** | #1 Bolgheri, #2 Gevrey, #3 Barolo | Come GT-10 |
| GT-13 | `vino che faccia figura` | **PASS** | #1 Bolgheri, #2 Gevrey, #3 Barolo | Come GT-10 |

## GT Tannicità (GT-14 → GT-17)

| GT | Query | Verdict | Top 3 | Note |
|----|-------|---------|-------|------|
| GT-14 | `vino tannico e strutturato` | **PASS** | #1 Bolgheri, #2 Barolo (m=0.75), #4 Sagrantino (m=0.85) | Sagrantino e Baroli in top 5 |
| GT-15 | `rosso leggero poco tannico` | **WARN** | #1 Frappato (m=0.9), #2 Valpolicella (m=0.85), #3 Torrette (m=0.85) | Rossi leggeri in top — Gevrey/Pinot non in top 5 (qualità base più bassa dei nuovi vini). Match alto sui leggeri, tannici penalizzati: logica corretta |
| GT-16 | `rosso elegante non troppo tannico` | **PASS** | #1 Bolgheri (m=0.74), #2 Gevrey (m=0.64), #5 Sagrantino (m=0.45) | Gevrey in top 2, Sagrantino penalizzato |
| GT-17 | `vino potente per cena importante` | **PASS** | #1 Bolgheri, #2 Barolo, #3 Pauillac, #5 Barolo Monforte | Vini potenti + important_dinner in top 5 |

## GT Copertura (GT-18 → GT-26)

| GT | Query | Verdict | Top 3 | Note |
|----|-------|---------|-------|------|
| GT-18 | `poco tannico` | **WARN** | #1 Bolgheri (m=0.19), #2 Gevrey (m=0.54), #5 Champagne (m=0.7) | Qualità domina match — Bolgheri #1 con match basso. Champagne (tannini low) #5 con match alto. Logica corretta ma qualità sovrasta |
| GT-19 | `vino tannico` | **PASS** | #1 Bolgheri (m=0.51), #3 Barolo (m=0.81) | Barolo con match alto, Sagrantino più basso nel ranking ma match coerente |
| GT-20 | `vino sotto 20 euro` | **PASS** | #1 Fiano €18, #2 Cerasuolo €16, #3 Vermentino €19 | Tutti sotto 20€ |
| GT-21 | `nebbiolo` | **PASS** | #1 Barolo (m=1.0), #2 Barolo Monforte (m=0.7), #4 Barbaresco (m=1.0), #5 Nebbiolo Langhe (m=1.0) | Nebbioli tutti presenti |
| GT-22 | `rosato per aperitivo` | **PASS** | #1 Cerasuolo (m=0.8), #2 Provence Rosé (m=0.75), #3 Chiaretto (m=1.0) | Perfetto |
| GT-23 | `prosecco` | **FAIL** | #1 Champagne, #2 Champagne, #3 Trento | Prosecco non in top — parse non riconosce "prosecco" come denominazione. Mappa a sparkling generico, qualità domina |
| GT-24 | `etna` | **PASS** | #1 Nerello Etna Rosso (m=1.0), #2 Etna Bianco (m=1.0), #3 Etna Rosso (m=1.0) | **FIX FUNZIONANTE** — 5 vini Etna con match=1.0 |
| GT-25 | `vino dolce` | **PASS** | #1 Recioto (m=1.0), #2 Vin Santo (m=0.9), #3 Moscato Scanzo (m=0.9) | Passito #5 — qualità base più bassa |
| GT-26 | `voglio stupire` | **PASS** | #1 Bolgheri (m=0.49), #2 Gevrey (m=0.574), #3 Barolo (m=0.49) | **FIX FUNZIONANTE** — prestige scoring attivo, vini di prestigio in top 3 |

---

## Fix Applicati

| # | Cosa | Riga | Classificazione |
|---|------|------|----------------|
| FIX 1 | REGION_PATTERNS: +20 zone/regioni | ~298 | B |
| FIX 2 | Filtro regione: AND→OR | ~2107 | B |
| FIX 3 | parse_prestige_intent: +5 pattern | ~440 | B |

## Miglioramenti Collaterali

- **GT-01**: ora filtra solo vini Barolo (prima era ranking generico per qualità)
- **GT-03**: ora filtra solo Franciacorta (prima Champagne sopra per qualità)

## Regressioni

Nessuna regressione rilevata. Tutti i GT pre-esistenti mantengono lo stesso verdict o migliorano.

---

## Gap Residui

| Gap | GT | Priorità | Fix |
|-----|-----|----------|-----|
| "prosecco" non matchato come denominazione | GT-23 | Alta | Aggiungere "prosecco" a REGION_PATTERNS o creare parse_denomination |
| Vitigni mancanti in KNOWN_GRAPES (14 vitigni) | — | Media | Sessione dedicata arricchimento parser |
| LLM intent parser non integrato | — | Alta | Backlog mandatorio |
| Passito non in top 2 per "vino dolce" | GT-25 | Bassa | Qualità base bassa — fix editoriale CSV o peso sweetness |
| Bolgheri domina tutti i ranking generici | — | Bassa | Qualità 4.87 è la più alta del dataset — comportamento corretto |

---

## Confronto Pre-Fix vs Post-Fix

| GT | Pre-Fix | Post-Fix | Delta |
|----|---------|----------|-------|
| GT-01 | WARN (Barolo #3) | **PASS** (Barolo #1) | Migliorato |
| GT-03 | WARN (Franciacorta #4) | **PASS** (Franciacorta #1) | Migliorato |
| GT-24 | FAIL (nessun Etna) | **PASS** (5 Etna m=1.0) | **Fixato** |
| GT-26 | FAIL (match=-1) | **PASS** (prestige attivo) | **Fixato** |
| Altri | Invariati | Invariati | — |

---

*Rapporto di collaudo definitivo — 2026-03-20 16:37 CET*
