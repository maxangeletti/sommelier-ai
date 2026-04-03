# 🍷 SommelierAI — Documento di Contesto Progetto

Versione 1.5 — 3 Aprile 2026

## 1. Identità del Prodotto

SommelierAI è un'app iOS per la raccomandazione intelligente di vini tramite linguaggio naturale. L'utente scrive query libere e riceve risultati ordinati per rilevanza, qualità e valore, con una spiegazione leggibile del ranking.

**Differenziatore centrale:** Il motore di ranking proprietario A9v2 è l'asset principale. Non è un filtro statico né un wrapper LLM: interpreta l'intento dell'utente (occasione, eleganza, food pairing, prestige, location, stile, tannicità) e lo trasforma in uno score composito multi-dimensionale con explainability.

**Promessa di prodotto:** "Il vino giusto, per il momento giusto."

## 2. Stato del Progetto (ciclo 2026-04-03)

| Componente | Stato | Note |
|---|---|---|
| Blocco tecnico (BE+FE) | ~96% ✅ | Foundation stabile, freeze confermato |
| Progetto end-to-end | ~90% 🔄 | MVP funzionante su iPhone, deploy live |
| GT-01 → GT-26 | 22 PASS, 4 WARN, 0 FAIL | Matrice v2.1 — WARN documentati, nessun FAIL |
| LLM Layer — Step 1 | ✅ Integrato | llm_intent_parser.py attivo [B-accepted] |
| LLM Layer — Step 2 | ✅ Completato | Explain personalizzato live [B-accepted] |
| Region alias map | ✅ Attiva | etna → Etna, langhe → Langhe, valtellina → Valtellina |
| Deploy backend | ✅ Live | Render.com — sommelier-ai.onrender.com |
| App iOS su iPhone | ✅ Funzionante | Build su device fisico via Xcode |
| Dataset | 100 vini ✅ | 35 regioni, 75 vitigni, 8 paesi — live su Render |
| Scala tannicità AIS | ✅ 5 livelli | low/medium_low/medium/medium_plus/high |
| Tannin matching separato | ✅ | tannin_req indipendente da intensity_req |
| Cold start | Mitigato 🔄 | cron-job.org ping ogni 5 min |
| App Store | ❌ Da fare | Richiede Apple Developer $99/anno |
| UI Cleanup | 85% ✅ | UI-A→F completati, estetica da fare |
| Claude Code | ✅ Operativo | Opus 4.6 → Sonnet 4.6 |

## 3. Stack Tecnico

- **Frontend:** SwiftUI (iOS) — architettura chat-style
- **Backend:** FastAPI (Python) — main.py (2800+ righe)
- **Hosting backend:** Render.com (free tier) — URL: https://sommelier-ai.onrender.com
- **Dataset:** data/wines.normalized.csv — 100 vini, 27 colonne (live su Render ✅)
- **LLM Runtime:** claude-haiku-4-5-20251001 (dual-step: parse + explain)
- **LLM Sviluppo:** claude-opus-4-6 → claude-sonnet-4-6
- **Claude Code:** v2.1.80, Sonnet, operativo nel repo locale
- **Repo:** github.com/maxangeletti/sommelier-ai
- **Monorepo:** backend/ (Python), ios-app/SommelierAI/SommelierAI/ (Swift), data/ (CSV), docs/ (documentazione)

## 4. Dogmi del Progetto (NON NEGOZIABILI)

- Il motore di ranking A9v2 NON si tocca senza sessione dedicata e GT baseline
- Il CSV non si modifica per logiche di ranking — solo per arricchimento editoriale
- L'LLM non sostituisce il ranking engine — lo arricchisce (dual-step: parse + explain)
- L'interfaccia è sempre in italiano — il codice in inglese
- Ogni modifica al motore richiede classificazione A/B/C/D e test GT-01→GT-26
- Mai esporre canonical tecnici con underscore all'utente finale
- Modificare solo ciò che è richiesto — no refactoring non richiesto
- Non eliminare o riscrivere nulla senza motivo chiaro

## 4.1. Architettura LLM Dual-Step [B-accepted]

Il sistema LLM è integrato con architettura **dual-step** per arricchire il motore A9v2:

### Step 1 — Parse Semantico (✅ Integrato)
- **Input:** Query utente in linguaggio naturale
- **Output:** Intent strutturati (regioni, occasioni laterali, prestige emotivo)
- **LLM:** claude-haiku-4-5-20251001
- **Fallback:** Parser rule-based sempre attivo in parallelo
- **Combinazione:** Il motore fonde segnali LLM + rule-based (OR logico)
- **Region alias map:** Attiva — normalizza "etna" → "Etna", "langhe" → "Langhe", etc.

### Step 2 — Explain Personalizzato (✅ Completato — 3 Aprile 2026)
- **Input:** Query utente + segnali ranking attivi
- **Output:** Reason contestuale e naturale (max 15 parole)
- **LLM:** claude-haiku-4-5-20251001
- **Implementazione:** `generate_reason_with_llm()` in `llm_intent_parser.py`
- **Segnali passati:** color, region, prestige_intent, elegance_intent, occasion, foods, intensity, tannin, style, value
- **Fallback:** Description CSV se LLM non disponibile
- **Costo:** ~$0.002 per search (10 vini) — $6/mese con 100 search/giorno
- **Esempio:** 
  - Query: "vino elegante per cena importante"
  - Reason: "Un rosso toscano di grande eleganza e prestigio, perfetto per occasioni importanti"

### Principi Architetturali
- L'LLM **arricchisce**, non sostituisce il motore A9v2
- Il ranking rimane deterministico e trasparente
- Fallback rule-based e CSV garantiscono robustezza
- Classificazione: `[B-accepted]` dopo test positivi (3 Aprile 2026)

## 5. Scala Tannicità AIS (5 livelli)

| Livello | Valore numerico | Sensazione in bocca | Esempi |
|---|---|---|---|
| low | 0.10 | Astringenza appena percettibile | Pinot Nero delicato, rossi giovani leggeri |
| medium_low | 0.30 | Tannino presente ma morbido | Valpolicella, Barbera, Dolcetto |
| medium | 0.50 | Tannino evidente ma equilibrato | Chianti Classico, Montepulciano |
| medium_plus | 0.75 | Astringenza marcata, struttura importante | Barolo, Aglianico, Cabernet Sauvignon |
| high | 1.00 | Tannino molto incisivo, asciugante | Sagrantino |

**Nota:** tannicità ≠ intensità. Il motore ha due componenti separati: `tannin_req` (confronto diretto col campo tannins del CSV) e `intensity_req` (derivato da body+tannins+alcohol).

## 6. Stato UI Cleanup (ciclo 2026-03-20)

| Componente | Stato | Note |
|---|---|---|
| UI-A: Debug scores | ✅ Done | rankingDebugMode = false |
| UI-B: Underscore e tag tecnici | ✅ Done | Filtro tag, food pairing pulito |
| UI-C: Card espandibile | ✅ Done | Più/Meno dettagli su tap |
| UI-D: Cestino in navbar + filtri post-risultati | ✅ Done | Filtri visibili solo con risultati |
| UI-E: Accordion filtri collassato | ✅ Done | Collassato di default, pallino filtri attivi |
| UI-F: Nascondi Rating/Popolari sort | ✅ Done | Rimossi dal menu sort |
| Estetica/icona/splash | ⏳ Da fare | Serve asset grafico |
| Pillola del giorno | ⏳ Da fare | Prossima sessione |

## 7. Backlog — MANDATORIO (alta priorità)

**🟢 LLM Step 1 — COMPLETATO ✅**

Parser semantico LLM integrato in run_search() con architettura dual-step. Copre query laterali come "voglio stupire", "etna", "qualcosa di estivo ma non banale". Region alias map attiva. GT-23, GT-24, GT-26 tutti PASS.

**🟢 LLM Step 2 — COMPLETATO ✅ (3 Aprile 2026)**

Reason personalizzate contestuali implementate e live. Il backend passa i segnali di ranking attivi (prestige_intent, occasion, color, tannin_req, intensity_req, etc.) all'LLM che genera reason naturali (max 15 parole). Fallback garantito a description CSV. Test positivi su 3 query chiave. Classificazione: [B-accepted].

**Esempio:**
- Query: "vino elegante per cena importante"
- Reason generata: "Un rosso toscano di grande eleganza e prestigio, perfetto per occasioni importanti"
- Query: "rosso strutturato"
- Reason generata: "Un rosso toscano dal carattere importante e profondo, con una struttura decisa e tannini ben presenti"

**🟡 Arricchimento KNOWN_GRAPES**

14 vitigni presenti nel dataset ma non nel vocabolario KNOWN_GRAPES. Da aggiungere per completezza semantic matching.

## 8. Backlog — Nice to Have

- Gerarchia prestige: iconico / di prestigio / leggendario
- Pillola del giorno / Vino del giorno
- Sezione educativa (feature premium)
- Banner pubblicitari — da valutare post-lancio
- Estetica app: background theme, icona, splash screen
- UptimeRobot o Render paid plan per eliminare cold start
- Apple Developer Account ($99/anno) per TestFlight e App Store
- Tuning peso intensity_score — sessione dedicata con GT baseline

## 9. Prossimi Passi — In Ordine di Priorità

**Sessione attuale (Sonnet — 2026-04-03):**
- ✅ LLM Step 2: explain personalizzato integrato e testato
- ✅ Fix bug sparkling (commit fbb271a)
- ✅ Deploy live funzionante
- ✅ Documentazione v1.5 aggiornata
- ⏳ Estetica app: icona, splash, background theme
- ⏳ Arricchimento KNOWN_GRAPES (14 vitigni)

**Prossima sessione:**
- Estetica app: icona, splash, background theme
- Arricchimento KNOWN_GRAPES
- Pillola del giorno / Vino del giorno

**Post-lancio:**
- LLM Step 3: personalizzazione multi-turn, preference memory
- Dataset enrichment: 200+ vini, expert ratings, prestige tier
- Sezione educativa (premium)
- Apple Developer Account per TestFlight e App Store

## 10. Infrastruttura e Accessi

- Backend live: https://sommelier-ai.onrender.com
- Render dashboard: dashboard.render.com
- Ping anti-sleep: cron-job.org — ogni 5 minuti su /suggestions
- API key Anthropic: console.anthropic.com
- GitHub repo: github.com/maxangeletti/sommelier-ai — branch main
- Claude Code: v2.1.80, Opus, operativo in ~/sommelier-ai

## 11. Regole Operative per Sessioni di Sviluppo

- Modificare solo ciò che è richiesto — no refactoring non richiesto
- In debug/iterazione: una sola istruzione per volta
- Non eliminare o riscrivere nulla senza motivo chiaro e documentato
- Classificare le modifiche: A=Freeze/Foundation, B=Accepted tuning, C=Experimental, D=Rejected
- Aggiornare sempre questo documento dopo ogni ciclo significativo
- I GT (01-26) sono la baseline obbligatoria — verificare dopo ogni modifica al motore
- Non toccare pesi/fallback/bonus senza classificazione e test dedicati
- Lato utente finale: testi leggibili, niente underscore, niente tecnicismi esposti

## 12. File Principali

| File | Ruolo |
|---|---|
| backend/main.py | Motore ranking A9v2 + API FastAPI (2900+ righe) |
| backend/llm_intent_parser.py | Parser semantico LLM (Step 1 + Step 2) — integrato |
| data/wines.normalized.csv | Dataset 100 vini |
| docs/technical/RANKING_TEST_MATRIX_v2.1.md | Matrice GT ufficiale (26 test) |
| docs/LLM_STEP2_IMPLEMENTATION.md | Documentazione LLM Step 2 |
| ios-app/SommelierAI/SommelierAI/ChatView.swift | UI chat principale |
| ios-app/SommelierAI/SommelierAI/ChatViewModel.swift | Logica chat |
| ios-app/SommelierAI/SommelierAI/ChatTypes.swift | Tipi dominio chat |
| ios-app/SommelierAI/SommelierAI/Models.swift | Modelli dati |
| ios-app/SommelierAI/SommelierAI/APIClient.swift | Networking |
| CLAUDE.md | Contesto progetto per Claude Code |

---

*SommelierAI Project Context v1.5 — Generato il 3 aprile 2026 — Aggiornare ad ogni ciclo significativo*
