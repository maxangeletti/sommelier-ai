# 🔄 Session Handoff — 31 Marzo 2026

**Per la prossima sessione (1 Aprile 2026)**

---

## ⚡ Contesto Veloce (30 secondi)

Oggi completata **paginazione iOS**: mostra 10 vini iniziali → bottone "Mostra altri 5" → max 20 vini. Commit `3fd4b44` pushato, build OK, funzionante su Xcode. Backend v1.6.0 già compatibile (fornisce `total_count`). **Prossimo**: test device reale + LLM Step 2 explain personalizzato.

---

## 📊 Stato Progetto (Snapshot)

### Backend
- **Versione**: v1.6.0 LIVE su Render
- **Status**: Stabile, no regressioni
- **Ground Truth**: v2.1 (22 PASS, 4 WARN, 0 FAIL)
- **LLM Layer**: Step 1 integrato (C-experimental) ✅
- **Dataset**: 100 vini, 35 regioni, 75 vitigni

### iOS
- **Versione**: v0.9.1 (post-paginazione)
- **Build**: OK su Xcode
- **Ultima feature**: Paginazione chat 10→15→20
- **Commit**: `3fd4b44` pushato su `main`

### Milestone Corrente
- **Nome**: LLM Enhancement + Polish UI
- **Progress**: 7/10 task (70%)
- **Deadline**: 30 Aprile 2026

---

## 🚨 Blockers / Warning

### Nessun blocker critico ✅

### Warning da monitorare:
1. **Backend `total_count`**: Attualmente conta vini PRIMA dei filtri locali
   - Se applichi filtro vitigno iOS, `total_count` può essere sovrastimato
   - **Fix pianificato**: Backend considera filtri attivi (PRIORITÀ 2)

2. **Cold start Render**: Mitigato (ping ogni 5 min) ma non risolto
   - Prima richiesta dopo inattività → 30-45s
   - **Soluzione**: Render paid plan o UptimeRobot (post-lancio)

3. **Estetica app**: Non pronta per pubblico consumer
   - Icona placeholder
   - Splash screen mancante
   - Theme colori da rifinire

---

## 📁 File da Caricare Next Session

### Obbligatori (sempre)
```
docs/GROUND_RULES.md
docs/TODO_NEXT_SESSION.md
docs/roadmap/ROADMAP_UFFICIALE_v1.5.md
docs/SommelierAI_ProjectContext_v1_4.md
```

### Per task specifici
**Se lavori su paginazione/test**:
```
ios-app/SommelierAI/SommelierAI/ChatViewModel.swift
ios-app/SommelierAI/SommelierAI/ChatView.swift
```

**Se lavori su LLM Step 2**:
```
backend/main.py
backend/llm_intent_parser.py (se esiste ancora separato)
docs/technical/RANKING_TEST_MATRIX_v2_1.md (per validazione)
```

**Se lavori su estetica**:
```
ios-app/SommelierAI/SommelierAI/Assets.xcassets/
ios-app/SommelierAI/SommelierAI/AppColors.swift (se esiste)
```

---

## 🎯 Task Prioritari (in ordine)

### P0 — Urgenti (questa settimana)
1. **Test paginazione device reale**
   - iPhone fisico, non simulatore
   - Verifica bottone appare/scompare correttamente
   - Test con filtri attivi (vitigno, prezzo)
   - Test con <10 risultati (bottone non deve apparire)

2. **LLM Step 2 — Explain personalizzato** (alta priorità roadmap)
   - Backend: passa segnali ranking attivi all'LLM
   - LLM genera reason contestuale alla query
   - Sostituisci template statici
   - Test su GT-01→GT-26 per verificare qualità explain
   - **Esempio target**:
     - Query: "vino elegante per cena importante"
     - Reason OLD: "Rosso prestigioso con tannini morbidi"
     - Reason NEW: "Un Borgogna raffinato perfetto per occasioni formali, con eleganza e struttura delicata"

### P1 — Importanti (settimana prossima)
3. **Estetica app**
   - Icona app (1024x1024, tutte size richieste)
   - Splash screen
   - Refinement theme colori AppColors

4. **Fix backend `total_count` con filtri**
   - total_count deve considerare filtri attivi
   - Attualmente: count PRIMA filtri
   - Target: count DOPO filtri applicati

### P2 — Nice to have (prossimo sprint)
5. **Arricchimento KNOWN_GRAPES**
   - 14 vitigni presenti nel dataset ma non nel vocabolario
   - Da aggiungere per completezza semantic matching

6. **Analytics setup base**
   - Firebase o Mixpanel (decisione da prendere)
   - Track: query, sort mode, filtri usati, paginazione clicks

---

## 🔍 Note / Warning per Next Developer

### Ground Rules (OBBLIGATORIE)
- ✅ **Leggere SEMPRE** `docs/GROUND_RULES.md` prima di modificare codice
- ❌ **NON toccare** motore ranking A9v2 senza sessione dedicata + GT baseline
- ✅ **Backup obbligatorio** prima di ogni modifica file critici
- ✅ **Test before push** (WORKFLOW: modifica → test → commit → push → test live)
- ❌ **NO cazzate**: se fallisce, dirlo subito. Mai dire "funziona" senza verifica reale

### Freeze Architettura
- **Backend**: v1.6.0 — modifiche solo additive, no breaking changes
- **iOS**: v0.9 POST-FREEZE — build stable, solo feature additive
- **Dataset**: 100 vini — modifiche solo editoriali, no logiche ranking
- **Engine A9v2**: FREEZE TOTALE — no touch senza GT full validation

### Decisioni da Prendere (prossima sessione)
1. **LLM Step 2**: Haiku (veloce, economico) vs Sonnet (qualità alta)?
2. **Estetica**: Contrattare designer esterno o AI-generated assets?
3. **Analytics**: Setup ora o aspettare lancio pubblico?
4. **TestFlight**: Quando apriamo beta pubblica?

---

## 🧪 Come Testare (Procedure)

### Test Paginazione iOS
```
1. Build su iPhone reale (non simulatore)
2. Cerca "vino rosso" o "Barolo"
3. VERIFICA: mostra 10 vini inizialmente
4. VERIFICA: bottone "Mostra altri 5 vini" appare in fondo
5. TAP bottone → VERIFICA: mostra 15 vini totali
6. TAP bottone → VERIFICA: mostra 20 vini totali
7. VERIFICA: bottone scompare (raggiunto max 20)
8. Applica filtro vitigno (es. Nebbiolo)
9. VERIFICA: bottone funziona correttamente con filtri
10. Cerca query con <10 risultati
11. VERIFICA: bottone NON appare
```

### Test LLM Step 2 (quando implementato)
```
1. Modifica backend: integra explain personalizzato
2. Run locale: python backend/main.py
3. Test query: "vino elegante per cena importante"
4. VERIFICA: reason contestuale (non template)
5. Test GT-01→GT-26: verifica qualità explain
6. VERIFICA: latency accettabile (<1s extra)
7. Push a Render → wait deploy (2-3 min)
8. Test live: same query su app iOS
9. CONFERMA funzionante
```

---

## 📚 Documentazione Chiave

| File | Scopo | Quando consultare |
|---|---|---|
| `docs/GROUND_RULES.md` | Regole obbligatorie | **SEMPRE prima di modificare** |
| `docs/roadmap/ROADMAP_UFFICIALE_v1.5.md` | Roadmap strategica | Per capire priorità |
| `docs/SommelierAI_ProjectContext_v1_4.md` | Context completo progetto | Setup iniziale sessione |
| `docs/TODO_NEXT_SESSION.md` | Task immediati | Inizio sessione |
| `docs/technical/RANKING_TEST_MATRIX_v2_1.md` | Ground Truth 26 query | Modifiche motore ranking |
| `docs/RELEASE_NOTES_v1_6_1.md` | Release notes oggi | Reference modifiche paginazione |

---

## 🐛 Bug Conosciuti (da fixare)

### Nessun bug critico ✅

### Minor issues:
1. **Backend total_count**: Non considera filtri locali (già pianificato fix)
2. **Cold start**: Prima richiesta lenta dopo inattività (mitigato, non risolto)

---

## 💡 Idee / Feature Request (backlog)

- [ ] "Cerca vini simili a questo" button (alta priorità utenti)
- [ ] Pillola del giorno / Vino del giorno
- [ ] Sezione educativa (feature premium)
- [ ] Condivisione raccomandazioni (social)
- [ ] Storico ricerche persistente
- [ ] Preferiti con sync cloud (iCloud)

---

## 🔗 Link Utili

- **Backend LIVE**: https://sommelier-ai.onrender.com
- **Render Dashboard**: https://dashboard.render.com
- **GitHub Repo**: https://github.com/maxangeletti/sommelier-ai
- **Anthropic Console**: https://console.anthropic.com (API keys)
- **Cron-Job Ping**: https://cron-job.org (anti-cold-start)

---

## 📞 Quick Start Next Session

```bash
# 1. Pull latest
cd ~/sommelier-ai
git pull

# 2. Verifica clean
git status  # deve essere clean

# 3. Carica documenti in Claude
# - GROUND_RULES.md
# - TODO_NEXT_SESSION.md
# - ROADMAP_UFFICIALE_v1.5.md
# - (altri file specifici per task)

# 4. Leggi questo handoff
# - Capisci contesto veloce
# - Identifica task prioritario
# - Segui procedure test

# 5. Go!
```

---

**Session ID**: 2026-03-31_paginazione-ios  
**Durata**: ~2.5h  
**Completato**: Paginazione iOS 100%  
**Next**: Test device + LLM Step 2  

**Handoff preparato da**: Claude Sonnet 4.6  
**Per**: Prossimo Claude o developer  
**Data**: 31 Marzo 2026, ore 23:45
