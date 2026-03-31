# 🗺️ ROADMAP SOMMELIERAI — Ufficiale

**Versione**: 1.5  
**Ultimo aggiornamento**: 2026-03-31  
**Baseline Backend**: v1.6.0 LIVE  
**Baseline iOS**: v0.9 POST-FREEZE  
**Ground Truth**: v2.1 (22 PASS, 4 WARN, 0 FAIL)

---

## 🟢 FASE 0 — FONDAZIONE (100% ✅)

**Obiettivo**: MVP funzionante con ranking proprietario

### Completato
- [x] Backend FastAPI → Flask (migrato)
- [x] Dataset 100 vini (35 regioni, 75 vitigni)
- [x] Ranking A9v2 Composite Engine
- [x] Match score moltiplicatore
- [x] SSE streaming
- [x] UI chat SwiftUI
- [x] Favorites
- [x] Tier Free
- [x] Grouping annate
- [x] Filtri principali
- [x] Deploy LIVE Render.com

**Status**: ✅ COMPLETATA — Production ready

---

## 🔵 FASE 1 — HARDENING & STABILITÀ (100% ✅)

**Obiettivo**: Motore deterministico, stabile, spiegabile

### Completato
- [x] A9v2 Composite Engine stabile
- [x] Match come moltiplicatore (no double count)
- [x] Target price mode + proximity bonus
- [x] Delta breakdown (delta_vs_top)
- [x] Flatten match_breakdown
- [x] Value intent override
- [x] Test Matrix v2.1 (GT-01 → GT-26)
- [x] Freeze + versioning
- [x] Repository restructuring
- [x] Golden query set (26 query)
- [x] Region alias map (etna→Etna, langhe→Langhe)
- [x] Tannin matching separato da intensity

**Status**: ✅ COMPLETATA — Engine STABILE v1.0

---

## 🟣 FASE 2 — QUALITÀ PERCEPITA (85% ✅)

**Obiettivo**: Rendere evidente il salto qualitativo del ranking

### Completato
- [x] Debug composito backend (100%)
- [x] Breakdown differenziale (100%)
- [x] UI Cleanup A→F (85%)
  - [x] UI-A: Debug scores nascosti
  - [x] UI-B: Tag tecnici filtrati
  - [x] UI-C: Card espandibile
  - [x] UI-D: Cestino + filtri post-risultati
  - [x] UI-E: Accordion filtri collassato
  - [x] UI-F: Rating/Popolari rimossi
- [x] Filtri locali iOS (vitigno, prezzo, colore, intensità)
- [x] Smart Ranking toggle (Standard/A9v2)
- [x] **Paginazione iOS 10→15→20** ← OGGI COMPLETATO

### Da Completare (15%)
- [ ] Estetica app (icona, splash, background theme)
- [ ] Badge semantici intelligenti
- [ ] Visual ranking bar composito
- [ ] Modalità ranking selezionabile utente (UI enhancement)

**Status**: 🔄 IN CORSO — UX polish residuo

---

## 🟡 FASE 3 — LLM & INTELLIGENCE (50% ✅)

**Obiettivo**: Arricchire motore con layer semantico

### Completato
- [x] **LLM Step 1**: Parser semantico (C-experimental) ✅
  - Intent strutturati (regioni, occasioni, prestige)
  - claude-haiku-4-5-20251001
  - Fallback rule-based parallelo
  - Region alias map integrata
  - GT-23, GT-24, GT-26 PASS

### In Corso
- [ ] **LLM Step 2**: Explain personalizzato (50%)
  - Reason contestuale query-aware
  - Sostituire template statici
  - Passare segnali ranking attivi all'LLM

### Pianificato
- [ ] LLM Step 3: Preferenze multi-turn
- [ ] Expert Ratings Integration (weight 0.15-0.25)
- [ ] Dataset cleanup + arricchimento
  - 14 vitigni mancanti in KNOWN_GRAPES
  - Campo "indicato per"
  - Aromi + sensoriali estesi
- [ ] Espansione 100 → 200+ vini

**Status**: 🔄 IN CORSO — LLM Step 2 prioritario

---

## 🔴 FASE 4 — PERFORMANCE & SCALE (25% ✅)

**Obiettivo**: Ottimizzazioni latenza + architettura scalabile

### Completato
- [x] Cold start mitigation (cron-job.org ping ogni 5 min)
- [x] SSE streaming ottimizzato

### Pianificato
- [ ] Cache risultati query frequenti
- [ ] Debounce intelligente client-side
- [ ] Ottimizzazione latenza backend
- [ ] Refactor modulare ranking engine
- [ ] UptimeRobot o Render paid plan (cold start definitivo)

**Status**: ⏳ PIANIFICATO — Post-lancio

---

## 🟠 FASE 5 — GO-TO-MARKET (15% ✅)

**Obiettivo**: Lancio pubblico + strategia commerciale

### Completato
- [x] Posizionamento prodotto definito
- [x] Backend LIVE production-ready

### Pianificato
- [ ] Apple Developer Account ($99/anno)
- [ ] TestFlight beta pubblica
- [ ] App Store submission
- [ ] Strategia pricing (Free/Premium)
- [ ] Target utenti identificati
- [ ] Marketing roadmap
- [ ] Eventi settore (Vinitaly, etc.)
- [ ] Sezione educativa (Premium feature)
- [ ] Banner pubblicitari (valutazione post-lancio)

**Status**: ⏳ PIANIFICATO — Post beta chiusa

---

## 🎯 MILESTONE CORRENTE

**Nome**: **LLM Enhancement + Polish UI**  
**Obiettivo**: Completare layer semantico + estetica finale  
**Deadline**: 2026-04-30  
**Progress**: 7/10 task (70%)

### Checklist Milestone
- [x] LLM Step 1 integrato (C-experimental)
- [x] GT v2.1 validato (22 PASS, 4 WARN, 0 FAIL)
- [x] UI Cleanup A→F
- [x] Paginazione iOS ← COMPLETATO OGGI
- [x] Filtri locali completi
- [ ] LLM Step 2 (explain personalizzato) ← NEXT
- [ ] Estetica app (icona, splash, theme)
- [ ] Arricchimento KNOWN_GRAPES (14 vitigni)
- [ ] Test completi device reale
- [ ] Beta TestFlight preparazione

---

## 📊 METRICHE ATTUALI

### Backend (v1.6.0 LIVE)
- **Uptime**: ~95% (cold start mitigato, non eliminato)
- **Response time**: <500ms avg, <800ms p95
- **GT Accuracy**: 84.6% (22/26 PASS)
- **Dataset**: 100 vini, 35 regioni, 75 vitigni
- **Error rate**: <1%

### iOS (v0.9 POST-FREEZE)
- **Crash rate**: <1%
- **Build stability**: 100% (freeze attivo)
- **User base**: ~10-20 beta testers interni
- **Feature completeness**: 88%

### Tech Debt
- **BASSO**: Engine stabile, freeze confermato
- **MEDIO**: Cold start Render (mitigato ma non risolto)
- **MEDIO**: Mancano analytics/monitoring
- **ALTO**: Estetica app non pronta per pubblico

---

## 🚀 NEXT SPRINT (1-7 Aprile 2026)

### Sprint Goal
**Completare LLM Step 2 + validare paginazione**

### Tasks Prioritari
1. **Test paginazione iOS** su device reale (P0) ← OGGI
2. **LLM Step 2** explain personalizzato (P0)
3. **Estetica app** icona + splash (P1)
4. **Arricchimento KNOWN_GRAPES** 14 vitigni (P2)
5. **Fix `total_count`** backend con filtri (P2)

---

## 💡 DECISIONI ARCHITETTURALI APERTE

### Urgenti (Q2 2026)
1. **LLM Step 2**: Haiku vs Sonnet per explain generation?
2. **Estetica**: Contrattare designer o AI-generated assets?
3. **Analytics**: Firebase vs Mixpanel vs Amplitude vs niente?
4. **Cache layer**: Redis, in-memory, o aspettare scale reale?

### Non urgenti (Q3 2026)
5. **Paginazione vera**: Implementare offset/limit backend o keep client-side?
6. **Android**: Native Kotlin vs React Native vs Flutter?
7. **Backend scale**: Keep monolith Flask vs microservizi?
8. **Expert ratings**: Source? API partner? Manual curation?

---

## 📝 RELEASE HISTORY

- **v1.6.0** (2026-03-30): Backend total_count + paginazione iOS
- **v1.5.0** (2026-03-28): LLM Step 1 integrato
- **v1.4.0** (2026-03-24): GT v2.1 validato, tannin separato
- **v1.3.0** (2026-03-20): UI Cleanup A→F completato
- **v1.2.0** (2026-03-15): A9v2 Engine freeze
- **v1.1.0** (2026-03-10): Dataset 100 vini LIVE
- **v1.0.0** (2026-02-15): MVP Alpha launch

---

## 🎓 RETROSPECTIVE & LEARNINGS

### Cosa Funziona Bene ✅
- **Freeze architettura**: Stabilità massima, zero breaking changes
- **Ground Truth**: Quality gate efficace, previene regressioni
- **Dual-step LLM**: Arricchisce senza sostituire motore deterministico
- **SSE streaming**: UX perceived performance eccellente
- **Filtri client-side**: UI snappy, no backend bottleneck

### Cosa Migliorare ⚠️
- **Cold start**: Ping mitigation non ideale, serve Render paid o UptimeRobot
- **Analytics mancanti**: Decisioni data-blind, no A/B testing
- **Estetica app**: Non pronta per pubblico consumer
- **Test coverage**: Troppo manual, serve automation GT
- **Documentation**: Sparsa, serve consolidamento

### Next Retrospective
**2026-04-30** (fine milestone corrente)

---

## 🔒 FREEZE UFFICIALE

**Baseline stabile**: v1.6.0  
**Engine**: A9v2 Composite — NO TOUCH senza sessione GT dedicata  
**Dataset**: 100 vini — modifiche solo additive  
**Architettura**: Flask monolith — keep semplice  

---

## 📋 BACKLOG STRATEGICO

### High Priority (Q2 2026)
- [ ] LLM Step 2 (explain personalizzato)
- [ ] Estetica app completa
- [ ] Apple Developer Account + TestFlight
- [ ] Analytics setup base
- [ ] Arricchimento KNOWN_GRAPES

### Medium Priority (Q3 2026)
- [ ] Expert ratings integration
- [ ] Dataset 100 → 200+ vini
- [ ] Paginazione vera (offset/limit backend)
- [ ] Cache layer Redis
- [ ] Sezione educativa (Premium)

### Low Priority (Q4 2026)
- [ ] Android app
- [ ] Web PWA
- [ ] API pubblica B2B
- [ ] Wine cellar management
- [ ] AR label recognition

---

**Domande strategiche prossima sessione**:
1. LLM Step 2: quanto effort vs ROI percepito?
2. Estetica: budget designer esterno?
3. TestFlight: quando apriamo beta pubblica?
4. Analytics: setup ora o aspettiamo lancio?

---

*Roadmap v1.5 — Aggiornata 31 Marzo 2026 — Prossimo update: fine milestone (30 Aprile)*
