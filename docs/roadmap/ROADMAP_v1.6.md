# 🗺️ ROADMAP SOMMELIERAI — Ufficiale

**Versione**: 1.6  
**Ultimo aggiornamento**: 2026-04-17  
**Baseline Backend**: v1.7.0 LIVE  
**Baseline iOS**: v0.9.3 POST-ONBOARDING  
**Ground Truth**: v2.1 (23 PASS, 3 WARN, 1 FAIL)

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
- [x] Test Matrix v2.1 (GT-01 → GT-27)
- [x] Freeze + versioning
- [x] Repository restructuring
- [x] Golden query set (27 query + GT-NEW)
- [x] Region alias map (etna→Etna, langhe→Langhe)
- [x] Tannin matching separato da intensity

**Status**: ✅ COMPLETATA — Engine STABILE v1.0

---

## 🟣 FASE 2 — QUALITÀ PERCEPITA (90% ✅)

**Obiettivo**: Rendere evidente il salto qualitativo del ranking

### Completato
- [x] Debug composito backend (100%)
- [x] Breakdown differenziale (100%)
- [x] UI Cleanup A→F (100%)
  - [x] UI-A: Debug scores nascosti
  - [x] UI-B: Tag tecnici filtrati
  - [x] UI-C: Card espandibile
  - [x] UI-D: Cestino + filtri post-risultati
  - [x] UI-E: Accordion filtri collassato
  - [x] UI-F: Rating/Popolari rimossi
- [x] Filtri locali iOS (vitigno, prezzo, colore, intensità)
- [x] Smart Ranking toggle (Standard/A9v2)
- [x] Paginazione iOS 10→15→20
- [x] **Onboarding WelcomeView** ← COMPLETATO 11 Apr 2026

### Da Completare (10%)
- [ ] Estetica app (icona, splash, background theme)
- [ ] Badge semantici intelligenti (enhancement)
- [ ] Visual ranking bar composito (enhancement)

**Status**: 🔄 IN CORSO — Estetica residua

---

## 🟡 FASE 3 — LLM & INTELLIGENCE (75% ✅)

**Obiettivo**: Arricchire motore con layer semantico

### Completato
- [x] **LLM Step 1**: Parser semantico (C-experimental) ✅
  - Intent strutturati (regioni, occasioni, prestige)
  - claude-haiku-4-5-20251001
  - Fallback rule-based parallelo
  - Region alias map integrata
  - GT-23, GT-24, GT-26 PASS

- [x] **LLM Step 2**: Explain personalizzato ✅ ← COMPLETATO 17 Apr 2026
  - Reason contestuale query-aware
  - Timeout 10s per generation
  - Skip query semplici per ottimizzare costi
  - Passaggio segnali ranking attivi all'LLM
  - **Status**: LIVE v1.7.0

- [x] **Suggestion Mode**: Query generiche ✅ ← COMPLETATO 17 Apr 2026
  - Query ≤5 char senza intent → suggestions
  - Fuzzy matching semplificato (dist < 3)
  - UI "Forse cercavi..."
  - **Status**: LIVE v1.7.0

### Pianificato
- [ ] LLM Step 3: Preferenze multi-turn
- [ ] Expert Ratings Integration (weight 0.15-0.25)
- [ ] Dataset cleanup + arricchimento
  - 14 vitigni mancanti in KNOWN_GRAPES
  - Campo "indicato per"
  - Aromi + sensoriali estesi
- [ ] Espansione 100 → 200+ vini

**Status**: 🔄 IN CORSO — LLM Step 3 next

---

## 🔴 FASE 4 — PERFORMANCE & SCALE (30% ✅)

**Obiettivo**: Ottimizzazioni latenza + architettura scalabile

### Completato
- [x] Cold start mitigation (cron-job.org ping ogni 5 min)
- [x] SSE streaming ottimizzato
- [x] Fuzzy matching ottimizzato (word boundary)

### Pianificato
- [ ] Cache risultati query frequenti
- [ ] Debounce intelligente client-side
- [ ] Ottimizzazione latenza backend (LLM adds ~500ms)
- [ ] Refactor modulare ranking engine
- [ ] UptimeRobot o Render paid plan (cold start definitivo)

**Status**: ⏳ PIANIFICATO — Post-lancio

---

## 🟠 FASE 5 — GO-TO-MARKET (20% ✅)

**Obiettivo**: Lancio pubblico + strategia commerciale

### Completato
- [x] Posizionamento prodotto definito
- [x] Backend LIVE production-ready
- [x] Onboarding flow completo

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
**Progress**: 11/13 task (85%) ⬆️ +38% da ultimo update

### Checklist Milestone
- [x] LLM Step 1 integrato (C-experimental)
- [x] GT v2.1 validato (23 PASS, 3 WARN, 1 FAIL)
- [x] UI Cleanup A→F
- [x] Paginazione iOS
- [x] Filtri locali completi
- [x] **LLM Step 2** (explain personalizzato) ✅ ← COMPLETATO 17 Apr
- [x] **Suggestion Mode** (query generiche) ✅ ← COMPLETATO 17 Apr
- [x] **Onboarding iOS** (WelcomeView) ✅ ← COMPLETATO 11 Apr
- [x] **Value Intent Badge** (fix v1.6.2) ✅ ← COMPLETATO 8 Apr
- [ ] **iOS Suggestion UI** integration ← IN CORSO (2-3 giorni)
- [ ] Estetica app (icona, splash, theme) ← 2-3 giorni
- [ ] Test completi device reale ← 1 giorno
- [ ] Beta TestFlight preparazione ← 1-2 giorni

**Buffer**: 13 giorni rimanenti → FATTIBILISSIMO ✅

---

## 📊 METRICHE ATTUALI

### Backend (v1.7.0 LIVE) ⬆️
- **Uptime**: ~95% (cold start mitigato, non eliminato)
- **Response time**: <500ms avg, <1s p95 (LLM adds latency)
- **GT Accuracy**: 85.2% (23/27 PASS) — +1 PASS vs baseline
- **Dataset**: 100 vini, 35 regioni, 75 vitigni
- **Error rate**: <1%
- **Latest Release**: v1.7.0 (17 Apr 2026)
  - LLM Step 2 deployed
  - Suggestion mode LIVE
  - Fuzzy matching improvements

### iOS (v0.9.3 POST-ONBOARDING) ⬆️
- **Crash rate**: <1%
- **Build stability**: 100% (freeze attivo)
- **User base**: ~10-20 beta testers interni
- **Feature completeness**: 95% (+7% da ultimo update)
- **Latest**: WelcomeView onboarding (11 Apr 2026)

### Tech Debt
- **BASSO**: Engine stabile, freeze confermato
- **MEDIO**: Cold start Render (mitigato ma non risolto)
- **MEDIO**: Mancano analytics/monitoring
- **MEDIO**: Estetica app non pronta per pubblico (in progress)

---

## 🚀 NEXT SPRINT (18-24 Aprile 2026)

### Sprint Goal
**iOS Suggestion UI + Estetica App + Device Testing**

### Tasks Prioritari
1. **iOS Suggestion UI** integration (P0) — 2-3 giorni
2. **Test device reale** completo (P0) — 1 giorno
3. **Estetica app** icona + splash (P0) — 2-3 giorni
4. **Commit docs** session 17 Apr (P0) — 5 min
5. **Cleanup docs structure** script execution (P1) — 5 min

---

## 💡 DECISIONI ARCHITETTURALI APERTE

### Urgenti (Q2 2026)
1. **Estetica**: Contrattare designer o AI-generated assets? ← PRIORITÀ
2. **Analytics**: Firebase vs Mixpanel vs Amplitude vs niente?
3. **Cache layer**: Redis, in-memory, o aspettare scale reale?
4. **TestFlight**: Beta pubblica subito o interna estesa?

### Non urgenti (Q3 2026)
5. **Paginazione vera**: Implementare offset/limit backend o keep client-side?
6. **Android**: Native Kotlin vs React Native vs Flutter?
7. **Backend scale**: Keep monolith Flask vs microservizi?
8. **Expert ratings**: Source? API partner? Manual curation?

---

## 📝 RELEASE HISTORY

- **v1.7.0** (2026-04-17): LLM Step 2 + Suggestion Mode + Fuzzy improvements
- **v1.6.2** (2026-04-08): Value intent badge fix
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
- **Suggestion Mode**: Risolve ambiguità query generiche efficacemente

### Cosa Migliorare ⚠️
- **Cold start**: Ping mitigation non ideale, serve Render paid o UptimeRobot
- **Analytics mancanti**: Decisioni data-blind, no A/B testing
- **Estetica app**: Non pronta per pubblico consumer
- **Test coverage**: Troppo manual, serve automation GT
- **Documentation**: Migliorata con GROUND_RULES v2, ma serve manutenzione continua
- **LLM latency**: Step 2 aggiunge ~500ms, da ottimizzare

### Nuove Best Practices (GROUND_RULES v2)
- **Token Budget Management**: Soglie warning (50%, 90%, 95%)
- **GT Tests Pre-chiusura**: Obbligatori se modificato ranking
- **Handoff Verificabile**: Checklist minima session closure
- **Docs Cleanup**: Script automation per naming conventions

### Next Retrospective
**2026-04-30** (fine milestone corrente)

---

## 🔒 FREEZE UFFICIALE

**Baseline stabile**: v1.7.0  
**Engine**: A9v2 Composite — NO TOUCH senza sessione GT dedicata  
**Dataset**: 100 vini — modifiche solo additive  
**Architettura**: Flask monolith — keep semplice  

---

## 📋 BACKLOG STRATEGICO

### High Priority (Q2 2026)
- [x] LLM Step 2 (explain personalizzato) ← DONE
- [x] Suggestion Mode (generic queries) ← DONE
- [x] Onboarding iOS ← DONE
- [ ] iOS Suggestion UI integration
- [ ] Estetica app completa
- [ ] Apple Developer Account + TestFlight
- [ ] Analytics setup base
- [ ] Arricchimento KNOWN_GRAPES

### Medium Priority (Q3 2026)
- [ ] LLM Step 3 (multi-turn preferences)
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

## 🔥 HOT ITEMS (ACTION REQUIRED)

### Immediate (Next Session)
1. **Commit docs** (release notes, handoff, roadmap) — 5 min
2. **Execute cleanup_docs_structure.sh** — 5 min
3. **iOS Suggestion UI** implementation — 2-3 giorni

### This Week (18-24 Apr)
4. **Icona app** design/commission — 2-3 giorni
5. **Test device** completo — 1 giorno
6. **Apple Developer Account** purchase — 30 min + Apple delay

---

**Domande strategiche prossima sessione**:
1. Estetica: budget designer esterno ($200-500) o DIY?
2. TestFlight: quando apriamo beta pubblica?
3. Analytics: setup ora o aspettiamo lancio?
4. LLM Step 3: priorità vs estetica?

---

*Roadmap v1.6 — Aggiornata 17 Aprile 2026 — Prossimo update: fine milestone (30 Aprile)*
