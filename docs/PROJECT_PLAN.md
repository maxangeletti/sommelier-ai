# 📊 PIANO DI PROGETTO — SommelierAI

**Versione**: 1.2  
**Ultimo aggiornamento**: 2026-04-17  
**Baseline Backend**: v1.7.0 LIVE (LLM + Suggestions + Fuzzy)  
**Baseline iOS**: v0.9.3 POST-ONBOARDING  
**Ground Truth**: v2.1 (23/27 PASS, +1 vs baseline)

---

## 🎯 DOVE SIAMO

### Backend v1.7.0 — PRODUCTION LIVE ✅
- **Deploy**: Render.com, stabile
- **Uptime**: ~95% (cold start mitigato)
- **Ground Truth**: 23/27 PASS, 3 WARN, 1 FAIL (85% accuracy)
- **Latest Release**: Suggestion mode + LLM Step 2 (17 Apr 2026)
- **LLM**: Step 2 deployed (personalized reasons, timeout 10s)
- **Suggestion Mode**: Generic queries → suggestions instead results
- **Dataset**: 100 vini, 35 regioni, 75 vitigni
- **Performance**: <500ms avg, <1s p95 (LLM adds latency)

### iOS v0.9.3 — POST-ONBOARDING ✅
- **Build**: Stabile, Xcode OK
- **Ultima feature**: WelcomeView onboarding + UI refinements (11 Apr 2026)
- **Crash rate**: <1%
- **Feature completeness**: 95%
- **Pending**: Suggestion mode UI integration, test device

### Milestone Corrente
**"LLM Enhancement + Polish UI"**
- **Progress**: 11/13 task (85%) - nuovi task identificati
- **Deadline**: 30 Aprile 2026
- **Remaining**: 13 giorni
- **Latest**: v1.7.0 deployed - Suggestion mode + LLM Step 2 (17 Apr 2026)

---

## 🚀 DOVE ANDIAMO

### Obiettivi Immediati (Q2 2026)
1. **Beta TestFlight** → Lancio pubblico App Store
2. **LLM Step 2** → Explain personalizzato (sostituisce template)
3. **Estetica finale** → Icona, splash, theme pronto consumer
4. **Analytics** → Tracking decisioni utente

### Obiettivi Medio Termine (Q3 2026)
- Dataset 100 → 200+ vini
- Feature "Cerca simili"
- Expert ratings integration
- Android app (se budget/tempo)

---

## ⚠️ COSA MANCA (PRIORITÀ)

### P0 — BLOCCANTI LANCIO (entro 25 Aprile)
1. ✅ **Paginazione iOS** — COMPLETATA 31 Marzo 2026
2. ✅ **Badge "Ottimo Valore"** — COMPLETATA 8 Aprile 2026 (v1.6.2)
3. ✅ **GT Tests v1.7.0** — COMPLETATA 17 Aprile 2026 (23/27 PASS)
4. ✅ **LLM Step 2** — COMPLETATA 17 Aprile 2026 (v1.7.0 deployed)
5. ✅ **Suggestion Mode** — COMPLETATA 17 Aprile 2026 (v1.7.0 deployed)
6. ✅ **Onboarding iOS** — COMPLETATA 11 Aprile 2026 (v0.9.3)
7. ⏳ **iOS Suggestion UI** — 2-3 giorni (integrazione)
8. ⏳ **Test device reale** — 1 giorno
9. ⏳ **Estetica app** — Icona 1024x1024, splash, theme (2-3 giorni)
10. ⏳ **Apple Developer Account** — $99/anno (acquisto immediato)

### P1 — IMPORTANTI (entro 25 Aprile)
6. ⏳ **TestFlight setup** — 1 giorno
7. ⏳ **Fix `total_count` backend** — Considera filtri (1 giorno)
8. ⏳ **KNOWN_GRAPES** — Arricchimento 14 vitigni (2 giorni)
9. ⏳ **Analytics base** — Firebase setup (1 giorno)

### P2 — NICE TO HAVE (post-lancio)
10. ⏳ **Feature "Cerca simili"** — 3-5 giorni
11. ⏳ **Expert ratings** — Source da identificare
12. ⏳ **Dataset expansion** — 200+ vini

---

## ⏱️ TEMPI RAGGIUNGIMENTO OBIETTIVI

### Milestone Corrente (30 Aprile) — FATTIBILE ✅
```
Task rimanenti: 2/10
- LLM Step 2: 3-4 giorni
- Estetica app: 2-3 giorni
─────────────────────────
TOTALE: 5-7 giorni

Deadline: 30 Aprile
Buffer: 21 giorni
Risk: BASSO 🟢

Completed:
✅ Badge fix v1.6.2 (8 Apr)
✅ GT tests verification (8 Apr)
✅ iOS JSON parsing fix (7 Apr)
```

### Beta TestFlight (15 Maggio) — PROBABILE ✅
```
Setup TestFlight: 1 giorno
Beta testing interno: 1 settimana
Fix critici beta: 3-5 giorni
─────────────────────────
TOTALE: 11-13 giorni

Start: 1 Maggio (post-milestone)
Deadline: 15 Maggio
Risk: MEDIO 🟡
```

### Lancio App Store (31 Maggio) — OTTIMISTICO ⚠️
```
Beta pubblica: 1-2 settimane
App Store submission: 1 giorno
Apple review: 1-2 settimane (IMPREVEDIBILE)
─────────────────────────
TOTALE: 3-5 settimane

Start: 15 Maggio (post-beta)
Target: 31 Maggio
Risk: ALTO 🔴 (dipende da Apple)
```

---

## 📅 ROADMAP VISUALE

```
APRILE 2026
────────────────────────────────────────────────
1-5:   Test paginazione + LLM Step 2 START
6-10:  LLM Step 2 completamento
11-15: Estetica app (icona, splash, theme)
16-20: KNOWN_GRAPES + Fix total_count
21-25: Analytics setup + Test completi
26-30: MILESTONE COMPLETATA ✅

MAGGIO 2026
────────────────────────────────────────────────
1-5:   TestFlight setup + Beta interna
6-10:  Fix critici da beta
11-15: Beta pubblica aperta
16-20: Monitoring + fix
21-25: App Store submission
26-31: Apple review (speranza) + LANCIO 🚀

GIUGNO+ 2026
────────────────────────────────────────────────
Post-lancio: Analytics, feature simili, ratings
```

---

## 🎯 CHECKLIST LANCIO (7/15 DONE)

### Pre-Requisiti Tecnici
- [x] Backend LIVE stabile
- [x] iOS build stable
- [x] GT accuracy >80%
- [x] Paginazione implementata
- [ ] LLM Step 2 integrato
- [ ] Test device reale PASS
- [ ] Analytics tracking attivo

### Pre-Requisiti Business
- [ ] Apple Developer Account ($99)
- [ ] Icona app 1024x1024
- [ ] Splash screen
- [ ] Privacy policy
- [ ] Terms of service

### Pre-Requisiti Beta
- [ ] TestFlight configurato
- [ ] 10-20 beta tester identificati
- [ ] Crash reporting setup
- [ ] Feedback channel attivo

---

## 💰 COSTI IMMINENTI

| Item | Costo | Quando | Necessità |
|------|-------|---------|-----------|
| Apple Developer Account | $99/anno | SUBITO | **OBBLIGATORIO** |
| Designer (icona/splash) | $500-1000 | Opzionale | Consigliato |
| Render paid plan | $7/mese | Opzionale | Elimina cold start |
| Firebase (analytics) | $0 | Free tier OK | Sufficiente per inizio |
| **TOTALE minimo** | **$99** | - | - |
| **TOTALE consigliato** | **$599-1099** | - | - |

---

## 📈 METRICHE SUCCESSO

### Milestone 30 Aprile (Target)
- [ ] LLM Step 2: explain quality score >8/10
- [ ] Estetica: icona approved da 3+ persone
- [ ] Test: 0 crash su 20+ test case
- [ ] Performance: <1s latency p95

### Beta 15 Maggio (Target)
- [ ] 10+ beta tester attivi
- [ ] Feedback score >4/5
- [ ] Crash rate <0.5%
- [ ] Retention D1 >60%

### Lancio 31 Maggio (Target)
- [ ] App Store approval
- [ ] Rating >4.2/5
- [ ] 100+ download settimana 1
- [ ] Review sentiment >70% positive

---

## 🚨 RISCHI & MITIGAZIONI

### ALTO RISCHIO 🔴
**Apple Review Delay**
- **Rischio**: Review può richiedere 1-4 settimane
- **Impatto**: Lancio slittato a Giugno/Luglio
- **Mitigazione**: Submit presto, risposta rapida a rejection

### MEDIO RISCHIO 🟡
**LLM Step 2 Quality**
- **Rischio**: Explain generati non all'altezza
- **Impatto**: Milestone ritardata 1 settimana
- **Mitigazione**: Fallback a template se necessario

**Estetica Non Pronta**
- **Rischio**: Icona/splash non professionali
- **Impatto**: Percezione brand negativa
- **Mitigazione**: Contrattare designer esterno ($500-1000)

### BASSO RISCHIO 🟢
**Bug Minori**
- **Rischio**: Bug post-paginazione su device
- **Impatto**: 1-2 giorni fix
- **Mitigazione**: Test completi + rollback rapido

---

## ✅ PROSSIME 48 ORE (Azioni Concrete)

### Oggi (1 Aprile)
- [ ] Test paginazione su iPhone reale
- [ ] Verifica funzionamento filtri + paginazione
- [ ] Commit cleanup docs
- [ ] Acquisto Apple Developer Account

### Domani (2 Aprile)
- [ ] LLM Step 2: design architettura
- [ ] LLM Step 2: implementazione base backend
- [ ] Test locale explain generation
- [ ] Icona app: brief per designer o AI-generation

---

## 💡 DECISIONI DA PRENDERE (URGENTI)

### Decisione 1: LLM Model
**Domanda**: Haiku vs Sonnet per explain generation?
- **Haiku**: Veloce (<500ms), economico ($0.25/M token), qualità 7/10
- **Sonnet**: Lento (1-2s), costoso ($3/M token), qualità 9/10
- **Raccomandazione**: **Haiku** per MVP, upgrade a Sonnet se feedback negativo

### Decisione 2: Estetica
**Domanda**: Designer esterno vs AI-generated?
- **Designer**: $500-1000, professionale, 3-5 giorni
- **AI-gen**: $0, veloce, qualità 6/10
- **Raccomandazione**: **Designer** se budget OK, altrimenti AI + refinement

### Decisione 3: Analytics
**Domanda**: Setup ora o post-lancio?
- **Ora**: Dati beta utili per iterare
- **Post-lancio**: Risparmio tempo ora, blind su beta
- **Raccomandazione**: **Setup ora** (1 giorno, Firebase free tier)

---

## 🎓 SUMMARY ESECUTIVO

### ✅ COMPLETATO (Sessione 31 Marzo 2026)
- Paginazione iOS completata e funzionante
- Cleanup documentazione (-58% file)
- GROUND_RULES aggiornate
- Release notes v1.6.1 create
- PROJECT_PLAN.md creato

### ⏳ PROSSIMI STEP
1. Test device reale (1-2 Aprile)
2. LLM Step 2 (settimana 1-5 Aprile)
3. Estetica app (settimana 8-12 Aprile)
4. Apple Developer Account (SUBITO)

### 🎯 OBIETTIVO
**Lancio pubblico App Store: 31 Maggio 2026**
- Milestone 30 Aprile: FATTIBILE ✅
- Beta 15 Maggio: PROBABILE ✅
- Lancio 31 Maggio: OTTIMISTICO ⚠️ (rischio Apple review)

### 💪 CONFIDENZA
**80%** — Milestone aprile fattibile, lancio maggio dipende da Apple.

---

## 📊 STORICO AGGIORNAMENTI

### v1.0 — 1 Aprile 2026
- **Creazione documento** iniziale
- **Baseline**: Backend v1.6.0, iOS v0.9.1
- **Milestone corrente**: LLM Enhancement + Polish UI (70%)
- **Prossima deadline**: 30 Aprile 2026

---

**Prossimo aggiornamento**: Fine sessione corrente  
**Aggiornato da**: Claude Sonnet 4.6  
**Data**: 1 Aprile 2026
