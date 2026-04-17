# 📋 TODO PROSSIMA SESSIONE

**Aggiornato**: 17 Aprile 2026 ore 21:00  
**Sessione precedente**: Suggestion Mode + Deploy v1.7.0 (100% COMPLETATA!)  
**Prossima sessione**: TBD

---

## ✅ COMPLETATO (Sessione 17 Aprile)

### Backend v1.7.0 LIVE 🎉
- [x] Suggestion mode implementato
- [x] fuzzy_match_query() semplificato
- [x] is_generic_query() aggiunto
- [x] GT tests: 23/27 PASS (+1 vs baseline)
- [x] Deploy verificato
- [x] BUILD_ID aggiornato

### Documentazione
- [x] GROUND_RULES v2 (token budget, GT tests, handoff)
- [x] Release notes v1.7.0
- [x] SESSION_HANDOFF_2026-04-17.md
- [x] Script cleanup_docs_structure.sh creato

---

## 🔥 PRIORITÀ ASSOLUTA (fare SUBITO prossima sessione)

### 1. 💾 Commit Documentazione

```bash
cd /Users/massimilianoangeletti/sommelier-ai

git add docs/releases/v1.7.0.md
git add docs/roadmap/SESSION_HANDOFF_2026-04-17.md
git add docs/TODO_NEXT_SESSION.md
git add docs/GROUND_RULES.md
git commit -m "docs: session 2026-04-17 - v1.7.0 release + handoff + ground rules v2"
git push
```

**Tempo stimato**: 2 minuti

---

### 2. 🧹 Cleanup Docs Structure

```bash
cd /Users/massimilianoangeletti/sommelier-ai
./cleanup_docs_structure.sh
git add docs/
git commit -m "docs: cleanup structure + archive obsolete files"
git push
```

**Tempo stimato**: 5 minuti

---

### 3. 📱 iOS Suggestion Mode UI

**Task**: Integrare suggestion mode in ChatView

**Implementation**:
```swift
// In ChatViewModel
if let suggestions = response.suggestions, !suggestions.isEmpty {
    // Show suggestions UI
    self.showSuggestions = true
    self.suggestions = suggestions
} else {
    // Show results
    self.wines = response.results
}
```

**UI**: Mostrare "Forse cercavi: Amarone?" con tap per ricercare

**Tempo stimato**: 2-3 ore

---

## 📊 PRIORITÀ ALTA

### 4. 🧪 Test Onboarding su Device

Con onboarding completato (v0.9.3), testare su iPhone:
- WelcomeView appare al primo avvio
- Tap suggerimento → ricerca automatica
- Riapri app → WelcomeView NON appare

**Tempo stimato**: 15 minuti

---

### 5. 📸 Screenshot App Store

Con onboarding + suggestion mode:
- WelcomeView
- ChatView con risultati
- ChatView con suggestions (NEW!)
- WineDetailView

**Tempo stimato**: 30 minuti

---

### 6. 🎨 Icona App Definitiva

Sostituire placeholder:
- Design icona 1024x1024
- **Opzioni**: DIY (Figma) o designer ($200-500)

**Tempo stimato**: 2-3 giorni (DIY) o 1 settimana (designer)

---

## 🔄 BACKLOG

### Apple Developer Account
- Acquisto $99/anno
- Setup per TestFlight
- **Tempo**: 30 minuti + attesa Apple

### ROADMAP v1.6 Update
- Aggiornare progress milestone
- Marcare task completati (suggestion mode, LLM Step 2)
- **Tempo**: 15 minuti

### PROJECT_PLAN Update
- Aggiornare metriche backend (v1.7.0)
- Aggiornare milestone progress (10/10 → nuovi task)
- **Tempo**: 20 minuti

---

## 📅 ROADMAP SETTIMANA (18-24 Aprile)

### Venerdì 18 Aprile (DOMANI - se sessione)
- ✅ Commit docs (2 min)
- ✅ Cleanup docs structure (5 min)
- 🔄 ROADMAP + PROJECT_PLAN update (35 min)

### Sabato-Domenica 19-20 Aprile
- iOS suggestion mode UI (2-3 ore)
- Test onboarding device (15 min)
- Screenshot App Store (30 min)

### Lunedì 21 Aprile
- Review iOS integration
- Fix eventuali bug
- Commit iOS final

### Martedì-Giovedì 22-24 Aprile
- Icona app (brief o DIY)
- Apple Developer Account
- TestFlight prep

---

## 🎯 OBIETTIVO MILESTONE (30 Aprile)

**Progress**: 10/10 original tasks (100%) 🎉

**Nuovi task identificati**:
1. ⏳ iOS suggestion mode UI (2-3 giorni)
2. ⏳ Icona app (2-3 giorni)
3. ⏳ Test device completo (1 giorno)

**Buffer**: 13 giorni → FATTIBILISSIMO ✅

---

## 💰 SPESE IMMINENTI

| Item | Costo | Quando | Status |
|------|-------|--------|--------|
| Apple Developer Account | $99/anno | SUBITO | ⏳ Pending |
| Icona app (designer) | $200-500 | Opzionale | 💡 Nice to have |
| Anthropic API (Haiku) | ~$5-10/mese | Attivo | ✅ Running |

---

## 📝 NOTE TECNICHE

### Backend v1.7.0 Status
- ✅ LIVE su Render
- ✅ Commit `fbadc64`
- ✅ GT tests PASS (23/27)
- ✅ Suggestion mode funzionante
- ✅ LLM Step 2 attivo

### iOS App Status
- Version: **0.9.3** (Onboarding complete)
- ✅ WelcomeView implementata
- ⏳ Suggestion mode UI pending
- ⏳ Commit pending (dopo test)

---

## 🔗 RISORSE

**Documentazione Sessione**:
- [Release v1.7.0](/docs/releases/v1.7.0.md)
- [Session Handoff 2026-04-17](/docs/roadmap/SESSION_HANDOFF_2026-04-17.md)
- [GROUND_RULES v2](/docs/GROUND_RULES.md)

**Scripts**:
- cleanup_docs_structure.sh (root)
- test_all_gt.sh (root)

**Backend**:
- URL: https://sommelier-ai.onrender.com
- Dashboard: https://dashboard.render.com

**Tests**:
- GT Results: gt_results_20260417.txt

---

**Prossimo aggiornamento**: Fine prossima sessione  
**Ultima modifica**: Claude Sonnet 4.5, 17 Aprile 2026 ore 21:00
