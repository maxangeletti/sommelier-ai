# 📋 TODO PROSSIMA SESSIONE

**Aggiornato**: 22 Aprile 2026  
**Sessione precedente**: Backend Bugfix (bollinger) - ⚠️ FIX NON TESTATO  
**Prossima sessione**: TBD

---

## 🚨 CRITICO - FARE SUBITO

### 1. ⚠️ TEST BACKEND FIX "BOLLINGER"

**MUST DO**: Verificare che ricerca "bollinger" restituisca SOLO Bollinger, NON Roederer

**Test**:
```bash
curl -X POST "https://sommelier-ai.onrender.com/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"bollinger", "limit": 5}'
```

**Expected**:
- ✅ SOLO Bollinger (ID 52) nei risultati
- ❌ Louis Roederer (ID 53) NON deve comparire

**Se test PASSA**:
- ✅ Marcare fix come verificato in v1.8.1.md
- Procedere con task successivi

**Se test FALLISCE**:
- 🚨 DEBUG approfondito keyword filter
- Verificare se backend ha ricaricato modifiche (--reload)
- Verificare pandas Series vs dict issue

**Tempo stimato**: 5 minuti test + eventuale debug

---

### 2. 🐛 FIX iOS "GLI IMPERDIBILI" - ID NUMERICI

**Issue**: "Gli Imperdibili" mostra "ID: 32" invece del nome vino

**Root Cause**: API `/wine/{id}/similar` non restituisce campo `name` popolato

**Debug Steps**:
1. Test endpoint: `curl https://sommelier-ai.onrender.com/wine/52/similar?limit=3`
2. Verificare response JSON - campo `name` presente?
3. Se `name` vuoto → fix backend `_build_wine_card()`
4. Se `name` presente → fix iOS parsing

**Tempo stimato**: 30 minuti - 1 ora

---

### 3. 💾 COMMIT DOCUMENTAZIONE

```bash
cd /Users/massimilianoangeletti/sommelier-ai

git add docs/releases/v1.8.1.md
git add docs/roadmap/SESSION_HANDOFF_2026-04-22.md
git add docs/TODO_NEXT_SESSION.md
git commit -m "docs: session 2026-04-22 - v1.8.1 backend bugfix (untested)"
git push
```

**Tempo stimato**: 2 minuti

---

## 📊 PRIORITÀ ALTA

### 4. 🧹 Cleanup Docs Structure

```bash
cd /Users/massimilianoangeletti/sommelier-ai
./cleanup_docs_structure.sh
git add docs/
git commit -m "docs: cleanup structure + archive obsolete files"
git push
```

**Tempo stimato**: 5 minuti

---

### 5. 📱 iOS Suggestion Mode UI

**Task**: Integrare suggestion mode in ChatView (da sessione precedente)

**Implementation**:
```swift
// In ChatViewModel
if let suggestions = response.suggestions, !suggestions.isEmpty {
    self.showSuggestions = true
    self.suggestions = suggestions
} else {
    self.wines = response.results
}
```

**UI**: Mostrare "Forse cercavi: Amarone?" con tap per ricercare

**Tempo stimato**: 2-3 ore

---

### 6. 🧪 Test Onboarding su Device

Con onboarding completato (v0.9.3), testare su iPhone:
- WelcomeView appare al primo avvio
- Tap suggerimento → ricerca automatica
- Riapri app → WelcomeView NON appare

**Tempo stimato**: 15 minuti

---

## 🔄 BACKLOG

### 7. 📸 Screenshot App Store

Con onboarding + suggestion mode:
- WelcomeView
- ChatView con risultati
- ChatView con suggestions
- WineDetailView

**Tempo stimato**: 30 minuti

---

### 8. 🎨 Icona App Definitiva

Sostituire placeholder:
- Design icona 1024x1024
- **Opzioni**: DIY (Figma) o designer ($200-500)

**Tempo stimato**: 2-3 giorni (DIY) o 1 settimana (designer)

---

### 9. Apple Developer Account
- Acquisto $99/anno
- Setup per TestFlight
- **Tempo**: 30 minuti + attesa Apple

---

### 10. ROADMAP + PROJECT_PLAN Update
- Aggiornare progress milestone
- Marcare task completati
- Aggiornare metriche backend
- **Tempo**: 30 minuti

---

## 📅 ROADMAP SETTIMANA (23-29 Aprile)

### Martedì 23 Aprile (SUBITO - se sessione)
- 🚨 Test backend "bollinger" (5 min)
- 🐛 Fix iOS "Gli Imperdibili" (30-60 min)
- ✅ Commit docs (2 min)

### Mercoledì-Giovedì 24-25 Aprile
- iOS suggestion mode UI (2-3 ore)
- Test onboarding device (15 min)
- Cleanup docs (5 min)

### Venerdì 26 Aprile
- Review iOS integration
- Screenshot App Store (30 min)
- Fix eventuali bug

### Sabato-Domenica 27-28 Aprile
- Icona app (brief o DIY)
- Apple Developer Account
- ROADMAP update

---

## 🎯 OBIETTIVO MILESTONE (30 Aprile)

**Progress**: ~75% → target 100%

**Task rimanenti** (7 giorni buffer):
1. ⏳ iOS suggestion mode UI (2-3 giorni)
2. ⏳ Icona app (2-3 giorni)
3. ⏳ Test device completo (1 giorno)
4. ⏳ Fix "Gli Imperdibili" (urgente)
5. ⏳ Test backend fix (urgente)

**Status**: ⚠️ Buffer ridotto, ma FATTIBILE se focus

---

## 💰 SPESE IMMINENTI

| Item | Costo | Quando | Status |
|------|-------|--------|--------|
| Apple Developer Account | $99/anno | SUBITO | ⏳ Pending |
| Icona app (designer) | $200-500 | Opzionale | 💡 Nice to have |
| Anthropic API (Haiku) | ~$5-10/mese | Attivo | ✅ Running |

---

## 📝 NOTE TECNICHE

### Backend v1.8.1 Status
- ⚠️ Code ready, NOT TESTED
- Fix: Keyword filter "bollinger"
- File: `backend/main.py` riga ~2450
- Deploy: Presumibilmente LIVE (se --reload)

### iOS App Status
- Version: **0.9.3** (Unchanged)
- Bug open: "Gli Imperdibili" ID numerici
- ⏳ Suggestion mode UI pending
- ⏳ Fix pending

---

## 🔗 RISORSE

**Documentazione Sessione**:
- [Release v1.8.1](/docs/releases/v1.8.1.md)
- [Session Handoff 2026-04-22](/docs/roadmap/SESSION_HANDOFF_2026-04-22.md)

**Backend**:
- URL: https://sommelier-ai.onrender.com
- Dashboard: https://dashboard.render.com

**Test**:
- Backend fix: `curl -X POST "https://sommelier-ai.onrender.com/search" -d '{"query":"bollinger"}'`
- Similar wines: `curl https://sommelier-ai.onrender.com/wine/52/similar?limit=3`

---

**Prossimo aggiornamento**: Fine prossima sessione  
**Ultima modifica**: Claude Sonnet 4.5, 22 Aprile 2026
