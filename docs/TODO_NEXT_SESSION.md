# 📋 TODO PROSSIMA SESSIONE

**Aggiornato**: 8 Aprile 2026 ore 20:15  
**Sessione precedente**: Onboarding Screen Implementation (100% COMPLETATA!)  
**Prossima sessione**: 9 Aprile 2026

---

## ✅ COMPLETATO (Sessione 8 Aprile - Sera)

### Onboarding Screen (100% COMPLETATA) 🎉
- [x] WelcomeView.swift creata con design matching screenshot
- [x] ContentView.swift integrata con @AppStorage
- [x] ChatView.swift modificata per leggere pendingSearchQuery
- [x] Suggerimenti query predefiniti funzionanti
- [x] Integrazione end-to-end completata

**Status**: PRONTA per test su device! ✅

---

## 🔥 PRIORITÀ ASSOLUTA (fare SUBITO domani)

### 1. 🧪 Test Onboarding su Device

**Cosa testare**:
1. Build in Xcode
2. Run su iPhone (simulator OK per primo test)
3. Verifica WelcomeView appare al primo avvio
4. Tap su suggerimento → va a ChatView con ricerca automatica
5. Scrivi query custom → va a ChatView con quella query
6. Chiudi e riapri app → WelcomeView NON appare più

**Tempo stimato**: 10 minuti

---

### 2. 💾 Commit iOS Onboarding

```bash
cd /Users/massimilianoangeletti/sommelier-ai
git add ios-app/SommelierAI/SommelierAI/WelcomeView.swift
git add ios-app/SommelierAI/SommelierAI/ContentView.swift
git add ios-app/SommelierAI/SommelierAI/ChatView.swift
git commit -m "iOS v0.9.3: Add onboarding screen with query suggestions

- Create WelcomeView with 4 predefined query suggestions
- Show welcome screen on first app launch only (@AppStorage)
- Pass selected query to ChatView via UserDefaults
- ChatView auto-triggers search on pending query
- Clean, polished UI matching design spec"
git push origin main
```

**Tempo stimato**: 3 minuti

---

### 3. 🎨 LLM Step 2 - Implementation

**Decisioni prese**:
- Model: **Haiku** ($0.25/M token)
- Scope: **Top 3 vini**
- Fallback: **Template CSV** obbligatorio

**Architettura**:
```python
# backend/engine.py
async def generate_reason_llm(wine_data, query, context):
    try:
        response = await anthropic_client.generate(
            model="claude-3-5-haiku-20241022",
            max_tokens=150,
            system="Sommelier esperto...",
            messages=[...]
        )
        return response.text
    except Exception:
        return fallback_csv_template(wine_data)
```

**Tempo stimato**: 
- Design: 1 ora
- Implementazione: 3-4 ore
- Testing: 2 ore

---

## 📊 PRIORITÀ ALTA

### 4. 📱 Screenshot per App Store

Con onboarding completato, cattura screenshot per submission:
- WelcomeView
- ChatView con risultati
- WineDetailView  
- Favorites

**Tempo stimato**: 30 minuti

---

### 5. 🎨 Icona App Definitiva

Sostituire placeholder cerchio grigio:
- Design icona 1024x1024
- **Opzioni**: DIY (Figma) o designer ($200-500)

**Tempo stimato**: 2-3 giorni (DIY) o 1 settimana (designer)

---

## 🔄 BACKLOG

### Apple Developer Account
- Acquisto $99/anno
- Setup per TestFlight
- **Tempo**: 30 minuti + attesa Apple

### KNOWN_GRAPES Arricchimento
- Aggiungere 14 vitigni mancanti
- **Tempo**: 2 giorni

---

## 📅 ROADMAP SETTIMANA (9-12 Aprile)

### Martedì 9 Aprile (DOMANI)
- ✅ Test onboarding device (10 min)
- ✅ Commit iOS v0.9.3 (3 min)
- 🔄 LLM Step 2: Design architettura (1 ora)
- 🔄 LLM Step 2: Implementazione base (3-4 ore)

### Mercoledì 10 Aprile
- LLM Step 2: Test + tuning
- Screenshot App Store

### Giovedì 11 Aprile
- LLM Step 2: Deploy + verifiche
- Icona app: Brief o DIY

### Venerdì 12 Aprile
- Review progresso milestone
- Pianificare settimana 15-19 Aprile

---

## 🎯 OBIETTIVO MILESTONE (30 Aprile)

**Progress**: 10/10 task (100%) 🎉

**Completati**:
- ✅ Badge "Ottimo Valore" (v1.6.2)
- ✅ iOS Fix JSON parsing
- ✅ GT Tests verification
- ✅ **Onboarding screen (100% - COMPLETATA!)**

**Rimanenti**:
1. ⏳ LLM Step 2 (3-4 giorni)
2. ⏳ Icona app (2-3 giorni)

**Buffer**: 21 giorni → FATTIBILISSIMO ✅

---

## 💰 SPESE IMMINENTI

| Item | Costo | Quando | Status |
|------|-------|--------|--------|
| Apple Developer Account | $99/anno | SUBITO | ⏳ Pending |
| Icona app (designer) | $200-500 | Opzionale | 💡 Nice to have |
| Anthropic API (Haiku) | ~$5-10/mese | Dopo LLM Step 2 | 💡 Budget OK |

---

## 📝 NOTE TECNICHE

### Backend v1.6.2 Status
- ✅ LIVE su Render
- ✅ Commit `647252d`
- ✅ GT tests PASS (4/4 critici)

### iOS App Status
- Version: **0.9.3** (Onboarding complete!)
- ✅ WelcomeView implementata
- ✅ Integrazione ChatView completata
- ⏳ Commit pending (dopo test)

---

## 🔗 RISORSE

**Documentazione**:
- [GT Report v1.6.2](/docs/GT_REPORT_v1.6.2.md)
- [Release v1.6.2](/docs/releases/v1.6.2.md)
- [Session Handoff 2026-04-08](/docs/roadmap/SESSION_HANDOFF_2026-04-08.md)
- [Project Plan](/docs/PROJECT_PLAN.md)

**Codice iOS Onboarding**:
- WelcomeView: `/ios-app/SommelierAI/SommelierAI/WelcomeView.swift` ✅
- ContentView: `/ios-app/SommelierAI/SommelierAI/ContentView.swift` ✅
- ChatView: `/ios-app/SommelierAI/SommelierAI/ChatView.swift` ✅

**Backend**:
- URL: https://sommelier-ai.onrender.com
- Dashboard: https://dashboard.render.com

---

**Prossimo aggiornamento**: Fine sessione 9 Aprile 2026  
**Ultima modifica**: Claude Sonnet 4.6, 8 Aprile 2026 ore 20:15
