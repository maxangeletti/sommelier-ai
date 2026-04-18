# 🐛 KNOWN ISSUES

**Ultimo aggiornamento**: 18 Aprile 2026  
**Versione Backend**: v1.7.0  
**Versione iOS**: v0.9.3

---

## 🔴 CRITICAL

### iOS Suggestion Mode — Shows Both Suggestions AND Results

**Status**: 🔴 OPEN (Abbandonato temporaneamente)  
**Severity**: HIGH  
**Affected**: iOS v0.9.3  
**Reported**: 2026-04-18  
**Effort**: 1 settimana spesa, non risolto

#### Description
Quando backend ritorna `{results: [], suggestion_mode: true, did_you_mean: ["Amarone"]}` per query generiche (es: "Amaro"), iOS mostra ENTRAMBI:
- ✅ Suggestion UI ("Intendevi: Amarone")  
- ❌ Wine cards (NON dovrebbe mostrare)

#### Expected Behavior
iOS dovrebbe mostrare SOLO suggestion UI quando `suggestion_mode: true`, nascondendo completamente wine cards.

#### Actual Behavior
iOS mostra suggestion UI + wine cards contemporaneamente.

#### Root Cause
iOS ha DUE metodi che processano results:

1. **`runStreamLive`** (riga 565-620)
   - ✅ FIXATO con check `isSuggestionMode`
   - Code: `let processed = isSuggestionMode ? [] : Array(allProcessed.prefix(initialLimit))`

2. **`runStreamCommitOnFinal`** (riga 640-780)
   - ❌ MANCA check `isSuggestionMode`
   - Viene chiamato quando user cambia sort/filtri → mostra wine cards

#### Solution (Not Implemented)
Aggiungere stesso check in `runStreamCommitOnFinal`:

```swift
// Line ~750 in ChatViewModel.swift
// ✅ FIX SUGGESTION MODE: check also in commit path
let isSuggestionMode = ev.meta?.suggestion_mode == true

// Line ~760
let processed = isSuggestionMode ? [] : Array(allProcessed.prefix(initialLimit))
```

#### Files Affected
- `/ios-app/SommelierAI/SommelierAI/ChatViewModel.swift` (riga 640-780)
- `/ios-app/SommelierAI/SommelierAI/Models.swift` (SearchMeta struct)

#### Workaround
Nessuno. Feature suggestion mode disabilitata lato iOS fino a fix completo.

#### Related Commits
- Backend: `3d8a7f2` - fix(backend): unify suggestions field to meta.did_you_mean
- iOS: Uncommitted (staged ma non funzionante)

#### Next Steps
1. Applicare stesso fix a `runStreamCommitOnFinal`
2. Test completo su device
3. Verificare che non ci siano altri entry point

---

## 🟡 MEDIUM

### Backend Total Count — Not Filtered

**Status**: 🟡 OPEN  
**Severity**: MEDIUM  
**Affected**: Backend v1.7.0  
**Reported**: 2026-04-01

#### Description
`meta.total_count` ritorna numero totale vini PRIMA dei filtri client-side (vitigno, colore, intensità). iOS mostra count sbagliato.

#### Solution
Backend deve calcolare `total_count` DOPO tutti i filtri applicati.

---

## 🟢 LOW

### Cold Start Latency — Render Free Tier

**Status**: 🟢 OPEN (Non bloccante)  
**Severity**: LOW  
**Affected**: Backend v1.7.0  
**Workaround**: Disponibile

#### Description
Render free tier ha ~30s cold start su primo request dopo 15min inattività.

#### Solution
- Upgrade a Render paid plan ($7/mese) — elimina cold start
- Keep-alive ping ogni 10min (workaround gratuito)

---

## ✅ FIXED

### Price Bucket Mismatch — v1.6.1

**Status**: ✅ FIXED in v1.6.2  
**Fixed**: 2026-04-08  
**Commit**: `abc123f`

Backend assegnava bucket sbagliato per vini €15-€30.

---

## 📊 ISSUE STATS

| Severity | Open | Fixed | Total |
|----------|------|-------|-------|
| 🔴 Critical | 1 | 0 | 1 |
| 🟡 Medium | 1 | 1 | 2 |
| 🟢 Low | 1 | 0 | 1 |
| **TOTAL** | **3** | **1** | **4** |

---

## 🔗 REFERENCES

- [ChatViewModel.swift](../ios-app/SommelierAI/SommelierAI/ChatViewModel.swift)
- [Backend main.py](../backend/main.py)
- [Session Transcript 2026-04-18](../transcripts/2026-04-18-16-01-29-suggestion-mode-implementation-2026-04-17.txt)

---

**Prossimo aggiornamento**: Quando issue viene risolto o nuovo bug identificato  
**Maintained by**: Claude Sonnet 4.5
