# 📋 TODO LIST - SommelierAI

**Data ultimo aggiornamento:** 27 Aprile 2026  
**Stato sessione:** Tasks 1-5 COMPLETATI ✅ - Polish finale completo, ready for build

---

## ✅ COMPLETATO SESSIONE 27 APRILE 2026

### Tasks 1-5: iOS App Polish (100%)
Sequenza completa di polish finale app iOS secondo HANDOFF.md:

---

## 📋 TASK IN SEQUENZA (NON SALTARE)

### ✅ Task 1: Tab bar in Welcome
**Status:** COMPLETATO ✅ (26 Aprile 2026)  
**File:** ContentView.swift  
**Cosa fatto:** Tab bar sempre visibile, navigazione diretta a Preferiti/Degustazioni

---

### ✅ Task 2: Fix paginazione
**Status:** COMPLETATO ✅ (27 Aprile 2026)  
**Files:** ChatViewModel.swift (initialLimit=5, loadMore()), ChatTypes.swift (+totalCount, +currentLimit), ChatView.swift (bottone paginazione)  
**Risultato:** Feature "Mostra altri vini" ripristinata - 5 vini iniziali, carica 5 alla volta, max 20 totali

---

### ✅ Task 3: Polish animazioni
**Status:** COMPLETATO ✅ (27 Aprile 2026)  
**Files:** ChatView.swift (2 modifiche), WineDetailView.swift (spring damping)  
**Risultato:** Timing standardizzato - toggles 0.15s, spring damping 0.6, target 60fps su tutti i device

---

### ✅ Task 4: Spacing consistency
**Status:** COMPLETATO ✅ (27 Aprile 2026)  
**Files:** WelcomeView.swift (12 sostituzioni), ChatView.swift (18 sostituzioni)  
**Risultato:** Zero hardcoded spacing, uso Spacing.swift (xs/sm/md/lg/xl), padding card 16→20px (luxury feel)

---

### ✅ Task 5: Accessibilità
**Status:** COMPLETATO ✅ (27 Aprile 2026)  
**Files:** Typography.swift (Dynamic Type), ChatView.swift (4 labels), WelcomeView.swift (5+ labels), WineDetailView.swift (6 labels)  
**Risultato:** 15 accessibility labels con hints dinamici, font hardcoded→Dynamic (.largeTitle, .title, .body), VoiceOver support completo

---

## ✅ COMPLETATO SESSIONE PRECEDENTE (25 Aprile 2026)

### Design & Animazioni
- [x] Typing indicator (3 dots bounce)
- [x] Wine cards fade-in con stagger animation
- [x] Haptic feedback su tap
- [x] Spring animations su bottoni
- [x] EmptyStateView premium con suggerimenti

### Feature Implementate
- [x] WineDetailView - Scheda premium con tabs
- [x] Share Vino - ShareSheet iOS
- [x] Search History - UserDefaults persistence
- [x] Ricerche Recenti - Display in WelcomeView
- [x] Bottone "Nuova Ricerca" - Reset chat
- [x] **Scheda Degustazione Completa** ⭐ (vedi SCHEMA_SCHEDA_DEGUSTAZIONE.md)
  - [x] Model + Store + View
  - [x] 4 sezioni collassabili (Visivo, Olfattivo, Gustativo, Finale)
  - [x] Multi-select aromi con emoji
  - [x] Progress bar completamento
  - [x] Salva + "Salva e Preferiti"
  - [x] Persistenza locale

### Bug Fix Precedenti
- [x] FavoritesTabView - Titolo duplicato
- [x] NavigationLink vini tappabili
- [x] SearchHistoryStore - import Combine
- [x] ContentView - Loop fix
- [x] FlowLayout - File standalone condiviso

---

## 🎨 DESIGN & UI POLISH

### Review Icone
- [ ] Audit icone app (SF Symbols consistency)
- [ ] Verificare icone tab bar
- [ ] Verificare icone filtri (vitigno, colore, intensità)
- [ ] Verificare icone WineDetailView (share, favorite, degustazione)
- [ ] Sostituire icone custom se necessario
- [ ] Verificare accessibilità icone (VoiceOver labels)

---

## 🐛 BUG FIX & CLEANUP

### Bug Attivi
- [ ] **Imperdibili (Vini Simili)** - Sezione "Gli Imperdibili" UI preparata in WineDetailView ma manca backend. Backend deve fornire array `similar_wines` in API response.

- [ ] **Backend: Query conversazionali** - "Un prosecco per brindare" ritorna vuoto, mentre "Barolo" funziona. Backend deve gestire linguaggio naturale.

### Cleanup
- [ ] Testing completo su device reali (non solo simulatore)
- [ ] Fix eventuali memory leaks
- [ ] Ottimizzazione immagini assets

---

## 🆕 NUOVE FEATURE (Roadmap Premium)

### **La Mia Cantina** 🍷
- [ ] Implementare tab "La Mia Cantina"
- [ ] Sistema salvataggio vini posseduti (DB locale)
- [ ] Differenziazione da Preferiti (possesso vs desiderio)
- [ ] UI gestione cantina (aggiungi, rimuovi, quantità)
- [ ] Filtri e ricerca in cantina
- [ ] **Feature Premium** - Gate per utenti free

### **I Miei Abbinamenti** 🍽
- [ ] Implementare sezione "I Miei Abbinamenti"
- [ ] Creazione abbinamenti personalizzati (piatto + vino)
- [ ] Salvataggio note abbinamenti
- [ ] Suggerimenti AI basati su abbinamenti salvati
- [ ] **Feature Premium** - Gate per utenti free

### **Scheda Degustazione - Export**
- [x] UI completa implementata (vedi SCHEMA_SCHEDA_DEGUSTAZIONE.md)
- [ ] Export scheda degustazione come PDF
- [ ] **Feature Premium opzionale** - Export PDF solo Premium

---

## 🧪 TESTING & QA

### Pre-Build (da fare ora)
- [ ] Build Xcode senza errori/warning
- [ ] **Test paginazione vini** (CRITICO - query con 7+ vini)
- [ ] Test Dynamic Type (Settings → Text Size → XXXL)
- [ ] Test VoiceOver (Settings → Accessibility → VoiceOver)
- [ ] Test 60fps scroll (iPhone SE simulatore)
- [ ] Edge cases: 6 vini (singolare), 25 vini (cap 20), 3 vini (no bottone)

### Post-TestFlight
- [ ] Test completo flow onboarding
- [ ] Test ricerca + filtri + ordinamento
- [ ] Test favorite + share + detail view
- [ ] Test search history
- [ ] Test scheda degustazione completa
- [ ] Test su iOS 15, 16, 17+
- [ ] Test su iPhone SE, 14, 15 Pro Max
- [ ] Performance testing (caricamento vini, scroll)

---

## 🚀 DEPLOY & RELEASE

- [ ] Review finale codice
- [ ] Rimuovere tutti i debug logs
- [ ] Rimuovere debug reset da ContentView
- [ ] Aggiornare version number
- [ ] Creare build di produzione
- [ ] Testing TestFlight
- [ ] Preparare screenshot App Store
- [ ] Scrivere release notes
- [ ] Submit App Store

---

## 📊 PRIORITÀ

**✅ COMPLETATO:** Tasks 1-5 (Polish finale)  
**🔴 P0 (URGENTE):** Build & Testing (questa settimana)  
**P1 (Alta):** TestFlight deployment, Bug fix  
**P2 (Media):** La Mia Cantina, Design review  
**P3 (Bassa):** I Miei Abbinamenti, Feature aggiuntive future

---

## 📝 NOTE

- **GROUND RULES:** Mai rimuovere feature senza permesso, sempre documentare, sempre backup
- **Task sequence:** 1 ✅ → 2 🔴 → 3 ⏭️ → 4 ⏭️ → 5 ⏭️ (NON SALTARE)
- Feature Premium richiedono integrazione sistema tier (già presente)
- Valutare backend storage per cantina/abbinamenti/degustazioni
- Considerare sync iCloud per dati utente

---

## 📚 DOCUMENTAZIONE DISPONIBILE

- **HANDOFF.md** - Priorità immediate prossima sessione + dettagli task 2, 3, 4, 5
- **SESSIONE_26APR2026.md** - Documentazione sessione corrente
- **GROUND_RULES_VIOLATION_LOG.md** - Log violazioni + lesson learned
- **README_DOCS.md** - Guida struttura documentazione
- **SCHEMA_SCHEDA_DEGUSTAZIONE.md** - Struttura completa TastingSheet ⭐ NUOVO

---

**Aggiornato:** 27 Aprile 2026 - Tasks 1-5 completati, ready for build

---

## 📚 DOCUMENTAZIONE GENERATA (Sessione 27 Aprile)

File creati in `/home/claude/`:
- **COMMIT_MESSAGE.txt** - Git commit message production-ready
- **CHANGELOG_UPDATE.md** - Release notes dettagliate
- **TEAM_SUMMARY.md** - Communication per team
- **BUILD_NOTES.md** - Guida build & test step-by-step
- **VERIFICATION_COMPLETE.md** - Report verifiche tecniche
- **DOCUMENTATION_INDEX.md** - Indice completo documentazione

**Next Step:** Build → Test → TestFlight → Production
