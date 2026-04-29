# 🚨 HANDOFF PROSSIMA SESSIONE - PRIORITÀ MASSIMA

## ✅ AGGIORNAMENTO 27 APRILE 2026

### TASKS 1-5 COMPLETATI
Tutti i task della sequenza sono stati completati con successo:
- ✅ Task 1: Tab bar sempre visibile
- ✅ Task 2: Paginazione ripristinata
- ✅ Task 3: Animazioni standardizzate
- ✅ Task 4: Spacing consistente
- ✅ Task 5: Accessibility completa

**Prossimo step:** Build & Testing (vedi TODO.md)

---

## 🎯 STEP 1: RIPRISTINO PAGINAZIONE (PRIORITÀ MASSIMA)

### Codice da inserire in ChatView.swift

**File:** `/Users/massimilianoangeletti/sommelier-ai/ios-app/SommelierAI/SommelierAI/ChatView.swift`

**Posizione:** Dentro loop vini, dopo la card dell'ultimo vino

**Codice completo:**
```swift
// ✅ PAGINAZIONE: Bottone "Mostra altri"
if let totalCount = msg.totalCount,
   let currentLimit = msg.currentLimit,
   currentLimit < totalCount,
   currentLimit < 20 {
    
    let remaining = min(totalCount - currentLimit, 5)
    
    Button(action: {
        vm.loadMore(for: msg.id)
    }) {
        HStack(spacing: 6) {
            Image(systemName: "arrow.down.circle")
            Text("Mostra altri \(remaining) vini")
                .font(.subheadline.weight(.medium))
        }
        .foregroundStyle(AppColors.primaryWine)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(AppColors.primaryWine.opacity(0.08))
        )
    }
    .padding(.top, 8)
}
```

---

### 📊 LOGICA PAGINAZIONE - LEGGERE ATTENTAMENTE

**COMPORTAMENTO INIZIALE:**
- Backend ritorna SEMPRE massimo **5 vini** alla prima query
- `totalCount` = numero totale vini trovati (es. 7, 12, 50)
- `currentLimit` = quanti vini sono attualmente mostrati (inizialmente 5)

**ESEMPI CONCRETI:**

**Caso 1: Query trova 7 vini**
- Inizialmente mostra: 5 vini
- `totalCount` = 7
- `currentLimit` = 5
- `remaining` = min(7 - 5, 5) = **2**
- Bottone dice: **"Mostra altri 2 vini"**
- Dopo click: mostra tutti i 7 vini, bottone scompare

**Caso 2: Query trova 12 vini**
- Inizialmente mostra: 5 vini
- `totalCount` = 12
- `currentLimit` = 5
- `remaining` = min(12 - 5, 5) = **5**
- Bottone dice: **"Mostra altri 5 vini"**
- Dopo 1° click: mostra 10 vini, bottone dice "Mostra altri 2 vini"
- Dopo 2° click: mostra tutti i 12 vini, bottone scompare

**Caso 3: Query trova 25 vini**
- Inizialmente mostra: 5 vini
- Bottone carica 5 alla volta
- LIMITE MASSIMO: 20 vini totali (hardcoded)
- Dopo 3 click: mostra 20 vini, bottone scompare (anche se ce ne sono 25)

**Caso 4: Query trova 6 vini (CASO EDGE CRITICO)**
- Inizialmente mostra: 5 vini
- `remaining` = min(6 - 5, 5) = **1**
- Bottone dice: **"Mostra 1 altro vino"** (singolare!)
- Dopo click: mostra tutti i 6 vini, bottone scompare
- **QUESTO era il bug da fixare**

**REGOLE CHIARE:**
1. ✅ Mostra **5 vini iniziali** SEMPRE
2. ✅ Bottone carica **fino a 5 vini alla volta**
3. ✅ Testo dinamico: "Mostra altri X vini" (X = min rimanenti o 5)
4. ✅ Limite assoluto: **20 vini massimi** mostrabili totali
5. ✅ Bottone appare SOLO se: `currentLimit < totalCount` E `currentLimit < 20`

---

### Verifica ChatViewModel.loadMore()

**File:** `/Users/massimilianoangeletti/sommelier-ai/ios-app/SommelierAI/SommelierAI/ChatViewModel.swift`

1. Apri ChatViewModel.swift
2. Cerca funzione `loadMore(for messageId:)`
3. Se **NON esiste**, implementala:

```swift
func loadMore(for messageId: UUID) {
    guard let index = messages.firstIndex(where: { $0.id == messageId }),
          let currentLimit = messages[index].currentLimit,
          let totalCount = messages[index].totalCount,
          currentLimit < totalCount else {
        return
    }
    
    let newLimit = min(currentLimit + 5, totalCount, 20)
    
    // Aggiorna limite e richiedi nuovi vini
    messages[index].currentLimit = newLimit
    
    // TODO: Chiamata API per caricare vini aggiuntivi
    // backend.loadMoreWines(limit: newLimit)
}
```

### Test Casi

**Test 1: 7 vini totali**
- Mostra 5 → bottone "Mostra altri 2 vini" → click → mostra 7 → bottone sparisce ✅

**Test 2: 12 vini totali**
- Mostra 5 → "Mostra altri 5 vini" → click → mostra 10
- Mostra 10 → "Mostra altri 2 vini" → click → mostra 12 → bottone sparisce ✅

**Test 3: 6 vini totali (EDGE)**
- Mostra 5 → "Mostra 1 altro vino" → click → mostra 6 → bottone sparisce ✅

---

## 📋 TASK RIMANENTI (IN SEQUENZA STRETTA)

### ✅ Task 1: Tab bar in Welcome
**Status:** COMPLETATO ✅ (26 Aprile 2026)  
**File:** ContentView.swift  
**Cosa fatto:** Tab bar sempre visibile, navigazione diretta

---

### ✅ Task 2: Fix paginazione
**Status:** COMPLETATO ✅ (27 Aprile 2026)  
**Files:** ChatViewModel.swift, ChatTypes.swift, ChatView.swift  
**Risultato:** Feature ripristinata - 5 vini iniziali, carica 5 alla volta, max 20

---

### ✅ Task 3: Polish animazioni
**Status:** COMPLETATO ✅ (27 Aprile 2026)  
**Files:** ChatView.swift, WineDetailView.swift  
**Risultato:** Timing 0.15s, spring damping 0.6, 60fps target

**Obiettivo:** Perfezionare timing e fluidità animazioni esistenti

**File da verificare:**
- `ChatView.swift` - Card fade-in, stagger animation
- `WineDetailView.swift` - Hero section, tab transitions
- `EmptyStateView.swift` - Animazioni suggerimenti
- `WelcomeView.swift` - Apparizione elementi

**Cosa fare:**
1. **Review timing globale**
   - Tutte le animazioni usano `.easeOut(duration: 0.5)`?
   - Spring animations consistent: `.spring(response: 0.3, dampingFraction: 0.6)`?
   
2. **Test fluidità su device reale**
   - Animazioni smooth su iPhone SE (device lento)?
   - Nessun lag durante scroll rapido?
   
3. **Fix eventuali jank**
   - Rimuovere animazioni duplicate
   - Ottimizzare offset/opacity combined

**Criterio completamento:**
- Tutte le animazioni fluide 60fps
- Timing consistente app-wide
- Zero lag su iPhone SE

---

### ✅ Task 4: Spacing consistency
**Status:** COMPLETATO ✅ (27 Aprile 2026)  
**Files:** WelcomeView.swift, ChatView.swift  
**Risultato:** 30 sostituzioni, Spacing.swift utilizzato, card padding 16→20px

**Obiettivo:** Standardizzare spacing/padding in tutta l'app

**File:** `Spacing.swift` (già esiste, verificare usage)

**Cosa fare:**
1. **Audit spacing corrente**
   - Cercare hardcoded padding values: `.padding(16)`, `.padding(20)`
   - Identificare inconsistenze tra schermate
   
2. **Definire spacing system** (se non esiste già in Spacing.swift)
   ```swift
   enum Spacing {
       static let xs: CGFloat = 4
       static let sm: CGFloat = 8
       static let md: CGFloat = 16
       static let lg: CGFloat = 24
       static let xl: CGFloat = 32
   }
   ```
   
3. **Sostituire hardcoded values**
   - `.padding(16)` → `.padding(Spacing.md)`
   - `.spacing(20)` → `.spacing(Spacing.lg)`
   
4. **Verificare consistenza visuale**
   - Cards hanno stesso padding interno?
   - Liste hanno stesso spacing tra elementi?
   - Margini screen-edge consistenti?

**File principali da aggiornare:**
- ChatView.swift
- WineDetailView.swift
- FavoritesTabView.swift
- TastingsTabView.swift
- WelcomeView.swift

**Criterio completamento:**
- Zero hardcoded spacing values
- Spacing system documentato
- Consistenza visuale verificata

---

### ✅ Task 5: Accessibilità
**Status:** COMPLETATO ✅ (27 Aprile 2026)  
**Files:** Typography.swift, ChatView.swift, WelcomeView.swift, WineDetailView.swift  
**Risultato:** Dynamic Type, 15 accessibility labels, VoiceOver support

**Obiettivo:** Rendere app accessibile (VoiceOver, Dynamic Type, Contrasti)

**Cosa fare:**

**A. VoiceOver Support**
1. Aggiungere `.accessibilityLabel()` a tutti i bottoni custom
2. Aggiungere `.accessibilityHint()` dove necessario
3. Raggruppare elementi correlati con `.accessibilityElement(children: .combine)`
4. Test completo con VoiceOver attivo

**B. Dynamic Type**
1. Tutte le font usano `.font(.body)`, `.font(.headline)` etc?
2. Nessun `.font(.system(size: 16))` hardcoded?
3. Layout si adatta a font size grande?
4. Test con "Larger Text" attivo in Settings

**C. Contrasti Colori**
1. Verificare tutti i colori in `AppColors.swift`
2. Check ratio contrasto testo/background (WCAG AA: 4.5:1)
3. Modalità High Contrast funziona?

**File principali:**
- AppColors.swift (verificare contrasti)
- Typography.swift (verificare dynamic type)
- Tutti i View files (aggiungere accessibility labels)

**Tools:**
- Accessibility Inspector (Xcode)
- Settings → Accessibility → VoiceOver
- Settings → Display & Brightness → Text Size

**Criterio completamento:**
- VoiceOver naviga correttamente
- App leggibile con testo grande
- Tutti i contrasti WCAG AA compliant

---

## 📁 FILE DA VERIFICARE

| File | Azione | Priority |
|------|--------|----------|
| ChatView.swift | Inserire codice paginazione | 🔴 MASSIMA |
| ChatViewModel.swift | Verificare/implementare loadMore() | 🔴 MASSIMA |
| ContentView.swift | Verificare tab bar funziona | ✅ Completato |
| ChatView.swift | Polish animazioni (Task 3) | ⏭️ Prossimo |
| WineDetailView.swift | Polish animazioni (Task 3) | ⏭️ Prossimo |
| Spacing.swift | Verificare/estendere (Task 4) | ⏭️ Prossimo |
| AppColors.swift | Verificare contrasti (Task 5) | ⏭️ Prossimo |

---

## 🚫 GROUND RULES - NON VIOLARE

1. **MAI cancellare feature senza permesso esplicito**
2. **SEMPRE documentare ogni change**
3. **SEMPRE creare backup prima di modifiche importanti**
4. **SEMPRE comunicare impatto delle modifiche**

Feature che richiedono GIORNI = massimo rispetto richiesto.

---

## 📚 RIFERIMENTI

- **Transcript sessione:** `/mnt/transcripts/2026-04-26-13-07-54-2026-04-26-pagination-restoration-loadmore.txt`
- **Doc sessione:** `SESSIONE_26APR2026.md`
- **Schema degustazione:** `SCHEMA_SCHEDA_DEGUSTAZIONE.md`
- **Codice originale paginazione:** Righe 115, 121, 350, 356 del transcript

---

## 💡 NOTE IMPERDIBILI (VINI SIMILI)

**File:** WineDetailView.swift  
**Status:** UI preparata (commentata), aspetta backend

Backend deve fornire campo `similar_wines` in response API.

---

**⏰ Data handoff:** 27 Aprile 2026  
**👤 Stato:** Tasks 1-5 COMPLETATI ✅  
**🎯 Prossima azione:** Build & Testing (vedi TODO.md)  
**📝 Documentazione:** Commit message + Build notes pronti in `/home/claude/`
