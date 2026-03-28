# TODO PROSSIMA SESSIONE - Paginazione Risultati

## FEATURE RICHIESTA
**Paginazione 5 → 10 → 15 → 20 risultati**

### Problema Attuale
- Backend può ritornare 47 vini per query "vino rosso"
- iOS mostra TUTTI i 47 vini → app inutilizzabile
- Utente interessato solo ai top 3-5

### Soluzione Richiesta
1. **Default: 5 vini** mostrati inizialmente
2. **Bottone "Mostra altri 5 vini"** se disponibili
3. **Incremento progressivo:** 5 → 10 → 15 → 20 (max)
4. **No più di 20 totali** (limite qualità risultati)

---

## STATO IMPLEMENTAZIONE

### ✅ BACKEND - COMPLETO
- `total_count` aggiunto al meta response (commit `d995638`)
- Backend ritorna: `{count: 5, total_count: 47}` quando `limit=5`
- API già supporta parametro `limit`

**File:** `backend/main.py`
**Test:**
```bash
curl -X POST https://sommelier-ai.onrender.com/search \
  -H "Content-Type: application/json" \
  -d '{"query":"vino rosso","limit":5}'
# Ritorna: meta.count=5, meta.total_count=47
```

### ❌ iOS - DA FARE
**File da modificare:**
1. `Models.swift` - aggiungere `total_count: Int?` a `SearchMeta`
2. `ChatTypes.swift` - aggiungere `totalCount: Int?` a `Message`
3. `ChatViewModel.swift` - salvare `totalCount` da meta, metodo `loadMore()`
4. `ChatView.swift` - bottone "Mostra altri", usa `msg.totalCount`

**STATO PRECEDENTE:**
- Tentativo implementazione fallito (2026-03-28)
- File rotti, ripristinati a commit `354088a`
- Problemi: errori compilazione, testo invisibile (Dark Mode)

---

## APPROCCIO PROSSIMA SESSIONE

### STRATEGIA SICURA
1. **BACKUP VERIFICABILE** di tutti e 4 file iOS prima di iniziare
2. **MODIFICHE INCREMENTALI:**
   - Prima: solo Models.swift + ChatTypes.swift (struct updates)
   - Test compilazione
   - Poi: ChatViewModel.swift (logica loadMore)
   - Test compilazione
   - Infine: ChatView.swift (UI bottone)
   - Test completo
3. **TEST AD OGNI STEP** - build Xcode dopo ogni file
4. **NO PUSH** finché utente non conferma tutto funzionante

### ALTERNATIVE SE iOS TROPPO COMPLESSO
**OPZIONE B - Solo Backend:**
- Default `limit=5` hardcoded in backend
- Nessuna modifica iOS
- Pro: zero rischio regression iOS
- Contro: utente non può espandere risultati (deve rifare query)

**DECISIONE:** Utente preferisce implementazione completa (bottone "Mostra altri").

---

## RISCHI NOTI
- iOS: modifiche a 4 file coordinati = alta probabilità errori
- ChatViewModel già complesso (750+ righe)
- Dark Mode issues (già risolto AppColors, ma attenzione)

## PREREQUISITI
- ✅ Backend total_count funzionante
- ✅ AppColors.swift adattivo Dark Mode
- ✅ Ground Rules definite
- ⏳ Backup iOS verificabili
