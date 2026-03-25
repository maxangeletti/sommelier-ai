# SommelierAI - Release Notes v1.5.0

**Data:** 25 Marzo 2026  
**Tag:** v1.5.0  
**Commit:** 589b74a

---

## 🎯 Obiettivi Release

Miglioramento UX iOS e fix ricerca vini frizzanti/spumanti.

---

## ✨ Nuove Funzionalità iOS

### UI Dinamica - Filtri Contestuali
- **Filtri nascosti all'avvio**: Barra filtri (Vitigno, Colore, Intensità) nascosta fino al primo risultato
- **Comparsa automatica**: Filtri appaiono dopo prima ricerca con risultati vini
- **UX migliorata**: Interfaccia pulita all'avvio, focus sulla ricerca

### Menu Tier-Aware
- **Free tier**: Solo "Standard", "Smart", "Rilevanza"
- **Pro tier**: Accesso completo a tutti i ranking modes

### Card Espandibile
- **Bottone "Più/Meno dettagli"**: Espande/collassa dettagli vino inline
- **Dettagli nascosti**: Tags, link acquisto visibili solo se espansi

### Miglioramenti Minori
- **Underscore fix**: `carne_rossa` → `carne rossa`
- **Tastiera dismiss**: Chiusura automatica dopo invio
- **Cestino unico**: Solo in navbar

---

## 🔧 Fix Backend - Frizzante/Spumante

**Problema:** Query "bianco frizzante" ritornava solo Moscato

**Fix:** Query frizzante include ANCHE vini spumante (entrambi effervescenti)

**Risultati:**
- "bianco frizzante" → 8 risultati (Prosecco, Franciacorta, Champagne, Moscato)
- "rosso frizzante" → Lambrusco

---

## 📊 Stato Sistema

**Backend:**
- ✅ LLM Step 2 attivo
- ✅ KNOWN_GRAPES 100%
- ✅ Fix frizzante deployato
- ⏳ Tannicità AIS - PENDING
- ⏳ GT v2.1 - PENDING

**iOS:**
- ✅ UI v1.5 completa

---

**Commit:** 589b74a, 354088a, d6365a0
