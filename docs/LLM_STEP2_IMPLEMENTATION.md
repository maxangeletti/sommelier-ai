# LLM Step 2 — Explain Personalizzato

**Data implementazione:** 2 Aprile 2026  
**Classificazione:** [C-experimental]  
**Stato:** Implementato — Da testare su Render

---

## Obiettivo

Sostituire le `reason` statiche (prese dal campo `description` del CSV) con spiegazioni personalizzate generate dall'LLM in base alla query dell'utente e ai segnali di ranking attivi.

## Esempio

**Query:** "vino elegante per cena importante"  
**Vino:** Gevrey-Chambertin Premier Cru

**Reason vecchia (CSV):**  
"Pinot Nero elegante con tannini morbidi"

**Reason nuova (LLM):**  
"Un Borgogna raffinato perfetto per occasioni formali, elegante e di grande finezza"

---

## Implementazione

### File modificati

1. **`llm_intent_parser.py`**
   - Aggiunta funzione `generate_reason_with_llm()`
   - Aggiunta funzione `_format_signals_for_llm()`
   - Nuovo prompt `EXPLAIN_SYSTEM_PROMPT`

2. **`main.py`**
   - Modificata firma di `_build_wine_card()` per accettare `query` e `ranking_signals`
   - Modificata chiamata a `_build_wine_card()` in `run_search()` per passare segnali

### Architettura

```
run_search()
  ↓
  Per ogni vino:
    1. Calcola ranking score
    2. Prepara ranking_signals dict
    3. Chiama _build_wine_card(query, ranking_signals)
       ↓
       _build_wine_card()
         ↓
         generate_reason_with_llm()
           ↓
           LLM API call (Haiku)
           ↓
           Reason personalizzata (max 15 parole)
```

### Fallback Strategy

- Se `LLM_ENABLED=0` → usa description CSV
- Se API key mancante → usa description CSV
- Se LLM timeout/errore → usa description CSV
- **Nessun crash garantito** — graceful degradation sempre attiva

### Segnali passati all'LLM

```python
ranking_signals = {
    "color_req": color_req,              # "rosso", "bianco", "rosato"
    "region": region,                    # "piemonte", "toscana", etc.
    "prestige_intent": prestige_intent,  # True/False
    "elegance_intent": elegance_intent,  # True/False
    "occasion_intent": occasion_intent,  # "important_dinner", "aperitif", etc.
    "foods_req": foods_req,              # ["pesce", "carne"]
    "intensity_req": intensity_req,      # "high", "low", etc.
    "tannin_req": typology_req.get("tannin"),
    "style_intent": style_intent,        # "fresco", "strutturato", etc.
    "value_intent": value_intent,        # True/False
}
```

---

## Prompt LLM

**Modello:** claude-haiku-4-5-20251001  
**Max tokens:** 100  
**Timeout:** 4 secondi

**System prompt:**
```
Sei un sommelier esperto italiano.
Genera una reason breve (max 15 parole) che spiega perché questo vino è rilevante per la query dell'utente.

Regole:
- Massimo 15 parole
- Tono naturale, non tecnico
- Enfatizza i match attivi
- Non ripetere nome del vino
- Non usare gergo enologico difficile
```

**User prompt template:**
```
Query utente: "{query}"

Vino: {wine_name}
Regione: {wine_region}

Segnali di ranking attivi:
{signals_formatted}

Genera una reason breve (max 15 parole).
```

---

## Test Cases

### GT-08 — Rosso elegante per cena importante
- **Query:** "rosso elegante per cena importante"
- **Vino:** Gevrey-Chambertin
- **Segnali attivi:** elegant_intent=True, occasion=important_dinner, color=rosso, prestige_intent
- **Reason attesa:** "Un Borgogna raffinato perfetto per occasioni formali, elegante e di grande finezza"

### GT-05 — Rosso strutturato
- **Query:** "rosso strutturato"
- **Vino:** Barolo DOCG
- **Segnali attivi:** intensity_req=high, style=strutturato, color=rosso
- **Reason attesa:** "Piemonte prestigioso con tannini importanti e struttura verticale"

### GT-06 — Bianco fresco per pesce
- **Query:** "bianco fresco per cena importante di pesce"
- **Vino:** Vermentino di Gallura
- **Segnali attivi:** color=bianco, foods=pesce, style=fresco
- **Reason attesa:** "Fresco e sapido, ideale per crostacei e piatti di mare"

### GT-26 — Voglio stupire
- **Query:** "voglio stupire"
- **Vino:** Barolo DOCG
- **Segnali attivi:** prestige_intent=True
- **Reason attesa:** "Vino iconico del Piemonte, prestigioso e di grande impatto"

---

## Metriche di Costo

**Per singola search con 10 risultati:**
- 10 chiamate LLM (una per vino)
- ~150 token input per chiamata
- ~20 token output per chiamata
- **Totale:** ~1,700 token per search

**Costo stimato (Haiku):**
- Input: $0.80 per 1M token
- Output: $4.00 per 1M token
- **Costo per search:** ~$0.002 (0.2 centesimi)

**Con 100 search/giorno:**
- **Costo mensile:** ~$6

---

## Next Steps

1. **Deploy su Render** — Push modifiche e verifica build
2. **Test manuale** — Verificare reason generate per GT-08, GT-05, GT-06, GT-26
3. **Validazione UX** — Verificare che le reason siano naturali e pertinenti
4. **Tuning prompt** (se necessario) — Aggiustare tono/lunghezza se le reason non sono ideali
5. **Promozione a [B-accepted]** se test positivi

---

## Rollback Plan

Se LLM Step 2 produce reason di bassa qualità:

1. **Quick fix:** `export SOMMELIERAI_LLM_ENABLED=0` su Render → torna a description CSV
2. **Code rollback:** Rimuovere parametri `query` e `ranking_signals` da `_build_wine_card()`

---

## Note

- **Non rompe mai** — fallback garantito
- **Aggiunge valore UX** — reason contestuali alla query
- **Costo contenuto** — ~$6/mese con 100 search/giorno
- **Compatibile con dual-step** — Step 1 (parse) e Step 2 (explain) indipendenti

---

*Generato il 2 Aprile 2026 — LLM Step 2 Implementation*
