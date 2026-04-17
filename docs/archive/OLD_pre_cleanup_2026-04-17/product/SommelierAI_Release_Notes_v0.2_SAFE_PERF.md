# SOMMELIERAI — RELEASE NOTES
## Versione: v0.2 STABILE + SAFE + PERF (Feb 2026)

---

## Backend – Location Engine v2

### Denomination Recognition — Data Driven
- Rimosso hardcode (KNOWN_DENOMINATIONS, BAROLO_COMMUNES)
- Denominazioni lette dinamicamente dal CSV
- Match robusto anche senza “di/del/della”
  - es: 'brunello montalcino' → denom corretta

### Zone/Comune Automatic Extraction
- Se viene trovata una denom:
  - estrazione automatica delle zone dal dataset
  - solo se coerenti con la query
- Nessuna mappa manuale
- Scalabile a tutte le DOC/DOCG del CSV

### Structured-only Location Terms
- Se presenti 'denom:' o 'comune:'
  - fallback plain escluso
- meta.filters.location_terms ora semanticamente pulito

### Robust Location Tokenizer (A7)
- Nuova funzione _loc_tokens():
  - normalizza apostrofi tipografici
  - gestisce 'd'alba' / 'd alba'
  - gestisce trattini/slash
- Riduce regressioni su input reali

---

## Performance Upgrade (SAFE+PERF – A8)

### LocationIndex
Indice in memoria con:
- denom_sig_tokens
- zone_sig_tokens
- denom_to_zones
- vocab_tokens

Caratteristiche:
- Build automatico
- Invalidazione su CSV mtime
- Warmup su startup
- Stats esposte in /stats

### Colli di bottiglia eliminati
Non viene più ricostruito per ogni query:
- set(df["denomination"])
- df.loc[...] per estrarre zone
- vocab dinamico

Risultato:
- Matching location O(1)
- Migliore scalabilità

---

## Cleanup (A6)

- Rimossi blocchi legacy non più usati
- Eliminati return duplicati
- Aggiunti mini test manuali
- BUILD_ID aggiornato

---

## Stato motore

Location matching: STABILE  
Performance location: OTTIMIZZATA  
Dataset scaling: PRONTO  
Ranking tuning: BASE  
Monetizzazione: NON ATTIVA  
