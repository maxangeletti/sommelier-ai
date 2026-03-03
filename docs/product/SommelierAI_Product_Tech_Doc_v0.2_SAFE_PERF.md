# SOMMELIERAI — PRODUCT / TECH DOC
## Engine Architecture — Location Matching v2

---

## Pipeline

Query → _extract_location_terms()

1. Tokenizzazione robusta (_loc_tokens)
2. Match denominazione via token set
3. Se denom trovata → estrazione zone via LocationIndex
4. Se presenti structured terms → fallback plain disabilitato
5. _row_location_match():
   - denom + comune → 50/50
   - solo denom → 1.0
   - solo comune → frazione
   - fallback → plain

---

## LocationIndex

Struttura in memoria:
- denom_sig_tokens: token significativi per denominazione
- zone_sig_tokens: token significativi per zone
- denom_to_zones: mapping denom → zone
- vocab_tokens: vocabolario token location

Invalidazione:
- basata su CSV mtime e numero righe

Warmup:
- eseguito allo startup

---

## Obiettivi architetturali

- Zero hardcode denominazioni
- Matching robusto e scalabile
- Performance O(1) rispetto al dataset
- Struttura pronta per alias/sinonimi futuri

---

## Prossimi sviluppi possibili

Backend:
- Ranking tuning dinamico
- Boost implicito denom → vitigno
- Cache derive_* se necessario

Prodotto:
- Persistenza sort iOS
- Gating premium
- Ranking mode selezionabili
