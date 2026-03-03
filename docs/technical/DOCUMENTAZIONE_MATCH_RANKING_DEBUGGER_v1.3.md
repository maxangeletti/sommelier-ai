# SommelierAI — Documentazione tecnica (estratto) v1.3

**Data:** 2026-03-01  
**Scope di questa patch:** formalizzare **Match** e **Ranking Debugger (A/B/C/D)** + decisione B.

---

## Match

### Cos’è
Il **Match** misura quanto un vino è coerente con la richiesta utente (testo libero) e/o filtri “impliciti” estratti dalla query.

### Componenti tipiche
> Nomi esatti possono variare, ma la semantica resta quella.

- **keyword/text match**: presenza di termini chiave rilevanti nella richiesta (es. “barolo”, “serralunga”).
- **vincoli strutturati** (quando presenti):
  - cibo (`foods_req`)
  - regione/zona (`region_req`)
  - vitigno (`grapes_req`)
  - colore (`color_req`)
  - intensità (`intensity_req`)
  - tipologia (`typology_req` → sparkling/sweetness)

### Output
- `match_score` (e/o fallback `__match_score`): valore numerico (tipicamente 0..1) usato come componente o segnale per il ranking (in base alla modalità sort).
- In debug (`debug=true`):
  - `match_breakdown`: breakdown per componente (pesi/score/contrib)
  - `match_explanation`: lista di spiegazioni “umane” (es. “match testo: parole chiave forti”)

### Uso nel ranking
- In modalità sort **match**: il match influenza direttamente l’ordinamento (primario o forte).
- In modalità sort **relevance**: il match può essere una componente del composite score (a pesi), o un boost condizionale.

> Importante: evitare double counting: se `score` già incorpora match, non sommarlo di nuovo in un “overall” parallelo.

### Uso nella UI
- UI può usare `match_score` e fallback su `__match_score` per stabilità.
- I dettagli (breakdown) **non** vanno mostrati in produzione: sono per dev/debug (flag).

---

## Ranking Debugger

### Obiettivo
Rendere chiaro “perché un vino sta sopra un altro” senza rompere UX né appesantire payload in produzione.

### Scenari A/B/C/D (decisione attuale: B)

- **A) Debug sempre ON** (sconsigliato)
  - Pro: sempre osservabile
  - Contro: payload più pesante, rischio UX/performance
- **B) Debug solo con flag `debug=true`** ✅ **SCELTA ATTUALE**
  - Pro: zero impatto produzione, controllabile via curl/app dev
  - Contro: richiede attivazione esplicita
- **C) Debug su build debug (app)**
  - Pro: automatico per dev, pulito per release
  - Contro: richiede logica iOS extra + gating
- **D) Debug su endpoint separato**
  - Pro: separazione netta
  - Contro: più superficie API, rischio divergenze

### Implementazione (B)
- POST `/search`: body include `"debug": true`
- GET `/search_stream`: query param `?debug=true`

Quando debug è attivo, ogni risultato include:
- `__match_score`
- `__quality_score` (normalizzato 0..1)
- `__value_score` (normalizzato 0..1)
- `__final_score` (**uguale al `score` usato per ordinare**)

Opzionali:
- `match_breakdown`
- `match_explanation`

In SSE `final.meta` può includere:
- `build_id`
- `debug.rows` e `debug.delta_vs_top` (strumenti dev)

---

## Anti double-counting (regola)
- `score` è il punteggio *effettivo* usato per ordinare (scala “stelle”, es. 0..5).
- `__final_score` in debug deve essere **identico** a `score` (stessa metrica).
- `__quality_score/__value_score/__match_score` sono componenti/diagnostica; non si sommano due volte.

---

## Nota rapida (non cambio ora)
- Dedup dataset (Franciacorta Brut ecc.) rimane pending.
- UX banner “Ordinamento non disponibile…” non viene modificato ora.
- Non introduciamo nuove modalità ranking selezionabili (Nice To Have) in questo step.
