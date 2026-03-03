# SommelierAI — Ranking Test Matrix
Versione: v1.0  
Data: 2026-03-03  
Scope: Backend `/search` — ranking + filtri + debug payload  
Build baseline: v0.2.x (relevance_v2 / value_intent override / match_breakdown flatten / proximity_bonus=0.06)

---

## Regole di esecuzione

- Sempre: `debug=true`
- Salvare: `meta.sort`, `meta.__debug_sort_after_override`
- Salvare per i primi 3 risultati:
  - `rank`, `id`, `name`, `price`
  - `score`, `__quality_score`, `__value_score`, `__match_score`
  - `__price_delta`
- Se presente `meta.debug.delta_vs_top`: salvare almeno `delta_contrib` per rank 2 e 3.

---

## Comando standard (template)

```bash
curl -s -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"<QUERY>","sort":"<SORT>","limit":10,"debug":true}'
