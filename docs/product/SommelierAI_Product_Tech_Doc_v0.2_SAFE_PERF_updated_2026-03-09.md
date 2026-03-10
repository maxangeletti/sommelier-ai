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


---

## Stato attuale del progetto

### Stato sintetico
- Blocco tecnico: circa **92%**
- Progetto completo end-to-end: circa **58–60%**
- Baseline attiva: build stabile post-fix output quality + fix ramo `occasion/elegance` combinato + consolidamento performance/cache
- Dataset runtime attuale: `data/wines.normalized.csv`
- Regola stabile: underscore solo interni/tecnici; output utente finale sempre user-friendly

### Completato nel ciclo recente
- **Task 2 — Dataset cleanup critico**: chiuso
- **Task 3 — Output data quality**: chiuso sul perimetro lavorato
- Regressioni post-CSV: rientrate
- Performance + cache consolidation: applicata e validata
- GT 01–13: tornati eseguibili e stabili nella verifica finale del ciclo

### Residui aperti
- **Task 5 — tuning qualitativo**
  - `GT-07` — `rosso elegante`
  - `GT-09` — `bianco fresco per cena importante di pesce`

Questi due punti restano **aperti come tuning qualitativo**, non come regressioni bloccanti.

### Blocco roadmap attuale
Ordine corretto dei lavori:
1. chiusura residuo tecnico
2. source strategy vini
3. schema/catalog governance
4. LLM
5. go-to-market / lancio

### Decisione ufficiale sulla source strategy
- Base catalogo: **proprietaria/curata**
- Enrichment esterno: solo **permissioned**
- Commerce layer: separato dal core
- Ratings layer: separato dal core

---

## Documentazione collegata

### 1) Strategy
Path:
`docs/product/strategy/`

Documenti:
- `sommelierai_source_strategy_decision_2026-03-09.txt`
- `sommelierai_source_evaluation_checklist_v1_2026-03-09.txt`
- `sommelierai_candidate_sources_shortlist_v1_2026-03-09.txt`
- `sommelierai_candidate_sources_evaluation_matrix_v1_2026-03-09.xlsx`
- `sommelierai_candidate_sources_first_pass_evaluation_2026-03-09.xlsx`

Ruolo:
- decisioni strategiche sulle fonti
- shortlist e confronto tra fonti candidate
- strumenti di valutazione

### 2) Catalog
Path:
`docs/product/catalog/`

Documenti:
- `sommelierai_catalog_v1_minimum_schema_2026-03-09.txt`
- `sommelierai_catalog_master_v1_template.csv`
- `sommelierai_catalog_master_v1_template_v2.xlsx`
- `sommelierai_master_to_normalized_mapping_2026-03-09.txt`
- `sommelierai_sourcing_policy_v1_2026-03-09_v2.txt`
- `sommelierai_catalog_ingestion_workflow_v1_2026-03-09_v2.txt`
- `sommelierai_catalog_update_process_v1_2026-03-09_v2.txt`
- `sommelierai_catalog_onboarding_pack_v1_2026-03-09_v2.txt`
- `sommelierai_source_specific_mapping_sheet_template_v1_2026-03-09.xlsx`
- `sommelierai_validation_log_template_v1_2026-03-09.xlsx`
- `sommelierai_post_export_checks_template_v1_2026-03-09.txt`

Ruolo:
- governance del catalogo
- template e processi
- mapping e validazione
- export verso `wines.normalized.csv`

### 3) Roadmap / Progress
Path:
`docs/product/roadmap/`

Documenti:
- `SommelierAI_Roadmap_v1.1.docx`
- `sommelierai_complete_handoff_roadmap_gt_2026-03-07_v2.txt`
- `sommelierai_freeze_cycle_note_2026-03-09.txt`
- `sommelierai_status_progress_2026-03-09.txt`
- `sommelierai_status_progress_2026-03-09_v2.txt`
- `sommelierai_roadmap_gantt_updated_2026-03-07_v2.xlsx`

Ruolo:
- stato del progetto
- priorità
- tracking del ciclo
- avanzamento percentuale

### 4) Release Notes
Path:
`docs/product/release-notes/`

Documenti:
- `sommelierai_release_notes_incremental_2026-03-07_refreshed.txt`
- `sommelierai_release_notes_incremental_2026-03-09.txt`
- `sommelierai_release_notes_incremental_2026-03-09_v3.txt`
- `sommelierai_batch_update_note_template_v1_2026-03-09.txt`

Ruolo:
- tracciare cosa è stato fatto
- cosa è stato accettato
- cosa è stato rollbackato
- cosa resta aperto

### 5) Technical Support Docs
Path:
`docs/technical/`

Documenti:
- `DOCUMENTAZIONE_MATCH_RANKING_DEBUGGER_v1.3.md`
- `RANKING_TEST_MATRIX_v1.0.md`

Ruolo:
- supporto tecnico al motore
- diagnostica
- test matrix

### 6) Legacy / Historical Docs
Path:
`docs/roadmap/`, `docs/archive/`

Ruolo:
- materiale storico
- versioni precedenti
- riferimento retrospettivo

### Regola operativa
Ogni nuovo documento ufficiale deve essere accompagnato da:
- nome file definitivo
- path di destinazione
- ruolo del documento
- decisione se è additivo o sostitutivo
- aggiornamento della sezione “Documentazione collegata” nella product doc

### Check rapido
Prima di chiudere un ciclo, verificare:
- i nuovi documenti sono nel path corretto
- i file non sono rimasti sciolti in `docs/`
- la product doc principale linka i documenti nuovi
- roadmap, release notes e catalog docs sono coerenti tra loro

### Nota rapida (non cambio ora)
Gli underscore restano ammessi nei nomi file e nei canonical interni.
Nell’output utente finale, testi ed etichette devono restare user-friendly.
