# SOMMELIERAI — PRODUCT / TECH DOC
## Engine Architecture — Location Matching v2

**Nome file:** `SommelierAI_Product_Tech_Doc_v0.2_SAFE_PERF_updated_2026-03-10.md`  
**Path:** `sommelier-ai/docs/product/SommelierAI_Product_Tech_Doc_v0.2_SAFE_PERF_updated_2026-03-10.md`

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
- Blocco tecnico: circa **93%**
- Progetto completo end-to-end: circa **61–63%**
- Stato generale: **pre-launch strutturato**
- Baseline attiva: build stabile post-fix output quality + consolidamento performance/cache + riallineamento scoring `food + occasion`
- Dataset runtime attuale: `data/wines.normalized.csv`
- Regola stabile: underscore solo interni/tecnici; output utente finale sempre user-friendly

### Lettura corretta dello stato
- il progetto non è più in caos strutturale
- la foundation tecnica, documentale e strategica è ormai forte
- non siamo ancora a progetto finito
- il prossimo valore viene da execution reale + tuning finale, non da nuova foundation generica

### Completato nel ciclo recente
- **Task 2 — Dataset cleanup critico**: chiuso
- **Task 3 — Output data quality**: chiuso sul perimetro lavorato
- Regressioni post-CSV: rientrate
- Performance + cache consolidation: applicata e validata
- GT 01–13: tornati eseguibili e stabili nella verifica finale del ciclo
- correzione del ramo di scoring `food + occasion`
- riallineamento parser `occasion` ai canonical runtime interni inglesi

### Stato GT / residui aperti

#### GT-07 — `rosso elegante`
- Stato: **rientrato**
- Classificazione ufficiale: **B Accepted tuning**
- Lettura: non più in rottura evidente; eventuale residuo solo qualitativo fine, non blocker

#### GT-09 — `bianco fresco per cena importante di pesce`
- Stato: **rientrato sul bug strutturale principale**
- Classificazione ufficiale: **B Accepted tuning**
- Lettura: non più in rottura evidente; eventuale residuo solo qualitativo fine, non blocker

### Fix strutturale rilevante del ciclo
Problema precedente:
- quando `food_present` era attivo, il peso `occasion` veniva azzerato
- questo poteva favorire risultati `aperitif` sopra risultati più coerenti con `dinner` / `important_dinner`

Correzione introdotta:
- nuova branch dedicata `food_present and occasion_intent`
- branch `food_present` pura mantenuta separata
- parser occasion riallineato ai canonical runtime:
  - `aperitif`
  - `important_dinner`
  - `dinner`
  - `lunch`
  - `meditation`
  - `summer`
  - `everyday`

Effetto osservato:
- GT-09 non mostra più un `aperitif` in top-1
- i risultati per query con occasione + cibo sono ora coerenti e difendibili
- query come `rosso elegante per cena importante` mostrano top set coerente con elegance + occasion

### Blocco roadmap attuale
Ordine corretto dei lavori:
1. riallineamento finale documentale
2. verifica file ufficiali roadmap / status / release notes del ciclo
3. execution reale asset / landing / materiali launch
4. eventuale tuning tecnico fine non urgente
5. evoluzioni successive: source strategy, catalog governance, LLM, go-to-market operativo continuo

### Decisione ufficiale sulla source strategy
- Base catalogo: **proprietaria/curata**
- Enrichment esterno: solo **permissioned**
- Commerce layer: separato dal core
- Ratings layer: separato dal core

### Roadmap reale dopo il ciclo 2026-03-10
Blocchi forti:
- hardening / documentation foundation
- source strategy
- catalog governance
- LLM foundation
- GTM foundation
- pricing foundation
- launch / soft launch foundation

Blocchi ancora aperti:
- execution reale
- ultimo consolidamento operativo
- tuning motore fine non urgente

Sintesi:
- vicino a **pre-launch ready** sul piano strutturale
- non ancora pienamente **soft launch ready** senza execution concreta degli asset e ultimo consolidamento operativo

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
- `sommelierai_gt_freeze_update_2026-03-10.rtf`
- `sommelierai_status_progress_2026-03-10.rtf`
- `sommelierai_roadmap_gantt_updated_2026-03-07_v2.xlsx`

Ruolo:
- stato del progetto
- priorità
- tracking del ciclo
- avanzamento percentuale
- freeze update e progress update ufficiali del ciclo 2026-03-10

### 4) Release Notes
Path:
`docs/product/release-notes/`

Documenti:
- `sommelierai_release_notes_incremental_2026-03-07_refreshed.txt`
- `sommelierai_release_notes_incremental_2026-03-09.txt`
- `sommelierai_release_notes_incremental_2026-03-09_v3.txt`
- `sommelierai_release_notes_incremental_2026-03-10.rtf`
- `sommelierai_batch_update_note_template_v1_2026-03-09.txt`

Ruolo:
- tracciare cosa è stato fatto
- cosa è stato accettato
- cosa è stato rollbackato
- cosa resta aperto
- consolidare il riallineamento finale del ciclo 2026-03-10

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
- Gli underscore restano ammessi nei nomi file e nei canonical interni.
- Nell’output utente finale, testi ed etichette devono restare user-friendly.
- I file del ciclo 2026-03-10 risultano al momento con estensione `.rtf` in alcuni casi; più avanti può valere un riallineamento formato/naming, ma non è il focus di questo step.
