# SOMMELIERAI --- ROADMAP STRATEGICA UFFICIALE

Versione: v1.4 Data aggiornamento: 2026-03-01

  --------------------------
  STATO REALE DEL PROGETTO
  --------------------------

FASE 0 --- FONDAZIONE - Backend FastAPI - Dataset iniziale - Ranking
A9 - Match score - Sort relevance / match / price / value - Streaming
SSE - UI Chat SwiftUI - Favorites - Tier Free limitato - Grouping
annate - Filtri (vitigno, colore, intensità, prezzo)

STATO: COMPLETATA (100%)

FASE 1 --- HARDENING & STABILITÀ

1A --- Hardening filtri - Separazione filtro vs ordinamento - Gestione
combinazioni multiple STATO: \~85% (serve validazione edge case finale)

1B --- Ranking Debugger (NUOVO STEP ATTIVO) Decisione: Debug attivabile
SOLO con flag ?debug=true - \_\_match_score - \_\_quality_score -
\_\_value_score - \_\_final_score - Nessun double counting STATO: 0% (da
implementare nella prossima sessione)

1C --- Dataset Cleanup - Normalizzazione vitigni - Deduplicazione logica
STATO: \~50%

1D --- Performance & Cache - Base funzionante - Non ancora strutturata
STATO: \~60%

STATO FASE 1 COMPLESSIVA: \~70%

FASE 2 --- PERCEZIONE INTELLIGENZA

2A --- Ranking visivo - Overall bar - Match badge - Evidenza primo
risultato STATO: \~75%

2B --- Badge semantici - Aperitivo - Cena importante - Top
qualità/prezzo STATO: \~65%

2C --- Micro-reason dinamica - topReasonLabel implementata base - Da
rendere più esplicita e strutturata STATO: \~60%

2D --- Modalità ranking (CASE A-B-C-D formalizzati) Decisione corrente:
CASE B --- Balanced Relevance (default) STATO: 100% definizione teorica,
60% strutturazione futura

STATO FASE 2 COMPLESSIVA: \~65%

FASE 3 --- TESTFLIGHT Prerequisiti: - Ranking Debugger stabile -
Hardening completo - Performance fluida STATO: 0%

FASE 4 --- EXPERT RATINGS INTEGRATION Campi previsti: -
expert_rating_avg - expert_rating_sources STATO: 0%

FASE 5 --- LLM LAYER - Interpretazione semantica avanzata - Descrizioni
naturali evolute STATO: 0%

FASE 6 --- POSIZIONAMENTO & MARKETING - Target - Pricing - Eventi
(Vinitaly) STATO: embrionale

  -----------------------------
  RANKING FILOSOFIA UFFICIALE
  -----------------------------

CASE A --- Match First CASE B --- Balanced Relevance (ATTUALE DEFAULT)
CASE C --- Quality First CASE D --- Dual Mode (Match Mode + Smart Mode)

Decisione attiva: CASE B

  ------------------------------------------------------------
  SEZIONE MATCH (DA DOCUMENTARE NEL DETTAGLIO)
  ------------------------------------------------------------
  Il Match rappresenta la coerenza semantica con la query
  utente. Non deve essere conteggiato due volte nel calcolo
  finale. Va sempre spiegato chiaramente in UI e
  documentazione.

  ------------------------------------------------------------

## PRIORITÀ IMMEDIATA PROSSIMA SESSIONE

Implementare Ranking Debugger con flag ?debug=true Validare assenza
double counting Aggiornare documentazione con breakdown algoritmo

Documento manutenuto centralmente. Ogni fine sessione verrà rigenerata
versione aggiornata.
