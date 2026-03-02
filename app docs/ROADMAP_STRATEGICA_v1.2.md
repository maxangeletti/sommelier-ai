# SOMMELIERAI --- ROADMAP STRATEGICA UFFICIALE

## Versione 1.2

**Data:** 01.03.2026\
**Baseline tecnica:** v1.0.1 + UI Explainability estesa\
**Stato progetto:** Alpha avanzata (non Beta-ready)

------------------------------------------------------------------------

# 🟢 FASE 0 --- FONDAZIONE (COMPLETATA)

## Obiettivo

Costruire MVP funzionante con ranking proprietario.

Include: - Backend FastAPI - Dataset iniziale vini - Ranking A9 - Match
score - Sort relevance / prezzo / rating - Streaming SSE - UI chat
base - Favorites - Tier Free limitato - Grouping annate - Filtri
(vitigno, colore, intensità, prezzo)

**Stato: 100%**

------------------------------------------------------------------------

# 🔵 FASE 1 --- HARDENING & STABILITÀ

## 1A --- Hardening Filtri

-   Separazione filtro vs ordinamento\
-   Combinazioni multiple\
-   Edge cases

**Stato: 85%**

## 1B --- Ranking Debugger

Implementato: - Debug UI dev-mode - match / overall / value visibili

Manca: - Logging strutturato backend - Boost explanation dettagliata

**Stato: 90%**

## 1C --- Dataset Cleanup

-   Normalizzazione vitigni\
-   Uniformare denominazioni\
-   Eliminare duplicati logici

**Stato: 70%**

## 1D --- Performance & Cache

-   Cache risultati query\
-   Riduzione ricalcoli ranking

**Stato: 75%**

### Stato Fase 1 Totale: 80%

------------------------------------------------------------------------

# 🟣 FASE 2 --- PERCEZIONE INTELLIGENZA

## 2A --- Ranking Visivo

-   Stelle paglierino\
-   Overall bar coerente\
-   Evidenziazione #1\
-   Match badge

**Stato: 95%**

## 2B --- Badge Semantici

-   Aperitivo\
-   Cena importante\
-   Top qualità/prezzo\
-   ui_highlights support

**Stato: 85%**

## 2C --- Micro-Reason Dinamica

Implementato: - Match perfetto - Sushi detection - Cena detection -
Strutturato detection - Value sort detection - Fallback coerente

Manca: - Boost explanation dettagliata - Coerenza futura con LLM

**Stato: 80%**

## 2D --- Modalità Ranking Future

-   Qualità\
-   Valore\
-   Match\
-   Sommy Smart A2

**Stato: 0% (Futuro)**

### Stato Fase 2 Totale: 75%

------------------------------------------------------------------------

# 🟢 FASE 3 --- TESTFLIGHT (NON AVVIATA)

Obiettivo: validazione reale utenti esterni.

Richiede: - Hardening chiuso - Dataset coerente - Performance fluida -
value_score robusto

**Stato: 0%**

------------------------------------------------------------------------

# 🟡 FASE 4 --- EXPERT RATINGS INTEGRATION

Previsto: - expert_rating_avg - expert_rating_sources

Peso ranking: 0.15--0.25 opzionale

**Stato: 0%**

------------------------------------------------------------------------

# 🟠 FASE 5 --- LLM LAYER

-   Interpretazione semantica avanzata\
-   Descrizioni naturali\
-   Suggerimenti evoluti

**Stato: 0%**

------------------------------------------------------------------------

# 🔴 FASE 6 --- POSIZIONAMENTO & MARKETING

-   Target
-   Pricing
-   Eventi
-   Strategia lancio

**Stato: embrionale**

------------------------------------------------------------------------

# 📊 FOTO REALISTICA PROGETTO

  Fase            Stato
  --------------- -------
  Fondazione      100%
  Hardening       80%
  Percezione      75%
  TestFlight      0%
  Expert Rating   0%
  LLM Layer       0%
  Marketing       0%

------------------------------------------------------------------------

# PROSSIMA PRIORITÀ

Chiudere FASE 1 (Hardening).\
Prossimo focus: Audit completo value_score.
