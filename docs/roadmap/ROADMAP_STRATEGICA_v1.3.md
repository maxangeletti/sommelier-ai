# SOMMELIERAI --- ROADMAP STRATEGICA

Versione: v1.3\
Data aggiornamento: 2026-03-01

------------------------------------------------------------------------

## 📌 Ranking Philosophy -- Modalità e Comportamento Futuro

Per garantire chiarezza architetturale e futura manutenibilità,
definiamo formalmente le modalità ranking (CASE A--B--C--D).

### CASE A --- Match First

-   Peso dominante al match semantico
-   Ideale per query specifiche (es. "Barolo Serralunga")
-   Percezione: "capisce esattamente cosa ho chiesto"
-   Rischio: può promuovere vini medi se altamente coerenti

### CASE B --- Balanced Relevance (ATTUALE DEFAULT)

-   Match + Quality combinati in modo bilanciato
-   Robusto su query generiche
-   Percezione: "consiglia bene anche se chiedo male"
-   Attualmente modalità attiva di default

### CASE C --- Quality First

-   Ranking guidato principalmente da qualità interna
-   Percezione: "best picks del sistema"
-   Rischio: può ignorare specificità della richiesta

### CASE D --- Dual Mode (Match Mode + Smart Mode)

-   Separazione esplicita tra:
    -   Match Mode (quasi puro match)
    -   Smart Mode (relevance bilanciata)
-   Massima trasparenza verso l'utente
-   Possibile implementazione futura in UI avanzata

------------------------------------------------------------------------

## 🎯 Decisione Corrente

✔ Modalità attiva: **CASE B --- Balanced Relevance**\
✔ Motivo: miglior compromesso per fase Beta\
✔ Nessun double counting tra match e quality

------------------------------------------------------------------------

## 🔮 Sezione da Rivalutare in Beta (Bassa Priorità)

Da riesaminare durante test utenti:

-   Peso relativo match vs quality
-   Introduzione Match Mode esplicita (CASE D)
-   Eventuale tuning aggressivo per query altamente specifiche
-   Micro-reason dinamica più esplicita in UI

------------------------------------------------------------------------

## 🧭 Stato Attuale del Progetto

  Fase                      Stato
  ------------------------- ----------
  Fondazione                ✅ 100%
  Hardening                 🟡 \~75%
  Percezione Intelligenza   🟡 \~55%
  TestFlight                🔴 0%
  Expert Rating             🔴 0%
  LLM Layer                 🔴 0%
  Marketing                 🔴 0%

------------------------------------------------------------------------

Documento manutenuto centralmente.\
Ogni fine sessione verrà rigenerata una versione aggiornata.
