# 🚀 STARTUP — Auto-Bootstrap Nuova Sessione

**QUESTO FILE VIENE LETTO ALL'INIZIO DI OGNI NUOVA SESSIONE**

---

## ISTRUZIONI PER CLAUDE

Quando l'utente dice **"ripartiamo da dove abbiamo lasciato"** o simile, esegui automaticamente questa sequenza:

### STEP 1: Lettura Documenti (OBBLIGATORIA)

Leggi **nell'ordine esatto**:

1. `docs/GROUND_RULES.md` — Regole operative
2. `docs/PROJECT_PLAN.md` — Piano progetto (obiettivi, tempi, rischi)
3. `docs/roadmap/SESSION_HANDOFF_[LAST].md` — Contesto ultima sessione
4. `docs/roadmap/ROADMAP_v[LAST].md` — Roadmap strategica
5. `docs/context/ProjectContext_v[LAST].md` — Context completo
6. `docs/TODO_NEXT_SESSION.md` — Task prioritari

**[LAST] = versione più recente** (es. SESSION_HANDOFF_2026-03-31.md)

### STEP 2: Conferma Lettura

Dopo aver letto tutti i 6 file, rispondi:

```
✅ Bootstrap completato!

📊 STATO PROGETTO
- Backend: [versione] [status]
- iOS: [versione] [status]
- Milestone: [nome] ([X/Y] task, [%]%)
- Deadline: [data]

🎯 PROSSIMI TASK
P0 (urgenti):
- [task 1]
- [task 2]

P1 (importanti):
- [task 3]

Cosa vuoi fare?
```

### STEP 3: Proposta Intelligente

Basandoti su `TODO_NEXT_SESSION.md`, proponi:
- Se ci sono task P0 non completati → proponi il primo P0
- Se P0 completati → proponi primo P1
- Se task in corso → chiedi se continuare o switchare

**NON aspettare che l'utente ti dica cosa fare. PROPONI ATTIVAMENTE.**

---

## ESEMPI DI AVVIO

### Esempio 1: Task P0 Presente
```
User: ripartiamo da dove abbiamo lasciato

Claude:
✅ Bootstrap completato!

📊 STATO PROGETTO
- Backend: v1.6.0 LIVE (stabile)
- iOS: v0.9.1 (post-paginazione)
- Milestone: LLM Enhancement + Polish UI (7/10 task, 70%)
- Deadline: 30 Aprile 2026

🎯 PROSSIMI TASK
P0 (urgenti):
- Test paginazione device reale
- LLM Step 2 explain personalizzato

Propongo di iniziare con **test paginazione su device reale**. 
Hai un iPhone a disposizione? Procedo con le istruzioni di test?
```

### Esempio 2: Utente Vuole Task Specifico
```
User: ripartiamo, voglio lavorare su LLM Step 2

Claude:
✅ Bootstrap completato!

[stato progetto...]

Perfetto! LLM Step 2 (explain personalizzato) è P0 nel TODO.

Prima di iniziare:
1. Hai già deciso: Haiku vs Sonnet?
2. Preferisci design architettura prima o prototipo diretto?

Mentre mi rispondi, leggo il codice backend attuale per capire 
dove integrare la feature.
```

---

## TRIGGER PHRASES

Claude deve attivarsi automaticamente quando l'utente dice:
- "ripartiamo da dove abbiamo lasciato"
- "continua da dove abbiamo smesso"
- "nuova sessione"
- "bootstrap"
- "reload context"
- "riprendiamo"
- "andiamo avanti"

**Anche se l'utente NON menziona esplicitamente questo file, Claude deve seguire questa procedura.**

---

## FALLBACK

Se non riesci a leggere uno dei 6 file:
1. Segnala quale file manca
2. Chiedi all'utente di caricarlo
3. Continua con gli altri file
4. Proponi comunque task basandoti su ciò che hai letto

---

## NOTE IMPLEMENTAZIONE

- **Non dire "sto leggendo i file"** → leggi silenziosamente e mostra solo il risultato
- **Non chiedere "vuoi che legga X?"** → leggilo e basta
- **Non fare preamble lunghi** → vai dritto al punto
- **Sii proattivo** → proponi sempre un'azione concreta

---

**File creato**: 1 Aprile 2026  
**Versione**: 1.0  
**Status**: ATTIVO per tutte le nuove sessioni
