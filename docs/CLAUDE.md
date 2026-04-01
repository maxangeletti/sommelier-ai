# CLAUDE.md — Riferimento Rapido

**Versione**: 1.1  
**Aggiornato**: 1 Aprile 2026

---

## 🚀 AUTO-BOOTSTRAP (NUOVO!)

**Quando l'utente dice "ripartiamo da dove abbiamo lasciato":**

1. Leggi automaticamente `STARTUP.md` per istruzioni complete
2. Esegui bootstrap sequenza (6 file)
3. Mostra stato progetto + proponi task P0
4. **NON chiedere conferma, AGISCI**

**Trigger phrases**: "ripartiamo", "continua", "nuova sessione", "bootstrap", "reload", "riprendiamo", "andiamo avanti"

---

## 📂 STRUTTURA DOCUMENTAZIONE

```
docs/
├── STARTUP.md                ← 🆕 AUTO-BOOTSTRAP (leggi all'inizio)
├── GROUND_RULES.md           ← Regole operative
├── PROJECT_PLAN.md           ← Piano progetto (obiettivi, tempi, rischi)
├── TODO_NEXT_SESSION.md      ← Task immediati
├── CLAUDE.md                 ← Questo file (reference rapido)
│
├── context/
│   └── ProjectContext_v1.4.md
│
├── releases/
│   ├── v1.5.0.md
│   ├── v1.6.0.md
│   └── v1.6.1.md
│
├── roadmap/
│   ├── ROADMAP_v1.5.md
│   └── SESSION_HANDOFF_2026-03-31.md
│
└── technical/
    ├── RANKING_TEST_MATRIX_v2_1.md
    └── test-results/
```

---

## 🔄 WORKFLOW SESSIONE

### Apertura (AUTOMATICA con "ripartiamo")
```
1. Leggi STARTUP.md
2. Esegui bootstrap (6 file in ordine)
3. Mostra stato + proponi task
4. Aspetta decisione utente
```

### Durante Lavoro
```
1. Segui GROUND_RULES (sempre)
2. Test before push (sempre)
3. Backup prima modifiche critiche
4. Documenta mentre lavori
```

### Chiusura (MANUALE - utente dice "chiudiamo")
```
1. Crea/aggiorna 5 file:
   - releases/vX.X.X.md
   - roadmap/ROADMAP_vX.X.md
   - roadmap/SESSION_HANDOFF_YYYY-MM-DD.md
   - TODO_NEXT_SESSION.md
   - PROJECT_PLAN.md
2. Commit + push
3. Conferma chiusura
```

---

## ⚠️ REGOLE CRITICHE

1. **NO TOUCH MOTORE** — Mai modificare ranking engine senza approval
2. **BACKUP OBBLIGATORIO** — Prima di modifiche critiche
3. **TEST BEFORE PUSH** — Mai pushare codice non testato
4. **NO CAZZATE** — Se fallisce, dirlo subito
5. **STARTUP.md PRIORITARIO** — Sempre primo file da leggere

---

## 🎯 STATO CORRENTE (Quick Ref)

- **Backend**: v1.6.0 LIVE, stabile
- **iOS**: v0.9.1, paginazione OK
- **Milestone**: LLM Enhancement + Polish UI (70%)
- **Deadline**: 30 Aprile 2026
- **Prossimo obiettivo**: Test device + LLM Step 2

---

## 📞 HELP

**Quick access**:
- Bootstrap? → `STARTUP.md` 🆕
- Regole? → `GROUND_RULES.md`
- Piano? → `PROJECT_PLAN.md`
- Task? → `TODO_NEXT_SESSION.md`
- Handoff? → `roadmap/SESSION_HANDOFF_[LAST].md`

**In dubbio?** Chiedi all'utente, non inventare.

---

*File di riferimento per Claude v1.1 — Auto-bootstrap abilitato*
