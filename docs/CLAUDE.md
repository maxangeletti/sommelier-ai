# CLAUDE.md — Riferimento Rapido

**Versione**: 1.0  
**Aggiornato**: 1 Aprile 2026

---

## 📂 STRUTTURA DOCUMENTAZIONE

```
docs/
├── GROUND_RULES.md           ← Regole operative (LEGGI SEMPRE)
├── PROJECT_PLAN.md           ← Piano progetto (obiettivi, tempi, rischi)
├── TODO_NEXT_SESSION.md      ← Task immediati
├── CLAUDE.md                 ← Questo file (reference rapido)
│
├── context/
│   └── ProjectContext_v1.4.md   ← Context completo progetto
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
    ├── RAPPORTO_COLLAUDO_DEFINITIVO.md
    ├── MATCH_DEBUGGER_v1.3.md
    ├── PATCH_GT24_GT26.md
    └── test-results/
        ├── gt_results_FINAL_v2.txt
        └── gt_runner.sh
```

---

## 🚀 QUICK START NUOVA SESSIONE

### 1. Leggi (in ordine):
1. `GROUND_RULES.md`
2. `PROJECT_PLAN.md`
3. `roadmap/SESSION_HANDOFF_[LAST].md`
4. `roadmap/ROADMAP_v[LAST].md`
5. `context/ProjectContext_v[LAST].md`
6. `TODO_NEXT_SESSION.md`

### 2. Conferma:
"Letti GROUND_RULES, PROJECT_PLAN, HANDOFF, ROADMAP, CONTEXT, TODO. Pronto."

### 3. Proponi:
Identifica task P0 da `TODO_NEXT_SESSION.md` o chiedi all'utente

---

## 📋 CHIUSURA SESSIONE

### File da creare/aggiornare:
1. `releases/vX.X.X.md` (nuovo)
2. `roadmap/ROADMAP_vX.X.md` (aggiornato)
3. `roadmap/SESSION_HANDOFF_YYYY-MM-DD.md` (nuovo)
4. `TODO_NEXT_SESSION.md` (aggiornato)
5. `PROJECT_PLAN.md` (aggiornato) ⭐

### Commit:
```bash
git add docs/
git commit -m "docs: session YYYY-MM-DD - release notes + roadmap + project plan update"
git push
```

---

## ⚠️ REGOLE CRITICHE

1. **NO TOUCH MOTORE** — Mai modificare ranking engine senza approval esplicita
2. **BACKUP OBBLIGATORIO** — Prima di modifiche critiche
3. **TEST BEFORE PUSH** — Mai pushare codice non testato
4. **NO CAZZATE** — Se fallisce, dirlo subito
5. **DOCUMENTA SEMPRE** — Aggiorna PROJECT_PLAN.md ad ogni sessione

---

## 🎯 STATO CORRENTE (Quick Ref)

- **Backend**: v1.6.0 LIVE, stabile
- **iOS**: v0.9.1, paginazione OK
- **Milestone**: LLM Enhancement + Polish UI (70%)
- **Deadline**: 30 Aprile 2026
- **Prossimo obiettivo**: Test device + LLM Step 2

---

## 📞 HELP

**File più importanti**:
- Regole? → `GROUND_RULES.md`
- Piano? → `PROJECT_PLAN.md`
- Task? → `TODO_NEXT_SESSION.md`
- Handoff? → `roadmap/SESSION_HANDOFF_[LAST].md`

**In dubbio?** Chiedi all'utente, non inventare.

---

*File di riferimento per Claude — Non modificare senza motivo*
