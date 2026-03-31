# 🧹 Piano Pulizia Documentazione - SommelierAI

**Data**: 31 Marzo 2026  
**Obiettivo**: Riorganizzare docs/ per chiarezza e manutenibilità

---

## 📋 FASE 1 - ELIMINAZIONE FILE OBSOLETI

### File da ELIMINARE (non servono più)

#### Root docs/
```bash
# Vecchi recap superseded da HANDOFF più recente
rm docs/SESSION_RECAP_2026-03-28.md
rm docs/SESSION_RECAP_2026-03-31.md

# Vecchi context superseded da v1.4
rm docs/SommelierAI_ProjectContext_v1.1.docx
rm docs/SommelierAI_ProjectContext_v1_2.docx
rm docs/SommelierAI_ProjectContext_v1_3.md

# Summary implementazioni vecchie (info già in release notes)
rm docs/STEP2_IMPLEMENTATION_SUMMARY.md
rm docs/GRAPES_ENRICHMENT_SUMMARY.md
rm docs/AESTHETICS_COMPLETE_SUMMARY.md

# KEEP ONLY: CLAUDE.md (versione concisa), v1.4 (più recente)
```

#### docs/technical/
```bash
# GT results vecchi (keep solo FINAL)
rm docs/technical/gt_results_20260320_*.txt
rm docs/technical/gt_results_20260325_*.txt
rm docs/technical/gt_results_post_fix.txt

# Vecchie versioni matrix (keep solo v2_1)
rm docs/technical/RANKING_TEST_MATRIX_v1.0.md
rm docs/technical/RANKING_TEST_MATRIX_v2.0.md

# Vecchi rapporti collaudo (keep solo DEFINITIVO)
rm docs/technical/RAPPORTO_COLLAUDO_GT_v2_0.md

# KEEP: RANKING_TEST_MATRIX_v2_1.md, gt_results_FINAL_v2.txt, RAPPORTO_COLLAUDO_GT_v2_0_DEFINITIVO.md
```

#### docs/roadmap/
```bash
# Vecchie roadmap (keep solo UFFICIALE v1.5)
rm docs/roadmap/ROADMAP_STRATEGICA_v1.2.md
rm docs/roadmap/ROADMAP_STRATEGICA_v1.3.md
rm docs/roadmap/ROADMAP_STRATEGICA_v1.3\(1\).md
rm docs/roadmap/ROADMAP_STRATEGICA_v1.4.md
rm docs/roadmap/SommelierAI_Roadmap_v1.2.1_FULL_02-03-2026.pdf

# KEEP: ROADMAP_UFFICIALE_v1.5.md (ora in docs/ root)
```

#### Cleanup macOS
```bash
# File di sistema inutili
rm docs/.DS_Store
rm docs/roadmap/.DS_Store
```

**Totale file da eliminare**: ~22 file

---

## 📁 FASE 2 - RIORGANIZZAZIONE DIRECTORY

### Struttura PRIMA
```
docs/
├── GROUND_RULES.md
├── TODO_NEXT_SESSION.md
├── CLAUDE.md
├── SommelierAI_ProjectContext_v1_4.md
├── ROADMAP_UFFICIALE_v1.5.md
├── SESSION_HANDOFF_2026-03-31.md
├── RELEASE_NOTES_v1_5_0.md
├── RELEASE_NOTES_v1_6_0.md
├── RELEASE_NOTES_v1_6_1.md
├── RANKING_TEST_MATRIX_v2_1.md (da spostare)
├── roadmap/ (vuota dopo cleanup)
├── technical/
│   ├── RANKING_TEST_MATRIX_v2_1.md
│   ├── RAPPORTO_COLLAUDO_GT_v2_0_DEFINITIVO.md
│   ├── DOCUMENTAZIONE_MATCH_RANKING_DEBUGGER_v1.3.md
│   ├── PATCH_GT24_GT26.md
│   ├── gt_results_FINAL_v2.txt
│   └── gt_runner.sh
├── archive/ (per file vecchi)
├── enterprise/ (?)
├── marketing/ (?)
└── product/ (?)
```

### Struttura DOPO (proposta)
```
docs/
├── GROUND_RULES.md              # Regole operative (CORE)
├── TODO_NEXT_SESSION.md         # Task prossima sessione (CORE)
├── CLAUDE.md                    # Context conciso per Claude Code (CORE)
│
├── context/                     # Context di progetto
│   └── ProjectContext_v1.4.md  # Rinominato per chiarezza
│
├── roadmap/                     # Planning strategico
│   ├── ROADMAP_v1.5.md         # Rinominato (spostato da root)
│   └── SESSION_HANDOFF_2026-03-31.md  # Spostato qui
│
├── releases/                    # Release notes (cronologico)
│   ├── v1.5.0.md
│   ├── v1.6.0.md
│   └── v1.6.1.md
│
├── technical/                   # Documentazione tecnica
│   ├── RANKING_TEST_MATRIX_v2_1.md
│   ├── RAPPORTO_COLLAUDO_DEFINITIVO.md  # Rinominato
│   ├── MATCH_DEBUGGER_v1.3.md           # Rinominato
│   ├── PATCH_GT24_GT26.md
│   └── test-results/            # Nuova subdir
│       ├── gt_results_FINAL_v2.txt
│       └── gt_runner.sh
│
└── archive/                     # File vecchi (se servono per reference)
    ├── session-recaps/
    │   ├── SESSION_RECAP_2026-03-28.md
    │   └── SESSION_RECAP_2026-03-31.md
    ├── old-context/
    │   └── SommelierAI_ProjectContext_v1_3.md
    └── old-roadmaps/
        └── ROADMAP_STRATEGICA_v1.4.md
```

---

## 🔧 FASE 3 - SCRIPT DI ESECUZIONE

### Step 1: Backup Completo
```bash
cd ~/sommelier-ai
tar -czf docs_backup_2026-03-31.tar.gz docs/
```

### Step 2: Eliminazione File Obsoleti
```bash
cd ~/sommelier-ai/docs

# Root
rm SESSION_RECAP_2026-03-28.md \
   SESSION_RECAP_2026-03-31.md \
   SommelierAI_ProjectContext_v1.1.docx \
   SommelierAI_ProjectContext_v1_2.docx \
   SommelierAI_ProjectContext_v1_3.md \
   STEP2_IMPLEMENTATION_SUMMARY.md \
   GRAPES_ENRICHMENT_SUMMARY.md \
   AESTHETICS_COMPLETE_SUMMARY.md

# Technical
rm technical/gt_results_20260320_*.txt \
   technical/gt_results_20260325_*.txt \
   technical/gt_results_post_fix.txt \
   technical/RANKING_TEST_MATRIX_v1.0.md \
   technical/RANKING_TEST_MATRIX_v2.0.md \
   technical/RAPPORTO_COLLAUDO_GT_v2_0.md

# Roadmap vecchie
rm roadmap/ROADMAP_STRATEGICA_v*.md \
   roadmap/SommelierAI_Roadmap_v1.2.1_FULL_02-03-2026.pdf

# macOS junk
rm .DS_Store roadmap/.DS_Store
```

### Step 3: Riorganizzazione
```bash
cd ~/sommelier-ai/docs

# Crea nuove directory
mkdir -p context releases technical/test-results archive/{session-recaps,old-context,old-roadmaps}

# Sposta file attuali
mv SommelierAI_ProjectContext_v1_4.md context/ProjectContext_v1.4.md
mv RELEASE_NOTES_v1_5_0.md releases/v1.5.0.md
mv RELEASE_NOTES_v1_6_0.md releases/v1.6.0.md
mv RELEASE_NOTES_v1_6_1.md releases/v1.6.1.md
mv ROADMAP_UFFICIALE_v1.5.md roadmap/ROADMAP_v1.5.md
mv SESSION_HANDOFF_2026-03-31.md roadmap/SESSION_HANDOFF_2026-03-31.md

# Rinomina file tecnici (per chiarezza)
mv technical/RAPPORTO_COLLAUDO_GT_v2_0_DEFINITIVO.md technical/RAPPORTO_COLLAUDO_DEFINITIVO.md
mv technical/DOCUMENTAZIONE_MATCH_RANKING_DEBUGGER_v1.3.md technical/MATCH_DEBUGGER_v1.3.md

# Sposta test results
mv technical/gt_results_FINAL_v2.txt technical/test-results/
mv technical/gt_runner.sh technical/test-results/

# Elimina directory roadmap vecchia se vuota
rmdir roadmap 2>/dev/null || echo "roadmap/ non vuota, skip"
```

### Step 4: Update Reference Paths (se necessario)
Aggiorna riferimenti in:
- `GROUND_RULES.md` (path dei file da leggere)
- `TODO_NEXT_SESSION.md` (se ci sono path)
- `CLAUDE.md` (se ci sono path)

---

## ✅ FASE 4 - VERIFICA POST-CLEANUP

### Checklist Verifica
```bash
cd ~/sommelier-ai/docs

# 1. Verifica file core presenti
ls -1 GROUND_RULES.md TODO_NEXT_SESSION.md CLAUDE.md

# 2. Verifica directory
ls -d context/ releases/ roadmap/ technical/ archive/

# 3. Conta file per directory
echo "Context:"; ls -1 context/ | wc -l
echo "Releases:"; ls -1 releases/ | wc -l
echo "Roadmap:"; ls -1 roadmap/ | wc -l
echo "Technical:"; ls -1 technical/ | wc -l
echo "Archive:"; find archive/ -type f | wc -l

# 4. Dimensione totale
du -sh .
```

### Risultato Atteso
```
✅ Core files: 3
✅ context/: 1 file
✅ releases/: 3 file
✅ roadmap/: 2 file
✅ technical/: 5 file + test-results/
✅ archive/: vuoto (opzionale)
✅ Totale: ~14 file attivi + 3 core
```

---

## 🔄 FASE 5 - COMMIT

### Git Status
```bash
git status
# Dovrebbe mostrare:
# - deleted: molti file obsoleti
# - renamed: file riorganizzati
# - modified: GROUND_RULES.md (se path aggiornati)
```

### Commit
```bash
git add -A
git commit -m "docs: cleanup + riorganizzazione directory

- Eliminati 22+ file obsoleti (vecchi recap, context, summary)
- Riorganizzate directory: context/, releases/, roadmap/, technical/
- Rinominati file per chiarezza (ProjectContext_v1.4, ROADMAP_v1.5)
- Consolidati GT results in technical/test-results/
- Struttura finale: 17 file attivi vs 41 precedenti (-58%)

Breaking: Aggiornare path in script/tool se referenziano vecchi path"

git push
```

---

## 📊 PRIMA/DOPO

### Prima
- **File totali**: 41 (root + subdirs)
- **Dimensione**: ~138 KB
- **Directory**: 6 (mescolate, non chiare)
- **Duplicati**: 8+ versioni vecchie

### Dopo
- **File attivi**: 17
- **Dimensione**: ~90 KB (-35%)
- **Directory**: 5 (purpose-driven)
- **Duplicati**: 0

### Benefici
- ✅ Più facile trovare documentazione
- ✅ Release notes organizzate cronologicamente
- ✅ Context e roadmap separati (purpose-driven)
- ✅ Technical docs isolata (facile per dev)
- ✅ Core files in root (GROUND_RULES, TODO, CLAUDE)

---

## ⚠️ NOTE

1. **Backup creato**: `docs_backup_2026-03-31.tar.gz` in project root
2. **Archive opzionale**: Puoi eliminare `archive/` se non serve history
3. **Path breaking**: Aggiorna script che usano path hardcoded
4. **GROUND_RULES**: Aggiorna path nella sezione "APERTURA SESSIONE"

---

**Piano preparato da**: Claude Sonnet 4.6  
**Data**: 31 Marzo 2026  
**Status**: ⏳ PRONTO PER ESECUZIONE
