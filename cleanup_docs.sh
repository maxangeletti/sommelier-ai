#!/bin/bash
# Script Cleanup Documentazione SommelierAI
# Data: 31 Marzo 2026

set -e  # Exit on error

echo "🧹 CLEANUP DOCUMENTAZIONE - SommelierAI"
echo "========================================"
echo ""

# STEP 1: Backup
echo "📦 STEP 1: Creazione backup..."
cd ~/sommelier-ai
tar -czf docs_backup_2026-03-31.tar.gz docs/
echo "✅ Backup creato: docs_backup_2026-03-31.tar.gz"
echo ""

# STEP 2: Eliminazione file obsoleti
echo "🗑️  STEP 2: Eliminazione file obsoleti..."
cd ~/sommelier-ai/docs

# Root docs/
rm -f SESSION_RECAP_2026-03-28.md \
      SESSION_RECAP_2026-03-31.md \
      SommelierAI_ProjectContext_v1.1.docx \
      SommelierAI_ProjectContext_v1_2.docx \
      SommelierAI_ProjectContext_v1_3.md \
      STEP2_IMPLEMENTATION_SUMMARY.md \
      GRAPES_ENRICHMENT_SUMMARY.md \
      AESTHETICS_COMPLETE_SUMMARY.md \
      .DS_Store

echo "✅ Eliminati 9 file da root"

# Technical (se esiste)
if [ -d "technical" ]; then
    cd technical
    rm -f gt_results_20260320_*.txt \
          gt_results_20260325_*.txt \
          gt_results_post_fix.txt \
          RANKING_TEST_MATRIX_v1.0.md \
          RANKING_TEST_MATRIX_v2.0.md \
          RAPPORTO_COLLAUDO_GT_v2_0.md
    echo "✅ Eliminati ~14 file da technical/"
    cd ..
fi

# Roadmap vecchie (se esiste)
if [ -d "roadmap" ]; then
    cd roadmap
    rm -f ROADMAP_STRATEGICA_v*.md \
          SommelierAI_Roadmap_v1.2.1_FULL_02-03-2026.pdf \
          .DS_Store
    echo "✅ Eliminati 5+ file da roadmap/"
    cd ..
fi

echo ""

# STEP 3: Riorganizzazione
echo "📁 STEP 3: Riorganizzazione directory..."

# Crea nuove directory
mkdir -p context releases roadmap_new technical/test-results archive/{session-recaps,old-context,old-roadmaps}
echo "✅ Directory create"

# Sposta file (se esistono)
[ -f "SommelierAI_ProjectContext_v1_4.md" ] && mv SommelierAI_ProjectContext_v1_4.md context/ProjectContext_v1.4.md
[ -f "RELEASE_NOTES_v1_5_0.md" ] && mv RELEASE_NOTES_v1_5_0.md releases/v1.5.0.md
[ -f "RELEASE_NOTES_v1_6_0.md" ] && mv RELEASE_NOTES_v1_6_0.md releases/v1.6.0.md
[ -f "RELEASE_NOTES_v1_6_1.md" ] && mv RELEASE_NOTES_v1_6_1.md releases/v1.6.1.md
[ -f "ROADMAP_UFFICIALE_v1.5.md" ] && mv ROADMAP_UFFICIALE_v1.5.md roadmap_new/ROADMAP_v1.5.md
[ -f "SESSION_HANDOFF_2026-03-31.md" ] && mv SESSION_HANDOFF_2026-03-31.md roadmap_new/SESSION_HANDOFF_2026-03-31.md
[ -f "RANKING_TEST_MATRIX_v2_1.md" ] && mv RANKING_TEST_MATRIX_v2_1.md technical/

echo "✅ File riorganizzati"

# Rinomina file tecnici (se esistono)
if [ -d "technical" ]; then
    cd technical
    [ -f "RAPPORTO_COLLAUDO_GT_v2_0_DEFINITIVO.md" ] && mv RAPPORTO_COLLAUDO_GT_v2_0_DEFINITIVO.md RAPPORTO_COLLAUDO_DEFINITIVO.md
    [ -f "DOCUMENTAZIONE_MATCH_RANKING_DEBUGGER_v1.3.md" ] && mv DOCUMENTAZIONE_MATCH_RANKING_DEBUGGER_v1.3.md MATCH_DEBUGGER_v1.3.md
    [ -f "gt_results_FINAL_v2.txt" ] && mv gt_results_FINAL_v2.txt test-results/
    [ -f "gt_runner.sh" ] && mv gt_runner.sh test-results/
    cd ..
    echo "✅ File tecnici rinominati"
fi

# Rimuovi directory roadmap vecchia e rinomina nuova
if [ -d "roadmap" ]; then
    rmdir roadmap 2>/dev/null && echo "✅ Directory roadmap/ vecchia rimossa" || echo "⚠️  roadmap/ non vuota, mantenuta"
fi
[ -d "roadmap_new" ] && mv roadmap_new roadmap && echo "✅ roadmap_new/ → roadmap/"

echo ""

# STEP 4: Verifica
echo "✅ STEP 4: Verifica finale..."
echo ""
echo "📊 Struttura finale:"
echo "-------------------"
ls -R | grep -E '^./$|^./[^/]+:$' | sed 's/:$//' | sed 's|^./||'
echo ""

echo "📈 File count per directory:"
echo "Core files (root): $(ls -1 *.md 2>/dev/null | wc -l)"
echo "context/: $(ls -1 context/ 2>/dev/null | wc -l)"
echo "releases/: $(ls -1 releases/ 2>/dev/null | wc -l)"
echo "roadmap/: $(ls -1 roadmap/ 2>/dev/null | wc -l)"
echo "technical/: $(find technical/ -type f 2>/dev/null | wc -l)"
echo ""

echo "💾 Dimensione totale:"
du -sh .
echo ""

echo "✅ CLEANUP COMPLETATO!"
echo ""
echo "📝 Prossimi passi:"
echo "1. Verifica con: cd ~/sommelier-ai/docs && ls -la"
echo "2. Test: leggi GROUND_RULES.md, verifica path"
echo "3. Commit: git add -A && git commit -m 'docs: cleanup + riorganizzazione'"
echo "4. Se tutto OK: rm ~/sommelier-ai/docs_backup_2026-03-31.tar.gz"
