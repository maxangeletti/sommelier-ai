#!/bin/bash
# =============================================================================
# SommelierAI — GT-01→GT-26 Full Test Runner
# Versione: v2.0 — Dataset 100 vini
# Data: 2026-03-20
# Endpoint: POST https://sommelier-ai.onrender.com/search
# =============================================================================

BASE="https://sommelier-ai.onrender.com/search"
OUT="gt_results_$(date +%Y%m%d_%H%M%S).txt"

echo "=============================================" | tee "$OUT"
echo " SommelierAI — GT Full Run ($(date))" | tee -a "$OUT"
echo "=============================================" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# --- Preflight: verify dataset size ---
echo ">>> PREFLIGHT: Checking dataset size..." | tee -a "$OUT"
curl -s -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -d '{"query":"vino","limit":100,"debug":true}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
n = len(d.get('results', []))
meta = d.get('meta', {})
print(f'Dataset results for \"vino\": {n} wines')
print(f'Meta: {json.dumps(meta, indent=2)}')
" 2>&1 | tee -a "$OUT"
echo "" | tee -a "$OUT"

# --- Preflight: verify frappato exists ---
echo ">>> PREFLIGHT: Searching 'frappato'..." | tee -a "$OUT"
curl -s -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -d '{"query":"frappato","limit":5,"debug":true}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
for w in d.get('results', [])[:3]:
    ms = w.get('match_score') or w.get('__match_score') or -1
    print(f'  rank={w.get(\"rank\",\"?\")} id={w[\"id\"]} name={w[\"name\"]} price={w.get(\"price\",\"?\")} score={w.get(\"score\",\"?\")} match={ms}')
" 2>&1 | tee -a "$OUT"
echo "" | tee -a "$OUT"

# =============================================================================
# GT QUERIES
# =============================================================================

declare -a GT_IDS=(
  "GT-01" "GT-02" "GT-03" "GT-04" "GT-05" "GT-06" "GT-07"
  "GT-08" "GT-09" "GT-10" "GT-11" "GT-12" "GT-13"
  "GT-14" "GT-15" "GT-16" "GT-17"
  "GT-18" "GT-19" "GT-20" "GT-21" "GT-22" "GT-23" "GT-24" "GT-25" "GT-26"
)

declare -a GT_QUERIES=(
  "barolo serralunga"
  "formaggi erborinati"
  "franciacorta brut"
  "vino per cena importante"
  "rosso strutturato"
  "bianco fresco"
  "rosso elegante"
  "rosso elegante per cena importante"
  "bianco fresco per cena importante di pesce"
  "vino importante"
  "vino importante di pesce"
  "bottiglia importante"
  "vino che faccia figura"
  "vino tannico e strutturato"
  "rosso leggero poco tannico"
  "rosso elegante non troppo tannico"
  "vino potente per cena importante"
  "poco tannico"
  "vino tannico"
  "vino sotto 20 euro"
  "nebbiolo"
  "rosato per aperitivo"
  "prosecco"
  "etna"
  "vino dolce"
  "voglio stupire"
)

declare -a GT_EXPECTED=(
  "Barolo DOCG Serralunga in top 1-2"
  "Vini con food_pairing formaggi in top 3 (Passito, Riesling dolce)"
  "Franciacorta Brut in top 3"
  "Vini con occasion=important_dinner in top 3 (Barolo, Pauillac, Gevrey)"
  "Rossi strutturati in top 3 (Barolo, Pauillac, Sagrantino)"
  "Bianchi freschi in top 3 (Riesling, Vermentino, Chablis)"
  "Gevrey-Chambertin in top 2. Pinot Nero, Barbaresco accettabili"
  "Gevrey-Chambertin in top 2 con match > 0.85"
  "Bianchi con food pesce e occasion importante in top 3"
  "Vini di prestigio in top 3 (Barolo, Pauillac, Gevrey)"
  "Bianchi/bollicine di prestigio con food pesce in top 3"
  "Come GT-10 — vini di prestigio in top 3"
  "Come GT-10 — vini di prestigio in top 3"
  "Sagrantino, Barolo, Taurasi, Brunello in top 5"
  "Gevrey, Pinot Nero, Barbera in top 5. Tannici penalizzati nel match"
  "Gevrey in top 2. Sagrantino penalizzato"
  "Barolo, Pauillac, Amarone, Brunello in top 5"
  "Vini con tannini bassi in top 3"
  "Sagrantino, Barolo, Taurasi in top 3"
  "Vini con price <= 20 in top 3"
  "Barolo, Barbaresco, Nebbiolo Langhe in top 3"
  "Cerasuolo, Chiaretto, Provence Rose in top 3"
  "Prosecco DOC in top 1"
  "Etna Rosso, Etna Bianco, Etna Rosato in top 3"
  "Passito di Pantelleria in top 1-2"
  "Vini di prestigio in top 3 (Barolo, Pauillac, Gevrey)"
)

# =============================================================================
# RUN ALL GTs
# =============================================================================

for i in "${!GT_IDS[@]}"; do
  gt="${GT_IDS[$i]}"
  query="${GT_QUERIES[$i]}"
  expected="${GT_EXPECTED[$i]}"
  
  echo "--- $gt: \"$query\" ---" | tee -a "$OUT"
  echo "  Expected: $expected" | tee -a "$OUT"
  
  curl -s -X POST "$BASE" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$query\",\"limit\":10,\"debug\":true}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
results = d.get('results', [])
if not results:
    print('  ⚠️  NO RESULTS')
for w in results[:5]:
    ms = w.get('match_score') or w.get('__match_score') or -1
    qs = w.get('__quality_score', '?')
    print(f'  #{w.get(\"rank\",\"?\")} id={w[\"id\"]} {w[\"name\"]} | price={w.get(\"price\",\"?\")} score={w.get(\"score\",\"?\")} match={ms} quality={qs}')
" 2>&1 | tee -a "$OUT"
  
  echo "  Verdict: ___" | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done

echo "=============================================" | tee -a "$OUT"
echo " Results saved to: $OUT" | tee -a "$OUT"
echo "=============================================" | tee -a "$OUT"
