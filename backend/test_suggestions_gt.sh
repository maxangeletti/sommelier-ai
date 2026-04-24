#!/bin/bash
# Test GT - Validazione Suggerimenti Sommy v1.8.4
# Verifica che tutti i 14 suggerimenti restituiscano risultati

set -e

BASE_URL="http://localhost:8000"
FAILED=0

echo "🧪 TEST GT: Suggerimenti Sommy (14 suggerimenti)"
echo "================================================"
echo ""

# Helper function
test_suggestion() {
    local num=$1
    local query=$2
    local min_count=$3
    
    echo "Test $num: \"$query\""
    
    response=$(curl -s -X POST "$BASE_URL/search" \
        -H 'Content-Type: application/json' \
        -d "{\"query\":\"$query\",\"limit\":10}")
    
    count=$(echo "$response" | jq -r '.meta.count')
    
    if [ "$count" -ge "$min_count" ]; then
        echo "  ✅ PASS: $count risultati (min: $min_count)"
    else
        echo "  ❌ FAIL: $count risultati (attesi: >=$min_count)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# Original 6
test_suggestion 1 "Un sangiovese di buona qualità" 1
test_suggestion 2 "Un vino intenso e strutturato sopra i 20€" 1
test_suggestion 3 "Uno spumante brut per aperitivo" 1
test_suggestion 4 "Uno Champagne sopra i 30€" 2
test_suggestion 5 "Un bianco con sentori agrumati sotto i 15€" 1
test_suggestion 6 "Un vino con buon rapporto qualità prezzo" 1

# New 8
test_suggestion 7 "Un Barolo o Brunello per una cena importante" 1
test_suggestion 8 "Un rosso corposo per carne alla griglia" 1
test_suggestion 9 "Un bianco minerale per pesce crudo" 1
test_suggestion 10 "Un rosato fresco per l'estate" 1
test_suggestion 11 "Un Prosecco per brindare" 1
test_suggestion 12 "Un passito per dessert" 1
test_suggestion 13 "Un vino biologico sotto i 20€" 1
test_suggestion 14 "Un Nebbiolo giovane e tannico" 1

# Summary
echo "================================================"
if [ $FAILED -eq 0 ]; then
    echo "✅ TUTTI I 14 TEST PASSATI!"
    exit 0
else
    echo "❌ $FAILED test falliti"
    exit 1
fi
