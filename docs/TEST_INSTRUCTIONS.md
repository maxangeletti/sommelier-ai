# 🧪 Istruzioni Test LLM Step 2

## Test Rapido (30 secondi)

```bash
./test_quick.sh
```

Questo script:
- Attende 30 secondi per cold start
- Testa 1 query: "vino elegante per cena importante"
- Mostra se le reason sono personalizzate (🎯) o generiche (📝)
- **Tempo totale:** ~35 secondi

---

## Test Completo (2 minuti)

```bash
./test_llm_step2.sh
```

Questo script:
- Attende 60 secondi per cold start
- Testa 5 query chiave:
  1. GT-08: "vino elegante per cena importante"
  2. GT-05: "rosso strutturato"
  3. GT-06: "bianco fresco per cena importante di pesce"
  4. GT-26: "voglio stupire"
  5. Baseline: "barolo"
- Mostra reason, match score, e ranking
- **Tempo totale:** ~2-3 minuti

---

## Test Manuale (se preferisci)

### 1. Attendi che il backend sia pronto
```bash
# Ping finché non risponde
while ! curl -s https://sommelier-ai.onrender.com/health > /dev/null; do
  echo "Attendo backend..."
  sleep 5
done
echo "Backend pronto!"
```

### 2. Testa una singola query
```bash
curl -s -X POST "https://sommelier-ai.onrender.com/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"vino elegante per cena importante","limit":3}' \
  | python3 -m json.tool | grep -A 1 "reason"
```

### 3. Output atteso
Dovresti vedere reason come:
- **Personalizzata:** "Un Borgogna raffinato perfetto per occasioni formali..."
- **Oppure fallback CSV:** "Pinot Nero elegante con tannini morbidi"

---

## Cosa verificare

✅ **Le reason sono contestuali alla query?**
- Query "elegante per cena" → reason parla di eleganza/occasione
- Query "strutturato" → reason parla di struttura/tannini
- Query "voglio stupire" → reason parla di prestigio/impatto

✅ **Le reason sono naturali e leggibili?**
- Max 15 parole
- Tono professionale ma accessibile
- No gergo tecnico eccessivo

✅ **Il fallback funziona?**
- Se LLM non disponibile → usa description CSV
- Nessun crash
- Response time accettabile

---

## Troubleshooting

### Backend non risponde
```bash
# Verifica stato su Render Dashboard
# Oppure attendi 2-3 minuti per il build
```

### Reason identiche al CSV
```bash
# Verifica che ANTHROPIC_API_KEY sia configurata su Render
# Vai su Render Dashboard → sommelier-ai → Environment
# Controlla che ANTHROPIC_API_KEY sia presente
```

### Reason troppo lunghe o generiche
```bash
# Segnala il problema — possiamo tuning il prompt
# Salva esempi di query e reason problematiche
```

---

## Prossimi Passi dopo Test

### ✅ Se test OK
1. Aggiornare `SommelierAI_ProjectContext_v1_4.md`
2. Spostare LLM Step 2 da `[C-experimental]` a `[B-accepted]`
3. Procedere con task successivo (Estetica App o KNOWN_GRAPES)

### ⚠️ Se test problematici
1. Raccogliere esempi di reason problematiche
2. Tuning del prompt `EXPLAIN_SYSTEM_PROMPT`
3. Test iterativo finché qualità OK

### ❌ Se fallback sempre attivo
1. Verificare ANTHROPIC_API_KEY su Render
2. Verificare `SOMMELIERAI_LLM_ENABLED=1`
3. Check logs backend per errori LLM

---

## Note

- **Costo:** ~$0.002 per search (10 vini)
- **Timeout LLM:** 4 secondi
- **Fallback:** Sempre attivo, nessun crash garantito
- **Classificazione attuale:** [C-experimental]

---

*Generato il 2 Aprile 2026*
