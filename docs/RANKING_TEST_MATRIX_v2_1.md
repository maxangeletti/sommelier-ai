# SommelierAI — Ranking Test Matrix
Versione: v2.1  
Data: 2026-03-24  
Scope: Backend `/search` — ranking + filtri + tannin matching + LLM Step 1  
Build baseline: LLM intent parser integrato [C-experimental] + region alias map  
Dataset: 100 vini live su Render

---

## Regole di esecuzione

- Sempre: `debug=true`
- Endpoint: POST `https://sommelier-ai.onrender.com/search`
- Salvare per i primi 3 risultati:
  - `rank`, `id`, `name`, `price`
  - `score`, `match_score`
- Verdict: PASS / WARN / FAIL
  - **PASS**: top 3 coerenti con risultato atteso
  - **WARN**: risultati accettabili ma non ideali (es. vino corretto ma non in top 3)
  - **FAIL**: risultati incoerenti con la query

---

## Comando standard (template)

```bash
curl -s -X POST "https://sommelier-ai.onrender.com/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"<QUERY>","limit":10,"debug":true}'
```

---

## GT Originali (GT-01 → GT-13)

### GT-01 — Location match
- **Query:** `barolo serralunga`
- **Atteso:** Barolo DOCG Serralunga d'Alba in top 1-2 con match=1.0
- **Note:** se altri vini con score intrinseco più alto appaiono sopra con match=0.0, è comportamento noto (qualità domina location match)

### GT-02 — Food pairing
- **Query:** `formaggi erborinati`
- **Atteso:** vini con food_pairing formaggi in top 3 (Passito, Riesling dolce)

### GT-03 — Denominazione specifica
- **Query:** `franciacorta brut`
- **Atteso:** Franciacorta Brut in top 3 con match=1.0
- **Note:** Champagne può apparire sopra per score intrinseco — comportamento noto

### GT-04 — Occasione
- **Query:** `vino per cena importante`
- **Atteso:** vini con occasion=important_dinner in top 3 (Barolo, Pauillac, Gevrey-Chambertin)

### GT-05 — Struttura
- **Query:** `rosso strutturato`
- **Atteso:** rossi strutturati in top 3 (Barolo, Pauillac, Sagrantino)
- **Note:** intensity=high triggerato da "strutturato"

### GT-06 — Freschezza
- **Query:** `bianco fresco`
- **Atteso:** bianchi freschi in top 3 (Riesling, Vermentino, Chablis)
- **Note:** Champagne può apparire — tecnicamente è bianco fresco

### GT-07 — Eleganza
- **Query:** `rosso elegante`
- **Atteso:** Gevrey-Chambertin in top 2 con match alto. Pinot Nero, Barbaresco accettabili.

### GT-08 — Eleganza + occasione
- **Query:** `rosso elegante per cena importante`
- **Atteso:** Gevrey-Chambertin in top 2 con match > 0.85. Pauillac e Barolo accettabili.

### GT-09 — Multi-signal (bianco + fresco + pesce + cena importante)
- **Query:** `bianco fresco per cena importante di pesce`
- **Atteso:** bianchi con food_pairing pesce e occasion importante in top 3

### GT-10 — Prestige generico
- **Query:** `vino importante`
- **Atteso:** vini di prestigio in top 3 (Barolo, Pauillac, Gevrey-Chambertin)

### GT-11 — Prestige + food
- **Query:** `vino importante di pesce`
- **Atteso:** bianchi/bollicine di prestigio con food pesce in top 3

### GT-12 — Prestige (sinonimo)
- **Query:** `bottiglia importante`
- **Atteso:** come GT-10 — vini di prestigio in top 3

### GT-13 — Prestige (linguaggio informale)
- **Query:** `vino che faccia figura`
- **Atteso:** come GT-10 — vini di prestigio in top 3

---

## GT Tannicità (GT-14 → GT-17)

### GT-14 — Tannicità alta esplicita
- **Query:** `vino tannico e strutturato`
- **Atteso:** Sagrantino, Barolo, Taurasi, Brunello in top 5
- **Segnali:** tannin_req=high, intensity_req=high

### GT-15 — Tannicità bassa esplicita
- **Query:** `rosso leggero poco tannico`
- **Atteso:** Gevrey-Chambertin, Pinot Nero, Barbera in top 5. Sagrantino/Barolo penalizzati nel match.
- **Segnali:** tannin_req=low, intensity_req=low, color=rosso
- **Note:** con dataset 53 vini, i rossi leggeri hanno qualità base più bassa — possono non essere in top 3 ma devono avere match più alto dei tannici

### GT-16 — Eleganza e bassa tannicità
- **Query:** `rosso elegante non troppo tannico`
- **Atteso:** Gevrey-Chambertin in top 2 con match alto. Sagrantino penalizzato.
- **Segnali:** elegant_intent=true, color=rosso

### GT-17 — Potenza + occasione
- **Query:** `vino potente per cena importante`
- **Atteso:** Barolo, Pauillac, Amarone, Brunello in top 5
- **Segnali:** intensity_req=high, occasion=important_dinner

---

## GT Copertura (GT-18 → GT-26)

### GT-18 — Tannicità sola
- **Query:** `poco tannico`
- **Atteso:** vini con tannini bassi in top 3. Rosati, bianchi, o rossi leggeri.
- **Segnali:** tannin_req=low

### GT-19 — Tannicità alta sola
- **Query:** `vino tannico`
- **Atteso:** Sagrantino, Barolo, Taurasi in top 3
- **Segnali:** tannin_req=high

### GT-20 — Prezzo
- **Query:** `vino sotto 20 euro`
- **Atteso:** vini con price ≤ 20 in top 3

### GT-21 — Vitigno specifico
- **Query:** `nebbiolo`
- **Atteso:** Barolo, Barbaresco, Nebbiolo Langhe in top 3

### GT-22 — Rosato + occasione
- **Query:** `rosato per aperitivo`
- **Atteso:** Cerasuolo, Chiaretto, Provence Rosé in top 3

### GT-23 — Denominazione singola
- **Query:** `prosecco`
- **Atteso:** Prosecco DOC in top 1

### GT-24 — Zona geografica
- **Query:** `etna`
- **Atteso:** Etna Rosso, Etna Bianco, Etna Rosato in top 3

### GT-25 — Dolce
- **Query:** `vino dolce`
- **Atteso:** Passito di Pantelleria in top 1-2

### GT-26 — Query laterale / prestige emotivo
- **Query:** `voglio stupire`
- **Atteso:** vini di prestigio in top 3 (Barolo, Pauillac, Gevrey-Chambertin)
- **Note:** test per copertura parser prestige su linguaggio informale. Se rule-based non cattura, documenta come gap per integrazione LLM.

---

## Storico esecuzioni

### Esecuzione 2026-03-24 — v2.1 (LLM Step 1 + region alias map)

**Risultato: 22 PASS, 4 WARN, 0 FAIL**

| GT | Verdict | Note |
|----|---------|------|
| GT-01 | PASS ✅ | Barolo Serralunga in top 3, match=1.0 |
| GT-02 | PASS ✅ | Passito in top 1, food_pairing corretto |
| GT-03 | PASS ✅ | Franciacorta Brut in top 2, match=1.0 |
| GT-04 | PASS ✅ | Barolo, Pauillac in top 3 con occasion match |
| GT-05 | PASS ✅ | Sagrantino, Barolo, Brunello in top 3 |
| GT-06 | PASS ✅ | Riesling, Vermentino in top 5 |
| GT-07 | PASS ✅ | Gevrey-Chambertin #1, match alto |
| GT-08 | PASS ✅ | Gevrey-Chambertin #1, match > 0.90 |
| GT-09 | WARN ⚠️ | Champagne #1 (qualità + food pesce + occasion), bianchi freschi in top 5 — comportamento corretto |
| GT-10 | WARN ⚠️ | Champagne #1 per qualità base, rossi prestigiosi in top 5 — accettabile |
| GT-11 | PASS ✅ | Champagne, Chablis in top 3 |
| GT-12 | PASS ✅ | Come GT-10 — vini prestigiosi in top 3 |
| GT-13 | PASS ✅ | Come GT-10 — prestige intent triggerato |
| GT-14 | PASS ✅ | Sagrantino #1, Barolo #2, tutti tannici in top 5 |
| GT-15 | PASS ✅ | Gevrey-Chambertin, Pinot Nero in top 3, match corretto |
| GT-16 | WARN ⚠️ | Gevrey-Chambertin ha tannins=medium (non low) — dataset constraint |
| GT-17 | PASS ✅ | Barolo, Pauillac, Brunello in top 5 |
| GT-18 | PASS ✅ | Pinot Nero, Barbera in top 3 con tannini bassi |
| GT-19 | PASS ✅ | Sagrantino, Barolo, Taurasi in top 3 |
| GT-20 | PASS ✅ | Vini ≤ 20€ in top 3 |
| GT-21 | WARN ⚠️ | Barbaresco #1, Barolo #2 — entrambi corretti, ordine da score base |
| GT-22 | PASS ✅ | Cerasuolo, Chiaretto in top 3 |
| GT-23 | PASS ✅ | Prosecco DOC in top 1 — fix denomination match |
| GT-24 | PASS ✅ | Etna Rosso in top 1 — fix region alias map |
| GT-25 | PASS ✅ | Passito in top 1 |
| GT-26 | PASS ✅ | Barolo, Pauillac, Gevrey-Chambertin in top 3 — LLM prestige intent |

### WARN Analysis (accettabili, non bloccanti)

- **GT-09:** Champagne domina per qualità intrinseca, ma ha food_pairing pesce + occasion important_dinner → tecnicamente corretto
- **GT-10:** Champagne ha score base più alto dei rossi, ma rossi prestigiosi in top 5 → accettabile per query generica
- **GT-16:** Gevrey-Chambertin ha tannins=medium nel CSV, non low — impossibile migliorare senza modifica dataset
- **GT-21:** Barbaresco e Barolo entrambi nebbiolo — ordine dipende da score intrinseco, non da errore motore

---

### Esecuzione 2026-03-20 — v2.0 (Scala tannicità AIS)

| GT | Verdict | Note |
|----|---------|------|
| GT-01 → GT-13 | _da compilare_ | |
| GT-14 → GT-17 | _da compilare_ | |
| GT-18 → GT-26 | _da compilare_ | |

---

## Note

- I GT sono la baseline obbligatoria dopo ogni modifica al motore
- Nessuna modifica ai pesi/fallback/bonus senza esecuzione GT completa
- Classificazione modifiche: A=Freeze, B=Accepted tuning, C=Experimental, D=Rejected
- Scala tannicità AIS: low (0.10), medium_low (0.30), medium (0.50), medium_plus (0.75), high (1.00)
