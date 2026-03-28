# GROUND RULES - SommelierAI Development

**REGOLE FONDAMENTALI DA RISPETTARE SEMPRE**

## 1. MOTORE = NO TOUCH
Il backend search engine (`backend/main.py` - logica ranking, scoring, filtering) si modifica **SOLO** se:
- Indispensabile per la feature richiesta
- Approval esplicita dell'utente
- Test completi preparati

**Mai modificare il motore "tanto per provare" o per ottimizzazioni non richieste.**

## 2. BACKUP OBBLIGATORIO
Prima di **OGNI** modifica a file critici:
- Creare backup verificabile (con md5sum)
- Annotare punto di ripristino (commit hash se in git)
- Verificare che il backup sia accessibile

**Mai iniziare modifiche senza backup confermato.**

## 3. LOCAL = REMOTE SEMPRE
Prima di ogni sessione di lavoro:
- `git status` deve essere clean
- `git pull` per allineamento
- Verificare branch corretta (`main`)

Dopo modifiche:
- Commit solo dopo test positivo
- Push solo dopo approval utente

## 4. TEST BEFORE PUSH
**WORKFLOW OBBLIGATORIO:**
1. Modifica locale
2. Test locale (esito positivo confermato)
3. Commit locale
4. Push a remote
5. Attesa deploy (2-3 min)
6. Test live
7. Conferma utente

**Mai pushare codice non testato.**

## 5. NO CAZZATE
- Se qualcosa fallisce → dirlo IMMEDIATAMENTE
- Mai dire "funziona perfettamente" senza verifica reale
- Se uno screenshot mostra errori → AMMETTERLO, non inventare scuse
- Suggerire sempre la soluzione migliore, non quella più veloce
- Se non sai → dire "non lo so", non inventare

## 6. PRAGMATICO, NON REATTIVO
- Pensare prima di agire
- Non provare soluzioni a caso sperando che funzionino
- Leggere i file REALI prima di modificarli
- Verificare che le modifiche abbiano senso nel contesto
- Non fare cambi "tanto per fare"

---

**VIOLAZIONE DI QUESTE REGOLE = SESSIONE FALLITA**

Queste regole esistono perché sono state violate ripetutamente in passato, causando:
- Regressioni nel motore di ricerca
- File iOS corrotti
- Deploy di codice rotto
- Perdita di tempo dell'utente

**ZERO TOLLERANZA.**
