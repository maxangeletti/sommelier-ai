# GROUND RULES - SommelierAI Development

**REGOLE FONDAMENTALI DA RISPETTARE SEMPRE**

---

## 🚀 APERTURA SESSIONE OBBLIGATORIA

**ALL'INIZIO DI OGNI SESSIONE, LEGGERE NELL'ORDINE**:

1. **`docs/GROUND_RULES.md`** (questo file) — Regole operative
2. **`docs/PROJECT_PLAN.md`** — Piano progetto (obiettivi, tempi, rischi)
3. **`docs/roadmap/SESSION_HANDOFF_[LAST].md`** — Contesto ultima sessione
4. **`docs/roadmap/ROADMAP_v[LAST].md`** — Stato progetto
5. **`docs/context/ProjectContext_v[LAST].md`** — Context completo
6. **`docs/TODO_NEXT_SESSION.md`** — Task prioritari

**Solo DOPO aver letto tutti questi file, iniziare a lavorare.**

---

## 📋 CHIUSURA SESSIONE OBBLIGATORIA

**PRIMA DI CHIUDERE OGNI SESSIONE, CREARE/AGGIORNARE**:

### A) Release Notes
**Path**: `docs/releases/vX.X.X.md`

**Contenuto minimo**:
- Modifiche applicate (feature/fix/refactor)
- File modificati con descrizione
- Commit hash + message
- Test eseguiti (OK/pending)
- Breaking changes (se presenti)
- Prossimi passi

### B) ROADMAP Aggiornata
**Path**: `docs/roadmap/ROADMAP_vX.X.md`

**Aggiornare**:
- Task completati oggi (spostare da "In corso" a "Completato")
- Nuovi task identificati (aggiungere a backlog)
- Milestone progress (X/Y task)
- Metriche attuali (se cambiate)

### C) SESSION_HANDOFF
**Path**: `docs/roadmap/SESSION_HANDOFF_YYYY-MM-DD.md`

**Contenuto minimo**:
- Contesto veloce (30 secondi - cosa fatto oggi)
- Stato progetto (backend, iOS, milestone)
- Blockers/warning
- File da caricare next session
- Task prioritari (P0, P1, P2)
- Note per next developer

### D) TODO_NEXT_SESSION.md
**Path**: `docs/TODO_NEXT_SESSION.md`

**Aggiornare**:
- Task completati oggi (rimuovere o marcare ✅)
- Nuovi task identificati (aggiungere)
- Priorità riordinate

### E) PROJECT_PLAN.md
**Path**: `docs/PROJECT_PLAN.md`

**Aggiornare**:
- Checklist P0/P1/P2 (marcare ✅ task completati)
- Prossime 48 ore (aggiornare azioni)
- Metriche attuali (backend/iOS)
- Milestone progress (X/Y task)
- Storico aggiornamenti (aggiungere entry v.X.X)
- Risk assessment (se cambiato)

---

## WORKFLOW APERTURA SESSIONE

```bash
# 1. Pull latest
cd ~/sommelier-ai
git pull

# 2. Verifica clean
git status  # deve essere clean

# 3. Claude legge 6 file obbligatori (vedi sopra)
# 4. Claude conferma: "Letti GROUND_RULES, PROJECT_PLAN, HANDOFF, ROADMAP, CONTEXT, TODO. Pronto."
# 5. Utente specifica task o Claude propone da TODO_NEXT_SESSION
# 6. Go!
```

---

## WORKFLOW CHIUSURA SESSIONE

```bash
# 1. Verifica tutto committato e pushato
git status  # deve essere clean
git log --oneline -5  # verifica commit presente

# 2. Claude genera/aggiorna 5 file obbligatori
# - releases/vX.X.X.md
# - roadmap/ROADMAP_vX.X.md (aggiornato)
# - roadmap/SESSION_HANDOFF_YYYY-MM-DD.md
# - TODO_NEXT_SESSION.md (aggiornato)
# - PROJECT_PLAN.md (aggiornato)

# 3. Claude salva nel progetto nelle path corrette

# 4. Commit documentazione
git add docs/releases/*.md
git add docs/roadmap/*.md
git add docs/TODO_NEXT_SESSION.md
git add docs/PROJECT_PLAN.md
git commit -m "docs: session YYYY-MM-DD - release notes + roadmap + project plan update"
git push

# 5. Conferma con utente
"✅ Sessione chiusa. Release notes, roadmap, handoff, TODO e PROJECT PLAN aggiornati e pushati."
```

---

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

## 7. WORKFLOW EFFICIENZA ⭐ NUOVO

**Regole per ridurre friction e migliorare efficienza:**

### A) NON Ripetere Ground Rules
- Workflow (backup, release notes, commit, test) è IMPLICITO
- Non serve che l'utente ripeta ogni volta
- Claude lo applica automaticamente

### B) NO "Siparietto Accesso"
- Claude legge direttamente filesystem locale senza dire "non posso"
- Se serve qualcosa da remoto (GitHub) → chiede UNA VOLTA esplicitamente
- NO pre-spiegazioni ripetitive su cosa può/non può fare

### C) File Pesanti - Workflow Semplificato
- File >1000 righe: utente fornisce → Claude modifica → restituisce
- File <1000 righe: Claude modifica direttamente filesystem
- NO discussioni preventive su dimensione file

### D) Istruzioni Sequenziali
- 1 azione → output → 1 azione successiva
- NO blocchi concatenati di istruzioni che dipendono da output intermedi
- Aspettare conferma utente tra step dipendenti

**Queste regole valgono per TUTTE le sessioni future.**

---

**VIOLAZIONE DI QUESTE REGOLE = SESSIONE FALLITA**

Queste regole esistono perché sono state violate ripetutamente in passato, causando:
- Regressioni nel motore di ricerca
- File iOS corrotti
- Deploy di codice rotto
- Perdita di tempo dell'utente
- **Mancanza di continuità tra sessioni**
- **Documentazione obsoleta o mancante**
- **Friction inutile e ripetitivo**

**ZERO TOLLERANZA.**

---

## COMMIT MESSAGE CONVENTIONS

```
feat(scope): descrizione breve
fix(scope): descrizione breve
docs: aggiornamento documentazione
chore: task manutenzione
test: aggiunta/modifica test
refactor: refactoring senza cambio funzionalità
perf: ottimizzazione performance
```

**Scope**: `ios`, `backend`, `docs`, `build`

---

## PRIORITÀ DECISIONI

1. **Stabilità > Feature**
2. **Performance > Estetica**
3. **UX > Complessità tecnica**
4. **Patch minime > Refactor massivi**
5. **Test reali > Teoria**
6. **Documentazione > Velocità**
7. **Continuità > Velocità**
8. **Efficienza > Spiegazioni** ⭐ NUOVO

---

**REGOLA D'ORO**: Leggi, capisci, agisci, documenta. In quest'ordine. Senza friction.
