# =============================================================================
# SommelierAI — Patch: Fix GT-24 (etna) + GT-26 (voglio stupire)
# Classificazione: B (Accepted tuning)
# Data: 2026-03-20
# Commit message: fix(parser): GT-24 etna zone match + GT-26 prestige "voglio stupire"
# =============================================================================
#
# CHANGES:
#
# 1. REGION_PATTERNS — Aggiunge zone geografiche rilevanti (etna, langhe, franciacorta, etc.)
#    Impatto: parse_region() ora riconosce anche zone, non solo regioni
#
# 2. Region filter — Cambia da AND sequenziale a OR (any column matches)
#    BUG: il for loop applicava _filter_by_text_contains su ogni colonna IN SEQUENZA,
#    il che equivaleva a un AND impossibile (es. "etna" doveva essere in region E zone E denomination)
#    FIX: usa mask OR su tutte le colonne
#
# 3. parse_prestige_intent — Aggiunge pattern "stupire" / "voglio stupire" / "far colpo"
#    Impatto: GT-26 "voglio stupire" ora triggera prestige_intent=true
#
# =============================================================================

# --- ISTRUZIONI MANUALI PER APPLICARE ---
# Applica le 3 modifiche al file backend/main.py come descritto sotto.
# Dopo ogni modifica, verifica che il server parta senza errori.
# Poi esegui gt_runner.sh per il collaudo completo.


# ===========================================================================
# FIX 1: REGION_PATTERNS — Riga ~298
# ===========================================================================
# PRIMA (originale):
# ---------------------------------------------------------------------------
# REGION_PATTERNS = [
#     r"\bpiemonte\b", r"\btoscana\b", r"\bveneto\b", r"\bsicilia\b", r"\bpuglia\b", r"\btrentino\b",
#     r"\bloira\b", r"\bborgogna\b", r"\bbordeaux\b", r"\bchampagne\b", r"\balsazia\b",
# ]
# ---------------------------------------------------------------------------
# DOPO (fix):
# ---------------------------------------------------------------------------
# REGION_PATTERNS = [
#     # Regioni
#     r"\bpiemonte\b", r"\btoscana\b", r"\bveneto\b", r"\bsicilia\b", r"\bpuglia\b", r"\btrentino\b",
#     r"\bloira\b", r"\bborgogna\b", r"\bbordeaux\b", r"\bchampagne\b", r"\balsazia\b",
#     r"\bcampania\b", r"\bcalabria\b", r"\bsardegna\b", r"\bumbria\b", r"\bfriuli\b",
#     r"\babruzzo\b", r"\blazio\b", r"\bliguria\b", r"\bmarche\b", r"\bprovence\b",
#     # Zone (match su colonna zone del CSV)
#     r"\betna\b", r"\blanghe\b", r"\bfranciacorta\b", r"\bvalpolicella\b", r"\bchianti\b",
#     r"\bmontalcino\b", r"\bbolgheri\b", r"\bbarolo\b", r"\bbarbaresco\b", r"\bsoave\b",
#     r"\bchablis\b", r"\bpauillac\b", r"\bsancerre\b", r"\bpriorat\b",
# ]
# ---------------------------------------------------------------------------


# ===========================================================================
# FIX 2: Region filter AND→OR — Riga ~2094
# ===========================================================================
# PRIMA (originale):
# ---------------------------------------------------------------------------
#     # region filter (match in region/zone/denomination/country)
#     if region:
#         for col in ["region", "zone", "denomination", "country"]:
#             filtered = _filter_by_text_contains(filtered, col, region)
# ---------------------------------------------------------------------------
# DOPO (fix):
# ---------------------------------------------------------------------------
#     # region filter (match in region/zone/denomination/country — OR logic)
#     if region:
#         r_lc = _norm_lc(region)
#         mask = pd.Series(False, index=filtered.index)
#         for col in ["region", "zone", "denomination", "country"]:
#             if col in filtered.columns:
#                 mask = mask | filtered[col].astype(str).str.lower().str.contains(r_lc, na=False)
#         filtered = filtered.loc[mask]
# ---------------------------------------------------------------------------


# ===========================================================================
# FIX 3: parse_prestige_intent — Riga ~427
# ===========================================================================
# PRIMA (originale):
# ---------------------------------------------------------------------------
# def parse_prestige_intent(query: str) -> bool:
#     q = _norm_lc(query)
#     patterns = [
#         r"\bvino\s+importante\b",
#         r"\bbottiglia\s+importante\b",
#         r"\b(fa(re)?|faccia)(\s+bella)?\s+figura\b",
#         r"\bdi\s+livello\b",
#         r"\bprestigios[oa]\b",
#         r"\bpremium\b",
#     ]
#     return any(re.search(pat, q) for pat in patterns)
# ---------------------------------------------------------------------------
# DOPO (fix):
# ---------------------------------------------------------------------------
# def parse_prestige_intent(query: str) -> bool:
#     q = _norm_lc(query)
#     patterns = [
#         r"\bvino\s+importante\b",
#         r"\bbottiglia\s+importante\b",
#         r"\b(fa(re)?|faccia)(\s+bella)?\s+figura\b",
#         r"\bdi\s+livello\b",
#         r"\bprestigios[oa]\b",
#         r"\bpremium\b",
#         # GT-26: linguaggio informale prestige
#         r"\bstupire\b",
#         r"\bfar\s+colpo\b",
#         r"\bimpressionare\b",
#         r"\bvino\s+wow\b",
#         r"\beffetto\s+wow\b",
#     ]
#     return any(re.search(pat, q) for pat in patterns)
# ---------------------------------------------------------------------------
