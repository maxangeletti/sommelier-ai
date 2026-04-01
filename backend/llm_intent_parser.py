# llm_intent_parser.py
# Layer di normalizzazione LLM per SommelierAI
# Posizionamento: si inserisce in run_search() PRIMA dei parser rule-based esistenti.
# Il motore di scoring/ranking NON cambia.

from __future__ import annotations

import json
import os
import re
import hashlib
import httpx
from typing import Any, Dict, List, Optional

# =========================
# Config
# =========================

LLM_ENABLED = os.getenv("SOMMELIERAI_LLM_ENABLED", "1") == "1"
LLM_MODEL = os.getenv("SOMMELIERAI_LLM_MODEL", "claude-haiku-4-5-20251001")  # modello leggero = costo basso
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_TIMEOUT_SEC = float(os.getenv("SOMMELIERAI_LLM_TIMEOUT_SEC", "4.0"))  # fallback veloce se LLM lento
LLM_MAX_TOKENS = 256  # output strutturato breve, non serve di più

# =========================
# LLM Reason Cache (Ottimizzazione Costi)
# =========================
# Cache in-memory per reason generate. Reset ad ogni deploy.
# Key: hash(query + top_wine_id) → reason string
# Risparmio stimato: 80-90% su query ripetute
REASON_CACHE: Dict[str, str] = {}

# =========================
# Prompt di sistema
# =========================

SYSTEM_PROMPT = """Sei un parser semantico per un'app di raccomandazione vini italiana.
Il tuo unico compito è trasformare una query utente in un oggetto JSON strutturato.
Non rispondere mai in testo libero. Restituisci SOLO JSON valido, nessun altro testo.

Estrai i seguenti campi (tutti opzionali, usa null se non presenti):

{
  "color": "rosso" | "bianco" | "rosato" | null,
  "region": string | null,             // es. "piemonte", "toscana", "borgogna", "loira"
  "occasion": "aperitif" | "dinner" | "important_dinner" | "lunch" | "meditation" | "summer" | "everyday" | null,
  "prestige_intent": true | false,     // vino importante, fare bella figura, regalo
  "elegant_intent": true | false,      // elegante, fine, raffinato
  "foods": [string] | [],              // lista canonical: "pesce", "carne", "pasta", "formaggi", "dolci", "aperitivo", "verdure", "salumi", "pizza"
  "style": "fresco" | "strutturato" | "leggero" | "potente" | "minerale" | null,
  "price_max": number | null,          // soglia massima prezzo
  "price_min": number | null,          // soglia minima prezzo
  "price_target": number | null,       // prezzo target (es. "sui 30 euro")
  "sparkling": "spumante" | "frizzante" | "fermo" | null,
  "sweetness": "secco" | "dolce" | "amabile" | null,
  "value_intent": true | false,        // rapporto qualità/prezzo
  "grapes": [string] | [],             // es. ["nebbiolo", "sangiovese"]
  "free_context": string | null        // contesto non mappabile ma utile (max 20 parole)
}

Regole:
- "rosso elegante" → color: rosso, elegant_intent: true
- "qualcosa di importante per mia suocera" → prestige_intent: true, occasion: important_dinner
- "aperitivo estivo leggero" → occasion: aperitif, style: leggero, color: bianco (implicito per aperitivo)
- "voglio stupire" → prestige_intent: true
- "cena da amici che amano la Francia" → region: loira o borgogna (scegli il più probabile), occasion: dinner
- "vino da meditazione potente" → occasion: meditation, style: potente
- "ho già preso il pesce, abbino la carne" → foods: ["carne"] (ignora ciò che è già stato ordinato)
- CRITICO: NON inferire vitigni da denominazioni! "barolo" NON deve produrre grapes: ["nebbiolo"], "brunello" NON deve produrre ["sangiovese"]. Estrai "grapes" SOLO se l'utente nomina esplicitamente il vitigno (es. "un nebbiolo", "vino di sangiovese").
- CRITICO: NON inferire region da denominazioni specifiche! "barolo", "brunello", "chianti", "franciacorta" sono denominazioni che saranno matchate tramite keyword, NON tramite filtro region. Estrai "region" SOLO se l'utente dice esplicitamente la regione (es. "vino del piemonte", "un toscano").
- CRITICO: NON inferire color da denominazioni! "barolo", "brunello", "chianti" sono denominazioni rosse note, ma NON estrarre color=rosso. Estrai "color" SOLO se l'utente dice esplicitamente il colore (es. "rosso", "bianco", "rosato").
- Non inventare valori. Se non sei sicuro, usa null.
"""

# =========================
# Output fallback (identico a "nessun intent rilevato")
# =========================

EMPTY_INTENT: Dict[str, Any] = {
    "color": None,
    "region": None,
    "occasion": None,
    "prestige_intent": False,
    "elegant_intent": False,
    "foods": [],
    "style": None,
    "price_max": None,
    "price_min": None,
    "price_target": None,
    "sparkling": None,
    "sweetness": None,
    "value_intent": False,
    "grapes": [],
    "free_context": None,
}


# =========================
# Core: chiama LLM e ritorna intent strutturato
# =========================

def parse_intent_with_llm(query: str) -> Dict[str, Any]:
    """
    Chiama il layer LLM per normalizzare la query in intent strutturato.
    In caso di errore o timeout, ritorna EMPTY_INTENT (graceful degradation).
    Il motore rule-based esistente continua a girare in parallelo come fallback.
    """
    if not LLM_ENABLED or not LLM_API_KEY or not query.strip():
        return EMPTY_INTENT

    try:
        with httpx.Client(timeout=LLM_TIMEOUT_SEC) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": LLM_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "max_tokens": LLM_MAX_TOKENS,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": query}],
                },
            )
        response.raise_for_status()
        data = response.json()
        raw = data["content"][0]["text"].strip()

        # Pulizia robusta: rimuovi eventuali backtick markdown
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        intent = json.loads(raw)
        return _validate_intent(intent)

    except Exception:
        # Fallback silenzioso: il motore rule-based gestisce tutto
        return EMPTY_INTENT


def _validate_intent(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Valida e normalizza l'output LLM. Mai crashare per input inatteso."""
    out = dict(EMPTY_INTENT)

    def _str_or_none(key: str) -> Optional[str]:
        v = raw.get(key)
        return str(v).lower().strip() if v else None

    def _bool(key: str) -> bool:
        return bool(raw.get(key, False))

    def _list_of_str(key: str) -> List[str]:
        v = raw.get(key, [])
        if not isinstance(v, list):
            return []
        return [str(x).lower().strip() for x in v if x]

    def _float_or_none(key: str) -> Optional[float]:
        v = raw.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    VALID_COLORS = {"rosso", "bianco", "rosato"}
    VALID_OCCASIONS = {"aperitif", "dinner", "important_dinner", "lunch", "meditation", "summer", "everyday"}
    VALID_STYLES = {"fresco", "strutturato", "leggero", "potente", "minerale"}
    VALID_SPARKLING = {"spumante", "frizzante", "fermo"}
    VALID_SWEETNESS = {"secco", "dolce", "amabile"}

    color = _str_or_none("color")
    out["color"] = color if color in VALID_COLORS else None

    out["region"] = _str_or_none("region")

    occasion = _str_or_none("occasion")
    out["occasion"] = occasion if occasion in VALID_OCCASIONS else None

    out["prestige_intent"] = _bool("prestige_intent")
    out["elegant_intent"] = _bool("elegant_intent")
    out["value_intent"] = _bool("value_intent")

    out["foods"] = _list_of_str("foods")
    out["grapes"] = _list_of_str("grapes")

    style = _str_or_none("style")
    out["style"] = style if style in VALID_STYLES else None

    sparkling = _str_or_none("sparkling")
    out["sparkling"] = sparkling if sparkling in VALID_SPARKLING else None

    sweetness = _str_or_none("sweetness")
    out["sweetness"] = sweetness if sweetness in VALID_SWEETNESS else None

    out["price_max"] = _float_or_none("price_max")
    out["price_min"] = _float_or_none("price_min")
    out["price_target"] = _float_or_none("price_target")

    free = raw.get("free_context")
    out["free_context"] = str(free)[:100] if free else None

    return out


# =========================
# Merge: LLM intent + rule-based esistente
# =========================

def merge_intent(llm_intent: Dict[str, Any], rule_based: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strategia di merge: LLM ha priorità sui campi semantici ambigui.
    Rule-based ha priorità sui campi numerici precisi (prezzo con range esatti).
    
    Questo garantisce che il parser LLM arricchisce senza rompere ciò che
    il motore rule-based già gestisce bene.
    """
    merged = dict(rule_based)

    # Campi semantici: LLM vince se ha trovato qualcosa che rule-based non ha
    for key in ["color", "region", "occasion", "prestige_intent", "elegant_intent",
                "value_intent", "sparkling", "sweetness", "style"]:
        llm_val = llm_intent.get(key)
        rb_val = rule_based.get(key)

        # LLM vince solo se rule-based è vuoto/None/False
        if llm_val and not rb_val:
            merged[key] = llm_val

    # Foods: unione (LLM può trovare cibi che rule-based non riconosce)
    llm_foods = set(llm_intent.get("foods", []))
    rb_foods = set(rule_based.get("foods", []))
    merged["foods"] = sorted(llm_foods | rb_foods)

    # Grapes: unione
    llm_grapes = set(llm_intent.get("grapes", []))
    rb_grapes = set(rule_based.get("grapes", []))
    merged["grapes"] = sorted(llm_grapes | rb_grapes)

    # Prezzo: rule-based ha priorità (è più preciso su range/target/min/max)
    # LLM interviene solo se rule-based non ha trovato nulla
    if rule_based.get("price_mode", "none") == "none":
        if llm_intent.get("price_target"):
            merged["price_target_llm"] = llm_intent["price_target"]
        if llm_intent.get("price_max"):
            merged["price_max_llm"] = llm_intent["price_max"]
        if llm_intent.get("price_min"):
            merged["price_min_llm"] = llm_intent["price_min"]

    # Free context: solo da LLM, utile per logging/debug futuri
    merged["llm_free_context"] = llm_intent.get("free_context")

    return merged


# =========================
# Step 2: Explain Personalizzato
# =========================

EXPLAIN_SYSTEM_PROMPT = """Sei un sommelier esperto che spiega perché un vino è la scelta giusta.
Ricevi la query dell'utente e i segnali di ranking che hanno guidato la selezione.
Genera una spiegazione breve (max 30 parole) in italiano naturale e conversazionale.

La spiegazione deve:
- Essere contestuale alla query specifica dell'utente
- Menzionare i segnali chiave che giustificano la scelta
- Evitare tecnicismi eccessivi e underscore
- Usare un tono caldo ma professionale

Esempi:
Query: "vino elegante per cena importante"
Signals: prestige_intent=true, occasion=important_dinner, color=rosso, elegant_intent=true
→ "Un rosso prestigioso perfetto per occasioni formali, con eleganza e grande finezza"

Query: "bianco fresco per pesce"
Signals: color=bianco, foods=["pesce"], style=fresco
→ "Bianco fresco e minerale, ideale per esaltare i sapori delicati del pesce"

Query: "vino tannico e strutturato"
Signals: tannin_req=high, intensity_req=high, color=rosso
→ "Rosso potente e tannico, con struttura importante e grande persistenza"

Query: "voglio stupire"
Signals: prestige_intent=true
→ "Una bottiglia di grande prestigio che lascerà il segno"

Restituisci SOLO il testo della spiegazione, senza introduzioni o commenti."""


def generate_personalized_reason(
    query: str,
    active_signals: Dict[str, Any],
    top_wine: Optional[Dict[str, Any]] = None
) -> str:
    """
    Step 2: Genera reason personalizzata usando LLM con cache per ottimizzazione costi.
    
    Args:
        query: Query originale dell'utente
        active_signals: Dizionario con i segnali attivi del ranking
        top_wine: Opzionale - info sul vino top per personalizzare ulteriormente
    
    Returns:
        Reason personalizzata (str). Cache hit se già generata. Fallback a template se LLM non disponibile.
    """
    if not LLM_ENABLED or not LLM_API_KEY or not query.strip():
        return _generate_fallback_reason(active_signals)
    
    # ✅ CACHE: Check se reason già generata per questa coppia query+wine
    wine_id = top_wine.get("id", "") if top_wine else ""
    cache_key = hashlib.sha256(f"{query}:{wine_id}".encode()).hexdigest()
    
    if cache_key in REASON_CACHE:
        return REASON_CACHE[cache_key]  # ✅ Cache hit - 0 costo API
    
    try:
        # Prepara contesto strutturato per l'LLM
        signals_text = _format_signals_for_llm(active_signals, top_wine)
        
        user_message = f"""Query utente: "{query}"

Segnali di ranking attivi:
{signals_text}

Genera una spiegazione breve (max 30 parole) di perché questo è il vino giusto."""

        with httpx.Client(timeout=LLM_TIMEOUT_SEC) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": LLM_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "max_tokens": 150,  # reason breve
                    "system": EXPLAIN_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_message}],
                },
            )
        response.raise_for_status()
        data = response.json()
        reason = data["content"][0]["text"].strip()
        
        # Validazione: max 40 parole, min 5 parole
        word_count = len(reason.split())
        if 5 <= word_count <= 40:
            # ✅ CACHE: Salva reason generata
            REASON_CACHE[cache_key] = reason
            return reason
        else:
            return _generate_fallback_reason(active_signals)
    
    except Exception:
        # Fallback silenzioso
        return _generate_fallback_reason(active_signals)


def _format_signals_for_llm(signals: Dict[str, Any], top_wine: Optional[Dict[str, Any]]) -> str:
    """Formatta i segnali attivi in testo leggibile per l'LLM."""
    lines = []
    
    if signals.get("color"):
        lines.append(f"- Colore: {signals['color']}")
    if signals.get("prestige_intent"):
        lines.append("- Intento: prestigio/importanza")
    if signals.get("elegant_intent"):
        lines.append("- Intento: eleganza")
    if signals.get("occasion"):
        lines.append(f"- Occasione: {signals['occasion']}")
    if signals.get("foods"):
        lines.append(f"- Abbinamento: {', '.join(signals['foods'])}")
    if signals.get("style"):
        lines.append(f"- Stile: {signals['style']}")
    if signals.get("tannin_req"):
        lines.append(f"- Tannicità richiesta: {signals['tannin_req']}")
    if signals.get("intensity_req"):
        lines.append(f"- Intensità richiesta: {signals['intensity_req']}")
    if signals.get("region"):
        lines.append(f"- Regione: {signals['region']}")
    if signals.get("grapes"):
        lines.append(f"- Vitigni: {', '.join(signals['grapes'])}")
    if signals.get("sparkling"):
        lines.append(f"- Tipologia: {signals['sparkling']}")
    if signals.get("sweetness"):
        lines.append(f"- Dolcezza: {signals['sweetness']}")
    
    if top_wine:
        if top_wine.get("name"):
            lines.append(f"- Vino selezionato: {top_wine['name']}")
        if top_wine.get("region"):
            lines.append(f"- Provenienza: {top_wine['region']}")
    
    return "\n".join(lines) if lines else "- Nessun segnale specifico"


def _generate_fallback_reason(signals: Dict[str, Any]) -> str:
    """Template fallback se LLM non disponibile."""
    parts = []
    
    if signals.get("prestige_intent"):
        parts.append("Vino di prestigio")
    elif signals.get("elegant_intent"):
        parts.append("Vino elegante")
    
    if signals.get("color"):
        color_map = {"rosso": "Rosso", "bianco": "Bianco", "rosato": "Rosato"}
        parts.append(color_map.get(signals["color"], "Vino"))
    
    if signals.get("style"):
        parts.append(signals["style"])
    
    if signals.get("tannin_req") == "high":
        parts.append("con tannini importanti")
    elif signals.get("tannin_req") == "low":
        parts.append("con tannini morbidi")
    
    if signals.get("foods"):
        foods_str = ", ".join(signals["foods"])
        parts.append(f"ideale per {foods_str}")
    
    if signals.get("occasion"):
        occasion_map = {
            "aperitif": "per aperitivo",
            "dinner": "per cena",
            "important_dinner": "per occasioni importanti",
            "meditation": "da meditazione",
        }
        parts.append(occasion_map.get(signals["occasion"], ""))
    
    if parts:
        return " ".join(parts)
    else:
        return "Selezione basata su qualità e caratteristiche del vino"
