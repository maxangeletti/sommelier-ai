# llm_intent_parser.py
# Layer di normalizzazione LLM per SommelierAI
# Posizionamento: si inserisce in run_search() PRIMA dei parser rule-based esistenti.
# Il motore di scoring/ranking NON cambia.

from __future__ import annotations

import json
import os
import re
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

EXPLAIN_SYSTEM_PROMPT = """Sei un sommelier esperto italiano.
Genera una reason breve (max 15 parole) che spiega perché questo vino è rilevante per la query dell'utente.

Usa un tono naturale, professionale ma accessibile. Enfatizza gli aspetti che l'utente sta cercando.

Regole:
- Massimo 15 parole
- Tono naturale, non tecnico
- Enfatizza i match attivi (eleganza, occasione, food pairing, struttura, etc.)
- Non ripetere nome del vino
- Non usare gergo enologico difficile

Esempi:
Query: "vino elegante per cena importante"
Vino: Gevrey-Chambertin
→ "Un Borgogna raffinato perfetto per occasioni formali, elegante e di grande finezza"

Query: "rosso strutturato"
Vino: Barolo
→ "Piemonte prestigioso con tannini importanti e struttura verticale"

Query: "bianco fresco per pesce"
Vino: Vermentino
→ "Fresco e sapido, ideale per crostacei e piatti di mare"
"""


def generate_reason_with_llm(
    query: str,
    wine_name: str,
    wine_region: str,
    wine_description: str,
    ranking_signals: Dict[str, Any],
) -> str:
    """
    Genera una reason personalizzata contestuale alla query dell'utente.
    
    Args:
        query: Query utente originale
        wine_name: Nome del vino
        wine_region: Regione del vino
        wine_description: Descrizione del CSV (fallback)
        ranking_signals: Dict con segnali attivi (prestige_intent, occasion, color, etc.)
    
    Returns:
        Reason personalizzata, oppure wine_description se LLM non disponibile
    """
    if not LLM_ENABLED or not LLM_API_KEY or not query.strip():
        return wine_description or ""
    
    # Costruisci contesto per LLM
    signals_text = _format_signals_for_llm(ranking_signals)
    
    user_prompt = f"""Query utente: "{query}"

Vino: {wine_name}
Regione: {wine_region}

Segnali di ranking attivi:
{signals_text}

Genera una reason breve (max 15 parole) che spiega perché questo vino è rilevante."""
    
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
                    "max_tokens": 100,  # Reason breve
                    "system": EXPLAIN_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
        response.raise_for_status()
        data = response.json()
        reason = data["content"][0]["text"].strip()
        
        # Cleanup: rimuovi virgolette se presenti
        reason = reason.strip('"').strip("'")
        
        return reason
        
    except Exception:
        # Fallback graceful: usa descrizione CSV
        return wine_description or ""


def _format_signals_for_llm(signals: Dict[str, Any]) -> str:
    """Formatta i segnali di ranking in testo leggibile per LLM."""
    parts = []
    
    if signals.get("color_req"):
        parts.append(f"- Colore: {signals['color_req']}")
    
    if signals.get("region"):
        parts.append(f"- Regione richiesta: {signals['region']}")
    
    if signals.get("prestige_intent"):
        parts.append("- Vino prestigioso/importante")
    
    if signals.get("elegance_intent"):
        parts.append("- Eleganza richiesta")
    
    if signals.get("occasion_intent"):
        parts.append(f"- Occasione: {signals['occasion_intent']}")
    
    if signals.get("foods_req"):
        parts.append(f"- Abbinamenti cibo: {', '.join(signals['foods_req'])}")
    
    if signals.get("intensity_req"):
        parts.append(f"- Intensità richiesta: {signals['intensity_req']}")
    
    if signals.get("tannin_req"):
        parts.append(f"- Tannicità richiesta: {signals['tannin_req']}")
    
    if signals.get("style_intent"):
        parts.append(f"- Stile: {signals['style_intent']}")
    
    if signals.get("value_intent"):
        parts.append("- Rapporto qualità/prezzo importante")
    
    return "\n".join(parts) if parts else "- Nessun segnale specifico"


# =========================
# Step 3: Tasting Notes (Note di Degustazione)
# =========================

TASTING_NOTES_SYSTEM_PROMPT = """Sei un sommelier esperto italiano.
Genera note di degustazione dettagliate (60-80 parole) per questo vino.

Struttura ideale:
1. Aspetto visivo (colore, consistenza)
2. Profilo olfattivo (aromi primari, secondari, terziari)
3. Gusto (struttura, equilibrio, persistenza)
4. Considerazioni finali (quando berlo, abbinamenti)

Regole:
- 60-80 parole totali
- Tono professionale ma accessibile
- Usa terminologia sommelier corretta ma comprensibile
- Enfatizza caratteristiche distintive del vino
- Considera la query utente per contestualizzare

Esempio:
"Rubino intenso con riflessi granato. Al naso emergono note di ciliegia matura, spezie dolci e un tocco di cuoio. 
In bocca è potente e strutturato, con tannini nobili e persistenza notevole. L'acidità vivace bilancia 
perfettamente la morbidezza. Un vino da meditazione, ideale con brasati e formaggi stagionati. 
Pronto ora ma con ottimo potenziale di invecchiamento."
"""


def generate_tasting_notes_with_llm(
    query: str,
    wine_name: str,
    wine_region: str,
    wine_grapes: str,
    wine_vintage: str,
    wine_description: str,
    wine_specs: Dict[str, Any],
) -> str:
    """
    Genera note di degustazione dettagliate (60-80 parole) per la schermata dettaglio vino.
    
    Args:
        query: Query utente originale
        wine_name: Nome del vino
        wine_region: Regione del vino
        wine_grapes: Vitigni
        wine_vintage: Annata
        wine_description: Descrizione CSV (fallback)
        wine_specs: Dict con specs (body, tannins, acidity, alcohol_level, etc.)
    
    Returns:
        Tasting notes dettagliate, oppure wine_description se LLM non disponibile
    """
    if not LLM_ENABLED or not LLM_API_KEY:
        return wine_description or ""
    
    # Costruisci contesto per LLM
    specs_text = "\n".join([f"- {k}: {v}" for k, v in wine_specs.items() if v])
    
    user_prompt = f"""Query utente: "{query}"

Vino: {wine_name}
Regione: {wine_region}
Vitigni: {wine_grapes}
Annata: {wine_vintage}

Caratteristiche tecniche:
{specs_text}

Genera note di degustazione dettagliate (60-80 parole) in tono professionale ma accessibile."""
    
    try:
        with httpx.Client(timeout=6.0) as client:  # Timeout più lungo per testo lungo
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": LLM_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "max_tokens": 256,  # Tasting notes più lunghe
                    "system": TASTING_NOTES_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
        response.raise_for_status()
        data = response.json()
        notes = data["content"][0]["text"].strip()
        
        # Cleanup: rimuovi virgolette se presenti
        notes = notes.strip('"').strip("'")
        
        return notes
        
    except Exception:
        # Fallback graceful: usa descrizione CSV
        return wine_description or ""


# =========================
# Integrazione in run_search: patch minima
# =========================

# Questo è lo snippet da inserire in run_search() in main.py
# SUBITO DOPO: q = _norm(query)
# E PRIMA DI: price_info = parse_price(q)

INTEGRATION_SNIPPET = '''
    # --- LLM Intent Layer (nuovo) ---
    # Chiama il layer LLM per normalizzare query ambigue/laterali.
    # Graceful degradation: se LLM non disponibile, tutto continua come prima.
    from llm_intent_parser import parse_intent_with_llm, merge_intent
    
    llm_intent = parse_intent_with_llm(q)
    
    # I parser rule-based girano comunque (invariati)
    price_info = parse_price(q)
    region = parse_region(q) or llm_intent.get("region")
    grapes_req = parse_grapes(q) or llm_intent.get("grapes", [])
    aromas_req = parse_aromas(q)
    intensity_req = parse_intensity_request(q)
    typology_req = parse_typology_request(q)
    
    # Foods: unione rule-based + LLM
    foods_rule = parse_food_request(q)
    foods_llm = llm_intent.get("foods", [])
    foods_req = sorted(set(foods_rule) | set(foods_llm))
    
    color_req = parse_color_request(q) or llm_intent.get("color")
    value_intent = parse_value_intent(q) or llm_intent.get("value_intent", False)
    style_intent = parse_style_intent(q)
    
    # Occasion: LLM arricchisce se rule-based non ha trovato nulla
    occasion_intent = parse_occasion_intent(q) or llm_intent.get("occasion")
    
    # Prestige/elegance: OR logico (basta uno dei due a rilevarlo)
    prestige_intent = parse_prestige_intent(q) or llm_intent.get("prestige_intent", False)
    elegance_intent = (
        bool(re.search(r"\\b(elegante|elegant|finezza|raffinato|raffinata)\\b", _norm_lc(q)))
        or llm_intent.get("elegant_intent", False)
    )
    # --- Fine LLM Intent Layer ---
'''

# =========================
# Test cases: query che il parser attuale non gestisce
# =========================

TEST_QUERIES = [
    # Query ambigue/laterali — rompono il rule-based
    "qualcosa di importante per mia suocera",
    "voglio stupire",
    "cena da amici che amano la Francia",
    "aperitivo estivo ma non banale",
    "ho già preso il pesce, cosa abbino alla carne?",
    "vino da portare come regalo",
    "qualcosa di elegante ma non troppo costoso",
    "un vino che faccia colpo",
    # Query che il rule-based gestisce già (devono restare invariate)
    "rosso elegante",
    "bianco fresco per cena importante di pesce",
    "Barolo di Serralunga",
    "spumante brut sotto 25 euro",
]

if __name__ == "__main__":
    print("=== SommelierAI LLM Intent Parser — Test ===\n")
    print(f"LLM_ENABLED: {LLM_ENABLED}")
    print(f"LLM_MODEL: {LLM_MODEL}")
    print(f"API_KEY present: {'YES' if LLM_API_KEY else 'NO — set ANTHROPIC_API_KEY'}\n")

    for q in TEST_QUERIES:
        print(f"Query: '{q}'")
        intent = parse_intent_with_llm(q)
        # Mostra solo i campi non-vuoti
        active = {k: v for k, v in intent.items() if v and v != [] and v != False}
        print(f"  → {json.dumps(active, ensure_ascii=False)}\n")
