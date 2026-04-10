# backend/main.py
from __future__ import annotations

import os
import re
import json
import time
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

import pandas as pd
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# LLM intent parser (dual-step: parse + explain)
from llm_intent_parser import parse_intent_with_llm, generate_personalized_reason, generate_tasting_notes
from ui_helpers import should_show_value_badge, get_aroma_icons, get_mock_reviews_count, get_mock_critic_score


# =========================
# Build signature (anti-confusione / anti-regressione)
# =========================

BUILD_ID = "SommelierAI v0.2 STABILE + A/B/D (CSV schema real) + cache-safe 2026-03-25-v1.6.0"


# =========================
# Config
# =========================

APP_TITLE = "SommelierAI Backend"

# Default: ../data/wines.csv (dato che backend/main.py sta in /backend e il CSV sta in /data)
DEFAULT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wines.normalized.csv"))
CSV_PATH = os.getenv("SOMMELIERAI_CSV_PATH", DEFAULT_CSV_PATH)

SEARCH_CACHE_TTL_SEC = float(os.getenv("SOMMELIERAI_SEARCH_CACHE_TTL_SEC", "45"))
SEARCH_CACHE_CAP = int(os.getenv("SOMMELIERAI_SEARCH_CACHE_CAP", "256"))
DISABLE_CACHE = os.getenv("SOMMELIERAI_DISABLE_CACHE", "0") == "1"

MAX_RESULTS_DEFAULT = 8
MAX_RESULTS_CAP = 30


# =========================
# App
# =========================

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Utilities
# =========================

def _now() -> float:
    return time.time()


def _norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _norm_lc(s: Optional[str]) -> str:
    return _norm(s).lower()


def _sse_data(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _parse_float_maybe(v: Any) -> Optional[float]:
    s = _norm(v)
    if not s:
        return None
    s = s.replace(",", ".")
    # strip non-numeric decorations
    s = re.sub(r"[^\d\.]+", "", s)
    try:
        return float(s)
    except Exception:
        return None


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(n, hi))

def _filter_by_text_contains(df: pd.DataFrame, col: str, val: str) -> pd.DataFrame:
    """Filter DataFrame rows where column contains val (case-insensitive)."""
    if col not in df.columns or not val:
        return df
    v = _norm_lc(val)
    return df.loc[df[col].astype(str).str.lower().str.contains(v, na=False)]


# =========================
# CSV cache (mtime + warmup)
# =========================

@dataclass
class CsvCache:
    df: Optional[pd.DataFrame] = None
    mtime: float = 0.0
    rows: int = 0
    last_load_ts: float = 0.0


CSV_CACHE = CsvCache()


# ---- CSV schema reale (quello che hai mostrato) ----
CSV_EXPECTED_COLS = [
    "id", "name", "producer", "country", "region", "zone", "denomination", "vintage",
    "grape_varieties", "color", "color_detail", "body", "tannins", "acidity",
    "quality", "balance", "persistence", "alcohol_level", "sweetness",
    "food_pairings", "occasion", "style_tags",
    "price_avg", "price_min", "availability",
    "purchase_url", "description",
]


def _read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=CSV_EXPECTED_COLS)

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Ensure expected columns exist
    for c in CSV_EXPECTED_COLS:
        if c not in df.columns:
            df[c] = ""

    return df


def get_wines_df() -> pd.DataFrame:
    try:
        mtime = os.path.getmtime(CSV_PATH)
    except Exception:
        mtime = 0.0

    if CSV_CACHE.df is None or (mtime and mtime != CSV_CACHE.mtime):
        df = _read_csv_safely(CSV_PATH)
        CSV_CACHE.df = df
        CSV_CACHE.mtime = mtime
        CSV_CACHE.rows = int(len(df))
        CSV_CACHE.last_load_ts = _now()
        # Invalidate search cache immediately when dataset changes
        SEARCH_CACHE.clear()
        global SEARCH_CACHE_LAST_PRUNE_TS
        SEARCH_CACHE_LAST_PRUNE_TS = 0.0

    return CSV_CACHE.df


@app.on_event("startup")
def _warmup_on_startup() -> None:
    _ = get_wines_df()


# =========================
# Search cache (TTL + cap) — cache-safe con BUILD_ID
# =========================

@dataclass
class CacheEntry:
    ts: float
    value: Any


SEARCH_CACHE: Dict[str, CacheEntry] = {}
SEARCH_CACHE_LAST_PRUNE_TS: float = 0.0
SEARCH_CACHE_PRUNE_INTERVAL_SEC = 15.0


def _cache_key(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_prune_if_needed(force: bool = False) -> None:
    global SEARCH_CACHE_LAST_PRUNE_TS
    if DISABLE_CACHE:
        return
    now = _now()
    if not force and (now - SEARCH_CACHE_LAST_PRUNE_TS) < SEARCH_CACHE_PRUNE_INTERVAL_SEC:
        return
    expired = [k for k, ent in SEARCH_CACHE.items() if now - ent.ts > SEARCH_CACHE_TTL_SEC]
    for k in expired:
        SEARCH_CACHE.pop(k, None)
    SEARCH_CACHE_LAST_PRUNE_TS = now


def _cache_get(key: str) -> Optional[Any]:
    if DISABLE_CACHE:
        return None
    _cache_prune_if_needed()
    ent = SEARCH_CACHE.get(key)
    if not ent:
        return None
    if _now() - ent.ts > SEARCH_CACHE_TTL_SEC:
        try:
            del SEARCH_CACHE[key]
        except KeyError:
            pass
        return None
    return ent.value


def _cache_set(key: str, value: Any) -> None:
    if DISABLE_CACHE:
        return
    _cache_prune_if_needed()
    if len(SEARCH_CACHE) >= SEARCH_CACHE_CAP:
        oldest_key = None
        oldest_ts = float("inf")
        for k, ent in SEARCH_CACHE.items():
            if ent.ts < oldest_ts:
                oldest_ts = ent.ts
                oldest_key = k
        if oldest_key is not None:
            try:
                del SEARCH_CACHE[oldest_key]
            except KeyError:
                pass
    SEARCH_CACHE[key] = CacheEntry(ts=_now(), value=value)


# =========================
# Parsing: prezzo / regione / A-B-D
# =========================

# --- prezzo (supporto base: range/min/max/target + economico/costoso + extreme) ---
PRICE_RANGE_RE = re.compile(r"(?P<min>\d{1,3})\s*[-–]\s*(?P<max>\d{1,3})")


def parse_price(query: str) -> Dict[str, Any]:
    q = _norm_lc(query)

    m = PRICE_RANGE_RE.search(q)
    if m:
        return {"min": float(m.group("min")), "max": float(m.group("max")), "mode": "range"}

    m2 = re.search(r"\btra\s+(\d{1,3})\s+e\s+(\d{1,3})\b", q)
    if m2:
        return {"min": float(m2.group(1)), "max": float(m2.group(2)), "mode": "range"}

    m3 = re.search(r"\b(sotto|fino a|entro|meno di|max)\s+(\d{1,3})\b", q)
    if m3:
        return {"min": None, "max": float(m3.group(2)), "mode": "max"}

    m4 = re.search(r"\b(sopra|oltre|almeno|min)\s+(\d{1,3})\b", q)
    if m4:
        return {"min": float(m4.group(2)), "max": None, "mode": "min"}

    m5 = re.search(r"\bda\s+(\d{1,3})\s+in\s+su\b", q)
    if m5:
        return {"min": float(m5.group(1)), "max": None, "mode": "min"}
 
    m6 = re.search(r"\b(intorno a|circa|sui|sul)\s+(\d{1,3})\b", q)
    if m6:
        val = float(m6.group(2))

        # ✅ leggi finestra tipo "+/- 20" oppure "±20"
        m6b = re.search(r"(?:\+/-|±)\s*(\d{1,3})", q)
        d = float(m6b.group(1)) if m6b else 10.0  # default consigliato (al posto di 1.0)

        return {"target": val, "delta": d, "mode": "target"}

    if re.search(r"\beconomic[oa]\b|\beconomico\b|\beconomica\b", q):
        return {"max": 15.99, "mode": "max", "fallback": "economico"}
    if re.search(r"\bcostos[oa]\b|\bcostoso\b|\bcostosa\b", q):
        return {"min": 30.0, "mode": "min", "fallback": "costoso"}

    if re.search(r"\b(pi[uù]\s+costoso|vino\s+pi[uù]\s+costoso)\b", q):
        return {"extreme": "max", "mode": "extreme"}
    if re.search(r"\b(pi[uù]\s+economico|vino\s+pi[uù]\s+economico)\b", q):
        return {"extreme": "min", "mode": "extreme"}

    if re.search(r"\b(euro|€)\b", q):
        m7 = re.search(r"(\d{1,3})", q)
        if m7:
            # fallback "50 euro" => target con delta più permissivo (coerente con "intorno a")
            return {"target": float(m7.group(1)), "delta": 10.0, "mode": "target"}

    return {"mode": "none"}

# --- region parsing (robusto ma semplice) ---
REGION_PATTERNS = [
    r"\bpiemonte\b", r"\btoscana\b", r"\bveneto\b", r"\bsicilia\b", r"\bpuglia\b", r"\btrentino\b",
    r"\bloira\b", r"\bborgogna\b", r"\bbordeaux\b", r"\bchampagne\b", r"\balsazia\b",
]


def parse_region(query: str) -> Optional[str]:
    q = _norm_lc(query)
    for pat in REGION_PATTERNS:
        if re.search(pat, q):
            return re.sub(r"\\b", "", pat).strip("\\")
    return None


# --- B: vitigni (match su query -> filtro su grape_varieties) ---
KNOWN_GRAPES = [
    # Italiani principali
    "sangiovese", "nebbiolo", "barbera", "montepulciano", "primitivo", "aglianico",
    "nero d'avola", "negroamaro", "corvina", "corvinone", "rondinella", "molinaro",
    "glera", "vermentino",
    "sagrantino", "dolcetto", "arneis", "grignolino", "lagrein", "cannonau",
    
    # Bianchi italiani
    "trebbiano", "verdicchio", "fiano", "greco", "grechetto", "garganega", "pecorino",
    "cortese", "turbiana", "friulano", "ribolla gialla", "pigato", "vernaccia",
    "malvasia", "moscato bianco", "moscato di scanzo",
    
    # Sicilia e Sud
    "nerello mascalese", "nerello cappuccio", "frappato", "carricante", "gaglioppo",
    "zibibbo", "grillo", "catarratto",
    
    # Valle d'Aosta e Alto Adige
    "petit rouge", "prie blanc", "schiava", "lagrein",
    
    # Friuli e confine est
    "vitovska", "rebula",
    
    # Emilia
    "lambrusco di sorbara",
    
    # Lazio
    "cesanese", "procanico",
    
    # Bianchi aromatici
    "gewurztraminer", "albarino",
    
    # Internazionali
    "chardonnay", "sauvignon", "sauvignon blanc", "pinot noir", "pinot nero", "pinot meunier",
    "cabernet sauvignon", "cabernet franc", "merlot", "petit verdot",
    "syrah", "shiraz", "grenache", "garnacha", "carignan", "cinsault",
    "riesling", "chenin blanc", "malbec", "tempranillo", "graciano", "carinena",
    "touriga nacional", "touriga franca", "tinta roriz",
]


def parse_grapes(query: str) -> List[str]:
    q = _norm_lc(query)
    found: List[str] = []
    for g in KNOWN_GRAPES:
        if re.search(rf"\b{re.escape(g)}\b", q):
            found.append(g)
    return sorted(set(found))


# --- A: sentori (match su description, dato che non hai aromi strutturati) ---
AROMA_KEYWORDS = {
    "agrumi": ["agrumi", "agrumato", "agrumata", "agrumati", "agrumatico", "agrumatica", "limone", "lime", "pompelmo", "cedro", "bergamotto"],
    "frutta rossa": ["frutta rossa", "ciliegia", "fragola", "lampone", "ribes"],
    "frutta nera": ["frutta nera", "mora", "mirtillo", "prugna", "amarena"],
    "fiori": ["floreale", "fiori", "violetta", "rosa", "gelsomino", "lavanda"],
    "spezie": ["speziato", "speziata", "pepe", "cannella", "chiodi di garofano", "noce moscata"],
    "vaniglia": ["vaniglia", "vanigliato", "vanigliata"],
    "tostato": ["tostato", "tostata", "caffè", "caffe", "cacao", "cioccolato", "tabacco"],
    "erbaceo": ["erbaceo", "erba", "fieno", "salvia"],
    "minerale": ["minerale", "pietra focaia", "gesso", "salino", "iodato"],
    "balsamico": ["balsamico", "menta", "eucalipto", "resina"],
}


# --- v1.0: Food pairing intent (match cibo-vino) ---
# Parsing SAFE: attivo solo se l'utente cita cibi/contesti espliciti nella query.
FOOD_KEYWORDS = {
    # categorie / sottocategorie specifiche prima delle generiche
    "formaggi_erborinati": ["formaggi erborinati", "erborinati", "blu cheese", "blue cheese", "roquefort", "stilton"],
    "pesce_crudo": ["pesce crudo", "crudo di pesce", "tartare di pesce", "carpaccio di pesce"],
    "pesce": ["pesce", "seafood", "sushi", "crudo", "ostriche", "crostacei", "gamberi", "scampi", "tonno", "salmone"],
    "carne": ["carne", "bistecca", "manzo", "vitello", "filetto", "tagliata", "grigliata", "bbq", "arrosto", "brasato", "maiale", "agnello", "cacciagione"],
    "pasta": ["pasta", "spaghetti", "tagliatelle", "lasagne", "risotto"],
    "pizza": ["pizza", "margherita", "diavola", "napoletana"],
    "formaggi": ["formaggi", "formaggio", "pecorino", "parmigiano", "grana", "gorgonzola", "toma", "brie"],
    "salumi": ["salumi", "prosciutto", "crudo", "cotto", "salame", "mortadella", "speck"],
    "verdure": ["verdure", "vegetariano", "veg", "insalata", "ortaggi", "asparagi", "carciofi", "funghi"],
    "dolci": ["dolce", "dolci", "dessert", "torta", "cioccolato", "pasticceria", "cantucci"],
    # contesti
    "aperitivo": ["aperitivo", "apericena", "stuzzichini"],
}
# Export: elenco canonical (utile per normalizzazione CSV / test)
FOOD_CANONICAL = sorted(list(FOOD_KEYWORDS.keys()))

FOOD_EXCLUSIVE_MAP = {
    "formaggi_erborinati": ["formaggi"],
    "pesce_crudo": ["pesce"],
}

def parse_food_request(query: str) -> List[str]:
    q = _norm_lc(query)
    found: List[str] = []
    for canonical, variants in FOOD_KEYWORDS.items():
        for v in variants:
            if re.search(rf"\b{re.escape(v)}\b", q):
                found.append(canonical)
                break

    found_set = set(found)
    for specific, generics in FOOD_EXCLUSIVE_MAP.items():
        if specific in found_set:
            for generic in generics:
                found_set.discard(generic)

    return sorted(found_set)

def parse_style_intent(q: str) -> Dict[str, bool]:
    qq = _norm_lc(q)

    def _has_any(terms: List[str]) -> bool:
        return any(t in qq for t in terms)

    return {
        # già esistenti (mantieni se li hai)
        "elegant": _has_any(["elegante", "fine", "raffinato", "delicato"]),
        "aperitivo": _has_any(["aperitivo", "aperitif"]),
        "meditation": _has_any(["meditazione", "da meditazione", "vino da meditazione"]),
        "important_dinner": _has_any(["cena importante", "occasione speciale", "serata importante"]),

        # ✅ nuovi (B1)
        "power": _has_any(["potente", "strutturato", "robusto", "corposo", "importante"]),
        "fresh": _has_any(["fresco", "leggero", "beverino", "snello"]),
    }

def parse_occasion_intent(query: str) -> Optional[str]:
    q = _norm_lc(query)

    if re.search(r"\baperitivo\b|\bapericena\b", q):
        return "aperitif"
    if re.search(r"\bcena\s+importante\b|\boccasione\s+speciale\b|\bserata\s+importante\b", q):
        return "important_dinner"
    if re.search(r"\bcena\b", q):
        return "dinner"
    if re.search(r"\bpranzo\b", q):
        return "lunch"
    if re.search(r"\bmeditazione\b", q):
        return "meditation"
    if re.search(r"\bestate\b|\bestivo\b", q):
        return "summer"
    if re.search(r"\bquotidiano\b|\bdi\s+tutti\s+i\s+giorni\b", q):
        return "everyday"

    return None


def parse_prestige_intent(query: str) -> bool:
    q = _norm_lc(query)
    patterns = [
        r"\bvino\s+importante\b",
        r"\bbottiglia\s+importante\b",
        r"\b(fa(re)?|faccia)(\s+bella)?\s+figura\b",
        r"\bdi\s+livello\b",
        r"\bprestigios[oa]\b",
        r"\bpremium\b",
    ]
    return any(re.search(pat, q) for pat in patterns)


def _prestige_match_score(row: Any) -> float:
    score = 0.0

    qv = _parse_float_maybe(getattr(row, "quality", "")) or _score_quality(row)
    pr = _price_effective(row) or 0.0
    denom = _norm_lc(getattr(row, "denomination", ""))
    name = _norm_lc(getattr(row, "name", ""))
    occ = _norm_lc(getattr(row, "occasion", ""))
    tags_raw = _norm_lc(getattr(row, "style_tags", ""))
    sparkling = derive_sparkling(
        _norm(getattr(row, "denomination", "")),
        _norm(getattr(row, "style_tags", "")),
        _norm(getattr(row, "name", "")),
        _norm(getattr(row, "description", "")),
    )
    hay = " ".join([denom, name, tags_raw])

    if qv >= 4.8:
        score += 0.35
    elif qv >= 4.6:
        score += 0.27
    elif qv >= 4.4:
        score += 0.18

    if pr >= 120:
        score += 0.30
    elif pr >= 80:
        score += 0.24
    elif pr >= 50:
        score += 0.18
    elif pr >= 30:
        score += 0.10

    prestige_tokens = [
        "barolo", "brunello", "barbaresco", "amarone", "taurasi",
        "champagne", "franciacorta", "pauillac", "gevrey-chambertin",
        "habemus", "lynch-bages", "bollinger", "roederer", "armand rousseau",
        "prum", "trimbach", "franz haas", "edi simcic", "edi simčič",
    ]
    if any(tok in hay for tok in prestige_tokens):
        score += 0.12

    if sparkling == "spumante" and ("champagne" in hay or "franciacorta" in hay):
        score += 0.08

    if "cena importante" in occ or "meditazione" in occ:
        score += 0.08

    if "elegante" in tags_raw or "longevo" in tags_raw:
        score += 0.05

    return _clamp01(score)

def food_match(row: Any, foods_req: List[str]) -> bool:
    if not foods_req:
        return False
    fp = _norm_lc(getattr(row, "food_pairings", ""))
    if not fp:
        return False
    # match per categoria: se il CSV contiene il canonical o una delle varianti
    for canonical in foods_req:
        if canonical in fp:
            return True
        for v in FOOD_KEYWORDS.get(canonical, []):
            if v and v in fp:
                return True
    return False

def food_match_strength(row: Any, foods_req: List[str]) -> float:
    """
    Strength 0.0–1.0 (deterministico).
    - 1.0 match diretto categoria o variante nel food_pairings
    - 0.5 match debole (keyword presente ma non canonical)
    """
    if not foods_req:
        return 0.0
    fp = _norm_lc(getattr(row, "food_pairings", ""))
    if not fp:
        return 0.0

    # match diretto
    for canonical in foods_req:
        if canonical and canonical in fp:
            return 1.0
        for v in FOOD_KEYWORDS.get(canonical, []):
            if v and v in fp:
                return 1.0

    # match debole: se compare una keyword di un'altra categoria ma vicina
    # (es. "brasato" -> carne) quando foods_req include "carne" ma fp non ha canonical
    weak_hits = 0
    for canonical in foods_req:
        for v in FOOD_KEYWORDS.get(canonical, []):
            if v and v in fp:
                weak_hits += 1
    if weak_hits > 0:
        return 0.5

    return 0.0

def parse_aromas(query: str) -> List[str]:
    q = _norm_lc(query)
    requested: List[str] = []
    for canonical, variants in AROMA_KEYWORDS.items():
        for v in variants:
            if v and re.search(rf"\b{re.escape(v)}\b", q):
                requested.append(canonical)
                break
    return sorted(set(requested))


# --- A: intensità (richiesta da query) + derivazione da body/tannins/alcohol_level ---
INTENSITY_WORDS = {
    "delicato": "low", "leggero": "low", "leggera": "low",
    "medio": "medium", "media": "medium", "equilibrato": "medium", "equilibrata": "medium",
    "intenso": "high", "intensa": "high", "strutturato": "high", "strutturata": "high", "potente": "high",
}


def parse_intensity_request(query: str) -> Optional[str]:
    q = _norm_lc(query)
    for w, v in INTENSITY_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", q):
            return v
    return None


def _normalize_level(s: str) -> str:
    # normalize italian/english-ish into low/medium/high
    v = _norm_lc(s)
    if not v:
        return ""
    if any(k in v for k in ["low", "basso", "legger", "delicat"]):
        return "low"
    if any(k in v for k in ["high", "alto", "intens", "strutturat", "power", "potent", "robusto"]):
        return "high"
    if any(k in v for k in ["medium", "medio", "media", "moderato"]):
        return "medium"
    return ""


def derive_intensity(body: str, tannins: str, alcohol_level: str) -> Optional[str]:
    # body/tannins in CSV are often categorical; alcohol may be numeric or text.
    b = _normalize_level(body)
    t = _normalize_level(tannins)

    alc = _parse_float_maybe(alcohol_level)
    # scoring: high if any strong signal
    high_signals = 0
    low_signals = 0

    if b == "high":
        high_signals += 1
    if t == "high":
        high_signals += 1
    if b == "low":
        low_signals += 1
    if t == "low":
        low_signals += 1

    if alc is not None:
        if alc >= 14.0:
            high_signals += 1
        if alc <= 11.5:
            low_signals += 1

    if high_signals >= 2:
        return "high"
    if low_signals >= 2:
        return "low"
    if high_signals == 1 and low_signals == 0:
        return "high"
    if low_signals == 1 and high_signals == 0:
        return "low"
    return "medium"



def derive_freshness(acidity: str, sparkling: str, alcohol_level: str) -> Optional[str]:
    """
    Deriva livello di freschezza (low/medium/high) da acidity, sparkling, alcohol.
    Freschezza = acidità + effervescenza - alcol.
    """
    # Normalizza acidity
    acid = _normalize_level(acidity)
    
    # Sparkling boost
    sparkling_boost = 0
    if "spumante" in _norm_lc(sparkling):
        sparkling_boost = 1
    elif "frizzante" in _norm_lc(sparkling):
        sparkling_boost = 1
    
    # Alcohol penalty
    alc = _parse_float_maybe(alcohol_level)
    alcohol_penalty = 0
    if alc is not None:
        if alc >= 14.0:
            alcohol_penalty = 1
        elif alc >= 13.0:
            alcohol_penalty = 0
    
    # Calcola freshness score
    score = 0
    if acid == "high":
        score += 2
    elif acid == "medium":
        score += 1
    
    score += sparkling_boost
    score -= alcohol_penalty
    
    # Map to low/medium/high
    if score >= 3:
        return "high"
    elif score >= 1:
        return "medium"
    else:
        return "low"


# --- D: tipologia (sparkling) derivata + sweetness normalizzata ---
def derive_sparkling(denomination: str, style_tags: str, name: str, description: str) -> str:
    hay = " ".join([_norm_lc(denomination), _norm_lc(style_tags), _norm_lc(name), _norm_lc(description)])
    # spumante
    if any(k in hay for k in [
        "champagne", "prosecco", "franciacorta", "cava", "cremant", "crémant",
        "spumante", "metodo classico", "méthode traditionnelle", "metodo tradizionale",
    ]):
        return "spumante"
    # frizzante
    if any(k in hay for k in ["frizzante", "pet-nat", "pét-nat", "col fondo"]):
        return "frizzante"
    return "fermo"


def normalize_sweetness(s: str) -> str:
    v = _norm_lc(s)
    if not v:
        return ""
    # map common variants
    if any(k in v for k in ["secco", "dry", "brut", "extra brut", "pas dos", "dosage zero", "nature"]):
        return "secco"
    if any(k in v for k in ["abbocc", "off-dry", "off dry"]):
        return "abboccato"
    if any(k in v for k in ["amabile", "semi-sweet", "semi sweet"]):
        return "amabile"
    if any(k in v for k in ["dolce", "sweet", "demi-sec", "demisec", "doux"]):
        return "dolce"
    return v  # fallback (non rompere)


def parse_typology_request(query: str) -> Dict[str, Optional[str]]:
    q = _norm_lc(query)

    sparkling: Optional[str] = None
    sweetness: Optional[str] = None
    tannin: Optional[str] = None

    # sparkling keywords
    if re.search(r"\bspumante\b", q) or re.search(r"\bchampagne\b", q) or re.search(r"\bprosecco\b", q) or re.search(r"\bbollicine\b", q) or re.search(r"\bfranciacorta\b", q):
        sparkling = "spumante"
    elif re.search(r"\bfrizzante\b", q):
        sparkling = "frizzante"
    elif re.search(r"\bfermo\b", q):
        sparkling = "fermo"

    # sweetness keywords
    if re.search(r"\bsecco\b|\bbrut\b|\bextra\s*brut\b|\bnature\b", q):
        sweetness = "secco"
    elif re.search(r"\babboccato\b", q):
        sweetness = "abboccato"
    elif re.search(r"\bamabile\b", q):
        sweetness = "amabile"
    elif re.search(r"\bdolce\b|\bdemi-?sec\b", q):
        sweetness = "dolce"

    # Brut implies spumante + secco if not specified
    if re.search(r"\bbrut\b|\bextra\s*brut\b|\bpas\s*dos", q):
        if sparkling is None:
            sparkling = "spumante"
        if sweetness is None:
            sweetness = "secco"

    # ✅ AIS Tannin levels (5 livelli) - ORDINE IMPORTANTE: più specifici prima
    if re.search(r"\bpoco\s+tannico\b|\btannini\s+leggeri\b", q):
        tannin = "low"
    elif re.search(r"\btannico\s+morbido\b|\bmorbido\b.*\btannic", q):
        tannin = "low-medium"
    elif re.search(r"\btannico\s+aggressivo\b|\baggressivo\b.*\btannic", q):
        tannin = "very_high"
    elif re.search(r"\bmolto\s+tannico\b|\btannicissimo\b", q):
        tannin = "high"
    elif re.search(r"\btannico\b", q):
        # "tannico" senza qualificatori → medium-high
        tannin = "medium-high"

    return {"sparkling": sparkling, "sweetness": sweetness, "tannin": tannin}


# --- Intent: "qualità/prezzo" SOLO su richiesta utente ---
VALUE_INTENT_RE = re.compile(
    r"\b("
    r"qualit[aà]\s*prezzo|prezzo\s*qualit[aà]|"
    r"rapporto\s+qualit[aà]\s*/?\s*prezzo|"
    r"rapporto\s+qualit[aà]\s+prezzo|"
    r"value\s*for\s*money|"
    r"q\s*/\s*p|q\/p"
    r")\b",
    re.IGNORECASE
)


# --- Color intent (bianco/rosso/rosato) ---
def parse_color_request(query: str) -> Optional[str]:
    """Ritorna uno tra: 'bianco', 'rosso', 'rosato' se l'utente lo chiede esplicitamente nel testo."""
    q = _norm(query)

    # Tokenizzazione semplice per evitare falsi positivi
    tokens = set(re.findall(r"[a-zàèéìòù]+", q))

    if {"bianco", "bianchi", "bianca", "bianche"} & tokens:
        return "bianco"
    if {"rosso", "rossi", "rossa", "rosse"} & tokens:
        return "rosso"
    if {"rosato", "rosati", "rosata", "rosate", "rosé", "rose"} & tokens:
        return "rosato"

    return None


def parse_value_intent(query: str) -> bool:
    return bool(VALUE_INTENT_RE.search(_norm_lc(query)))


# =========================
# Filtering helpers
# =========================

def _filter_by_price(df: pd.DataFrame, price_info: Dict[str, Any]) -> pd.DataFrame:
    mode = price_info.get("mode", "none")
    if mode == "none":
        return df

    # use price_avg fallback price_min
    prices = df["price_avg"].map(_parse_float_maybe)
    prices_min = df["price_min"].map(_parse_float_maybe)

    # choose avg if exists else min
    effective = prices.copy()
    effective[effective.isna()] = prices_min[effective.isna()]

    if mode == "range":
        mn = price_info.get("min")
        mx = price_info.get("max")
        if mn is not None:
            df = df.loc[effective >= float(mn)]
        if mx is not None:
            df = df.loc[effective <= float(mx)]
        return df

    if mode == "max":
        mx = price_info.get("max")
        if mx is None:
            return df
        return df.loc[effective <= float(mx)]

    if mode == "min":
        mn = price_info.get("min")
        if mn is None:
            return df
        return df.loc[effective >= float(mn)]

    if mode == "target":
        tgt = price_info.get("target")
        delta = float(price_info.get("delta", 1.0))
        if tgt is None:
            return df
        return df.loc[(effective >= float(tgt) - delta) & (effective <= float(tgt) + delta)]

    if mode == "extreme":
        return df

    return df


def _filter_by_color(df: pd.DataFrame, color_req: Optional[str]) -> pd.DataFrame:
    """Filtra per colore SOLO se richiesto esplicitamente.

    Supporta dataset IT + EN dopo normalizzazione CSV.
    """
    if not color_req:
        return df

    want = _norm_lc(str(color_req))
    alias = {
        "rossa": "rosso", "rosse": "rosso", "rossi": "rosso",
        "red": "rosso", "ruby_red": "rosso",
        "bianca": "bianco", "bianche": "bianco", "bianchi": "bianco",
        "white": "bianco", "straw_yellow": "bianco", "golden_yellow": "bianco",
        "rosé": "rosato", "rose": "rosato", "pink": "rosato", "salmon": "rosato",
    }
    want = alias.get(want, want)

    accepted = {
        "rosso": {"rosso", "red", "ruby_red"},
        "bianco": {"bianco", "white", "straw_yellow", "golden_yellow"},
        "rosato": {"rosato", "rose", "rosé", "pink", "salmon"},
    }.get(want, {want})

    cols = []
    if "color" in df.columns:
        cols.append("color")
    if "color_detail" in df.columns and "color_detail" not in cols:
        cols.append("color_detail")
    if not cols:
        return df

    import re as _re
    mask = None
    for c in cols:
        series = df[c].astype(str).map(_norm_lc)
        m = series.apply(lambda x: any(tok in _re.split(r"[\s,;|/]+", x) for tok in accepted))
        m_bool = m.astype(bool); mask = m_bool if mask is None else (mask | m_bool)

    return df.loc[mask] if mask is not None else df

def _filter_new_A_B_D(
    df: pd.DataFrame,
    grapes_req: List[str],
    aromas_req: List[str],
    intensity_req: Optional[str],
    typology_req: Dict[str, Optional[str]],
) -> pd.DataFrame:
        
    # B) grapes: filter on grape_varieties (token-aware, robusto su separatori)
    if grapes_req:
        series = df["grape_varieties"].astype(str).map(_norm_lc)

        # separatori supportati: spazio, virgola, pipe, slash, punto e virgola
        mask = None
        for g in grapes_req:
            g = _norm_lc(g)
            if not g:
                continue
            token_re = re.compile(rf"(?:^|[\s,;|/]+){re.escape(g)}(?:$|[\s,;|/]+)")
            m = series.str.contains(token_re, regex=True, na=False)
            mask = m if mask is None else (mask | m)

        if mask is not None:
            df = df.loc[mask]

    # A) aromas: match on description (free text)
    if aromas_req:
        lc = df["description"].astype(str).str.lower()
        mask = False
        for a in aromas_req:
            # we search canonical word too (it's what parse_aromas returns)
            mask = mask | lc.str.contains(re.escape(a), na=False)
            # also search for any variant of that canonical in description for robustness
            for v in AROMA_KEYWORDS.get(a, []):
                mask = mask | lc.str.contains(re.escape(_norm_lc(v)), na=False)
        df = df.loc[mask]

    # D) typology: derived sparkling + normalized sweetness + tannin
    sp_req = typology_req.get("sparkling")
    sw_req = typology_req.get("sweetness")
    tannin_req = typology_req.get("tannin")

    if sp_req or sw_req or intensity_req or tannin_req:
        # compute derived columns on the fly (vectorized apply is fine at this scale; dataset is cached)
        derived_sp = df.apply(
            lambda r: derive_sparkling(r.get("denomination", ""), r.get("style_tags", ""), r.get("name", ""), r.get("description", "")),
            axis=1,
        )
        derived_sw = df["sweetness"].astype(str).map(normalize_sweetness)

        if sp_req:
            # ✅ FIX: "frizzante" query includes both frizzante AND spumante (both effervescent)
            if sp_req == "frizzante":
                df = df.loc[derived_sp.isin(["frizzante", "spumante"])]
            else:
                df = df.loc[derived_sp.eq(sp_req)]
        if sw_req:
            df = df.loc[derived_sw.eq(sw_req)]
        
        # ✅ Tannin filter (AIS 5 levels)
        if tannin_req:
            # Map CSV tannins (low/medium/high) to AIS levels
            tannin_map = {
                "low": ["low"],
                "low-medium": ["low", "medium"],
                "medium-high": ["medium", "high"],
                "high": ["high"],
                "very_high": ["high"]  # very_high = highest tannins (Sagrantino, Nebbiolo)
            }
            allowed_levels = tannin_map.get(tannin_req, [])
            if allowed_levels:
                df = df.loc[df["tannins"].isin(allowed_levels)]

    # A) intensity: derived from body/tannins/alcohol_level
    if intensity_req:
        derived_int = df.apply(
            lambda r: derive_intensity(r.get("body", ""), r.get("tannins", ""), r.get("alcohol_level", "")) or "",
            axis=1,
        )
        df = df.loc[derived_int.eq(intensity_req)]

    return df


# =========================
# Scoring
# =========================

def _score_quality(row: Any) -> float:
    """
    Usa qualità/balance/persistence se numeriche.
    Se assenti, fallback su rating_overall (0..5).
    """

    q = _parse_float_maybe(getattr(row, "quality", ""))
    b = _parse_float_maybe(getattr(row, "balance", ""))
    p = _parse_float_maybe(getattr(row, "persistence", ""))

    score = 0.0
    parts = 0

    for v in [q, b, p]:
        if v is not None:
            score += v
            parts += 1

    if parts > 0:
        return score / parts

    # ✅ Fallback serio: usa rating_overall se disponibile
    r = _parse_float_maybe(getattr(row, "rating_overall", ""))
    if r is not None and r > 0:
        return max(0.0, min(5.0, r))

    return 0.0


def _price_effective(row: Any) -> Optional[float]:
    a = _parse_float_maybe(getattr(row, "price_avg", ""))
    m = _parse_float_maybe(getattr(row, "price_min", ""))
    return a if a is not None else m

def _score_row(row: Any, price_info: Dict[str, Any], boosts: Dict[str, bool], value_intent: bool = False) -> Tuple[float, float]:
    base = _score_quality(row)

    # =========================
    # VALUE FOR MONEY (attivo SOLO se richiesto) — più realistico e stabile
    # =========================
    pr = _price_effective(row)
    if value_intent and pr and pr > 0:
        import math

        # Qualità "vera" se presente, altrimenti base
        q = _parse_float_maybe(getattr(row, "quality", "")) or base

        # Diminishing returns sul prezzo: il prezzo aiuta ma non domina
        value_index = (q + 0.75) / math.log(pr + 2.0)

        # Boost morbido e cappato (non deve ribaltare la qualità pura)
        base += min(value_index * 0.55, 1.25)

    # =========================
    # Price delta (target mode)
    # =========================
    price_delta = 0.0
    if price_info.get("mode") == "target" and pr is not None:
        tgt = float(price_info.get("target", 0.0))
        price_delta = abs(pr - tgt)
        base += max(0.0, 1.2 - price_delta) * 0.25

    # =========================
    # Boost A/B/D
    # =========================
    if boosts.get("grape_match"):
        base += 0.6
    if boosts.get("aroma_match"):
        base += 0.35
    if boosts.get("intensity_match"):
        base += 0.25
    if boosts.get("typology_match"):
        base += 0.35
        # v1.0: cibo-vino
    # v1.0: cibo-vino (penalty SOLO se foods_req presente)
    if boosts.get("food_match"):
        base += 0.45
    elif boosts.get("foods_present"):
        base -= 0.25

    return base, price_delta

def _score_row_a9v1(
    row: Any,
    price_info: Dict[str, Any],
    boosts: Dict[str, bool],
    style_intent: Optional[Dict[str, bool]] = None,
) -> Tuple[float, float]:
    """A9v1/A9v2 (opzionale): A9 + micro value boost; A9v2 aggiunge intent semantico (se style_intent presente)."""
    base = _score_quality(row)

    pr = _price_effective(row)
    price_delta = 0.0

    if price_info.get("mode") == "target" and pr is not None:
        tgt = float(price_info.get("target", 0.0))
        price_delta = abs(pr - tgt)
        base += max(0.0, 1.2 - price_delta) * 0.25

    if boosts.get("grape_match"):
        base += 0.6
    if boosts.get("aroma_match"):
        base += 0.35
    if boosts.get("intensity_match"):
        base += 0.25
    if boosts.get("typology_match"):
        base += 0.35
    
    # v1: cibo-vino (boolean)
    if boosts.get("food_match"):
        base += 0.45
    elif boosts.get("foods_present"):
        base -= 0.25

    # v2: cibo-vino (strength) SOLO se style_intent presente (cioè relevance_a9v2)
    if style_intent and boosts.get("foods_present"):
        fs = food_match_strength(row, boosts.get("foods_req", [])) if isinstance(boosts.get("foods_req"), list) else 0.0
        # micro boost cappato: 0.0..+0.25
        base += min(max(fs, 0.0), 1.0) * 0.25

    if pr is not None and pr > 0:
        qv = _parse_float_maybe(getattr(row, "quality", ""))
        q = qv if qv is not None else base
        v = (q + 0.75) / math.log(pr + 2.0)
        base += min(v * 0.10, 0.20)

    # ---- A9v2 Style intent (optional) ----
    if style_intent:
        sb = 0.0
        di = _norm_lc(getattr(row, "__derived_intensity", "")) or _norm_lc(
            derive_intensity(
                _norm(getattr(row, "body", "")),
                _norm(getattr(row, "tannins", "")),
                _norm(getattr(row, "alcohol_level", "")),
            )
            or ""
        )
        sparkling = _norm_lc(getattr(row, "__derived_sparkling", "")) or _norm_lc(
            derive_sparkling(
                _norm(getattr(row, "denomination", "")),
                _norm(getattr(row, "style_tags", "")),
                _norm(getattr(row, "name", "")),
                _norm(getattr(row, "description", "")),
            )
            or ""
        )
        denom = _norm_lc(getattr(row, "denomination", ""))

        if style_intent.get("elegant"):
            if di in ("low", "medium"):
                sb += 0.10
            elif di == "high":
                sb -= 0.05

        if style_intent.get("important_dinner"):
            # Prestige score derivato (no hardcoding): usa qualità + fascia prezzo
            qv = _parse_float_maybe(getattr(row, "quality", ""))
            pr_eff = _price_effective(row)

            prestige = 0.0
            if qv is not None:
                if qv >= 4.75:
                    prestige += 0.08
                elif qv >= 4.6:
                    prestige += 0.05

            if pr_eff is not None:
                if pr_eff >= 70:
                    prestige += 0.06
                elif pr_eff >= 45:
                    prestige += 0.04

            # micro boost cappato
            sb += min(prestige, 0.14)

        if style_intent.get("aperitivo"):
            if "spumante" in sparkling:
                sb += 0.12
            if di == "low":
                sb += 0.06
        if style_intent.get("meditation"):
            if di == "high":
                sb += 0.10
            qv = _parse_float_maybe(getattr(row, "quality", ""))
            if qv is not None and qv >= 4.7:
                sb += 0.06

        # Occasion intent: micro boost se occasion del vino matcha la richiesta
        occ_req = boosts.get("occasion_intent")
        if occ_req:
            occ_row = _norm_lc(getattr(row, "occasion", ""))
            if occ_row and occ_req in occ_row:
                sb += 0.10
        base += max(-0.10, min(sb, 0.25))
        
    return base, price_delta

def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x != x or x == float("inf") or x == float("-inf"):
        return 0.0
    return max(0.0, min(1.0, x))


def _score_row_a9v2_composite(
    row: Any,
    price_info: Dict[str, Any],
    boosts: Dict[str, Any],
    style_intent: Optional[Dict[str, bool]] = None,
    debug_out: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """
    A9v2 Composite (relevance_v2) — Food-smart (Opzione 3)
    Score = 0.52*M + 0.18*Q + 0.08*V + 0.17*F + 0.03*O + 0.02*I

    Note:
    - M qui è il match "UI-stable" che già usi per la barra (0..1), senza cambiare struttura.
    - F è food strength (0..1) solo se foods_present.
    - O/I sono boost semplici (0..1) per occasione e intensità.
    - Output finale lo riportiamo su scala ~0..5 per coerenza UI.
    """
    # ---------- components (0..1) ----------
    # Q: usa score qualità esistente ma normalizza (assumiamo scala ~0..5)
    q_raw = _score_quality(row)
    Q = _clamp01(q_raw / 5.0)

    # V: value coerente con tua formula attuale, normalizzato soft
    pr = _price_effective(row)
    price_delta = 0.0

    # price target (come prima) -> solo per price_delta e micro-spinta indiretta via V/Q
    if price_info.get("mode") == "target" and pr is not None:
        tgt = float(price_info.get("target", 0.0))
        price_delta = abs(pr - tgt)

    V = 0.0
    if pr is not None and pr > 0:
        qv = _parse_float_maybe(getattr(row, "quality", ""))
        q_for_value = qv if qv is not None else q_raw
        v_raw = (q_for_value + 0.75) / math.log(pr + 2.0)
        # normalizzazione conservativa: v_raw tipicamente ~0..2+
        V = _clamp01(v_raw / 2.0)

    # F: food strength (0..1) solo se cibo presente
    foods_present = bool(boosts.get("foods_present"))
    foods_req = boosts.get("foods_req", [])
    F = 0.0
    if foods_present and isinstance(foods_req, list) and foods_req:
        try:
            F = _clamp01(food_match_strength(row, foods_req))
        except Exception:
            F = 0.0

    # O: occasion alignment semplice
    O = 0.0
    occ_req = boosts.get("occasion_intent")
    if occ_req:
        occ_row = _norm_lc(getattr(row, "occasion", ""))
        if occ_row and _norm_lc(str(occ_req)) in occ_row:
            O = 1.0
    # anche intent semantico, se presente
    if style_intent:
        if style_intent.get("important_dinner") or style_intent.get("aperitivo") or style_intent.get("meditation"):
            # se c'è un intent, O non deve restare sempre 0: lo "accendiamo" leggero
            O = max(O, 0.5)

    # I: intensity alignment (bool)
    I = 1.0 if boosts.get("intensity_match") else 0.0

    # M: match "stabile" (0..1). Usiamo lo stesso che hai per la barra.
    # Se per qualche motivo non è disponibile, fallback neutro 0.5
    try:
        M = _clamp01(float(boosts.get("__match_score_ui", 0.0)))
    except Exception:
        M = 0.0
    if M <= 0.0:
        M = 0.5

    # ---------- weights (Opzione 3 Food-smart) ----------
    # Base
    Wq = 0.28
    Wv = 0.15
    Wf = 0.34
    Wo = 0.10
    Wi = 0.03

    # ✅ se l'utente chiede "qualità/prezzo" aumentiamo il peso Value (V) in modo controllato
    if boosts.get("value_intent"):
        Wq = 0.20  # ridotto da 0.26
        Wv = 0.50  # AUMENTATO da 0.20 a 0.50 - value diventa dominante
        Wf = 0.20  # ridotto da 0.30
        Wo = 0.07  # ridotto da 0.11
        Wi = 0.03  # invariato

    # normalizza pesi base (senza match)
    Wsum = (Wq + Wv + Wf + Wo + Wi) or 1.0
    Wq /= Wsum
    Wv /= Wsum
    Wf /= Wsum
    Wo /= Wsum
    Wi /= Wsum

    overall_base = (
        Wq * Q +
        Wv * V +
        Wf * F +
        Wo * O +
        Wi * I
    )

     # ✅ price_delta: SOLO target mode (abs(pr - tgt)), altrimenti 0.0
    price_delta_out = 0.0
    if price_info.get("mode") == "target" and pr is not None:
        try:
            tgt = float(price_info.get("target", 0.0))
            price_delta_out = abs(float(pr) - tgt)
        except Exception:
            price_delta_out = 0.0
    price_delta_out = round(price_delta_out, 6)

    # Match come fattore (non additivo) → evita ridondanza in UI se mostri anche "Match %"
    match_factor = 0.55 + 0.45 * M  # range: [0.75 .. 1.00]
    composite01 = _clamp01(overall_base * match_factor)

    # --- B1 semantic boost (micro, safe) ---
    sem_boost = 0.0

    body = _norm_lc(str(getattr(row, "body", "") or ""))
    tan  = _norm_lc(str(getattr(row, "tannins", "") or ""))
    acid = _norm_lc(str(getattr(row, "acidity", "") or ""))
    occ  = _norm_lc(str(getattr(row, "occasion", "") or ""))
    tags_raw = _norm_lc(str(getattr(row, "style_tags", "") or ""))
    tags = set([t.strip() for t in re.split(r"[;|,]", tags_raw) if t.strip()])

    def _lvl(x: str) -> str:
        if x in ("alto", "alta", "high"): return "high"
        if x in ("medio", "media", "medium"): return "medium"
        if x in ("basso", "bassa", "low"): return "low"
        return x

    body = _lvl(body)
    tan = _lvl(tan)
    acid = _lvl(acid)

    try:
        alc = float(getattr(row, "alcohol_level", 0.0) or 0.0)
    except Exception:
        alc = 0.0

    # elegant / elegante
    if style_intent and style_intent.get("elegant"):
        if "elegante" in tags:
            sem_boost += 0.06
        elif body == "medium" and tan in ("low", "medium") and acid in ("medium", "high"):
            sem_boost += 0.03
    # meditation / meditazione
    if style_intent and style_intent.get("meditation"):
        if "meditazione" in occ:
            sem_boost += 0.05
        # fallback se nel dataset non c'è "meditazione" come occasion
        if body == "high" and alc >= 14.0:
            sem_boost += 0.01
    # power / potente
    if style_intent and style_intent.get("power"):
        if "strutturato" in tags:
            sem_boost += 0.04
        if body == "high" and tan in ("medium", "high"):
            sem_boost += 0.02
        if alc >= 14.0:
            sem_boost += 0.01

    # fresh / fresco
    if style_intent and style_intent.get("fresh"):
        if ("fresco" in tags) or ("minerale" in tags):
            sem_boost += 0.04
        if body in ("low", "medium") and acid == "high" and alc <= 13.0:
            sem_boost += 0.02

    # cap conservativo
    sem_boost = min(0.06, sem_boost)

    # ✅ debug/components: traccia semantic boost e flags
    try:
        target = dbg_comp.get("components") if isinstance(dbg_comp, dict) and isinstance(dbg_comp.get("components"), dict) else dbg_comp
        if isinstance(target, dict):
            target["__semantic_boost"] = round(float(sem_boost), 4)
            target["__semantic_flags"] = {k: bool(v) for k, v in (style_intent or {}).items()}
    except Exception:
        pass

    # applica al punteggio finale
    composite01 += sem_boost

    # ---- Target proximity bonus (solo se mode=target) ----
    if price_info.get("mode") == "target":
        try:
            delta = float(price_info.get("delta", 1.0))
            if delta > 0:
                delta_norm = min(price_delta_out / delta, 1.0)  # 0=perfetto, 1=al bordo finestra
                proximity_bonus = (1.0 - delta_norm) * 0.06     # peso leggero (tuning)
                composite01 = _clamp01(composite01 + proximity_bonus)
        except Exception:
            pass

    score_out = round(5.0 * composite01, 6)

    if debug_out is not None:
        debug_out.update({
            "__quality_score": round(Q, 6),
            "__value_score": round(V, 6),
            "__food_score": round(F, 6),
            "__other_score": round(O, 6),
            "__intensity_score": round(I, 6),
            "__match_score_ui": round(M, 6),

            "__overall_base_0_1": round(_clamp01(overall_base), 6),
            "__match_factor": round(match_factor, 6),
            "__composite_0_1": round(composite01, 6),

            "__composite_score": score_out,
            "__price_delta": price_delta_out,
            
            "__semantic_boost": round(sem_boost, 4),
            "__semantic_flags": {k: bool(v) for k, v in (style_intent or {}).items()},
        })

    return score_out, price_delta_out

def _tokenize_query(q: str) -> List[str]:
    q = _norm_lc(q)
    toks = re.split(r"[^a-z0-9àèéìòù]+", q)
    stop = {
        "un","una","uno","dei","delle","della","del","di","da","per","con","senza","e","o","ma",
        "il","lo","la","i","gli","le","su","sopra","sotto","oltre","molto","poco","più","meno",
        "vino","vini","dammi","voglio","cerca","trova","consiglia","consigliami"
    }
    out: List[str] = []
    for t in toks:
        t = t.strip()
        if len(t) < 3:
            continue
        if t in stop:
            continue
        out.append(t)
    return out


def _keyword_match_score(row: Any, query: str) -> float:
    toks = _tokenize_query(query)
    if not toks:
        return 0.5
    hay = " ".join([
        _norm_lc(getattr(row, "name", "")),
        _norm_lc(getattr(row, "producer", "")),
        _norm_lc(getattr(row, "denomination", "")),
        _norm_lc(getattr(row, "zone", "")),
        _norm_lc(getattr(row, "region", "")),
        _norm_lc(getattr(row, "description", "")),
    ])
    
    # Check full phrase match first (e.g., "barolo serralunga")
    q_norm = _norm_lc(query)
    if q_norm in hay:
        return 1.0
    
    # Fallback: token-by-token matching
    hits = 0
    for t in toks:
        if t in hay:
            hits += 1
    return min(1.0, max(0.0, hits / max(1, len(toks))))


def _structured_match_components(
    row: Any,
    region: str,
    grapes_req: List[str],
    color_req: Optional[str],
    intensity_req: Optional[str],
    typology_req: Dict[str, Optional[str]],
) -> Dict[str, float]:
    comps: Dict[str, float] = {}

    if region:
        hay = " ".join([
            _norm_lc(getattr(row, "region", "")),
            _norm_lc(getattr(row, "zone", "")),
            _norm_lc(getattr(row, "denomination", "")),
            _norm_lc(getattr(row, "country", "")),
        ])
        comps["region"] = 1.0 if _norm_lc(region) in hay else 0.0

    if grapes_req:
        gv = _norm_lc(getattr(row, "grape_varieties", "")) or _norm_lc(getattr(row, "grapes", ""))
        comps["grapes"] = 1.0 if any((_norm_lc(g) and _norm_lc(g) in gv) for g in grapes_req) else 0.0

    if color_req:
        c = _norm_lc(getattr(row, "color", "")) or _norm_lc(getattr(row, "color_detail", ""))
        req = _norm_lc(color_req)
        if req in ("red","rosso"):
            comps["color"] = 1.0 if ("red" in c or "rosso" in c) else 0.0
        elif req in ("white","bianco"):
            comps["color"] = 1.0 if ("white" in c or "bianco" in c or "giallo" in c) else 0.0
        elif req in ("rose","rosé","rosato"):
            comps["color"] = 1.0 if ("rose" in c or "ros" in c) else 0.0
        elif req in ("orange","aranc","amber"):
            comps["color"] = 1.0 if ("orange" in c or "aranc" in c or "amber" in c) else 0.0
        else:
            comps["color"] = 0.0

    if intensity_req:
        di = derive_intensity(
            _norm(getattr(row, "body", "")),
            _norm(getattr(row, "tannins", "")),
            _norm(getattr(row, "alcohol_level", "")),
        ) or ""
        comps["intensity"] = 1.0 if _norm_lc(di) == _norm_lc(intensity_req) else 0.0

    if typology_req:
        sp_req = typology_req.get("sparkling")
        sw_req = typology_req.get("sweetness")
        tannin_req = typology_req.get("tannin")
        
        if sp_req:
            sp = derive_sparkling(
                _norm(getattr(row, "denomination", "")),
                _norm(getattr(row, "style_tags", "")),
                _norm(getattr(row, "name", "")),
                _norm(getattr(row, "description", "")),
            ) or ""
            comps["sparkling"] = 1.0 if _norm_lc(sp) == _norm_lc(sp_req) else 0.0
        if sw_req:
            sw = normalize_sweetness(_norm(getattr(row, "sweetness", ""))) or ""
            comps["sweetness"] = 1.0 if _norm_lc(sw) == _norm_lc(sw_req) else 0.0
        
        # ✅ Tannin matching (AIS 5 levels)
        if tannin_req:
            wine_tannin = _norm_lc(getattr(row, "tannins", ""))
            # Exact match gets full score
            if tannin_req == "very_high" and wine_tannin == "high":
                comps["tannin"] = 1.0  # very_high = high tannins wines
            elif tannin_req == "high" and wine_tannin == "high":
                comps["tannin"] = 1.0
            elif tannin_req == "medium-high" and wine_tannin in ("medium", "high"):
                comps["tannin"] = 1.0 if wine_tannin == "medium" else 0.8
            elif tannin_req == "low-medium" and wine_tannin in ("low", "medium"):
                comps["tannin"] = 1.0 if wine_tannin == "low" else 0.8
            elif tannin_req == "low" and wine_tannin == "low":
                comps["tannin"] = 1.0
            else:
                comps["tannin"] = 0.0

    return comps


def _match_score_row_explain(
    row: Any,
    query: str,
    region: str,
    grapes_req: List[str],
    color_req: Optional[str],
    intensity_req: Optional[str],
    typology_req: Dict[str, Optional[str]],
    foods_req: List[str],
    occasion_intent: Optional[str] = None,
    prestige_intent: bool = False,
) -> Tuple[float, Dict[str, Any], List[str]]:
    """Compute match score (0..1) + breakdown + short explanation (deterministico).

    Breakdown is meant for UI/debug and is stable across runs.
    """
    # base weights (inside match)
    w_food = 0.50
    w_struct = 0.35
    w_kw = 0.15
    w_occ = 0.0
    w_prestige = 0.0
    w_eleg = 0.0

    comps_struct = _structured_match_components(row, region, grapes_req, color_req, intensity_req, typology_req)
    kw = _keyword_match_score(row, query)
    elegant_intent = bool(re.search(r"\b(elegante|elegant|finezza|raffinato|raffinata)\b", _norm_lc(query)))

    food_present = bool(foods_req)
    food_score = 0.0
    if food_present:
        try:
            food_score = float(food_match_strength(row, foods_req))
        except Exception:
            food_score = 0.0
        food_score = max(0.0, min(1.0, food_score))
        # tiny floor so "food requested" is still visible in UI even when no pairing is found
        if food_score <= 0.0:
            food_score = 0.05

    occasion_score = 0.0
    if occasion_intent:
        occ_row = _norm_lc(getattr(row, "occasion", ""))
        occ_req = _norm_lc(occasion_intent)
        if occ_row and occ_req:
            if occ_req in occ_row:
                occasion_score = 1.0
            elif occ_req == "cena" and "cena importante" in occ_row:
                occasion_score = 1.0
            elif occ_req == "cena importante" and occ_row == "cena":
                occasion_score = 0.7

    prestige_score = _prestige_match_score(row) if prestige_intent else 0.0

    elegant_score = 0.0
    if elegant_intent:
        tags_raw = _norm_lc(getattr(row, "style_tags", ""))
        tags = set([t.strip() for t in re.split(r"[;|,]", tags_raw) if t.strip()])
        body_row = _norm_lc(getattr(row, "body", ""))
        tan_row = _norm_lc(getattr(row, "tannins", ""))
        acid_row = _norm_lc(getattr(row, "acidity", ""))

        if "elegante" in tags:
            elegant_score = 1.0
        elif any(tok in tags for tok in ["finezza", "raffinato", "raffinata", "minerale", "fresco", "teso", "equilibrato", "salino", "agrumi"]):
            elegant_score = 0.75
        elif body_row in ("medium", "medio", "media") and tan_row in ("low", "medium", "basso", "bassa", "medio", "media") and acid_row in ("medium", "high", "media", "alta", "alto"):
            elegant_score = 0.45

    struct_score = 0.0
    if comps_struct:
        struct_score = sum(comps_struct.values()) / float(len(comps_struct))

    # dynamic redistribution
    if food_present and prestige_intent:
        # premium bottle within the right family
        w_food = 0.55
        w_prestige = 0.30
        w_struct = 0.10
        w_kw = 0.05
        w_occ = 0.0
    elif prestige_intent and occasion_intent:
        w_prestige = 0.50
        w_occ = 0.25
        w_struct = 0.15
        w_kw = 0.10
        w_food = 0.0
    elif prestige_intent:
        # default fast path for "vino importante"
        w_prestige = 0.70
        w_struct = 0.15
        w_kw = 0.15
        w_food = 0.0
        w_occ = 0.0
    elif food_present and occasion_intent:
        # Food stays primary, but occasion must still matter.
        w_food = 0.55
        w_occ = 0.20
        w_struct = 0.15
        w_kw = 0.10
        w_prestige = 0.0
        w_eleg = 0.0
    elif food_present:
        # Food becomes dominant when explicitly requested
        w_food = 0.70
        w_struct = 0.20
        w_kw = 0.10
        w_occ = 0.0
        w_prestige = 0.0
        w_eleg = 0.0
    elif occasion_intent and elegant_intent:
        # Combined branch: keep both occasion and elegance active.
        w_occ = 0.40
        w_eleg = 0.30
        w_struct = 0.20
        w_kw = 0.10
        w_food = 0.0
        w_prestige = 0.0
    elif occasion_intent:
        # Occasion becomes a primary driver when explicitly requested and no food is present
        w_occ = 0.65
        w_struct = 0.20
        w_kw = 0.15
        w_food = 0.0
        w_prestige = 0.0
        w_eleg = 0.0
    elif elegant_intent:
        # Elegant should not be read only as generic structure.
        w_eleg = 0.35
        w_struct = 0.40
        w_kw = 0.25
        w_food = 0.0
        w_occ = 0.0
        w_prestige = 0.0
    else:
        # redistribute food weight
        extra = w_food
        denom = (w_struct + w_kw) or 1.0
        w_struct = w_struct + extra * (w_struct / denom)
        w_kw = w_kw + extra * (w_kw / denom)
        w_food = 0.0
        w_occ = 0.0
        w_prestige = 0.0
        w_eleg = 0.0

    if not comps_struct:
        w_kw += w_struct
        w_struct = 0.0

    total = (w_food * food_score) + (w_struct * struct_score) + (w_kw * kw) + (w_occ * occasion_score) + (w_prestige * prestige_score) + (w_eleg * elegant_score)
    wsum = w_food + w_struct + w_kw + w_occ + w_prestige + w_eleg
    if wsum <= 0:
        score = 0.5
    else:
        score = max(0.0, min(1.0, total / wsum))

    # ---- breakdown (normalized weights + contributions) ----
    w_food_n = (w_food / wsum) if wsum > 0 else 0.0
    w_struct_n = (w_struct / wsum) if wsum > 0 else 0.0
    w_kw_n = (w_kw / wsum) if wsum > 0 else 0.0
    w_occ_n = (w_occ / wsum) if wsum > 0 else 0.0
    w_prestige_n = (w_prestige / wsum) if wsum > 0 else 0.0
    w_eleg_n = (w_eleg / wsum) if wsum > 0 else 0.0

    breakdown: Dict[str, Any] = {
        "food": {"w": round(w_food_n, 4), "s": round(food_score, 4), "c": round(w_food_n * food_score, 4)},
        "structured": {"w": round(w_struct_n, 4), "s": round(struct_score, 4), "c": round(w_struct_n * struct_score, 4)},
        "keyword": {"w": round(w_kw_n, 4), "s": round(kw, 4), "c": round(w_kw_n * kw, 4)},
        "occasion": {"w": round(w_occ_n, 4), "s": round(occasion_score, 4), "c": round(w_occ_n * occasion_score, 4)},
        "prestige": {"w": round(w_prestige_n, 4), "s": round(prestige_score, 4), "c": round(w_prestige_n * prestige_score, 4)},
        "elegance": {"w": round(w_eleg_n, 4), "s": round(elegant_score, 4), "c": round(w_eleg_n * elegant_score, 4)},
        "structured_components": comps_struct or {},
        "foods_req": foods_req or [],
        "occasion_req": 1.0 if occasion_intent else 0.0,
        "prestige_req": 1.0 if prestige_intent else 0.0,
        "elegance_req": 1.0 if elegant_intent else 0.0,
        "region_req": region or None,
        "grapes_req": grapes_req or [],
        "color_req": color_req or None,
        "intensity_req": intensity_req or None,
        "typology_req": typology_req or {},
    }

    # ---- short explanation (max 3 bullets) ----
    expl: List[str] = []

    if prestige_intent:
        if prestige_score >= 0.85:
            expl.append("👑 Bottiglia di rilievo / premium")
        elif prestige_score >= 0.55:
            expl.append("👑 Profilo premium coerente")
        else:
            expl.append("👑 Intent premium richiesto")

    if elegant_intent:
        if elegant_score >= 0.95:
            expl.append("🪶 Profilo elegante esplicito")
        elif elegant_score >= 0.7:
            expl.append("🪶 Profilo di finezza coerente")
        elif elegant_score > 0.0:
            expl.append("🪶 Eleganza derivata dal profilo")

    if food_present:
        if food_score >= 0.95:
            expl.append("🍽 Abbinamento: match diretto con i tuoi cibi")
        elif food_score >= 0.45:
            expl.append("🍽 Abbinamento: match parziale (pairing non perfetto)")
        else:
            expl.append("🍽 Abbinamento: richiesto ma non trovato nei pairing (fallback)")

    if occasion_intent:
        if occasion_score >= 0.95:
            expl.append("🎯 Occasione: centrato per l'occasione richiesta")
        else:
            expl.append("🎯 Occasione: richiesta ma non centrale nei metadati")

    if comps_struct:
        hits = [k for k, v in comps_struct.items() if float(v) >= 0.95]
        if hits:
            expl.append("🔎 Match strutturato: " + ", ".join(hits[:3]))
        else:
            expl.append("🔎 Match strutturato: parziale")

    if kw >= 0.70:
        expl.append("📝 Match testo: parole chiave forti")
    elif kw <= 0.30:
        expl.append("📝 Match testo: debole")

    expl = expl[:3]

    return score, breakdown, expl


def _ui_highlights_for_relevance_v2(components: Dict[str, Any]) -> List[str]:
    """
    Explainability light (UI): massimo 3 badge, deterministico. Solo per relevance_v2.
    A1 Budget highlights:
      - 3 slot totali
      - priorità: Match (se forte) → (Quality/Value) → (Occasione/Intensità)
      - evita "riempimento" con badge deboli
    Compatibile con:
      - chiavi "M/Q/V/O/I"
      - chiavi "__match_score_ui/__quality_score/__value_score/__other_score/__intensity_score"
    """
    def _f(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = components.get(k, None)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
        return float(default)

    M = _f("M", "__match_score_ui", default=0.0)
    Q = _f("Q", "__quality_score", default=0.0)
    V = _f("V", "__value_score", default=0.0)
    O = _f("O", "__other_score", default=0.0)
    I = _f("I", "__intensity_score", default=0.0)

    picks: List[str] = []
    BUDGET = 2

    def _add(label: str) -> None:
        if len(picks) < BUDGET and label not in picks:
            picks.append(label)

    # 1) 🍽 Match (solo se significativo)
    if M >= 0.90:
        _add("🍽 Abbinamento centrato")
    elif M >= 0.60:
        _add("🍽 Buon abbinamento")

    # 2) ⭐ Qualità (badge) + 💰 Value (badge) con budget/peso
    q_label = None
    q_strength = 0.0
    if Q >= 0.85:
        q_label = "⭐ Alta qualità"
        q_strength = 2.0
    elif Q >= 0.75:
        q_label = "⭐ Qualità sopra la media"
        q_strength = 1.0

    v_label = None
    v_strength = 0.0
    if V >= 0.85:
        v_label = "💰 Ottimo rapporto qualità/prezzo"
        v_strength = 2.0
    elif V >= 0.75:
        v_label = "💰 Buon rapporto qualità/prezzo"
        v_strength = 1.0

    # Se entrambi forti (>=2.0) aggiungili entrambi.
    # Se uno è forte e l'altro borderline, preferisci il forte.
    # Se entrambi borderline, prendi quello con score più alto.
    if q_label and v_label:
        if q_strength >= 2.0 and v_strength >= 2.0:
            _add(q_label)
            _add(v_label)
        elif q_strength != v_strength:
            _add(q_label if q_strength > v_strength else v_label)
        else:
            _add(q_label if Q >= V else v_label)
    else:
        if q_label:
            _add(q_label)
        if v_label:
            _add(v_label)

    # 3) 🎯 Occasione e 🔥 Intensità solo se resta budget
    if O > 0.0 and len(picks) < BUDGET:
        _add("🎯 Ideale per l'occasione")

    if I > 0.0 and len(picks) < BUDGET:
        _add("🔥 Intensità coerente")

    return picks[:BUDGET]

def _match_score_row(
    row: Any,
    query: str,
    region: str,
    grapes_req: List[str],
    color_req: Optional[str],
    intensity_req: Optional[str],
    typology_req: Dict[str, Optional[str]],
    foods_req: List[str],
    occasion_intent: Optional[str] = None,
    prestige_intent: bool = False,
) -> float:
    score, _, _ = _match_score_row_explain(row, query, region, grapes_req, color_req, intensity_req, typology_req, foods_req, occasion_intent, prestige_intent)
    return score




def _normalize_color_detail_output(row: Any) -> str:
    """Normalize color_detail for API output only.

    Prevents non-color values like sweetness/sparkling from leaking into color_detail.
    """
    raw_cd = _norm_lc(getattr(row, "color_detail", ""))
    raw_c = _norm_lc(getattr(row, "color", ""))
    sparkling = _norm_lc(getattr(row, "sparkling", ""))
    sweetness = _norm_lc(getattr(row, "sweetness", ""))
    hay = " ".join([raw_cd, raw_c])

    if any(k in hay for k in ["ruby_red", "rosso rubino", "rubino"]):
        return "ruby_red"
    if any(k in hay for k in ["straw_yellow", "giallo paglierino", "paglierino"]):
        return "straw_yellow"
    if any(k in hay for k in ["golden_yellow", "giallo dorato", "dorato"]):
        return "golden_yellow"
    if any(k in hay for k in ["rose", "rosé", "rosato", "rosa", "salmon"]):
        return "rose"
    if any(k in hay for k in ["red", "rosso"]):
        return "red"
    if any(k in hay for k in ["white", "bianco"]):
        return "white"

    # If color_detail was polluted by sweetness / sparkling, fall back to best-effort color.
    if sweetness in ("dolce", "sweet"):
        return "white"
    if sparkling in ("spumante", "frizzante"):
        return "white"

    return ""

def _build_wine_card(row: Any, rank: int, score: float, price_delta: float, match_score: float = 0.0) -> Dict[str, Any]:
    # prezzo mostrato: preferiamo price_avg, fallback price_min
    pr = _price_effective(row)
    price_str = ""
    if pr is not None:
        # format semplice
        price_str = f"{pr:.2f}"

    # tags: usa style_tags + color_detail normalizzato + body (senza duplicare troppo)
    tags_parts: List[str] = []
    st = _norm(getattr(row, "style_tags", ""))
    if st:
        tags_parts.append(st)
    normalized_color_detail = _normalize_color_detail_output(row)
    if normalized_color_detail:
        tags_parts.append(normalized_color_detail)
    b = _norm(getattr(row, "body", ""))
    if b:
        tags_parts.append(b)

    tags = ", ".join([t for t in tags_parts if t])

    card: Dict[str, Any] = {
        # base
        "id": _norm(getattr(row, "id", "")),
        "name": _norm(getattr(row, "name", "")),
        "price": price_str,
        "reason": _norm(getattr(row, "description", "")),  # fallback utile; puoi cambiarlo dopo se vuoi
        "purchase_url": _norm(getattr(row, "purchase_url", "")),
        "tags": tags,
        "rank": rank,
        "score": round(float(score), 4),

        "match_score": round(float(max(0.0, min(1.0, match_score))), 4),
        "__match_score": round(float(max(0.0, min(1.0, match_score))), 4),
        # rating_overall/popularity non presenti nel CSV: lasciamo 0.0 (stabile per iOS)
        "rating_overall": 0.0,
        "popularity": 0.0,

        # new fields già previsti dalla tua app
        "producer": _norm(getattr(row, "producer", "")),
        "country": _norm(getattr(row, "country", "")),
        "region": _norm(getattr(row, "region", "")),
        "zone": _norm(getattr(row, "zone", "")),
        "denomination": _norm(getattr(row, "denomination", "")),
        "vintage": _norm(getattr(row, "vintage", "")),
        "food_pairings": _norm(getattr(row, "food_pairings", "")),
        "occasion": _norm(getattr(row, "occasion", "")),
    }

    # opzionali: quality/balance/persistence/tannins/acidity
    for k in ["quality", "balance", "persistence", "tannins", "acidity"]:
        v = _norm(getattr(row, k, ""))
        if v:
            card[k] = v

    normalized_color_detail = normalized_color_detail or _normalize_color_detail_output(row)
    if normalized_color_detail:
        card["color_detail"] = normalized_color_detail

    # A/B/D: riportiamo anche i campi "veri" del dataset + derivati
    gv = _norm(getattr(row, "grape_varieties", ""))
    if gv:
        card["grapes"] = gv  # name stabile verso iOS (grapes)

    # intensity derivata
    intensity = derive_intensity(
        _norm(getattr(row, "body", "")),
        _norm(getattr(row, "tannins", "")),
        _norm(getattr(row, "alcohol_level", "")),
    )
    if intensity:
        card["intensity"] = intensity

    # sparkling derivato (PRIMA di usarlo in freshness)
    sparkling = derive_sparkling(
        _norm(getattr(row, "denomination", "")),
        _norm(getattr(row, "style_tags", "")),
        _norm(getattr(row, "name", "")),
        _norm(getattr(row, "description", "")),
    )
    if sparkling:
        card["sparkling"] = sparkling

    # freshness derivata (per barra UI Screen 4)
    freshness = derive_freshness(
        _norm(getattr(row, "acidity", "")),
        sparkling or "",
        _norm(getattr(row, "alcohol_level", "")),
    )
    if freshness:
        card["freshness"] = freshness

    # sweetness normalizzata
    sw = normalize_sweetness(_norm(getattr(row, "sweetness", "")))
    if sw:
        card["sweetness"] = sw

    # aroma: non essendoci lista strutturata, non la mettiamo come campo "aromas"
    # (il requisito "sentori" è gestito come filtro su description)

    card["__price_delta"] = round(float(price_delta), 4)

    
    # UI Helpers: aromi icons, value badge, mock data
    aromas_text = _norm(getattr(row, "aromas", ""))
    if aromas_text:
        card["aroma_icons"] = get_aroma_icons(aromas_text)
    
    # Quality-based mock data per UI
    quality_val = _parse_float_maybe(_norm(getattr(row, "quality", "")))
    if quality_val > 0:
        card["reviews_count"] = get_mock_reviews_count(quality_val)
        card["critic_score"] = get_mock_critic_score(quality_val)
    
    return card

def _apply_sort(cards: List[Dict[str, Any]], sort: str, value_intent: bool = False) -> List[Dict[str, Any]]:
    sort = sort or "relevance"
    
    if sort == "match":
        # match desc, poi food/prestige contribution, poi score desc, poi price_delta asc
        return sorted(
            cards,
            key=lambda c: (
                -float(c.get("__match_score", 0.0)),
                -float((c.get("match_breakdown", {}) or {}).get("food", 0.0)),
                -float((c.get("match_breakdown", {}) or {}).get("prestige", 0.0)),
                -float(c.get("score", 0.0)),
                float(c.get("__price_delta", 0.0)),
            ),
        )

    if sort == "price_asc":
        return sorted(cards, key=lambda c: _parse_float_maybe(c.get("price", "")) or float("inf"))
    if sort == "price_desc":
        return sorted(cards, key=lambda c: _parse_float_maybe(c.get("price", "")) or -1.0, reverse=True)
    if sort == "rating":
        return sorted(cards, key=lambda c: float(c.get("rating_overall", 0.0)), reverse=True)
    if sort == "popular":
        return sorted(cards, key=lambda c: float(c.get("popularity", 0.0)), reverse=True)

    # relevance (default)
    if value_intent and sort == "relevance":
    # In modalità qualità/prezzo usiamo solo score composito (già include V in relevance_v2)
        return sorted(
            cards,
            key=lambda c: (-float(c.get("score", 0.0)),)
    )

    # relevance normale (come prima)
    return sorted(cards, key=lambda c: (-float(c.get("score", 0.0)), float(c.get("__price_delta", 0.0))))


# =========================
# Core search
# =========================


def _is_test_wine_row(r: Any) -> bool:
    """Heuristic to hide dataset fixtures/placeholder wines from normal users."""
    name = _norm_lc(getattr(r, "name", ""))
    producer = _norm_lc(getattr(r, "producer", ""))
    region = _norm_lc(getattr(r, "region", ""))
    zone = _norm_lc(getattr(r, "zone", ""))

    if name.startswith("wine test"):
        return True
    if producer == "sommelierai dataset":
        return True
    if region == "test region" or zone == "test zone":
        return True
    return False

def _explain_mode_b(card: Dict[str, Any], dbg_comp: Optional[Dict[str, Any]], mexpl: Any) -> List[str]:
    """
    Explain Mode B (prodotto): massimo 3 righe, leggibili e UI-ready.
    Obiettivi:
    - non duplicare ui_highlights
    - niente score numerici
    - spiegazioni corte, coerenti col ranking
    """
    out: List[str] = []

    comps = dbg_comp.get("components") if isinstance(dbg_comp, dict) and isinstance(dbg_comp.get("components", None), dict) else (dbg_comp or {})

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(comps.get(key, default) or default)
        except Exception:
            return float(default)

    m = _f("__match_score_ui", _f("__match_score", float(card.get("__match_score", card.get("match_score", 0.0)) or 0.0)))
    q = _f("__quality_score", 0.0)
    v = _f("__value_score", 0.0)
    o = _f("__other_score", 0.0)
    i = _f("__intensity_score", 0.0)
    s = _f("__semantic_boost", 0.0)

    # 1) Match / semantic intent
    if s >= 0.05:
        out.append("Perfetto per lo stile che cerchi")
    elif m >= 0.90:
        out.append("Perfetto per la tua richiesta")
    elif m >= 0.60:
        out.append("Buon match per la tua richiesta")

    # 2) Qualità
    if q >= 0.85:
        out.append("Qualità alta e profilo convincente")
    elif q >= 0.75:
        out.append("Qualità sopra la media")

    # 3) Value
    if len(out) < 3:
        if v >= 0.85:
            out.append("Ottimo rapporto qualità/prezzo")
        elif v >= 0.75:
            out.append("Buon rapporto qualità/prezzo")

    # 4) Occasione / intensità come fallback
    if len(out) < 3 and o > 0.0:
        out.append("Particolarmente adatto all'occasione")
    if len(out) < 3 and i > 0.0:
        out.append("Profilo intenso e coerente con la richiesta")

    # 5) fallback su match explanation solo se siamo ancora corti
    if len(out) < 3 and not any("richiesta" in x.lower() for x in out):
        if isinstance(mexpl, list) and mexpl:
            txt = str(mexpl[0]).strip()
            if "parole chiave forti" in txt:
                out.append("Molto coerente con la tua ricerca")
            elif txt:
                out.append(txt)
        elif isinstance(mexpl, str) and mexpl.strip():
            out.append(mexpl.strip())
        
    # de-dup + cap
    dedup: List[str] = []
    for sline in out:
        sline = (sline or "").strip()
        if sline and sline not in dedup:
            dedup.append(sline)

    return dedup[:3]

def run_search(
    query: str,
    sort: str = "relevance",
    limit: int = MAX_RESULTS_DEFAULT,
    include_test: bool = False,
    debug: bool = False,
    explain: bool = False,  # ✅ Explain Mode B toggle
) -> Dict[str, Any]:    
    import time
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    df = get_wines_df()
    timings["load_df"] = round(time.perf_counter() - t0, 6)
    t0 = time.perf_counter()
    
    limit = _clamp(int(limit or MAX_RESULTS_DEFAULT), 1, MAX_RESULTS_CAP)
    q = _norm(query)
    
    # --- LLM Intent Layer Step 1: Parse ---
    llm_intent = parse_intent_with_llm(q)
    
    price_info = parse_price(q)
    region = parse_region(q) or llm_intent.get("region")

    # A/B/D requests
    grapes_req = parse_grapes(q) or llm_intent.get("grapes", [])
    aromas_req = parse_aromas(q)
    intensity_req = parse_intensity_request(q)
    typology_req = parse_typology_request(q)
    
    # Foods: unione rule-based + LLM
    foods_rule = parse_food_request(q)
    foods_llm = llm_intent.get("foods", [])
    foods_req = sorted(set(foods_rule) | set(foods_llm))

    color_req = parse_color_request(q) or llm_intent.get("color")

    # VALUE intent (solo se richiesto dall'utente)
    value_intent = parse_value_intent(q) or llm_intent.get("value_intent", False)

    # ✅ Opzione 2: se l'utente chiede qualità/prezzo e non ha scelto un sort specifico, usa relevance_v2
    if value_intent and (not sort or sort == "relevance"):
        sort = "relevance_v2"

    style_intent = parse_style_intent(q)
    
    # Occasion: LLM arricchisce se rule-based non ha trovato nulla
    occasion_intent = parse_occasion_intent(q) or llm_intent.get("occasion")
    # DEBUG: check Barolo pre-filter
    if "barolo" in q.lower():
        barolo_pre = df[df["name"].str.contains("Barolo", case=False, na=False)]
    prestige_intent = parse_prestige_intent(q) or llm_intent.get("prestige_intent", False)
    elegance_intent = (
        bool(re.search(r"\b(elegante|elegant|finezza|raffinato|raffinata)\b", _norm_lc(q)))
        or llm_intent.get("elegant_intent", False)
    )

    filtered = df

    # price filter
    filtered = _filter_by_price(filtered, price_info)

    # region filter (match in region/zone/denomination/country) - OR logic
    if region:
        mask = pd.Series([False] * len(filtered), index=filtered.index)
        for col in ["region", "zone", "denomination", "country"]:
            if col in filtered.columns:
                v = _norm_lc(region)
                # ✅ Normalize country aliases (italia→italy, francia→france, etc.)
                if col == "country":
                    country_aliases = {
                        "italia": "italy",
                        "francia": "france",
                        "spagna": "spain",
                        "germania": "germany",
                        "portogallo": "portugal"
                    }
                    v = country_aliases.get(v, v)
                mask |= filtered[col].astype(str).str.lower().str.contains(v, na=False)
        filtered = filtered.loc[mask]

    # Tannin → color rosso implicito (tannino è rilevante solo per rossi)
    if typology_req.get("tannin") and not color_req:
        color_req = "rosso"

    # color filter (bianco/rosso/rosato)
    filtered = _filter_by_color(filtered, color_req)

    # Keyword filter: se query matcha esattamente una denominazione, mostra SOLO quella
    keyword_matches = []
    for idx, row in filtered.iterrows():
        name_lower = str(row.get('name', '')).lower()
        denom_lower = str(row.get('denomination', '')).lower()
        q_lower = q.lower().strip()
        
        # Match esatto su name o denomination
        if q_lower in name_lower or q_lower in denom_lower:
            keyword_matches.append(idx)
    
    # Se ci sono match esatti, mostra SOLO quelli
    if len(keyword_matches) > 0 and len(keyword_matches) < len(filtered):
        filtered = filtered.loc[keyword_matches]
    # A/B/D filters
    filtered = _filter_new_A_B_D(filtered, grapes_req, aromas_req, intensity_req, typology_req)
    timings["filters"] = round(time.perf_counter() - t0, 6)
    t0 = time.perf_counter()

    # extreme price within current filters
    if price_info.get("mode") == "extreme":
        prices = filtered["price_avg"].map(_parse_float_maybe)
        prices_min = filtered["price_min"].map(_parse_float_maybe)
        effective = prices.copy()
        effective[effective.isna()] = prices_min[effective.isna()]
        if len(filtered) > 0:
            if price_info.get("extreme") == "max":
                idx = effective.idxmax()
            else:
                idx = effective.idxmin()
            filtered = filtered.loc[[idx]]

    rows = list(filtered.itertuples(index=False))

    # hide fixture/test wines unless explicitly requested
    if not include_test:
        rows = [r for r in rows if not _is_test_wine_row(r)]

    # debug var sempre definita
    _debug_sort_after_override = sort

    # Opzione 2
    if value_intent and sort == "relevance":
        sort = "relevance_v2"

    # aggiorna dopo eventuale override
    _debug_sort_after_override = sort

    # QUESTE DEVONO ESSERE FUORI DA OGNI IF
    scored: List[Dict[str, Any]] = []
    debug_map: Dict[str, Any] = {}

    for r in rows:
        dbg_comp: Dict[str, Any] = {}

        boosts = {
            "grape_match": bool(grapes_req),
            "aroma_match": bool(aromas_req),
            "intensity_match": bool(intensity_req),
            "typology_match": bool(typology_req.get("sparkling") or typology_req.get("sweetness")),
            "food_match": food_match(r, foods_req),
            "foods_present": bool(foods_req),
            "foods_req": foods_req,
            "occasion_intent": occasion_intent,
            "value_intent": value_intent,
        }

        # ✅ UI match score (0..1) + available to relevance_v2 composite as M
        mscore, mbd, mexpl = _match_score_row_explain(r, q, region, grapes_req, color_req, intensity_req, typology_req, foods_req, occasion_intent, prestige_intent)
        boosts["__match_score_ui"] = mscore
        
        # ✅ Flatten match_breakdown per iOS (converte i dict in valori numerici)
        flatten_mbd = {
        k: (
        float(v.get("c", 0.0)) if isinstance(v, dict)
        else (float(v) if isinstance(v, (int, float)) else (_parse_float_maybe(v) or 0.0))
        )
        for k, v in (mbd or {}).items()
        }
        
        if sort == "relevance_a9v1":
            s, pdlt = _score_row_a9v1(r, price_info, boosts)
        elif sort == "relevance_a9v2":
            s, pdlt = _score_row_a9v1(r, price_info, boosts, style_intent=style_intent)
        elif sort == "relevance_v2":
            # Always compute composite components (for UI highlights). Debugger may also reuse them.
            s, pdlt = _score_row_a9v2_composite(r, price_info, boosts, style_intent=style_intent, debug_out=dbg_comp)

        else:
            s, pdlt = _score_row(r, price_info, boosts, value_intent=value_intent)

        card = _build_wine_card(r, rank=0, score=s, price_delta=pdlt, match_score=mscore)
        
        # ✅ Assegna match_breakdown flattenato (MAI il dict originale)
        card["match_breakdown"] = flatten_mbd
        
        # ✅ Debug fields (B): only when debug=true (do NOT change default payload/UX)
        if debug:
            # Normalize components to 0..1 for easy comparison
            q_raw_dbg = _score_quality(r)
            Q_dbg = _clamp01(q_raw_dbg / 5.0)

            pr_dbg = _price_effective(r)
            V_dbg = 0.0
            if pr_dbg is not None and pr_dbg > 0:
                qv_dbg = _parse_float_maybe(getattr(r, "quality", ""))
                q_for_value_dbg = qv_dbg if qv_dbg is not None else q_raw_dbg
                v_raw_dbg = (q_for_value_dbg + 0.75) / math.log(pr_dbg + 2.0)
                V_dbg = _clamp01(v_raw_dbg / 2.0)
            
            card["__quality_score"] = round(float(Q_dbg), 6)
            card["__value_score"] = round(float(V_dbg), 6)
            card["__final_score"] = card.get("score")  # score used for ordering (avoid double counting)
            card["__semantic_boost"] = round(float(card.get("__semantic_boost", 0.0) or 0.0), 6)
            card["__components"] = dbg_comp if isinstance(dbg_comp, dict) else {}
            # card["__match_factor"] = round(float(match_factor), 6)

            # include explainability when debugging (già flattenato)
            card["match_explanation"] = mexpl
               
        if sort == "relevance_v2":
            card["match_explanation"] = mexpl
            try:
                # ✅ Passiamo dbg_comp completo: contiene __quality_score/__value_score/__match_score_ui...
                card["ui_highlights"] = _ui_highlights_for_relevance_v2(dbg_comp or {})
            except Exception:
                card["ui_highlights"] = []

        # ✅ Explain Mode B: solo se richiesto (default OFF)
        if explain:
            try:
                card["explain"] = list(_explain_mode_b(card, dbg_comp, mexpl))                
            except Exception:
                card["explain"] = []

        scored.append(card)

        if debug:
            cid = card.get("id") or _norm(getattr(r, "id", ""))
            dbg = {
                "id": cid,
                "name": card.get("name"),
                "sort": sort,
                "score": card.get("score"),
                "match_score": card.get("__match_score"),
                "price": card.get("price"),
                "__price_delta": card.get("__price_delta"),
                "filters": {
                    "foods_req": foods_req,
                    "prestige_req": prestige_intent,
                    "region_req": region or None,
                    "grapes_req": grapes_req,
                    "color_req": color_req,
                    "intensity_req": intensity_req,
                    "typology_req": typology_req,
                },
                # ✅ debug-only: campi raw del dataset (per semantic parsing & tuning)
                "row_fields": {
                    "color": _norm(getattr(r, "color", "")),
                    "body": _norm(getattr(r, "body", "")),
                    "tannins": _norm(getattr(r, "tannins", "")),
                    "acidity": _norm(getattr(r, "acidity", "")),
                    "alcohol_level": _norm(getattr(r, "alcohol_level", "")),
                    "sweetness": _norm(getattr(r, "sweetness", "")),
                    "style_tags": _norm(getattr(r, "style_tags", "")),
                    "occasion": _norm(getattr(r, "occasion", "")),
                },
            }
            
            dbg["components"] = dbg_comp if isinstance(dbg_comp, dict) else {}

            if sort == "relevance_v2":
                # composite components if available
                try:
                    dbg["composite"] = dbg_comp
                    # ✅ NON estrarre .get("components"): dbg_comp è flat e contiene già tutto
                    dbg["components"] = dbg_comp or {}
                except NameError:
                    dbg["composite"] = {}
                    dbg["components"] = {}
                dbg["match_breakdown"] = flatten_mbd  # ✅ flattenato anche qui
                dbg["match_explanation"] = mexpl
            else:
                dbg["match_breakdown"] = flatten_mbd  # ✅ flattenato anche qui
            debug_map[cid] = dbg
            timings["score"] = round(time.perf_counter() - t0, 6)
            t0 = time.perf_counter()
    
    # Salva count totale PRIMA del limit
    total_count = len(scored)
    sorted_cards = _apply_sort(scored, sort, value_intent=value_intent)[:limit]
    timings["sort"] = round(time.perf_counter() - t0, 6)
    t0 = time.perf_counter()
    for i, c in enumerate(sorted_cards, start=1):
        c["rank"] = i
    
    # --- LLM Intent Layer Step 2: Explain ---
    # Genera reason personalizzata per il vino top-ranked
    if sorted_cards:
        # Estrai tannin_req dalla query (parser veloce)
        tannin_req = None
        q_lc = _norm_lc(q)
        # Pattern LOW: negazioni esplicite (poco/non/senza + tannico)
        if re.search(r"\b(poco|non|senza)\s+(tannic|tannin)", q_lc):
            tannin_req = "low"
        # Pattern LOW: tannini + aggettivo morbido/basso/delicato
        elif re.search(r"\btannin[oi]\s+(bass[oi]|morbid[oi]|delicat[oi]|legg?er[oi])\b", q_lc):
            tannin_req = "low"
        # Pattern HIGH: tannico/astringente (dopo aver escluso i LOW)
        elif re.search(r"\b(tannic[oi]|astringent[ei])\b", q_lc):
            tannin_req = "high"
        # Pattern HIGH: tannini + aggettivo forte (alti/importanti/marcati)
        elif re.search(r"\btannin[oi]\s+(alt[oi]|important[ei]|marcati|evident[ei])\b", q_lc):
            tannin_req = "high"
        
        # Raccogliere segnali attivi dal ranking
        active_signals = {
            "color": color_req,
            "prestige_intent": prestige_intent,
            "elegant_intent": elegance_intent,
            "occasion": occasion_intent,
            "foods": foods_req,
            "style": style_intent.get("style") if style_intent else None,
            "tannin_req": tannin_req,
            "intensity_req": intensity_req,
            "region": region,
            "grapes": grapes_req,
            "sparkling": typology_req.get("sparkling") if typology_req else None,
            "sweetness": typology_req.get("sweetness") if typology_req else None,
            "value_intent": value_intent,  # ✅ AGGIUNTO
        }
        
        # Info vino top per contestualizzare la reason
        top_wine_info = {
            "name": sorted_cards[0].get("name"),
            "region": sorted_cards[0].get("region"),
        }
        
        # Genera reason personalizzata
        personalized_reason = generate_personalized_reason(
            query=q,
            active_signals=active_signals,
            top_wine=top_wine_info
        )
        
        # Sostituisci reason statica con quella personalizzata
        sorted_cards[0]["reason"] = personalized_reason
        
        # ✅ UI Badge: "Ottimo Valore" per tutti i vini
        for card in sorted_cards:
            card["show_value_badge"] = should_show_value_badge(card, active_signals)
    
    timings["llm_explain"] = round(time.perf_counter() - t0, 6)
    t0 = time.perf_counter()

    meta = {
        "build_id": BUILD_ID,
        "query": q,
        "sort": sort,
        "limit": limit,
        "filters": {
            "price": price_info,
            "region": region,
            "color": color_req,
            "grapes": grapes_req,
            "aromas": aromas_req,
            "intensity": intensity_req,
            "typology": typology_req,
            "foods": foods_req,
            "occasion_intent": bool(occasion_intent),
            "prestige_intent": prestige_intent,
            "elegance_intent": elegance_intent,
            "value_intent": value_intent,
        },
        "count": len(sorted_cards),
        "total_count": total_count,
        "timestamp": int(_now()),
    }
    if explain:
        meta["explain_mode"] = "B"

    if debug:
        meta["__debug_sort_after_override"] = _debug_sort_after_override

    if debug:
        debug_rows = [
            {**({"rank": c.get("rank")}), **(debug_map.get(c.get("id", ""), {}))}
            for c in sorted_cards
        ]

        # ✅ Delta vs top: utile per tuning (perché #1 batte #2, #3, ...)
        delta_vs_top: List[Dict[str, Any]] = []
        if debug_rows:
            top = debug_rows[0]
            top_comp = (top.get("composite") or {})
            top_contrib = (top_comp.get("contrib") or {})
            top_c01 = float(top_comp.get("composite_0_1") or 0.0)

            for r in debug_rows[1:]:
                comp = (r.get("composite") or {})
                contrib = (comp.get("contrib") or {})
                c01 = float(comp.get("composite_0_1") or 0.0)

                keys = set(top_contrib.keys()) | set(contrib.keys())
                contrib_delta = {k: float((top_contrib.get(k) or 0.0) - (contrib.get(k) or 0.0)) for k in sorted(keys)}

                delta_vs_top.append({
                    "rank": r.get("rank"),
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "delta_composite_0_1": top_c01 - c01,
                    "delta_contrib": contrib_delta,
                })

        # ---- Delta vs Top (composite + breakdown) ----
        delta_vs_top = []

        if debug_rows:
            top = debug_rows[0]
            top_comp = (top.get("composite") or {})
            top_c01 = float(top_comp.get("__composite_0_1", 0.0))

            for r in debug_rows[1:]:
                comp = (r.get("composite") or {})
                c01 = float(comp.get("__composite_0_1", 0.0))

                delta = round(top_c01 - c01, 6)

                delta_vs_top.append({
                    "rank": r.get("rank"),
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "delta_composite_0_1": delta,
                    "delta_contrib": {
                        "quality": round(float(top_comp.get("__quality_score", 0.0)) - float(comp.get("__quality_score", 0.0)), 6),
                        "value": round(float(top_comp.get("__value_score", 0.0)) - float(comp.get("__value_score", 0.0)), 6),
                        "food": round(float(top_comp.get("__food_score", 0.0)) - float(comp.get("__food_score", 0.0)), 6),
                        "match": round(float(top_comp.get("__match_score_ui", 0.0)) - float(comp.get("__match_score_ui", 0.0)), 6),
                    }
                })

        # ✅ Ranking Debugger (A): vista per-rank pulita per tuning (solo debug)
        def _fnum(x, default=0.0) -> float:
            try:
                return float(x)
            except Exception:
                return float(default)
        def _lvl_norm(x: Any) -> str:
            s = _norm_lc(str(x or ""))
            if s in ("alto", "alta", "high"):
                return "high"
            if s in ("medio", "media", "medium"):
                return "medium"
            if s in ("basso", "bassa", "low"):
                return "low"
            return s
        ranking_rows = []
        for c in sorted_cards:
            cid = c.get("id")
            dbg = debug_map.get(cid, {}) if cid else {}
            comps = dbg.get("components", {}) if isinstance(dbg.get("components", None), dict) else {}
            rf = dbg.get("row_fields", {}) if isinstance(dbg.get("row_fields", None), dict) else {}

            m_ui = _fnum(comps.get("__match_score_ui", c.get("__match_score", c.get("match_score", 0.0))), 0.0)
            q_s = _fnum(comps.get("__quality_score", c.get("__quality_score", 0.0)), 0.0)
            v_s = _fnum(comps.get("__value_score", c.get("__value_score", 0.0)), 0.0)
            f_s = _fnum(comps.get("__food_score", 0.0), 0.0)
            o_s = _fnum(comps.get("__other_score", 0.0), 0.0)
            i_s = _fnum(comps.get("__intensity_score", 0.0), 0.0)

            row = {
                "rank": int(c.get("rank") or 0),
                "id": cid,
                "name": c.get("name"),
                "sort": sort,
                "score": _fnum(c.get("score", 0.0), 0.0),
                "M": round(m_ui, 4),
                "Q": round(q_s, 4),
                "V": round(v_s, 4),
                "F": round(f_s, 4),
                "O": round(o_s, 4),
                "I": round(i_s, 4),
                "S": round(_fnum(comps.get("__semantic_boost", 0.0), 0.0), 4),
                "price": c.get("price"),
                "price_delta": c.get("__price_delta"),

                # ✅ campi utili per semantic parsing / tuning (debug-only)
                "color": rf.get("color"),
                "body": _lvl_norm(rf.get("body")),
                "tannins": _lvl_norm(rf.get("tannins")),
                "acidity": _lvl_norm(rf.get("acidity")),
                "alcohol_level": rf.get("alcohol_level"),
                "sweetness": rf.get("sweetness"),
                "style_tags": rf.get("style_tags"),
                "occasion": rf.get("occasion"),
            }
        
            if sort == "relevance_v2":
                row.update({
                    "overall_base_0_1": _fnum(comps.get("__overall_base_0_1", 0.0), 0.0),
                    "match_factor": _fnum(comps.get("__match_factor", 0.0), 0.0),
                    "composite_0_1": _fnum(comps.get("__composite_0_1", 0.0), 0.0),
                    "composite_score": _fnum(comps.get("__composite_score", 0.0), 0.0),
                })

            ranking_rows.append(row)
            # ✅ delta_vs_prev: distanza dal vino precedente (debug tuning)
            prev = None
            for rr in ranking_rows:
                cur = rr.get("score", 0.0) or 0.0
                if prev is None:
                    rr["delta_vs_prev"] = 0.0
                else:
                    rr["delta_vs_prev"] = round(float(prev) - float(cur), 4)
                prev = cur

        meta["debug"] = {
            "rows": debug_rows,
            "delta_vs_top": delta_vs_top,
            "ranking_rows": ranking_rows,
        }
        meta["timings"] = timings

    # ✅ single return path (no duplicati)
    sorted_cards = dedup_strict(sorted_cards)
    return {"results": sorted_cards, "meta": meta}

# =========================
# API helpers
# =========================

def _normalize_sort(sort: Optional[str]) -> str:
    allowed = {"relevance", "match", "relevance_a9v1", "relevance_a9v2", "price_asc", "price_desc", "rating", "popular", "relevance_v2"}
    s = (sort or "relevance").strip()
    return s if s in allowed else "relevance"

def dedup_strict(results: list[dict]) -> list[dict]:
    """
    Dedup STRICT: elimina solo copie certe.
    Chiave: producer + denomination + zone + vintage + price (tutti normalizzati).
    Tie-break: tiene quello con score più alto, poi match_score più alto.
    """
    def norm(x):
        return (str(x).strip().lower() if x is not None else "")

    best_by_key: dict[tuple, dict] = {}

    for r in results:
        key = (
            norm(r.get("producer")),
            norm(r.get("denomination")),
            norm(r.get("zone")),
            norm(r.get("vintage")),
            norm(r.get("price")),
        )

        cur = best_by_key.get(key)
        if cur is None:
            best_by_key[key] = r
            continue

        # tie-break deterministico
        s_new = float(r.get("score", 0.0) or 0.0)
        s_old = float(cur.get("score", 0.0) or 0.0)

        m_new = float(r.get("match_score", r.get("__match_score", 0.0)) or 0.0)
        m_old = float(cur.get("match_score", cur.get("__match_score", 0.0)) or 0.0)

        if (s_new, m_new) > (s_old, m_old):
            best_by_key[key] = r

    return list(best_by_key.values())

# =========================
# Endpoints
# =========================
@app.post("/search")
def post_search(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    query = _norm(payload.get("query", ""))
    sort = _normalize_sort(payload.get("sort"))
    limit = int(payload.get("limit") or MAX_RESULTS_DEFAULT)

    include_test = bool(payload.get("include_test", False))
    debug = bool(payload.get("debug", False))
    explain = bool(payload.get("explain", False))  # ✅ Explain Mode B toggle (default: OFF)

    # Cache: disable when debug=true (must be fresh and include meta.debug)
    if not debug:
        _ = get_wines_df()
        cache_key = _cache_key({
            "build": BUILD_ID,
            "csv_mtime": CSV_CACHE.mtime,
            "ep": "search",
            "query": query,
            "sort": sort,
            "limit": limit,
            "include_test": include_test,
            "explain": explain,
        })
        cached = _cache_get(cache_key)
        if cached is not None:
            return JSONResponse(cached)
    else:
        cache_key = None

    data = run_search(query=query, sort=sort, limit=limit, include_test=include_test, debug=debug, explain=explain)

    if cache_key is not None:
        _cache_set(cache_key, data)

    return JSONResponse(data)
    
@app.get("/search_stream")
def get_search_stream(
    query: str = Query(...),
    sort: str = Query("relevance"),
    limit: int = Query(MAX_RESULTS_DEFAULT),
    include_test: bool = Query(False),
    debug: bool = Query(False),
    explain: bool = Query(False),  # ✅ Explain Mode B toggle (default: OFF)
) -> StreamingResponse:

    sort = _normalize_sort(sort)
    limit = int(limit or MAX_RESULTS_DEFAULT)

    # Cache: disable when debug=true (must be fresh and include meta.debug)
    if not debug:
        _ = get_wines_df()
        cache_key = _cache_key({
            "build": BUILD_ID,
            "csv_mtime": CSV_CACHE.mtime,
            "ep": "search_stream",
            "query": query,
            "sort": sort,
            "limit": limit,
            "include_test": include_test,
            "explain": explain,
        })
        data = _cache_get(cache_key)
        if data is None:
            data = run_search(
                query=query,
                sort=sort,
                limit=limit,
                include_test=include_test,
                debug=debug,
                explain=explain,
            )
            _cache_set(cache_key, data)
    else:
        data = run_search(
            query=query,
            sort=sort,
            limit=limit,
            include_test=include_test,
            debug=debug,
            explain=explain,
        )

    # Freeze snapshot (mai più None dentro gen)
    results = dedup_strict((data or {}).get("results", []))
    meta = (data or {}).get("meta", {})

    def gen() -> Iterable[str]:
        yield ":ok\n\n"

        for card in results:
            yield _sse_data({"type": "delta", "wine": card})

        yield _sse_data({"type": "final", "results": results, "meta": meta})
        yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@app.get("/stats")
def get_stats() -> JSONResponse:
    df = get_wines_df()
    payload = {
        "build_id": BUILD_ID,
        "csv_path": CSV_PATH,
        "rows": int(len(df)),
        "csv_mtime": CSV_CACHE.mtime,
        "last_load_ts": CSV_CACHE.last_load_ts,
        "search_cache_size": len(SEARCH_CACHE),
        "search_cache_ttl_sec": SEARCH_CACHE_TTL_SEC,
        "search_cache_cap": SEARCH_CACHE_CAP,
        "cache_disabled": DISABLE_CACHE,
        "columns": list(df.columns),
    }
    return JSONResponse(payload)


@app.get("/suggestions")
def get_suggestions() -> JSONResponse:
    suggestions = [
        "Un sangiovese di buona qualità",
        "Un vino intenso e strutturato sopra i 20€",
        "Uno spumante brut per aperitivo",
        "Un frizzante dolce francese sopra i 30€",
        "Un bianco con sentori agrumati sotto i 15€",
        "Un vino con buon rapporto qualità prezzo",
    ]
    return JSONResponse({"suggestions": suggestions})

# --- CLI: normalize CSV ---


# =========================
# Wine Details Endpoint
# =========================

@app.get("/wine/{wine_id}/details")
def get_wine_details(wine_id: str) -> JSONResponse:
    """
    Endpoint per dettaglio vino completo con tasting notes LLM.
    """
    df = get_wines_df()
    wine_row = df[df["id"] == wine_id]
    
    if wine_row.empty:
        return JSONResponse({"error": "Wine not found"}, status_code=404)
    
    row = wine_row.iloc[0]
    wine_dict = {
        "id": _norm(getattr(row, "id", "")),
        "name": _norm(getattr(row, "name", "")),
        "producer": _norm(getattr(row, "producer", "")),
        "region": _norm(getattr(row, "region", "")),
        "denomination": _norm(getattr(row, "denomination", "")),
        "vintage": _norm(getattr(row, "vintage", "")),
        "grapes": _norm(getattr(row, "grape_varieties", "")),
        "aromas": _norm(getattr(row, "aromas", "")),
        "food_pairings": _norm(getattr(row, "food_pairings", "")),
        "quality": _norm(getattr(row, "quality", "")),
        "tannins": _norm(getattr(row, "tannins", "")),
        "acidity": _norm(getattr(row, "acidity", "")),
        "price": _price_effective(row),
    }
    
    wine_dict["tasting_notes"] = generate_tasting_notes(wine_dict)
    
    aromas_text = wine_dict.get("aromas", "")
    if aromas_text:
        wine_dict["aroma_icons"] = get_aroma_icons(aromas_text)
    
    quality_val = _parse_float_maybe(wine_dict.get("quality", ""))
    if quality_val > 0:
        wine_dict["reviews_count"] = get_mock_reviews_count(quality_val)
        wine_dict["critic_score"] = get_mock_critic_score(quality_val)
    
    # Deriva campi intensity, sparkling, freshness per UI Screen 4
    intensity = derive_intensity(
        _norm(getattr(row, "body", "")),
        _norm(getattr(row, "tannins", "")),
        _norm(getattr(row, "alcohol_level", "")),
    )
    if intensity:
        wine_dict["intensity"] = intensity
    
    sparkling = derive_sparkling(
        _norm(getattr(row, "denomination", "")),
        _norm(getattr(row, "style_tags", "")),
        _norm(getattr(row, "name", "")),
        _norm(getattr(row, "description", "")),
    )
    if sparkling:
        wine_dict["sparkling"] = sparkling
    
    freshness = derive_freshness(
        _norm(getattr(row, "acidity", "")),
        sparkling or "",
        _norm(getattr(row, "alcohol_level", "")),
    )
    if freshness:
        wine_dict["freshness"] = freshness
    
    return JSONResponse({"wine": wine_dict})

if __name__ == "__main__":
    import sys
    import pandas as pd

    if "--normalize-csv" in sys.argv:
        print("Normalizing CSV...")

        src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wines.csv"))
        dst = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wines.normalized.csv"))

        df = pd.read_csv(src)

        # normalizza food_pairings (split su ; e |)
        def clean_fp(x):
            if not isinstance(x, str):
                return ""
            tokens = [t.strip() for t in re.split(r"[;|]", x) if t.strip()]
            return "|".join(sorted(set(tokens)))

        if "food_pairings" in df.columns:
            df["food_pairings"] = df["food_pairings"].apply(clean_fp)

        df.to_csv(dst, index=False)
        print("WROTE:", dst)
        sys.exit(0)