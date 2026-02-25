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


# =========================
# Build signature (anti-confusione / anti-regressione)
# =========================

BUILD_ID = "SommelierAI v0.2 STABILE + A/B/D (CSV schema real) + cache-safe 2026-02-17"


# =========================
# Config
# =========================

APP_TITLE = "SommelierAI Backend"

# Default: ../data/wines.csv (dato che backend/main.py sta in /backend e il CSV sta in /data)
DEFAULT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wines.csv"))
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


def _cache_key(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[Any]:
    if DISABLE_CACHE:
        return None
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
        return {"target": val, "delta": 1.0, "mode": "target"}

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
            return {"target": float(m7.group(1)), "delta": 1.0, "mode": "target"}

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
    "sangiovese", "nebbiolo", "barbera", "montepulciano", "primitivo", "aglianico",
    "nero d'avola", "negroamaro", "corvina", "glera", "vermentino",
    "chardonnay", "sauvignon", "sauvignon blanc", "pinot noir",
    "cabernet sauvignon", "cabernet franc", "merlot", "syrah", "shiraz",
    "grenache", "riesling", "chenin blanc", "malbec", "tempranillo",
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
# Parsing SAFE: attivo solo se l’utente cita cibi/contesti espliciti nella query.
FOOD_KEYWORDS = {
    # categorie
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
def parse_food_request(query: str) -> List[str]:
    q = _norm_lc(query)
    found: List[str] = []
    for canonical, variants in FOOD_KEYWORDS.items():
        for v in variants:
            if re.search(rf"\b{re.escape(v)}\b", q):
                found.append(canonical)
                break
    return sorted(set(found))

def parse_style_intent(query: str) -> Dict[str, bool]:
    q = _norm_lc(query)
    return {
        "elegant": bool(re.search(r"\belegant[eaio]?\b", q)),
        "important_dinner": bool(re.search(r"\bcena\s+importante\b|\boccasione\s+speciale\b", q)),
        "aperitivo": bool(re.search(r"\baperitivo\b|\bapericena\b", q)),
        "meditation": bool(re.search(r"\bmeditazione\b", q)),
    }


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

    # sparkling keywords
    if re.search(r"\bspumante\b", q) or re.search(r"\bchampagne\b", q) or re.search(r"\bprosecco\b", q):
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

    return {"sparkling": sparkling, "sweetness": sweetness}


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
    """Ritorna uno tra: 'bianco', 'rosso', 'rosato' se l’utente lo chiede esplicitamente nel testo."""
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

    Fix robusto:
    - gestisce valori "color" non normalizzati (es. "rosso, strutturato")
    - fallback su "color_detail" se presente
    - supporta sinonimi (white/red/rosé/rosato)
    """
    if not color_req:
        return df

    # normalizza richiesta
    want = _norm(str(color_req))
    synonyms = {
        "white": "bianco",
        "red": "rosso",
        "rose": "rosato",
        "rosé": "rosato",
        "rosa": "rosato",
        "rosato": "rosato",
        "bianca": "bianco",
        "bianche": "bianco",
        "bianchi": "bianco",
        "rossa": "rosso",
        "rosse": "rosso",
        "rossi": "rosso",
    }
    want = synonyms.get(want, want)

    # colonne candidate
    cols: list[str] = []
    if "color" in df.columns:
        cols.append("color")
    if "color_detail" in df.columns and "color_detail" not in cols:
        cols.append("color_detail")
    if not cols:
        return df

    # match "token" (separatore: spazio, virgola, pipe, slash, punto e virgola)
    # usa contains invece di == per tollerare formati diversi nel CSV
    token_re = re.compile(rf"(?:^|[\s,;|/]+){re.escape(want)}(?:$|[\s,;|/]+)")

    mask = None
    for c in cols:
        series = df[c].astype(str).map(_norm)
        m = series.map(lambda s: bool(token_re.search(s)))
        mask = m if mask is None else (mask | m)

    return df.loc[mask] if mask is not None else df
def _filter_new_A_B_D(
    df: pd.DataFrame,
    grapes_req: List[str],
    aromas_req: List[str],
    intensity_req: Optional[str],
    typology_req: Dict[str, Optional[str]],
) -> pd.DataFrame:
    # B) grapes: filter on grape_varieties
    if grapes_req:
        lc = df["grape_varieties"].astype(str).str.lower()
        mask = False
        for g in grapes_req:
            mask = mask | lc.str.contains(re.escape(g), na=False)
        df = df.loc[mask]

    # A) aromas: match on description (free text)
    if aromas_req:
        lc = df["description"].astype(str).str.lower()
        mask = False
        for a in aromas_req:
            # we search canonical word too (it’s what parse_aromas returns)
            mask = mask | lc.str.contains(re.escape(a), na=False)
            # also search for any variant of that canonical in description for robustness
            for v in AROMA_KEYWORDS.get(a, []):
                mask = mask | lc.str.contains(re.escape(_norm_lc(v)), na=False)
        df = df.loc[mask]

    # D) typology: derived sparkling + normalized sweetness
    sp_req = typology_req.get("sparkling")
    sw_req = typology_req.get("sweetness")

    if sp_req or sw_req or intensity_req:
        # compute derived columns on the fly (vectorized apply is fine at this scale; dataset is cached)
        derived_sp = df.apply(
            lambda r: derive_sparkling(r.get("denomination", ""), r.get("style_tags", ""), r.get("name", ""), r.get("description", "")),
            axis=1,
        )
        derived_sw = df["sweetness"].astype(str).map(normalize_sweetness)

        if sp_req:
            df = df.loc[derived_sp.eq(sp_req)]
        if sw_req:
            df = df.loc[derived_sw.eq(sw_req)]

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
    Usa qualità/balance/persistence se numeriche; altrimenti fallback semplice basato su presenza.
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

    # fallback: se ci sono campi testuali non vuoti, assegna un minimo
    txt_hits = 0
    for k in ["quality", "balance", "persistence"]:
        if _norm(getattr(row, k, "")):
            txt_hits += 1
    return float(txt_hits) * 0.5


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
    if boosts.get("food_match"):
        base += 0.45
    elif boosts.get("foods_present"):
        base -= 0.25

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
            if denom in ("barolo", "brunello di montalcino", "amarone della valpolicella", "barbaresco"):
                sb += 0.08
            qv = _parse_float_maybe(getattr(row, "quality", ""))
            if qv is not None and qv >= 4.6:
                sb += 0.05

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

        base += max(-0.10, min(sb, 0.25))

    return base, price_delta


def _build_wine_card(row: Any, rank: int, score: float, price_delta: float) -> Dict[str, Any]:
    # prezzo mostrato: preferiamo price_avg, fallback price_min
    pr = _price_effective(row)
    price_str = ""
    if pr is not None:
        # format semplice
        price_str = f"{pr:.2f}"

    # tags: usa style_tags + color + body (senza duplicare troppo)
    tags_parts: List[str] = []
    st = _norm(getattr(row, "style_tags", ""))
    if st:
        tags_parts.append(st)
    c = _norm(getattr(row, "color_detail", "")) or _norm(getattr(row, "color", ""))
    if c:
        tags_parts.append(c)
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
    }

    # opzionali: quality/balance/persistence/color_detail
    for k in ["quality", "balance", "persistence", "color_detail"]:
        v = _norm(getattr(row, k, ""))
        if v:
            card[k] = v

    # A/B/D: riportiamo anche i campi “veri” del dataset + derivati
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

    # sparkling derivato
    sparkling = derive_sparkling(
        _norm(getattr(row, "denomination", "")),
        _norm(getattr(row, "style_tags", "")),
        _norm(getattr(row, "name", "")),
        _norm(getattr(row, "description", "")),
    )
    if sparkling:
        card["sparkling"] = sparkling

    # sweetness normalizzata
    sw = normalize_sweetness(_norm(getattr(row, "sweetness", "")))
    if sw:
        card["sweetness"] = sw

    # aroma: non essendoci lista strutturata, non la mettiamo come campo “aromas”
    # (il requisito “sentori” è gestito come filtro su description)

    card["__price_delta"] = round(float(price_delta), 4)

    return card

def _apply_sort(cards: List[Dict[str, Any]], sort: str, value_intent: bool = False) -> List[Dict[str, Any]]:
    sort = sort or "relevance"

    if sort == "price_asc":
        return sorted(cards, key=lambda c: _parse_float_maybe(c.get("price", "")) or float("inf"))
    if sort == "price_desc":
        return sorted(cards, key=lambda c: _parse_float_maybe(c.get("price", "")) or -1.0, reverse=True)
    if sort == "rating":
        return sorted(cards, key=lambda c: float(c.get("rating_overall", 0.0)), reverse=True)
    if sort == "popular":
        return sorted(cards, key=lambda c: float(c.get("popularity", 0.0)), reverse=True)

    # relevance (default)
    if value_intent:
        # tie-break: a parità di score -> prezzo più basso prima, poi __price_delta
        return sorted(
            cards,
            key=lambda c: (
                -float(c.get("score", 0.0)),
                _parse_float_maybe(c.get("price", "")) or float("inf"),
                float(c.get("__price_delta", 0.0)),
            ),
        )

    # relevance normale (come prima)
    return sorted(cards, key=lambda c: (-float(c.get("score", 0.0)), float(c.get("__price_delta", 0.0))))


# =========================
# Core search
# =========================

def run_search(query: str, sort: str = "relevance", limit: int = MAX_RESULTS_DEFAULT) -> Dict[str, Any]:
    df = get_wines_df()
    limit = _clamp(int(limit or MAX_RESULTS_DEFAULT), 1, MAX_RESULTS_CAP)

    q = _norm(query)

    price_info = parse_price(q)
    region = parse_region(q)

    # A/B/D requests
    grapes_req = parse_grapes(q)
    aromas_req = parse_aromas(q)
    intensity_req = parse_intensity_request(q)
    typology_req = parse_typology_request(q)
    foods_req = parse_food_request(q)


    color_req = parse_color_request(q)

    # VALUE intent (solo se richiesto dall'utente)
    value_intent = parse_value_intent(q)

    style_intent = parse_style_intent(q)

    filtered = df

    # price filter
    filtered = _filter_by_price(filtered, price_info)

    # region filter (match in region/zone/denomination/country)
    if region:
        for col in ["region", "zone", "denomination", "country"]:
            filtered = _filter_by_text_contains(filtered, col, region)

    # color filter (bianco/rosso/rosato)
    filtered = _filter_by_color(filtered, color_req)

    # A/B/D filters
    filtered = _filter_new_A_B_D(filtered, grapes_req, aromas_req, intensity_req, typology_req)

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

    scored: List[Dict[str, Any]] = []
    for r in rows:
        boosts = {
            "grape_match": bool(grapes_req),
            "aroma_match": bool(aromas_req),
            "intensity_match": bool(intensity_req),
            "typology_match": bool(typology_req.get("sparkling") or typology_req.get("sweetness")),
            "food_match": food_match(r, foods_req),
            "foods_present": bool(foods_req),
        }
        if sort == "relevance_a9v1":
            s, pdlt = _score_row_a9v1(r, price_info, boosts)
        elif sort == "relevance_a9v2":
            s, pdlt = _score_row_a9v1(r, price_info, boosts, style_intent=style_intent)
        else:
            s, pdlt = _score_row(r, price_info, boosts, value_intent=value_intent)
        scored.append(_build_wine_card(r, rank=0, score=s, price_delta=pdlt))

    sorted_cards = _apply_sort(scored, sort, value_intent=value_intent)[:limit]
    for i, c in enumerate(sorted_cards, start=1):
        c["rank"] = i

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
            "value_intent": value_intent,
        },
        "count": len(sorted_cards),
        "timestamp": int(_now()),
    }

    return {"results": sorted_cards, "meta": meta}


# =========================
# API helpers
# =========================

def _normalize_sort(sort: Optional[str]) -> str:
    allowed = {"relevance", "relevance_a9v1", "relevance_a9v2", "price_asc", "price_desc", "rating", "popular"}
    s = (sort or "relevance").strip()
    return s if s in allowed else "relevance"


# =========================
# Endpoints
# =========================

@app.post("/search")
def post_search(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    query = _norm(payload.get("query", ""))
    sort = _normalize_sort(payload.get("sort"))
    limit = int(payload.get("limit") or MAX_RESULTS_DEFAULT)

    cache_key = _cache_key({"build": BUILD_ID, "ep": "search", "query": query, "sort": sort, "limit": limit})
    cached = _cache_get(cache_key)
    if cached is not None:
        return JSONResponse(cached)

    data = run_search(query=query, sort=sort, limit=limit)
    _cache_set(cache_key, data)
    return JSONResponse(data)


@app.get("/search_stream")
def get_search_stream(
    query: str = Query(...),
    sort: str = Query("relevance"),
    limit: int = Query(MAX_RESULTS_DEFAULT),
) -> StreamingResponse:
    sort = _normalize_sort(sort)
    limit = int(limit or MAX_RESULTS_DEFAULT)

    cache_key = _cache_key({"build": BUILD_ID, "ep": "search_stream", "query": query, "sort": sort, "limit": limit})
    cached = _cache_get(cache_key)
    if cached is None:
        cached = run_search(query=query, sort=sort, limit=limit)
        _cache_set(cache_key, cached)

    def gen() -> Iterable[str]:
        yield ":ok\n\n"
        results = cached.get("results", [])
        for card in results:
            yield _sse_data({"type": "delta", "wine": card})
        yield _sse_data({"type": "final", "results": results, "meta": cached.get("meta", {})})
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
