# -----------------------------------------------------------------------------
# main.py
# SommelierAI Backend
#
# NOTE:
# - Questo file è pensato per rimanere stabile e compilabile.
# - Modifica applicata in questo step: FIX location_terms/match
#   - RIMOSSA duplicazione _row_location_match (prima veniva shadowata)
#   - Normalizzazione minima termini location per match stabile
# -----------------------------------------------------------------------------

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

BUILD_ID = "SommelierAI v0.2 STABILE + A/B/D (CSV schema real) + cache-safe 2026-02-22-intensityfix + locmatchfix-2026-02-23 + A7-loctokens-2026-02-23 + A8-locindex-2026-02-23 + A9-1-zonematchfix-2026-02-23 + A9-2-locweight-2026-02-23 + A9-3-valuesqrt-2026-02-23 + A9-4-rankmodes-2026-02-23"

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


def _filter_by_text_contains(df: pd.DataFrame, col: str, needle: str) -> pd.DataFrame:
    """Filtro robusto: contiene (case-insensitive), tollerante su stringhe vuote."""
    if col not in df.columns:
        return df
    n = _norm_lc(needle)
    if not n:
        return df
    s = df[col].astype(str).str.lower()
    return df.loc[s.str.contains(re.escape(n), na=False)]


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

# =========================
# Location index (SAFE+PERF)
# - accelera _extract_location_terms (denom/zone + vocab)
# - cache invalidata su CSV mtime
# =========================

LOCATION_STOP = {
    "d", "di", "da", "del", "della", "dell", "de", "la", "le", "il", "lo", "al", "alla", "alle", "dei",
    "doc", "docg", "igt", "dop", "dopg",
}

@dataclass
class LocationIndex:
    # denom normalizzata -> token significativi (stop rimossi)
    denom_sig_tokens: Dict[str, Tuple[str, ...]]
    # zone normalizzata -> token significativi (stop rimossi)
    zone_sig_tokens: Dict[str, Tuple[str, ...]]
    # denom normalizzata -> set(zone normalizzate)
    denom_to_zones: Dict[str, set]
    # vocab tokens (region/zone/denomination) per fallback
    vocab_tokens: set


@dataclass
class LocationIndexCache:
    idx: Optional[LocationIndex] = None
    mtime: float = 0.0
    rows: int = 0
    last_build_ts: float = 0.0


LOCATION_INDEX_CACHE = LocationIndexCache()


def _build_location_index(df: pd.DataFrame) -> LocationIndex:
    # vocab tokens (fallback)
    vocab: set[str] = set()
    for col in ["region", "zone", "denomination"]:
        if col in df.columns:
            for v in df[col].astype(str).tolist():
                vv = _norm_lc(v).strip()
                if not vv:
                    continue
                for t in _loc_tokens(vv):
                    if t:
                        vocab.add(t)

    # denominazioni
    denom_sig_tokens: Dict[str, Tuple[str, ...]] = {}
    denom_values: set[str] = set()
    if "denomination" in df.columns:
        denom_values = set(_norm_lc(x).strip() for x in df["denomination"].astype(str).tolist() if x)

    for d in denom_values:
        if not d:
            continue
        toks = tuple([t for t in _loc_tokens(d) if t and t not in LOCATION_STOP])
        if toks:
            denom_sig_tokens[d] = toks

    # zone
    zone_sig_tokens: Dict[str, Tuple[str, ...]] = {}
    zone_values: set[str] = set()
    if "zone" in df.columns:
        zone_values = set(_norm_lc(x).strip() for x in df["zone"].astype(str).tolist() if x)

    for z in zone_values:
        if not z:
            continue
        toks = tuple([t for t in _loc_tokens(z) if t and t not in LOCATION_STOP])
        if toks:
            zone_sig_tokens[z] = toks

    # mapping denom -> zones
    denom_to_zones: Dict[str, set] = {}
    if "denomination" in df.columns and "zone" in df.columns and len(df) > 0:
        dcol = df["denomination"].astype(str).map(_norm_lc)
        zcol = df["zone"].astype(str).map(_norm_lc)
        for d, z in zip(dcol.tolist(), zcol.tolist()):
            dd = (d or "").strip()
            zz = (z or "").strip()
            if not dd or not zz:
                continue
            denom_to_zones.setdefault(dd, set()).add(zz)

    return LocationIndex(
        denom_sig_tokens=denom_sig_tokens,
        zone_sig_tokens=zone_sig_tokens,
        denom_to_zones=denom_to_zones,
        vocab_tokens=vocab,
    )


def get_location_index(df: pd.DataFrame) -> LocationIndex:
    # Invalida su mtime o change di rows (fallback extra)
    mtime = float(CSV_CACHE.mtime or 0.0)
    rows = int(len(df))
    if (LOCATION_INDEX_CACHE.idx is None) or (mtime and mtime != LOCATION_INDEX_CACHE.mtime) or (rows != LOCATION_INDEX_CACHE.rows):
        LOCATION_INDEX_CACHE.idx = _build_location_index(df)
        LOCATION_INDEX_CACHE.mtime = mtime
        LOCATION_INDEX_CACHE.rows = rows
        LOCATION_INDEX_CACHE.last_build_ts = _now()
    return LOCATION_INDEX_CACHE.idx



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
    df = get_wines_df()
    _ = get_location_index(df)


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


def parse_aromas(query: str) -> List[str]:
    q = _norm_lc(query)
    requested: List[str] = []
    for canonical, variants in AROMA_KEYWORDS.items():
        for v in variants:
            if v and re.search(rf"\b{re.escape(v)}\b", q):
                requested.append(canonical)
                break
    return sorted(set(requested))


INTENSITY_WORDS = {
    "delicato": "low", "leggero": "low", "leggera": "low",
    "medio": "medium", "media": "medium", "equilibrato": "medium", "equilibrata": "medium",
    "intenso": "high", "intensa": "high", "potente": "high",
}


def parse_intensity_request(query: str) -> Optional[str]:
    q = _norm_lc(query)
    # "strutturato" è ambiguo: NON deve diventare un filtro hard.
    # Lo trattiamo come "high" SOLO se rafforzato esplicitamente.
    for w, v in INTENSITY_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", q):
            return v
    return None


def _normalize_level(s: str) -> str:
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
    b = _normalize_level(body)
    t = _normalize_level(tannins)

    alc = _parse_float_maybe(alcohol_level)

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


def derive_sparkling(denomination: str, style_tags: str, name: str, description: str) -> str:
    hay = " ".join([_norm_lc(denomination), _norm_lc(style_tags), _norm_lc(name), _norm_lc(description)])
    if any(k in hay for k in [
        "champagne", "prosecco", "franciacorta", "cava", "cremant", "crémant",
        "spumante", "metodo classico", "méthode traditionnelle", "metodo tradizionale",
    ]):
        return "spumante"
    if any(k in hay for k in ["frizzante", "pet-nat", "pét-nat", "col fondo"]):
        return "frizzante"
    return "fermo"


def normalize_sweetness(s: str) -> str:
    v = _norm_lc(s)
    if not v:
        return ""
    if any(k in v for k in ["secco", "dry", "brut", "extra brut", "pas dos", "dosage zero", "nature"]):
        return "secco"
    if any(k in v for k in ["abbocc", "off-dry", "off dry"]):
        return "abboccato"
    if any(k in v for k in ["amabile", "semi-sweet", "semi sweet"]):
        return "amabile"
    if any(k in v for k in ["dolce", "sweet", "demi-sec", "demisec", "doux"]):
        return "dolce"
    return v


def parse_typology_request(query: str) -> Dict[str, Optional[str]]:
    q = _norm_lc(query)

    sparkling: Optional[str] = None
    sweetness: Optional[str] = None

    if re.search(r"\bspumante\b", q) or re.search(r"\bchampagne\b", q) or re.search(r"\bprosecco\b", q):
        sparkling = "spumante"
    elif re.search(r"\bfrizzante\b", q):
        sparkling = "frizzante"
    elif re.search(r"\bfermo\b", q):
        sparkling = "fermo"

    if re.search(r"\bsecco\b|\bbrut\b|\bextra\s*brut\b|\bnature\b", q):
        sweetness = "secco"
    elif re.search(r"\babboccato\b", q):
        sweetness = "abboccato"
    elif re.search(r"\bamabile\b", q):
        sweetness = "amabile"
    elif re.search(r"\bdolce\b|\bdemi-?sec\b", q):
        sweetness = "dolce"

    if re.search(r"\bbrut\b|\bextra\s*brut\b|\bpas\s*dos", q):
        if sparkling is None:
            sparkling = "spumante"
        if sweetness is None:
            sweetness = "secco"

    return {"sparkling": sparkling, "sweetness": sweetness}


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


def parse_color_request(query: str) -> Optional[str]:
    """Ritorna uno tra: 'bianco', 'rosso', 'rosato' se l’utente lo chiede esplicitamente nel testo."""
    q = _norm(query)
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
# Match helpers (T2: match sort)

# -------------------------
# A9.1 — Implicit denom → grape boost (SAFE):
# - NON modifica __match_score (per non rompere i test location 1.0 / 0.5)
# - Applica un piccolo boost al campo "score" SOLO quando sort=match e l’utente non ha chiesto vitigni espliciti.
# -------------------------

IMPLICIT_DENOM_GRAPES: Dict[str, List[str]] = {
    # Piemonte
    "barolo": ["nebbiolo"],
    "barbaresco": ["nebbiolo"],
    # Toscana
    "brunello di montalcino": ["sangiovese"],
    "chianti classico": ["sangiovese"],
    # Veneto
    "amarone della valpolicella": ["corvina", "corvinone", "rondinella"],
    # Campania
    "taurasi": ["aglianico"],
    # Sicilia
    "etna rosso": ["nerello mascalese", "nerello cappuccio"],
}

def _implied_grapes_from_location_terms(location_terms: List[str]) -> List[str]:
    denoms: List[str] = []
    for t in location_terms or []:
        tt = _norm_lc(t).strip()
        if tt.startswith("denom:"):
            d = tt.split(":", 1)[1].strip()
            if d:
                denoms.append(d)
    out: List[str] = []
    for d in denoms:
        out.extend(IMPLICIT_DENOM_GRAPES.get(d, []))
    # dedup stabile
    seen = set()
    uniq: List[str] = []
    for g in out:
        gg = _norm_lc(g).strip()
        if gg and gg not in seen:
            seen.add(gg)
            uniq.append(gg)
    return uniq

def _row_implied_grape_boost(row: Any, implied_grapes: List[str]) -> float:
    """Ritorna un piccolo boost (0.05) se il vino contiene uno dei vitigni impliciti."""
    if not implied_grapes:
        return 0.0
    gv = _norm_lc(getattr(row, "grape_varieties", ""))
    if not gv:
        return 0.0
    for g in implied_grapes:
        if g and re.search(rf"\b{re.escape(g)}\b", gv):
            return 0.05
    return 0.0

# =========================

STRUCTURED_KEYWORDS_RE = re.compile(
    r"\b(strutturato|strutturata|strutturati|strutturate|robusto|robusta|robusti|robuste|potente|potenti)\b",
    re.IGNORECASE
)


def parse_structured_keyword(query: str) -> bool:
    """True se l'utente chiede esplicitamente 'strutturato' (o sinonimi).
    NOTA: questo NON deve filtrare, ma solo contribuire al match score.
    """
    return bool(STRUCTURED_KEYWORDS_RE.search(_norm(query)))


def _row_structured_match(row: Any) -> float:
    """Ritorna 1.0 se il vino 'sembra' strutturato dai dati (non perfetto, ma stabile)."""
    hay = " ".join([
        _norm_lc(getattr(row, "description", "")),
        _norm_lc(getattr(row, "style_tags", "")),
        _norm_lc(getattr(row, "body", "")),
        _norm_lc(getattr(row, "tannins", "")),
    ])
    if any(k in hay for k in ["struttur", "robust", "potent", "tannic", "tannico", "corposo", "full"]):
        return 1.0
    # fallback: se body/tannins derivano high, spesso è “strutturato”
    intensity = derive_intensity(
        _norm(getattr(row, "body", "")),
        _norm(getattr(row, "tannins", "")),
        _norm(getattr(row, "alcohol_level", "")),
    )
    return 1.0 if intensity == "high" else 0.0

# --- Location hierarchy (minimo, estendibile) ---
# Canonical -> varianti che possono apparire nella query (gestiamo anche "d alba" senza apostrofo)
# NOTE (A6.1): blocco legacy hardcoded disattivato — ora location_terms è data-driven (denomination/zone dal CSV)
# BAROLO_COMMUNES: Dict[str, List[str]] = {
#     "serralunga d'alba": ["serralunga", "serralunga d'alba", "serralunga d alba"],
#     "monforte d'alba": ["monforte", "monforte d'alba", "monforte d alba"],
#     "la morra": ["la morra", "lamorra"],
#     "castiglione falletto": ["castiglione falletto"],
#     "barolo": ["barolo"],
#     "novello": ["novello"],
#     "verduno": ["verduno"],
#     "grinzane cavour": ["grinzane cavour", "grinzane"],
#     "cherasco": ["cherasco"],
#     "diano d'alba": ["diano d'alba", "diano d alba", "diano"],
#     "roddi": ["roddi"],
# }


# Denominazioni note (per ora: solo Barolo, come richiesto)
# NOTE (A6.1): blocco legacy hardcoded disattivato — ora location_terms è data-driven (denomination/zone dal CSV)
# KNOWN_DENOMINATIONS: Dict[str, List[str]] = {
#     "barolo": ["barolo"],
# }



def _loc_tokens(s: str) -> List[str]:
    """
    Tokenizzazione robusta per location:
    - normalizza apostrofi tipografici e separa "d'alba" -> "d", "alba"
    - mantiene lettere accentate
    """
    v = _norm_lc(s)
    if not v:
        return []
    v = v.replace("’", "'").replace("`", "'")
    v = v.replace("'", " ")
    v = v.replace("-", " ").replace("/", " ")
    return [t for t in re.findall(r"[a-zàèéìòù]+", v) if t]

def _extract_location_terms(df: pd.DataFrame, query: str) -> List[str]:
    """
    Estrae termini di location con un minimo di "struttura" (data-driven):
    - denom:<x> se l'utente cita una denominazione (token match tollerante a "di/del/..." e apostrofi)
    - comune:<x> se l'utente cita una zone/comune presente nel dataset per quella denominazione
    - fallback: tokens presenti nel vocab (come prima), per non perdere copertura
    """
    q = _norm_lc(query)
    if not q:
        return []

    idx = get_location_index(df)

    # 1) Denominazioni (data-driven) — token match
    out: List[str] = []
    found_denoms: List[str] = []

    q_tokens = set(_loc_tokens(q))

    for denom, d_tokens in idx.denom_sig_tokens.items():
        if not d_tokens:
            continue
        if all(t in q_tokens for t in d_tokens):
            found_denoms.append(denom)
            out.append(f"denom:{denom}")

    # 1b) Zone/Comuni data-driven: solo per denom matchate
    if found_denoms:
        for d in found_denoms:
            zones = idx.denom_to_zones.get(d, set()) or set()
            for z in sorted(zones):
                z_tokens = idx.zone_sig_tokens.get(z)
                if not z_tokens:
                    continue

                # Match "comune/zone" robusto:
                # - se la zone ha almeno un token "forte" (len>=5), basta che UNO di quelli compaia nella query
                #   (es. "serralunga" deve matchare "serralunga d'alba" anche se manca "alba")
                # - altrimenti (zone corta) richiediamo tutti i token
                strong = [t for t in z_tokens if len(t) >= 5]
                if strong:
                    if any(t in q_tokens for t in strong):
                        out.append(f"comune:{z}")
                else:
                    if all(t in q_tokens for t in z_tokens):
                        out.append(f"comune:{z}")

    # 3) Fallback tokens dal vocab (come prima) — ma evitiamo rumore e doppioni
    fallback_terms: List[str] = []
    for t in q_tokens:
        tt = _norm_lc(t).strip()
        if not tt:
            continue
        if tt in LOCATION_STOP or len(tt) < 4:
            continue
        if tt in idx.vocab_tokens:
            fallback_terms.append(tt)

    # Se ho termini strutturati (denom/comune), i fallback plain non servono e possono creare rumore.
    has_structured = any(t.startswith("denom:") or t.startswith("comune:") for t in out)
    if has_structured:
        return sorted(set(out))

    return sorted(set(out + fallback_terms))

def _row_location_match(row: Any, location_terms: List[str]) -> float:
    """
    Match location 0..1 con priorità strutturata:
    - Se l'utente chiede denom + comune => media 50/50 (preserva test: Monforte=0.5 se manca comune)
    - Se chiede solo denom => 1.0 se denom match
    - Se chiede solo comune => frazione comuni matchati
    - Altrimenti fallback “vecchio” sui termini liberi
    """
    if not location_terms:
        return 0.0

    denom_terms: List[str] = []
    comune_terms: List[str] = []
    plain_terms: List[str] = []

    for t in location_terms:
        tt = _norm_lc(t).strip()
        if not tt:
            continue
        if tt.startswith("denom:"):
            denom_terms.append(tt.split(":", 1)[1].strip())
        elif tt.startswith("comune:"):
            comune_terms.append(tt.split(":", 1)[1].strip())
        else:
            plain_terms.append(tt)

    denom_terms = sorted(set([d for d in denom_terms if d]))
    comune_terms = sorted(set([c for c in comune_terms if c]))
    plain_terms = sorted(set([p for p in plain_terms if p]))

    # Evita doppio conteggio quando una keyword è sia denominazione che comune (es. "barolo")
    if denom_terms and comune_terms:
        comune_terms = [c for c in comune_terms if c not in set(denom_terms)]

    # Haystack: separiamo denom/zone per match più “semantico”
    denom_hay = " ".join([
        _norm_lc(getattr(row, "denomination", "")),
        _norm_lc(getattr(row, "name", "")),
    ])
    zone_hay = " ".join([
        _norm_lc(getattr(row, "zone", "")),
        _norm_lc(getattr(row, "region", "")),
        _norm_lc(getattr(row, "name", "")),
        _norm_lc(getattr(row, "description", "")),
        _norm_lc(getattr(row, "denomination", "")),
    ])

    # 1) denom score (0/1): basta un match
    denom_score = 0.0
    if denom_terms:
        for d in denom_terms:
            if d and re.search(rf"\b{re.escape(d)}\b", denom_hay):
                denom_score = 1.0
                break

    # 2) comune score (0..1): frazione comuni matchati
    comune_score = 0.0
    if comune_terms:
        hit = 0
        for c in comune_terms:
            if c and re.search(rf"\b{re.escape(c)}\b", zone_hay):
                hit += 1
        comune_score = float(hit) / float(len(comune_terms)) if comune_terms else 0.0

    # 3) plain fallback (0..1) come prima
    plain_score = 0.0
    if plain_terms:
        hit = 0
        for p in plain_terms:
            if p and re.search(rf"\b{re.escape(p)}\b", zone_hay):
                hit += 1
        plain_score = float(hit) / float(len(plain_terms)) if plain_terms else 0.0

    # Composizione:
    # Se esiste almeno un termine strutturato (denom/comune),
    # IGNORIAMO i plain_terms per evitare match gonfiati.
    if denom_terms or comune_terms:

        # denom+comune richiesti => 50/50
        if denom_terms and comune_terms:
            return max(0.0, min(1.0, 0.5 * denom_score + 0.5 * comune_score))

        # solo denom richiesto
        if denom_terms and not comune_terms:
            return max(0.0, min(1.0, denom_score))

        # solo comune richiesto
        if comune_terms and not denom_terms:
            return max(0.0, min(1.0, comune_score))

    # Nessun termine strutturato => fallback plain
    return max(0.0, min(1.0, plain_score))


def _row_grape_match(row: Any, grapes_req: List[str]) -> float:
    if not grapes_req:
        return 0.0
    gv = _norm_lc(getattr(row, "grape_varieties", ""))
    if not gv:
        return 0.0
    for g in grapes_req:
        if g and re.search(rf"\b{re.escape(g)}\b", gv):
            return 1.0
    return 0.0


def _row_aroma_match(row: Any, aromas_req: List[str]) -> float:
    """Frazione di aromi richiesti che matchano la description (0..1)."""
    if not aromas_req:
        return 0.0
    desc = _norm_lc(getattr(row, "description", ""))
    if not desc:
        return 0.0

    hit = 0
    for a in aromas_req:
        variants = [a] + AROMA_KEYWORDS.get(a, [])
        ok = any(v and re.search(rf"\b{re.escape(_norm_lc(v))}\b", desc) for v in variants)
        if ok:
            hit += 1

    return float(hit) / float(len(aromas_req)) if aromas_req else 0.0


def _row_typology_match(row: Any, typology_req: Dict[str, Optional[str]]) -> float:
    sp_req = typology_req.get("sparkling")
    sw_req = typology_req.get("sweetness")
    if not (sp_req or sw_req):
        return 0.0

    sp = derive_sparkling(
        _norm(getattr(row, "denomination", "")),
        _norm(getattr(row, "style_tags", "")),
        _norm(getattr(row, "name", "")),
        _norm(getattr(row, "description", "")),
    )
    sw = normalize_sweetness(_norm(getattr(row, "sweetness", "")))

    ok = 0
    tot = 0
    if sp_req:
        tot += 1
        if sp == sp_req:
            ok += 1
    if sw_req:
        tot += 1
        if sw == sw_req:
            ok += 1

    return float(ok) / float(tot) if tot > 0 else 0.0


def _row_intensity_match(row: Any, intensity_req: Optional[str]) -> float:
    if not intensity_req:
        return 0.0
    got = derive_intensity(
        _norm(getattr(row, "body", "")),
        _norm(getattr(row, "tannins", "")),
        _norm(getattr(row, "alcohol_level", "")),
    )
    return 1.0 if got == intensity_req else 0.0



# =========================
# Ranking modes (groundwork) — A9.4
# =========================
#
# Nota: per ora NON cambia l'API e NON introduce nuove modalità lato client.
# Serve solo a:
# - rendere esplicito in meta quale "rank_mode" è stato effettivamente applicato
# - centralizzare (in modo leggibile) profili di peso / strategie per futuri switch (Smart A2, ecc.)
#
# Regola: nessuna regressione di comportamento nello step A9.4.

def _resolve_rank_mode(sort: str, value_intent: bool) -> str:
    """Modalità logica effettiva (solo meta)."""
    s = (sort or "relevance").strip() or "relevance"
    if s in {"quality", "price_value", "match"}:
        return s
    # relevance con intento value => di fatto "value-first" (già presente in _apply_sort)
    if s == "relevance" and value_intent:
        return "price_value"
    return "relevance"


def _is_location_driven_query(
    grapes_req: list[str],
    aromas_req: list[str],
    intensity_req: str | None,
    typology_req: dict[str, str | None],
    structured_req: bool,
    location_terms: list[str],
) -> bool:
    """True se la query è quasi solo geografica."""
    if not location_terms:
        return False
    if grapes_req or aromas_req or intensity_req:
        return False
    if typology_req.get("sparkling") or typology_req.get("sweetness"):
        return False
    if structured_req:
        return False
    return True


# Profili pesi (match_score) — groundwork per ranking modes futuri.
# NOTA: i pesi sono normalizzati in _compute_match_score in base alle parti presenti.
MATCH_WEIGHT_PROFILES = {
    "default": {
        "grapes": 0.35,
        "aromas": 0.25,
        "intensity": 0.15,
        "typology": 0.15,
        "structured": 0.10,
        "location": 0.20,
    },
    # location-driven: aumenta importanza location (A9.2)
    "location_driven": {
        "grapes": 0.35,
        "aromas": 0.25,
        "intensity": 0.15,
        "typology": 0.15,
        "structured": 0.10,
        "location": 0.40,
    },
}

def _compute_match_score(
    row: Any,
    grapes_req: List[str],
    aromas_req: List[str],
    intensity_req: Optional[str],
    typology_req: Dict[str, Optional[str]],
    structured_req: bool,
    location_terms: List[str],
) -> float:
    """
    Match score continuo 0..1.
    Pesi normalizzati sulle richieste effettive (non penalizza se l’utente non chiede una dimensione).
    """

    profile = "location_driven" if _is_location_driven_query(
        grapes_req=grapes_req,
        aromas_req=aromas_req,
        intensity_req=intensity_req,
        typology_req=typology_req,
        structured_req=structured_req,
        location_terms=location_terms,
    ) else "default"
    W = MATCH_WEIGHT_PROFILES.get(profile, MATCH_WEIGHT_PROFILES["default"])

    parts: List[Tuple[float, float]] = []


    if grapes_req:
        parts.append((W.get('grapes', 0.35), _row_grape_match(row, grapes_req)))
    if aromas_req:
        parts.append((W.get('aromas', 0.25), _row_aroma_match(row, aromas_req)))
    if intensity_req:
        parts.append((W.get('intensity', 0.15), _row_intensity_match(row, intensity_req)))
    if typology_req.get("sparkling") or typology_req.get("sweetness"):
        parts.append((W.get('typology', 0.15), _row_typology_match(row, typology_req)))
    if structured_req:
        parts.append((W.get('structured', 0.10), _row_structured_match(row)))
    if location_terms:
        parts.append((W.get('location', 0.20), _row_location_match(row, location_terms)))

    if not parts:
        return 0.0

    wsum = sum(w for (w, _) in parts)
    if wsum <= 0:
        return 0.0

    score = sum(w * v for (w, v) in parts) / wsum
    return max(0.0, min(1.0, float(score)))


# =========================
# Filtering helpers
# =========================
# =========================
# Filtering helpers
# =========================
def _filter_by_price(df: pd.DataFrame, price_info: Dict[str, Any]) -> pd.DataFrame:
    mode = price_info.get("mode", "none")
    if mode == "none":
        return df

    prices = df["price_avg"].map(_parse_float_maybe)
    prices_min = df["price_min"].map(_parse_float_maybe)

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
    if not color_req:
        return df

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

    cols: list[str] = []
    if "color" in df.columns:
        cols.append("color")
    if "color_detail" in df.columns and "color_detail" not in cols:
        cols.append("color_detail")
    if not cols:
        return df

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
    if grapes_req:
        lc = df["grape_varieties"].astype(str).str.lower()
        mask = False
        for g in grapes_req:
            mask = mask | lc.str.contains(re.escape(g), na=False)
        df = df.loc[mask]

    if aromas_req:
        lc = df["description"].astype(str).str.lower()
        mask = False
        for a in aromas_req:
            mask = mask | lc.str.contains(re.escape(a), na=False)
            for v in AROMA_KEYWORDS.get(a, []):
                mask = mask | lc.str.contains(re.escape(_norm_lc(v)), na=False)
        df = df.loc[mask]

    sp_req = typology_req.get("sparkling")
    sw_req = typology_req.get("sweetness")

    if sp_req or sw_req or intensity_req:
        derived_sp = df.apply(
            lambda r: derive_sparkling(
                r.get("denomination", ""),
                r.get("style_tags", ""),
                r.get("name", ""),
                r.get("description", ""),
            ),
            axis=1,
        )
        derived_sw = df["sweetness"].astype(str).map(normalize_sweetness)

        if sp_req:
            df = df.loc[derived_sp.eq(sp_req)]
        if sw_req:
            df = df.loc[derived_sw.eq(sw_req)]

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

    pr = _price_effective(row)
    if value_intent and pr and pr > 0:
        value_index = (base + 0.75) / math.log(pr + 2.0)
        base += min(value_index * 0.55, 1.25)

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

    return base, price_delta


def _build_wine_card(row: Any, rank: int, score: float, price_delta: float) -> Dict[str, Any]:
    pr = _price_effective(row)
    price_str = ""
    if pr is not None:
        price_str = f"{pr:.2f}"

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
        "id": _norm(getattr(row, "id", "")),
        "name": _norm(getattr(row, "name", "")),
        "price": price_str,
        "reason": _norm(getattr(row, "description", "")),
        "purchase_url": _norm(getattr(row, "purchase_url", "")),
        "tags": tags,
        "rank": rank,
        "score": round(float(score), 4),

        # rating_overall/popularity non presenti nel CSV: lasciamo 0.0 (stabile per iOS)
        "rating_overall": 0.0,
        "popularity": 0.0,

        "producer": _norm(getattr(row, "producer", "")),
        "country": _norm(getattr(row, "country", "")),
        "region": _norm(getattr(row, "region", "")),
        "zone": _norm(getattr(row, "zone", "")),
        "denomination": _norm(getattr(row, "denomination", "")),
        "vintage": _norm(getattr(row, "vintage", "")),
        "food_pairings": _norm(getattr(row, "food_pairings", "")),
    }

    for k in ["quality", "balance", "persistence", "color_detail"]:
        v = _norm(getattr(row, k, ""))
        if v:
            card[k] = v

    gv = _norm(getattr(row, "grape_varieties", ""))
    if gv:
        card["grapes"] = gv

    intensity = derive_intensity(
        _norm(getattr(row, "body", "")),
        _norm(getattr(row, "tannins", "")),
        _norm(getattr(row, "alcohol_level", "")),
    )
    if intensity:
        card["intensity"] = intensity

    sparkling = derive_sparkling(
        _norm(getattr(row, "denomination", "")),
        _norm(getattr(row, "style_tags", "")),
        _norm(getattr(row, "name", "")),
        _norm(getattr(row, "description", "")),
    )
    if sparkling:
        card["sparkling"] = sparkling

    sw = normalize_sweetness(_norm(getattr(row, "sweetness", "")))
    if sw:
        card["sweetness"] = sw

    card["__price_delta"] = round(float(price_delta), 4)

    return card


# =========================
# Apply sort + Core search
# =========================

def _apply_sort(cards: List[Dict[str, Any]], sort: str, value_intent: bool = False) -> List[Dict[str, Any]]:
    sort = sort or "relevance"

    if sort == "price_asc":
        return sorted(cards, key=lambda c: _parse_float_maybe(c.get("price", "")) or float("inf"))

    if sort == "price_desc":
        return sorted(
            cards,
            key=lambda c: _parse_float_maybe(c.get("price", "")) or -1.0,
            reverse=True,
        )

    # ✅ legacy compat: "rating" = qualità pura (se rating_overall è 0 usa __quality_score)
    if sort == "rating":
        def _rating_value(c: Dict[str, Any]) -> float:
            r = float(c.get("rating_overall") or 0.0)
            if r > 0.0:
                return r
            return float(c.get("__quality_score") or 0.0)

        return sorted(
            cards,
            key=lambda c: (-_rating_value(c), str(c.get("name", "")).lower()),
        )

    if sort == "popular":
        return sorted(cards, key=lambda c: float(c.get("popularity", 0.0)), reverse=True)

    # ✅ premium sorts (T2)
    if sort == "quality":
        return sorted(
            cards,
            key=lambda c: (-float(c.get("__quality_score") or 0.0), str(c.get("name", "")).lower()),
        )

    if sort == "price_value":
        return sorted(
            cards,
            key=lambda c: (
                -float(c.get("__value_score") or 0.0),
                _parse_float_maybe(c.get("price", "")) or float("inf"),
                str(c.get("name", "")).lower(),
            ),
        )

    if sort == "match":
        return sorted(
            cards,
            key=lambda c: (
                -float(c.get("__match_score") or 0.0),
                -float(c.get("score", 0.0)),
                str(c.get("name", "")).lower(),
            ),
        )

    # ✅ relevance (default)
    # Se l’utente ha chiesto "rapporto qualità prezzo" => priorità a __value_score
    if value_intent:
        return sorted(
            cards,
            key=lambda c: (
                -float(c.get("__value_score") or 0.0),
                -float(c.get("score") or 0.0),
                _parse_float_maybe(c.get("price", "")) or float("inf"),
                float(c.get("__price_delta", 0.0)),
            ),
        )

    return sorted(
        cards,
        key=lambda c: (
            -float(c.get("score", 0.0)),
            float(c.get("__price_delta", 0.0)),
        ),
    )


def run_search(query: str, sort: str = "relevance", limit: int = MAX_RESULTS_DEFAULT) -> Dict[str, Any]:
    df = get_wines_df()
    limit = _clamp(int(limit or MAX_RESULTS_DEFAULT), 1, MAX_RESULTS_CAP)

    q = _norm(query)

    price_info = parse_price(q)
    region = parse_region(q)

    grapes_req = parse_grapes(q)
    aromas_req = parse_aromas(q)
    intensity_req = parse_intensity_request(q)
    typology_req = parse_typology_request(q)

    color_req = parse_color_request(q)

    value_intent = parse_value_intent(q)
    structured_req = parse_structured_keyword(q)
    location_terms = _extract_location_terms(df, q)

    # A9.1 — vitigni impliciti da denominazione (boost leggero solo per sort=match)
    implied_grapes = _implied_grapes_from_location_terms(location_terms) if (sort == "match" and not grapes_req) else []


    filtered = df

    filtered = _filter_by_price(filtered, price_info)

    if region:
        for col in ["region", "zone", "denomination", "country"]:
            filtered = _filter_by_text_contains(filtered, col, region)

    filtered = _filter_by_color(filtered, color_req)

    filtered = _filter_new_A_B_D(filtered, grapes_req, aromas_req, intensity_req, typology_req)

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
        }

        # ✅ qualità "pura" (senza boost) per il sort Quality
        quality_raw = _score_quality(r)

        s, pdlt = _score_row(r, price_info, boosts, value_intent=value_intent)
        card = _build_wine_card(r, rank=0, score=s, price_delta=pdlt)

        card["__quality_score"] = round(float(quality_raw), 6)

        pr_eff = _price_effective(r)
        if pr_eff and pr_eff > 0:
            # A9.3a — value_score più realistico: diminishing returns ~ sqrt(prezzo)
            card["__value_score"] = round(float(quality_raw / math.sqrt(pr_eff + 10.0)), 6)
        else:
            card["__value_score"] = 0.0

        # ✅ match score (T2)
        card["__match_score"] = round(
            float(_compute_match_score(
                r,
                grapes_req=grapes_req,
                aromas_req=aromas_req,
                intensity_req=intensity_req,
                typology_req=typology_req,
                structured_req=structured_req,
                location_terms=location_terms,
            )),
            6,
        )

        # A9.1 — piccolo boost sullo score (solo sort=match, no vitigni espliciti)
        if implied_grapes:
            _b = _row_implied_grape_boost(r, implied_grapes)
            if _b > 0.0:
                card["__implicit_grape_boost"] = round(float(_b), 6)
                card["score"] = round(float(card.get("score", 0.0)) + float(_b), 4)

        scored.append(card)

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
            "value_intent": value_intent,
            "location_terms": location_terms,   
        },
        "count": len(sorted_cards),
        "timestamp": int(_now()),
    }

    return {"results": sorted_cards, "meta": meta}


# =========================
# API helpers
# =========================

def _normalize_sort(sort: Optional[str]) -> str:
    allowed = {
        "relevance",
        "quality",
        "price_value",
        "match",
        "price_asc",
        "price_desc",
        "rating",   # legacy / compat
        "popular",
    }
    s = (sort or "relevance").strip()
    return s if s in allowed else "relevance"


# =========================
# Endpoints
# =========================

@app.post("/search")
def post_search(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    query = _norm(payload.get("query", ""))
    sort = _normalize_sort(payload.get("sort") or payload.get("sort_mode"))
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
    _ = get_location_index(df)
    payload = {
        "build_id": BUILD_ID,
        "csv_path": CSV_PATH,
        "rows": int(len(df)),
        "csv_mtime": CSV_CACHE.mtime,
        "last_load_ts": CSV_CACHE.last_load_ts,
        "location_index_mtime": LOCATION_INDEX_CACHE.mtime,
        "location_index_rows": LOCATION_INDEX_CACHE.rows,
        "location_index_last_build_ts": LOCATION_INDEX_CACHE.last_build_ts,
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


# =============================================================================
# Mini test manuali (A6.3) — regressioni rapide (da eseguire a mano)
# =============================================================================
# 1) Match location automatico (denom + comune) — atteso:
#    - Barolo Serralunga ~1.0
#    - Barolo Monforte ~0.5
# curl -s -X POST "http://127.0.0.1:8000/search" \
#   -H "Content-Type: application/json" \
#   -d '{"query":"barolo serralunga","sort":"match","limit":10}' \
# | python3 -c "import sys,json; d=json.load(sys.stdin); \
# print('location_terms:', d.get('meta',{}).get('filters',{}).get('location_terms')); \
# print('\n'.join([f\"{r.get('__match_score',0):>4}  {r.get('name','')} | denom={r.get('denomination','')} | zone={r.get('zone','')}\" for r in d.get('results',[])]))"
#
# 2) Denominazione tollerante a 'di' mancante — atteso: denom + comune strutturati
# curl -s -X POST "http://127.0.0.1:8000/search" \
#   -H "Content-Type: application/json" \
#   -d '{"query":"brunello montalcino","sort":"match","limit":5}' \
# | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('meta',{}).get('filters',{}).get('location_terms'))"
#
# 3) Sort premium — smoke test
# curl -s -X POST "http://127.0.0.1:8000/search" \
#   -H "Content-Type: application/json" \
#   -d '{"query":"vino rosso strutturato","sort":"quality","limit":5}' | python3 -m json.tool
#
# -----------------------------------------------------------------------------
# Nota rapida (non cambio ora)
# - La normalizzazione delle zone/denominazioni è token-based; se nel dataset arrivano
#   forme molto divergenti (es. abbreviazioni), potremmo aggiungere una normalizzazione
#   leggera (apostrofi/spazi) mantenendo il comportamento data-driven.
# -----------------------------------------------------------------------------

# =========================
# Mini test manuali (regressioni rapide) — A9.4
# =========================
# 1) Location match (deve restare: Serralunga=1.0, Monforte=0.5)
# curl -s -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" #   -d '{"query":"barolo serralunga","sort":"match","limit":10}' | python3 -c "import sys,json; d=json.load(sys.stdin); # print('rank_mode:', d.get('meta',{}).get('rank_mode')); # print('location_terms:', d.get('meta',{}).get('filters',{}).get('location_terms')); # print('\n'.join([f\"{r.get('__match_score',0):>4}  {r.get('name','')} | denom={r.get('denomination','')} | zone={r.get('zone','')}\" for r in d.get('results',[])[:5]]))"
#
# 2) Denom+comune (deve produrre termini strutturati)
# curl -s -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" #   -d '{"query":"brunello montalcino","sort":"match","limit":5}' | python3 -c "import sys,json; d=json.load(sys.stdin); # print('rank_mode:', d.get('meta',{}).get('rank_mode')); # print(d.get('meta',{}).get('filters',{}).get('location_terms'))"
#
# 3) Price/Value sorting sanity (controllo rapido value_score)
# curl -s -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" #   -d '{"query":"rosso strutturato","sort":"price_value","limit":10}' | python3 -c "import sys,json; d=json.load(sys.stdin); # print('rank_mode:', d.get('meta',{}).get('rank_mode')); # print('\n'.join([f\"{r.get('__value_score',0):>8}  €{r.get('price','')}  {r.get('name','')}\" for r in d.get('results',[])[:10]]))"
#
# Nota rapida (non cambio ora):
# - rank_mode è solo meta e non cambia il comportamento dell'API; serve per futuri switch UI (modalità ranking).
