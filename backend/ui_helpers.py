# ui_helpers.py
# UI-related helper functions for SommelierAI
# Badge logic, icon mapping, formatting utilities

from typing import Any, Dict, List, Optional


# =========================
# Aroma Icons Mapping
# =========================

AROMA_ICONS = {
    # Frutta
    "agrumi": "🍋",
    "limone": "🍋",
    "pompelmo": "🍋",
    "cedro": "🍋",
    "frutta_rossa": "🍒",
    "ciliegia": "🍒",
    "fragola": "🍓",
    "lampone": "🍒",
    "ribes": "🍒",
    "frutta_nera": "🫐",
    "mora": "🫐",
    "mirtillo": "🫐",
    "prugna": "🫐",
    "marasca": "🍒",
    "mela": "🍎",
    "pera": "🍐",
    "pesca": "🍑",
    "albicocca": "🍑",
    "frutta_tropicale": "🍍",
    "ananas": "🍍",
    "mango": "🥭",
    "passion": "🥭",
    
    # Fiori
    "fiori": "🌹",
    "floreale": "🌹",
    "rosa": "🌹",
    "violetta": "🪻",
    "viola": "🪻",
    "gelsomino": "🌸",
    "lavanda": "🌸",
    "camomilla": "🌼",
    "acacia": "🌼",
    
    # Spezie ed erbe
    "spezie": "🌿",
    "anice": "🌿",
    "anice_stellato": "🌿",
    "pepe": "🌶️",
    "pepe_nero": "🌶️",
    "pepe_verde": "🌶️",
    "cannella": "🌰",
    "chiodi_garofano": "🌰",
    "noce_moscata": "🌰",
    "vaniglia": "🥥",
    "zenzero": "🌿",
    "menta": "🌿",
    "basilico": "🌿",
    "timo": "🌿",
    "salvia": "🌿",
    "rosmarino": "🌿",
    "erbe": "🌿",
    "erbaceo": "🌿",
    
    # Terziario (invecchiamento)
    "cuoio": "🧳",
    "pelle": "🧳",
    "tabacco": "🍂",
    "foglia_secca": "🍂",
    "tostato": "☕",
    "caffe": "☕",
    "cacao": "🍫",
    "cioccolato": "🍫",
    "liquirizia": "⚫",
    "balsamico": "🌲",
    "resina": "🌲",
    "pino": "🌲",
    "legno": "🪵",
    "affumicato": "🪵",
    "minerale": "⛰️",
    "pietra": "⛰️",
    "grafite": "⛰️",
    "idrocarburi": "⛰️",
    
    # Dolci/confettura
    "miele": "🍯",
    "marmellata": "🍯",
    "confettura": "🍯",
    "frutta_secca": "🌰",
    "nocciola": "🌰",
    "mandorla": "🌰",
    "noce": "🌰",
}


def get_aroma_icons(aromas_text: Optional[str]) -> List[Dict[str, str]]:
    """Estrae aromi dal testo CSV e restituisce lista con icon mapping."""
    if not aromas_text or not isinstance(aromas_text, str):
        return []
    
    aromas = [a.strip().lower().replace(" ", "_") for a in aromas_text.split(",")]
    
    result = []
    for aroma in aromas[:5]:  # Max 5 aromi
        if not aroma:
            continue
            
        icon = AROMA_ICONS.get(aroma)
        
        if not icon:
            for key, val in AROMA_ICONS.items():
                if aroma in key or key in aroma:
                    icon = val
                    break
        
        if icon:
            result.append({
                "name": aroma.replace("_", " ").title(),
                "icon": icon
            })
    
    return result


# =========================
# Food Pairing Icons Mapping
# =========================

FOOD_PAIRING_ICONS = {
    # Carne
    "carne": "🥩",
    "bistecca": "🥩",
    "manzo": "🥩",
    "vitello": "🥩",
    "filetto": "🥩",
    "tagliata": "🥩",
    "grigliata": "🥩",
    "bbq": "🥩",
    "arrosto": "🍖",
    "brasato": "🍖",
    "stufato": "🍖",
    "maiale": "🐖",
    "agnello": "🐑",
    "selvaggi": "🦌",
    "selvaggina": "🦌",
    "cacciagione": "🦌",
    "cinghiale": "🦌",
    
    # Pesce
    "pesce": "🐟",
    "pesce_griglia": "🐟",
    "pesce_crudo": "🍣",
    "sushi": "🍣",
    "crudo": "🍣",
    "tartare": "🍣",
    "carpaccio": "🍣",
    "ostriche": "🦪",
    "frutti_mare": "🦐",
    "crostacei": "🦐",
    "gamberi": "🦐",
    "scampi": "🦐",
    "aragosta": "🦞",
    "tonno": "🐟",
    "salmone": "🐟",
    
    # Pasta & Pizza
    "pasta": "🍝",
    "spaghetti": "🍝",
    "tagliatelle": "🍝",
    "lasagne": "🍝",
    "pasta_sugo_verdure": "🍝",
    "pasta_sugo_pomodoro": "🍝",
    "risotto": "🍚",
    "pizza": "🍕",
    
    # Formaggi
    "formaggi": "🧀",
    "formaggio": "🧀",
    "pecorino": "🧀",
    "parmigiano": "🧀",
    "grana": "🧀",
    "gorgonzola": "🧀",
    "erborinati": "🧀",
    
    # Salumi
    "salumi": "🥓",
    "prosciutto": "🥓",
    "salame": "🥓",
    "mortadella": "🥓",
    "speck": "🥓",
    
    # Verdure
    "verdure": "🥗",
    "vegetariano": "🥗",
    "insalata": "🥗",
    "ortaggi": "🥗",
    
    # Dolci
    "dolci": "🍰",
    "dessert": "🍰",
    "torta": "🍰",
    "cioccolato": "🍫",
    "pasticceria": "🧁",
    
    # Contesti
    "aperitivo": "🥂",
    "apericena": "🥂",
    "stuzzichini": "🧀",
}


def get_food_pairing_icons(food_pairings_text: Optional[str]) -> List[Dict[str, str]]:
    """Estrae food pairings dal testo CSV e restituisce lista con icon mapping."""
    if not food_pairings_text or not isinstance(food_pairings_text, str):
        return []
    
    # Split su pipe o virgola
    separators = ["|", ",", ";"]
    foods = [food_pairings_text]
    for sep in separators:
        if sep in food_pairings_text:
            foods = food_pairings_text.split(sep)
            break
    
    foods = [f.strip().lower().replace(" ", "_") for f in foods]
    
    result = []
    seen_icons = set()
    
    for food in foods[:4]:  # Max 4 food pairings
        if not food:
            continue
            
        icon = FOOD_PAIRING_ICONS.get(food)
        
        if not icon:
            for key, val in FOOD_PAIRING_ICONS.items():
                if food in key or key in food:
                    icon = val
                    break
        
        if icon and icon not in seen_icons:
            result.append({
                "name": food.replace("_", " ").title(),
                "icon": icon
            })
            seen_icons.add(icon)
    
    return result


# =========================
# Badge "Ottimo Valore" Logic
# =========================

def should_show_value_badge(wine: Dict[str, Any], active_signals: Dict[str, Any]) -> bool:
    """Determina se mostrare badge 'Ottimo Valore' verde."""
    value_score = wine.get("value_score", 0)
    value_intent = active_signals.get("value_intent", False)
    
    if value_score > 0.75:
        return True
    
    if value_intent:
        return True
    
    return False


# =========================
# Mock Data Helpers
# =========================

def get_mock_reviews_count(quality: float) -> int:
    """Genera mock reviews count basato su quality score."""
    if not quality or quality < 0 or quality > 1:
        return 100
    
    return int(quality * 500)


def get_mock_critic_score(quality: float) -> int:
    """Genera mock critic score (0-100) basato su quality."""
    if not quality or quality < 0 or quality > 1:
        return 70
    
    return int(quality * 100)
