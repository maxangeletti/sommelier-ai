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
    """
    Estrae aromi dal testo CSV e restituisce lista con icon mapping.
    
    Args:
        aromas_text: Stringa aromi dal CSV (es. "ciliegia, spezie, cuoio")
    
    Returns:
        Lista di dict [{"name": "ciliegia", "icon": "🍒"}, ...]
        Massimo 5 aromi per evitare UI clutter
    
    Examples:
        >>> get_aroma_icons("ciliegia, spezie, cuoio")
        [{"name": "ciliegia", "icon": "🍒"}, 
         {"name": "spezie", "icon": "🌿"},
         {"name": "cuoio", "icon": "🧳"}]
    """
    if not aromas_text or not isinstance(aromas_text, str):
        return []
    
    # Split e normalizza aromi
    aromas = [a.strip().lower().replace(" ", "_") for a in aromas_text.split(",")]
    
    result = []
    for aroma in aromas[:5]:  # Max 5 aromi
        if not aroma:
            continue
            
        # Cerca match esatto
        icon = AROMA_ICONS.get(aroma)
        
        # Se non trovato, cerca match parziale (es. "frutta rossa" → "frutta_rossa")
        if not icon:
            for key, val in AROMA_ICONS.items():
                if aroma in key or key in aroma:
                    icon = val
                    break
        
        # Aggiungi solo se trovato icon
        if icon:
            result.append({
                "name": aroma.replace("_", " ").title(),
                "icon": icon
            })
    
    return result


# =========================
# Badge "Ottimo Valore" Logic
# =========================

def should_show_value_badge(wine: Dict[str, Any], active_signals: Dict[str, Any]) -> bool:
    """
    Determina se mostrare badge 'Ottimo Valore' verde.
    
    Trigger logic:
    - value_score > 0.75 (qualità alta + prezzo contenuto)
    - OR value_intent=True (user cerca esplicitamente value)
    
    Args:
        wine: Dizionario vino con value_score
        active_signals: Segnali di ranking attivi
    
    Returns:
        True se badge deve essere mostrato
    
    Examples:
        >>> wine = {"value_score": 0.8, "price": 15, "quality": 0.85}
        >>> signals = {"value_intent": False}
        >>> should_show_value_badge(wine, signals)
        True
        
        >>> wine = {"value_score": 0.5, "price": 50}
        >>> signals = {"value_intent": True}
        >>> should_show_value_badge(wine, signals)
        True
    """
    value_score = wine.get("value_score", 0)
    value_intent = active_signals.get("value_intent", False)
    
    # Trigger 1: High value score
    if value_score > 0.75:
        return True
    
    # Trigger 2: User explicitly seeking value
    if value_intent:
        return True
    
    return False


# =========================
# Mock Data Helpers
# =========================

def get_mock_reviews_count(quality: float) -> int:
    """
    Genera mock reviews count basato su quality score.
    
    Args:
        quality: Quality score 0-1
    
    Returns:
        Reviews count (50-500 range)
    
    Examples:
        >>> get_mock_reviews_count(0.9)
        450
        >>> get_mock_reviews_count(0.5)
        250
    """
    if not quality or quality < 0 or quality > 1:
        return 100  # Default
    
    # Scala: quality 0.5 → 250 reviews, 1.0 → 500 reviews
    return int(quality * 500)


def get_mock_critic_score(quality: float) -> int:
    """
    Genera mock critic score (0-100) basato su quality.
    
    Args:
        quality: Quality score 0-1
    
    Returns:
        Critic score 0-100
    
    Examples:
        >>> get_mock_critic_score(0.9)
        90
        >>> get_mock_critic_score(0.75)
        75
    """
    if not quality or quality < 0 or quality > 1:
        return 70  # Default
    
    # Scala diretta: quality 0.8 → critic 80
    return int(quality * 100)
