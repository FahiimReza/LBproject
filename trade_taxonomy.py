# trade_taxonomy.py
import re
from typing import Dict, List, Optional

TRADES: List[str] = [
    "plumbing",
    "electrical",
    "masonry",
    "carpentry",
    "painting",
    "general_labor",
    "appliance_repair",
    "welding_metalwork",
    "cleaning_sanitation",
    "moving_shifting",
]

TRADE_TO_ID: Dict[str, int] = {t: i for i, t in enumerate(TRADES)}
ID_TO_TRADE: Dict[int, str] = {i: t for i, t in enumerate(TRADES)}

ALIASES: Dict[str, str] = {
    # plumbing
    "plumber": "plumbing",
    "plumbing": "plumbing",
    "pani": "plumbing",
    # electrical (including dataset code "El" and common spelling/typo variants)
    "el": "electrical",
    "electrician": "electrical",
    "electracian": "electrical",
    "elecrician": "electrical",
    "electrical": "electrical",
    # masonry
    "mason": "masonry",
    "raj_mistri": "masonry",
    "masonry": "masonry",
    # carpentry
    "carpenter": "carpentry",
    "kath_mistri": "carpentry",
    "carpentry": "carpentry",
    # painting
    "painter": "painting",
    "painting": "painting",
    # general_labor
    "general": "general_labor",
    "general_labour": "general_labor",
    "general_labor": "general_labor",
    "labor": "general_labor",
    "labour": "general_labor",
    "kuli": "general_labor",
    "helper": "general_labor",
    # appliance_repair
    "appliance": "appliance_repair",
    "appliance_repair": "appliance_repair",
    "fridge_repair": "appliance_repair",
    "ac_repair": "appliance_repair",
    # welding_metalwork
    "welding": "welding_metalwork",
    "welder": "welding_metalwork",
    "metalwork": "welding_metalwork",
    "grill": "welding_metalwork",
    "welding_metalwork": "welding_metalwork",
    # cleaning_sanitation
    "cleaning": "cleaning_sanitation",
    "cleaner": "cleaning_sanitation",
    "sanitation": "cleaning_sanitation",
    "deep_clean": "cleaning_sanitation",
    "cleaning_sanitation": "cleaning_sanitation",
    # moving_shifting
    "moving": "moving_shifting",
    "shifting": "moving_shifting",
    "mover": "moving_shifting",
    "basa_bodol": "moving_shifting",
    "moving_shifting": "moving_shifting",
}

WAGE_PRIORS: Dict[str, Dict[str, float]] = {
    "general_labor": {"min": 500, "median": 900, "max": 1500},
    "cleaning_sanitation": {"min": 650, "median": 1150, "max": 2000},
    "moving_shifting": {"min": 750, "median": 1300, "max": 2300},
    "plumbing": {"min": 700, "median": 1300, "max": 2200},
    "painting": {"min": 750, "median": 1350, "max": 2200},
    "electrical": {"min": 800, "median": 1450, "max": 2400},
    "carpentry": {"min": 800, "median": 1500, "max": 2500},
    "masonry": {"min": 850, "median": 1550, "max": 2500},
    "welding_metalwork": {"min": 850, "median": 1600, "max": 2600},
    "appliance_repair": {"min": 900, "median": 1650, "max": 2800},
}

_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NORM_PATTERNS = [
    (r"\bbash?y\b|\bbash?ay\b", "basa"),
    (r"\bpanir\b|\bpani\b", "pani"),
    (r"\bsomoss?ha?\b", "somossa"),
    (r"\bmistri\b|\bmistry\b", "mistri"),
    (r"\belectracian\b|\belecrician\b|\belectrican\b", "electrician"),
    (r"\blagbe\b|\bdorkar\b", "lagbe"),
]


def canonical_trade(raw_label: object, label_map: Optional[Dict[str, str]] = None) -> str:
    key = str(raw_label).strip()
    if label_map and key in label_map:
        key = str(label_map[key]).strip()
    norm = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if norm in ALIASES:
        return ALIASES[norm]
    if norm in TRADES:
        return norm
    return "general_labor"


def normalize_text(text: str) -> str:
    s = str(text).lower().translate(_BANGLA_DIGITS)
    for pat, rep in _NORM_PATTERNS:
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s).strip()