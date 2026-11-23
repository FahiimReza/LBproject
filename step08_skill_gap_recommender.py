# step08_skill_gap_recommender.py
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from trade_taxonomy import TRADES, WAGE_PRIORS, canonical_trade

BASE_DIR = Path(__file__).resolve().parent
SKILL_PATH = BASE_DIR / "checkpoints" / "skill_gap_recommender.joblib"

TRADE_SUBSKILLS: Dict[str, List[str]] = {
    "plumbing": [
        "pipe_fitting",
        "leak_repair",
        "commode_flush",
        "basin_tap",
        "tank_line",
    ],
    "electrical": [
        "house_wiring",
        "switchboard_repair",
        "breaker_mcb",
        "fan_light_setup",
        "voltage_check",
    ],
    "masonry": [
        "brick_laying",
        "wall_plaster",
        "tile_installation",
        "slab_concrete",
        "crack_repair",
    ],
    "carpentry": [
        "door_fitting",
        "window_frame",
        "wardrobe_hinges",
        "furniture_repair",
        "wood_polishing",
    ],
    "painting": [
        "wall_putty",
        "primer_coat",
        "interior_paint",
        "damp_treatment",
        "distemper_finish",
    ],
    "general_labor": [
        "material_loading",
        "site_helper",
        "sack_unloading",
        "debris_clearing",
        "basic_lifting",
    ],
    "appliance_repair": [
        "fridge_cooling",
        "ac_gas_service",
        "washing_machine",
        "geyser_repair",
        "compressor_check",
    ],
    "welding_metalwork": [
        "arc_welding",
        "grill_repair",
        "gate_shutter",
        "iron_fabrication",
        "railing_work",
    ],
    "cleaning_sanitation": [
        "deep_cleaning",
        "water_tank_clean",
        "drain_sanitation",
        "pest_control",
        "disinfection",
    ],
    "moving_shifting": [
        "furniture_packing",
        "home_relocation",
        "loading_crew",
        "fragile_wrapping",
        "office_shifting",
    ],
}

TRADE_WAGE_LIFT: Dict[str, float] = {
    "plumbing": 250.0,
    "electrical": 300.0,
    "masonry": 300.0,
    "carpentry": 280.0,
    "painting": 220.0,
    "general_labor": 150.0,
    "appliance_repair": 400.0,
    "welding_metalwork": 350.0,
    "cleaning_sanitation": 200.0,
    "moving_shifting": 220.0,
}


def extract_subskills(text: str, trade: str) -> List[str]:
    t = canonical_trade(trade)
    low = str(text).lower()
    matched = [
        sk
        for sk in TRADE_SUBSKILLS.get(t, [])
        if sk.replace("_", " ") in low or sk.split("_")[0] in low
    ]
    return matched or TRADE_SUBSKILLS.get(t, ["general_basics"])[:2]


def build_and_save_recommender(out_path: Path = SKILL_PATH) -> dict:
    corpus = [" ".join(TRADE_SUBSKILLS[t]).replace("_", " ") for t in TRADES]
    vec = TfidfVectorizer()
    mat = vec.fit_transform(corpus)
    artifact = {
        "trades": list(TRADES),
        "subskills": TRADE_SUBSKILLS,
        "wage_lift": TRADE_WAGE_LIFT,
        "wage_priors": WAGE_PRIORS,
        "vectorizer": vec,
        "trade_matrix": mat,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    return artifact


def recommend_skill_gaps(
    artifact: Optional[dict],
    trade: str,
    current_skills: List[str],
    current_avg_wage: float,
    profile_text: str = "",
    target_trade: Optional[str] = None,
) -> dict:
    art = artifact or {
        "trades": list(TRADES),
        "subskills": TRADE_SUBSKILLS,
        "wage_lift": TRADE_WAGE_LIFT,
    }
    cur_t = canonical_trade(trade)
    tgt_t = canonical_trade(target_trade) if target_trade else cur_t

    cur_set = set(current_skills or [])
    tgt_skills = art["subskills"].get(tgt_t, [f"{tgt_t}_basics"])
    missing = [s for s in tgt_skills if s not in cur_set][:2] or tgt_skills[:2]

    sim = 0.85 if cur_t == tgt_t else 0.30
    if "vectorizer" in art and "trade_matrix" in art and tgt_t in TRADES:
        try:
            query_str = " ".join(cur_set).replace("_", " ") + " " + str(profile_text)
            q_vec = art["vectorizer"].transform([query_str])
            t_idx = TRADES.index(tgt_t)
            sim = float(cosine_similarity(q_vec, art["trade_matrix"][t_idx : t_idx + 1])[0, 0])
            sim = float(np.clip(sim, 0.05, 0.99))
        except Exception:
            pass

    lift = float(art["wage_lift"].get(tgt_t, 200.0))
    per_skill_lift = round(lift / max(len(missing), 1), 0)

    return {
        "current_trade": cur_t,
        "target_trade": tgt_t,
        "cosine_similarity": round(sim, 3),
        "recommended_skills": [
            {"skill": s, "est_wage_lift_bdt": per_skill_lift} for s in missing
        ],
        "projected_daily_wage": round(float(current_avg_wage) + lift, 0),
    }


if __name__ == "__main__":
    build_and_save_recommender()
    print(f"Saved skill-gap recommender ({len(TRADES)} trades) -> {SKILL_PATH}")