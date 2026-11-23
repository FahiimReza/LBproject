# step07_dual_objective_ranker.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from trade_taxonomy import TRADES, WAGE_PRIORS

BASE_DIR = Path(__file__).resolve().parent
RANK_PATH = BASE_DIR / "checkpoints" / "gbdt_dual_objective_ranker.joblib"

FEATURE_COLS = [
    "offered_wage",
    "wage_ratio_to_worker",
    "distance_km",
    "est_commute_min",
    "wage_per_km",
    "nlp_max_conf",
    "nlp_worker_trade_prob",
    "nlp_trade_match",
    "legitimacy_score",
    "is_flagged_scam",
    "skill_overlap_ratio",
    "skill_wage_lift_norm",
    "worker_reliability_score",
]


def train_ranker(n_samples: int = 30_000) -> None:
    rng = np.random.default_rng(42)

    # Sample wages across realistic 10-trade Dhaka priors
    trade_choices = rng.choice(TRADES, size=n_samples)
    med_arr = np.array([WAGE_PRIORS[t]["median"] for t in trade_choices], dtype=float)
    offered_wage = np.clip(med_arr * rng.uniform(0.55, 1.65, size=n_samples), 400.0, 3200.0)

    wage_ratio = np.clip(rng.normal(1.05, 0.25, size=n_samples), 0.2, 3.0)
    distance_km = rng.uniform(0.2, 15.0, size=n_samples)
    est_commute_min = distance_km * 3.5 + 10.0
    wage_per_km = offered_wage / np.maximum(distance_km, 0.1)

    nlp_max_conf = rng.uniform(0.55, 0.99, size=n_samples)
    trade_match = rng.integers(0, 2, size=n_samples).astype(float)
    worker_trade_prob = np.where(
        trade_match > 0.5,
        nlp_max_conf,
        rng.uniform(0.01, 0.25, size=n_samples),
    )

    legitimacy = rng.uniform(0.55, 1.0, size=n_samples)
    is_scam = (legitimacy < 0.50).astype(float)
    skill_overlap = np.where(
        trade_match > 0.5,
        rng.uniform(0.5, 1.0, size=n_samples),
        rng.uniform(0.0, 0.4, size=n_samples),
    )
    skill_lift_norm = rng.uniform(0.05, 0.50, size=n_samples)
    reliability = rng.uniform(0.45, 0.99, size=n_samples)

    X = pd.DataFrame(
        {
            "offered_wage": offered_wage,
            "wage_ratio_to_worker": wage_ratio,
            "distance_km": distance_km,
            "est_commute_min": est_commute_min,
            "wage_per_km": wage_per_km,
            "nlp_max_conf": nlp_max_conf,
            "nlp_worker_trade_prob": worker_trade_prob,
            "nlp_trade_match": trade_match,
            "legitimacy_score": legitimacy,
            "is_flagged_scam": is_scam,
            "skill_overlap_ratio": skill_overlap,
            "skill_wage_lift_norm": skill_lift_norm,
            "worker_reliability_score": reliability,
        }
    )[FEATURE_COLS]

    fit_target = np.clip(
        0.45 * trade_match
        + 0.25 * np.clip(skill_overlap, 0.0, 1.0)
        + 0.15 * np.clip(1.0 - distance_km / 15.0, 0.0, 1.0)
        + 0.15 * legitimacy
        + rng.normal(0.0, 0.04, size=n_samples),
        0.0,
        1.0,
    )

    accept_logit = (
        -0.8
        + 1.6 * (wage_ratio - 1.0)
        - 0.12 * distance_km
        + 1.8 * (reliability - 0.7)
        + 0.9 * trade_match
        + rng.normal(0.0, 0.25, size=n_samples)
    )
    accept_prob = 1.0 / (1.0 + np.exp(-accept_logit))
    accept_target = (rng.uniform(0.0, 1.0, size=n_samples) < accept_prob).astype(int)

    X_tr, X_te, y_fit_tr, y_fit_te, y_acc_tr, y_acc_te = train_test_split(
        X, fit_target, accept_target, test_size=0.2, random_state=42
    )

    fit_model = GradientBoostingRegressor(random_state=42)
    fit_model.fit(X_tr, y_fit_tr)

    accept_model = GradientBoostingClassifier(random_state=42)
    accept_model.fit(X_tr, y_acc_tr)

    print(
        f"Fit R^2 (holdout): {fit_model.score(X_te, y_fit_te):.4f} | "
        f"Accept Acc (holdout): {accept_model.score(X_te, y_acc_te):.4f}"
    )

    RANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "trades": list(TRADES),
            "feature_cols": FEATURE_COLS,
            "fit_model": fit_model,
            "accept_model": accept_model,
        },
        RANK_PATH,
    )
    print(f"Saved dual-objective ranker ({len(TRADES)} trades) -> {RANK_PATH}")


if __name__ == "__main__":
    train_ranker()