# train_step06_fraud_guardrail.py
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from dataset_loader import load_2m_full
from trade_taxonomy import TRADES, TRADE_TO_ID, WAGE_PRIORS, canonical_trade

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
GUARD_PATH = CKPT_DIR / "lgbm_fraud_guardrail.joblib"

SCAM_PATTERNS = [
    "advance registration fee",
    "bkash 500 tk age pathan",
    "nagad fee dile kaj paben",
    "otp share korun",
    "বিকাশে অগ্রিম টাকা পাঠান",
]


def _extract_features(df: pd.DataFrame, trade_medians: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    t_series = df["trade"].map(canonical_trade)
    med_series = t_series.map(trade_medians).astype(float).clip(lower=100.0)
    wage = pd.to_numeric(df["offered_wage"], errors="coerce").fillna(med_series)
    lat = pd.to_numeric(df["lat"], errors="coerce").fillna(23.78)
    lon = pd.to_numeric(df["lon"], errors="coerce").fillna(90.39)
    txt = df["raw_text"].astype(str).str.lower()

    out["trade_id"] = t_series.map(TRADE_TO_ID).fillna(5).astype(int)
    out["offered_wage"] = wage
    out["wage_ratio_to_med"] = (wage / med_series).clip(0.05, 10.0)
    out["text_len"] = txt.str.len().clip(0, 1000)
    out["has_scam_kw"] = txt.str.contains(
        r"advance|bkash|nagad|registration\s*fee|otp|pin|বিকাশ|নগদ|অগ্রিম",
        regex=True,
    ).astype(int)
    out["out_of_dhaka"] = ((lat < 23.68) | (lat > 23.92) | (lon < 90.30) | (lon > 90.48)).astype(int)
    return out


def score_listings_for_guardrail(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    model = artifact["model"]
    feat_cols = artifact["feature_cols"]
    trade_medians = artifact["trade_medians"]
    X = _extract_features(df, trade_medians)[feat_cols]
    probs = model.predict_proba(X)[:, 1]
    res = df.copy()
    res["legitimacy_score"] = np.clip(probs, 0.0, 1.0)
    res["is_flagged_scam"] = res["legitimacy_score"] < 0.50
    return res


def train_guardrail(sample_per_trade: int = 15_000) -> None:
    rng = np.random.default_rng(42)
    raw_df = load_2m_full()

    legit_parts = []
    for t in TRADES:
        sub = raw_df[raw_df["trade"] == t]
        if len(sub) > 0:
            legit_parts.append(sub.sample(n=min(sample_per_trade, len(sub)), random_state=42))
    legit_df = pd.concat(legit_parts, ignore_index=True)
    legit_df["is_legit"] = 1

    trade_medians = {
        t: float(legit_df.loc[legit_df["trade"] == t, "offered_wage"].median())
        if (legit_df["trade"] == t).any()
        else WAGE_PRIORS[t]["median"]
        for t in TRADES
    }

    n_scam = len(legit_df) // 3
    scam_trades = rng.choice(TRADES, size=n_scam)
    scam_rows = []
    for t in scam_trades:
        med = trade_medians[t]
        mode = int(rng.integers(0, 3))
        if mode == 0:
            txt = f"{rng.choice(SCAM_PATTERNS)} for {t} job in dhaka"
            wage = float(med * rng.uniform(2.2, 4.5))
            lat, lon = 23.78, 90.39
        elif mode == 1:
            txt = f"urgent {t} work needed today"
            wage = float(med * rng.uniform(0.15, 0.40))
            lat, lon = 23.78, 90.39
        else:
            txt = f"remote {t} shift"
            wage = float(med)
            lat, lon = 24.50, 91.20
        scam_rows.append({"raw_text": txt, "trade": t, "offered_wage": round(wage, 2), "lat": lat, "lon": lon, "is_legit": 0})

    full_df = pd.concat(
        [legit_df[["raw_text", "trade", "offered_wage", "lat", "lon", "is_legit"]], pd.DataFrame(scam_rows)],
        ignore_index=True,
    )
    X = _extract_features(full_df, trade_medians)
    y = full_df["is_legit"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    print(f"Guardrail Validation ROC-AUC: {auc:.4f}")

    joblib.dump(
        {"model": clf, "feature_cols": list(X.columns), "trade_medians": trade_medians},
        GUARD_PATH,
    )
    print(f"Saved guardrail -> {GUARD_PATH}")


if __name__ == "__main__":
    train_guardrail()