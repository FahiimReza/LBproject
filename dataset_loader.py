# dataset_loader.py
import glob
from pathlib import Path
import pandas as pd

from trade_taxonomy import TRADES, TRADE_TO_ID, canonical_trade, normalize_text

BASE_DIR = Path(__file__).resolve().parent


def load_2m_full(csv_name: str = "dhaka_trade_dataset_2m_full.csv") -> pd.DataFrame:
    full_path = BASE_DIR / csv_name
    if not full_path.exists():
        raise FileNotFoundError(f"Expected dataset at {full_path}")

    df = pd.read_csv(full_path)

    # Drop empty texts/labels if any exist
    df = df.dropna(subset=["raw_text", "trade"]).copy()
    df["raw_text"] = df["raw_text"].astype(str)
    df["trade"] = df["trade"].map(canonical_trade)
    df["norm_text"] = df["raw_text"].map(normalize_text)
    df = df[df["norm_text"].str.len() > 0].copy()

    # Join template_group_id from part metadata files if row_id is present
    if "template_group_id" not in df.columns and "row_id" in df.columns:
        meta_files = sorted(glob.glob(str(BASE_DIR / "dhaka_trade_dataset_2m_part*of4_meta.csv")))
        if meta_files:
            meta_df = pd.concat(
                [pd.read_csv(m, usecols=["row_id", "template_group_id"]) for m in meta_files],
                ignore_index=True,
            ).drop_duplicates(subset=["row_id"])
            df = df.merge(meta_df, on="row_id", how="left")

    if "template_group_id" in df.columns:
        df["template_group"] = (
            df["template_group_id"]
            .fillna(df["norm_text"].str.slice(0, 24))
            .astype(str)
        )
    elif "template_group" not in df.columns:
        df["template_group"] = df["norm_text"].str.slice(0, 24)

    # Deduplicate on normalized text to prevent exact/near-duplicate leakage
    df = df.drop_duplicates(subset=["norm_text"]).reset_index(drop=True)
    df["label_id"] = df["trade"].map(TRADE_TO_ID)

    missing_trades = set(TRADES) - set(df["trade"].unique())
    if missing_trades:
        raise ValueError(f"Missing trades after loading/dedup: {sorted(missing_trades)}")

    return df


if __name__ == "__main__":
    loaded = load_2m_full()
    print(f"Loaded {len(loaded):,} unique rows across {loaded['trade'].nunique()} trades.")
    print(loaded["trade"].value_counts())