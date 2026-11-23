# train_step03_classifiers.py
import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion

from dataset_loader import load_2m_full
from trade_taxonomy import ID_TO_TRADE, TRADES, TRADE_TO_ID

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"
NB_DIR = CKPT_DIR / "tfidf_nb_trade"
BERT_DIR = CKPT_DIR / "banglishbert_trade"
NB_DIR.mkdir(parents=True, exist_ok=True)
BERT_DIR.mkdir(parents=True, exist_ok=True)


def train_tfidf_nb(df) -> None:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(df, df["label_id"], groups=df["template_group"]))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=150_000, sublinear_tf=True)),
    ])

    X_train = vectorizer.fit_transform(train_df["norm_text"])
    X_test = vectorizer.transform(test_df["norm_text"])
    y_train = train_df["label_id"].to_numpy()
    y_test = test_df["label_id"].to_numpy()

    best_alpha = 0.5
    best_f1 = -1.0
    best_clf = None
    for alpha in [0.1, 0.3, 0.5, 1.0]:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        macro_f1 = f1_score(y_test, preds, average="macro")
        print(f"[NB alpha={alpha}] Grouped Holdout Macro-F1: {macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_alpha = alpha
            best_clf = clf

    preds = best_clf.predict(X_test)
    print(f"\nBest NB alpha={best_alpha} | Macro-F1={best_f1:.4f}")
    print(classification_report(y_test, preds, target_names=TRADES, digits=4))

    joblib.dump(vectorizer, NB_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(best_clf, NB_DIR / "multinomial_nb.joblib")
    (NB_DIR / "labels.json").write_text(
        json.dumps({"id2label": {str(k): v for k, v in ID_TO_TRADE.items()}, "label2id": TRADE_TO_ID}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved TF-IDF + MultinomialNB -> {NB_DIR}")


def train_banglishbert(df, model_name: str = "csebuetnlp/banglishbert", max_rows: int = 120_000) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("Skipping BanglishBERT: torch/transformers not installed.")
        return

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(df, df["label_id"], groups=df["template_group"]))

    train_sub = df.iloc[train_idx].sample(n=min(max_rows, len(train_idx)), random_state=42)
    val_sub = df.iloc[val_idx].sample(n=min(max_rows // 5, len(val_idx)), random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(list(texts), padding=True, truncation=True, max_length=96)
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TRADES),
        id2label=ID_TO_TRADE,
        label2id=TRADE_TO_ID,
    ).to(device)

    train_loader = DataLoader(TextDataset(train_sub["raw_text"], train_sub["label_id"]), batch_size=64, shuffle=True)
    val_loader = DataLoader(TextDataset(val_sub["raw_text"], val_sub["label_id"]), batch_size=128)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    best_val_f1 = -1.0

    for epoch in range(2):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels").numpy()
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(labels)

        val_f1 = f1_score(all_true, all_preds, average="macro")
        print(f"[BanglishBERT epoch {epoch + 1}] Grouped Val Macro-F1: {val_f1:.4f}")
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(BERT_DIR)
            tokenizer.save_pretrained(BERT_DIR)

    print(f"Saved BanglishBERT -> {BERT_DIR} (best Macro-F1={best_val_f1:.4f})")


if __name__ == "__main__":
    data = load_2m_full()
    print(f"Loaded {len(data):,} unique rows across {data['trade'].nunique()} trades.")
    train_tfidf_nb(data)
    train_banglishbert(data)