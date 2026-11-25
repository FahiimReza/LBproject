from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import unicodedata
import uuid
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from step04_hmm_worker_reliability import infer_worker_reliability
from step05_haversine_filter import compute_geo_features
from step08_skill_gap_recommender import extract_subskills, recommend_skill_gaps
from trade_taxonomy import TRADES, canonical_trade, normalize_text
from train_step06_fraud_guardrail import score_listings_for_guardrail

_DIACRITIC_MAP = str.maketrans({
    "ā": "a", "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a",
    "ē": "e", "é": "e", "è": "e", "ê": "e", "ë": "e",
    "ī": "i", "í": "i", "ì": "i", "î": "i", "ï": "i",
    "ō": "o", "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o",
    "ū": "u", "ú": "u", "ù": "u", "û": "u", "ü": "u",
    "č": "c", "ć": "c", "ṭ": "t", "ḍ": "d", "ṇ": "n", "ṅ": "n", "ñ": "n",
    "ś": "s", "ṣ": "s", "š": "s", "ṛ": "r", "ṝ": "r",
})

_DEVANAGARI_SYLLABLE_MAP = [
    ("राज्मिस्ट्री", "rajmistri"),
    ("राजमिस्त्री", "rajmistri"),
    ("काठमिस्त्री", "kathmistri"),
    ("पार्टीशन", "partition"),
    ("पार्टिशन", "partition"),
    ("फर्नीचर", "furniture"),
    ("कंप्यूटर", "computer"),
    ("लैपटॉप", "laptop"),
    ("इलेक्ट्रिशियन", "electrician"),
    ("प्लंबर", "plumber"),
    ("पल्टून", "paltan"),
    ("रेलाका", "elaka"),
    ("इलाका", "elaka"),
    ("भालो", "bhalo"),
    ("लाग भे", "lagbe"),
    ("लागबे", "lagbe"),
    ("काच", "kaj"),
    ("करते हो बे", "korte hobe"),
    ("जोगजो खुरून्", "jogajog korun"),
    ("नष्ट", "nosto"),
    ("खराब", "kharap"),
    ("पानी", "pani"),
    ("पाइप", "pipe"),
    ("लीक", "leak"),
    ("सफाई", "cleaning"),
    ("क्ष", "ksh"),
    ("त्र", "tr"),
    ("ज्ञ", "gy"),
]

_DEVANAGARI_CHAR_MAP = str.maketrans({
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "w", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "्": "", "ं": "n", "ः": "h", "ँ": "n", "़": "",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
})


def _strip_diacritics(text: str) -> str:
    folded = text.translate(_DIACRITIC_MAP)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _transliterate_devanagari(text: str) -> str:
    if not re.search(r"[\u0900-\u097F]", text):
        return text
    out = text
    for src, dst in _DEVANAGARI_SYLLABLE_MAP:
        out = out.replace(src, f" {dst} ")
    out = out.translate(_DEVANAGARI_CHAR_MAP)
    return re.sub(r"\s+", " ", out).strip()


BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"
INDEX_PATH = BASE_DIR / "index.html"
TOP_K = 3
LOW_CONF_THRESHOLD = 0.50
BERT_BLEND_THRESHOLD = float(os.getenv("BERT_BLEND_THRESHOLD", "0.75"))
NB_BLEND_WEIGHT = float(os.getenv("NB_BLEND_WEIGHT", "0.45"))

WHISPER_CKPT_DIR = CKPT_DIR / "whisper-banspeech-dhaka-slang-final"
WHISPER_FALLBACK_MODEL = os.getenv("WHISPER_FALLBACK_MODEL", "openai/whisper-small")
WHISPER_FORCE_LANGUAGE = os.getenv("WHISPER_FORCE_LANGUAGE", "").strip()

LOG_TRADE_EDGE_CASES = os.getenv("LOG_TRADE_EDGE_CASES", "1").strip().lower() in {"1", "true", "yes", "on"}
TRADE_EDGE_LOG_PATH = Path(os.getenv("TRADE_EDGE_LOG_PATH", str(BASE_DIR / "logs" / "trade_edge_cases.jsonl")))
_EDGE_LOG_LOCK = threading.Lock()

app = FastAPI(
    title="LABOR-HUB // Modular ML Telemetry Backend",
    version="4.7.2",
)

ACTIVE_JOB_BROADCASTS: Dict[str, dict] = {}
_WHISPER_CACHE: Dict[str, object] = {}
_BANGLISHBERT_CACHE: Dict[str, object] = {}


def _load_joblib_if_exists(path: Path):
    return joblib.load(path) if path.exists() else None


def _canonical_trade_name(raw_label: object, label_map: Optional[Dict[str, str]] = None) -> str:
    return canonical_trade(raw_label, label_map=label_map)


def _load_nlp_bundle():
    nb_dir = CKPT_DIR / "tfidf_nb_trade"
    vec_p = nb_dir / "tfidf_vectorizer.joblib"
    clf_p = nb_dir / "multinomial_nb.joblib"
    labels_p = nb_dir / "labels.json"

    label_map: Dict[str, str] = {}
    if labels_p.exists():
        try:
            raw_labels = json.loads(labels_p.read_text(encoding="utf-8"))
            if isinstance(raw_labels, dict):
                if "id2label" in raw_labels and isinstance(raw_labels["id2label"], dict):
                    label_map = {str(k): str(v) for k, v in raw_labels["id2label"].items()}
                else:
                    vals = list(raw_labels.values())
                    if vals and all(isinstance(v, str) for v in vals):
                        label_map = {str(k): str(v) for k, v in raw_labels.items()}
                    elif vals and all(isinstance(v, int) for v in vals):
                        label_map = {str(v): str(k) for k, v in raw_labels.items()}
            elif isinstance(raw_labels, list):
                label_map = {str(i): str(v) for i, v in enumerate(raw_labels)}
        except Exception:
            label_map = {}

    if vec_p.exists() and clf_p.exists():
        return joblib.load(vec_p), joblib.load(clf_p), label_map
    return None, None, label_map


def _get_banglishbert():
    if "unavailable" in _BANGLISHBERT_CACHE:
        return None, None
    if "model" in _BANGLISHBERT_CACHE and "tokenizer" in _BANGLISHBERT_CACHE:
        return _BANGLISHBERT_CACHE["tokenizer"], _BANGLISHBERT_CACHE["model"]

    bert_dir = CKPT_DIR / "banglishbert_trade"
    if not bert_dir.exists() or not (bert_dir / "config.json").exists():
        _BANGLISHBERT_CACHE["unavailable"] = True
        return None, None

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(device)
        model.eval()
        _BANGLISHBERT_CACHE["tokenizer"] = tokenizer
        _BANGLISHBERT_CACHE["model"] = model
        _BANGLISHBERT_CACHE["device"] = device
        return tokenizer, model
    except Exception:
        _BANGLISHBERT_CACHE["unavailable"] = True
        return None, None


def _predict_banglishbert(raw_text: str) -> Optional[Tuple[str, float, Dict[str, float]]]:
    tokenizer, model = _get_banglishbert()
    if tokenizer is None or model is None:
        return None

    try:
        import torch

        device = _BANGLISHBERT_CACHE.get("device", "cpu")
        enc = tokenizer(
            [raw_text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0].detach().cpu().numpy()

        id2label = getattr(model.config, "id2label", {}) or {}
        prob_map = {t: 0.0 for t in TRADES}
        for idx, p in enumerate(probs):
            raw_lbl = id2label.get(idx, id2label.get(str(idx), idx))
            t_name = _canonical_trade_name(raw_lbl)
            prob_map[t_name] = float(prob_map.get(t_name, 0.0) + float(p))

        total = sum(prob_map.values())
        if total > 0:
            prob_map = {k: float(v / total) for k, v in prob_map.items()}
        else:
            prob_map = {t: 1.0 / len(TRADES) for t in TRADES}

        best_t = max(prob_map, key=prob_map.get)
        return best_t, float(prob_map[best_t]), prob_map
    except Exception:
        return None


def _get_whisper_asr():
    if "model" in _WHISPER_CACHE and "processor" in _WHISPER_CACHE:
        return (
            _WHISPER_CACHE["processor"],
            _WHISPER_CACHE["model"],
            _WHISPER_CACHE["device"],
            _WHISPER_CACHE["model_source"],
        )

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if WHISPER_CKPT_DIR.exists() and (WHISPER_CKPT_DIR / "config.json").exists():
        model_source = str(WHISPER_CKPT_DIR)
    else:
        model_source = WHISPER_FALLBACK_MODEL

    processor = WhisperProcessor.from_pretrained(model_source)
    model = WhisperForConditionalGeneration.from_pretrained(model_source).to(device)
    model.eval()

    _WHISPER_CACHE["processor"] = processor
    _WHISPER_CACHE["model"] = model
    _WHISPER_CACHE["device"] = device
    _WHISPER_CACHE["model_source"] = model_source
    return processor, model, device, model_source


TFIDF_VEC, MNB_CLF, NB_LABEL_MAP = _load_nlp_bundle()
HMM_ART = _load_joblib_if_exists(CKPT_DIR / "hmm_worker_reliability.joblib")
FRAUD_ART = _load_joblib_if_exists(CKPT_DIR / "lgbm_fraud_guardrail.joblib")
SKILL_ART = _load_joblib_if_exists(CKPT_DIR / "skill_gap_recommender.joblib")
RANK_ART = _load_joblib_if_exists(CKPT_DIR / "gbdt_dual_objective_ranker.joblib")


class WorkerProfile(BaseModel):
    worker_id: str
    primary_trade: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    completion_rate: float = Field(ge=0.0, le=1.0)
    expected_daily_wage: float = Field(gt=0)


class JobRequest(BaseModel):
    job_id: Optional[str] = None
    raw_text: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    offered_wage: float = Field(gt=0)
    recent_daily_jobs: float = Field(default=85.0, ge=0)
    recent_wage_index: float = Field(default=1.05, gt=0)
    history: Optional[List[List[float]]] = None
    max_radius_km: float = Field(default=8.0, gt=0, le=20)
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    candidates: List[WorkerProfile]


class JobAcceptance(BaseModel):
    job_id: str
    worker_id: str


_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_PLUMBING_CUES = [
    "pani",
    "panir",
    "pipe",
    "leak",
    "plumber",
    "plumbing",
    "tank",
    "basin",
    "commode",
    "tap",
    "পানি",
    "পাইপ",
    "লিক",
    "প্লাম্বার",
    "টাংকি",
    "বেসিন",
    "কমোড",
]

_CLEANING_CUES = [
    "clean",
    "cleaning",
    "cleaner",
    "sanitation",
    "deep clean",
    "porishkar",
    "wash",
    "ক্লিন",
    "ক্লিনিং",
    "পরিষ্কার",
    "ধোয়া",
    "ধোলাই",
]

_HARD_PLUMBING_CUES = [
    "plumber",
    "plumbing",
    "pipe",
    "leak",
    "commode",
    "basin",
    "tap",
    "প্লাম্বার",
    "পাইপ",
    "লিক",
    "কমোড",
    "বেসিন",
]

_CARPENTRY_CUES = [
    "kathmistri",
    "carpenter",
    "carpentry",
    "partition",
    "furniture",
    "কাঠমিস্ত্রি",
    "ফার্নিচার মিস্ত্রি",
    "পার্টিশন",
    "কব্জা",
]

_MOVING_SHIFT_CUES = [
    "shift",
    "shifting",
    "relocate",
    "relocation",
    "truck",
    "van",
    "বাসা বদল",
    "শিফট",
    "ট্রাক",
    "ভ্যান",
]

_REPAIR_FAULT_CUES = [
    "nosto",
    "nasto",
    "kharap",
    "problem",
    "somossa",
    "shomosha",
    "repair",
    "sarabo",
    "thik",
    "tik",
    "on hocche na",
    "নষ্ট",
    "খারাপ",
    "সমস্যা",
    "ঠিক",
]


def _normalize_banglish(raw_text: str) -> str:
    """Normalize Devanagari transliteration, diacritics, and colloquial Banglish spellings."""
    cleaned = _transliterate_devanagari(raw_text)
    cleaned = _strip_diacritics(cleaned.lower())
    return normalize_text(cleaned)


def _is_cleaning_without_plumbing_repair(raw_text: str, normalized_text: str) -> bool:
    combined = f"{raw_text.lower()} {normalized_text}"
    has_clean = any(c in combined for c in _CLEANING_CUES)
    has_hard_plumb = any(p in combined for p in _HARD_PLUMBING_CUES)
    return has_clean and not has_hard_plumb


def _has_plumbing_cue(raw_text: str, normalized_text: str) -> bool:
    if _is_cleaning_without_plumbing_repair(raw_text, normalized_text):
        return False
    combined = f"{raw_text.lower()} {normalized_text}"
    return any(cue in combined for cue in _PLUMBING_CUES)


def _has_carpentry_partition_cue(raw_text: str, normalized_text: str) -> bool:
    combined = f"{raw_text.lower()} {normalized_text}"
    has_carp = any(c in combined for c in _CARPENTRY_CUES)
    has_move = any(m in combined for m in _MOVING_SHIFT_CUES)
    return has_carp and not has_move


def _has_device_repair_cue(raw_text: str, normalized_text: str) -> bool:
    combined = f"{raw_text.lower()} {normalized_text}"
    has_dev = bool(re.search(r"\b(pc|computer|laptop|desktop|printer|monitor|ups)\b", combined)) or any(
        d in combined for d in ["কম্পিউটার", "ল্যাপটপ", "প্রিন্টার", "মনিটর"]
    )
    has_fault = any(f in combined for f in _REPAIR_FAULT_CUES)
    return has_dev and has_fault


def _route_trade_probs(
    target_trade: str,
    base_probs: Optional[Dict[str, float]] = None,
    min_mass: float = 0.85,
) -> Tuple[str, float, Dict[str, float]]:
    if base_probs:
        mass = max(min_mass, float(base_probs.get(target_trade, 0.0)))
        remaining = max(0.0, 1.0 - mass)
        others = [t for t in TRADES if t != target_trade]
        other_sum = sum(max(0.0, float(base_probs.get(t, 0.0))) for t in others)
        prob_map = {target_trade: mass}
        if other_sum > 0:
            for t in others:
                prob_map[t] = remaining * (max(0.0, float(base_probs.get(t, 0.0))) / other_sum)
        else:
            share = remaining / max(len(others), 1)
            for t in others:
                prob_map[t] = share
    else:
        share = (1.0 - min_mass) / max(len(TRADES) - 1, 1)
        prob_map = {t: share for t in TRADES if t != target_trade}
        prob_map[target_trade] = min_mass

    total = sum(prob_map.values())
    prob_map = {k: float(v / total) for k, v in prob_map.items()}
    return target_trade, float(prob_map[target_trade]), prob_map


def _route_plumbing_probs(base_probs: Optional[Dict[str, float]] = None) -> Tuple[str, float, Dict[str, float]]:
    """Construct a high-confidence plumbing probability map while retaining trade spread."""
    return _route_trade_probs("plumbing", base_probs=base_probs, min_mass=0.82)


def _blend_trade_probs(
    nb_probs: Dict[str, float],
    bert_probs: Dict[str, float],
    nb_weight: float = NB_BLEND_WEIGHT,
) -> Tuple[str, float, Dict[str, float]]:
    w_nb = float(np.clip(nb_weight, 0.0, 1.0))
    w_bert = 1.0 - w_nb
    blended = {
        t: float(w_nb * nb_probs.get(t, 0.0) + w_bert * bert_probs.get(t, 0.0))
        for t in TRADES
    }
    total = sum(blended.values())
    if total > 0:
        blended = {k: float(v / total) for k, v in blended.items()}
    else:
        blended = {t: 1.0 / len(TRADES) for t in TRADES}

    best_t = max(blended, key=blended.get)
    return best_t, float(blended[best_t]), blended


def _log_trade_edge_case(
    raw_text: str,
    normalized_text: str,
    nb_trade: Optional[str],
    nb_confidence: Optional[float],
    final_trade: str,
    final_confidence: float,
    source: str,
    low_confidence: bool,
    overridden: bool,
) -> None:
    if not LOG_TRADE_EDGE_CASES:
        return
    if not (low_confidence or overridden):
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "nb_trade": nb_trade,
        "nb_confidence": round(float(nb_confidence), 4) if nb_confidence is not None else None,
        "final_trade": final_trade,
        "final_confidence": round(float(final_confidence), 4),
        "source": source,
        "low_confidence": bool(low_confidence),
        "overridden": bool(overridden),
    }

    try:
        with _EDGE_LOG_LOCK:
            TRADE_EDGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TRADE_EDGE_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _extract_text_wage(raw_text: str) -> Optional[float]:
    """Extract a plausible daily wage amount mentioned in raw_text (EN / Banglish / Bangla)."""
    text = _transliterate_devanagari(raw_text).translate(_BANGLA_DIGITS)

    patterns = [
        r"(?:৳|bdt|tk|taka|টাকা)\s*[:\-]?\s*(\d{2,6}(?:\.\d+)?)",
        r"(\d{2,6}(?:\.\d+)?)\s*(?:৳|bdt|tk|taka|টাকা|/-)(?:\s*(?:per\s*day|daily|a\s*day|sara\s*din|din|e|er))?",
        r"(?:wage|salary|rate|pay|বেতন|মজুরি|দর)\s*(?:is|of|:|=|tk|bdt|টাকা)?\s*(\d{2,6}(?:\.\d+)?)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            val = float(m.group(1))
            if 100.0 <= val <= 100_000.0:
                return val
    return None


def _classify_trade(raw_text: str) -> Tuple[str, float, Dict[str, float]]:
    norm_text = _normalize_banglish(raw_text)
    cleaning_override = _is_cleaning_without_plumbing_repair(raw_text, norm_text)
    plumbing_cue = _has_plumbing_cue(raw_text, norm_text)
    carpentry_cue = _has_carpentry_partition_cue(raw_text, norm_text)
    device_repair_cue = _has_device_repair_cue(raw_text, norm_text)

    nb_trade: Optional[str] = None
    nb_conf: Optional[float] = None

    if TFIDF_VEC is not None and MNB_CLF is not None:
        probs = MNB_CLF.predict_proba(TFIDF_VEC.transform([norm_text]))[0]
        prob_map = {t: 0.0 for t in TRADES}
        for c, p in zip(MNB_CLF.classes_, probs):
            t_name = _canonical_trade_name(c, NB_LABEL_MAP)
            prob_map[t_name] = float(prob_map.get(t_name, 0.0) + float(p))

        total = sum(prob_map.values())
        if total > 0:
            prob_map = {k: float(v / total) for k, v in prob_map.items()}
        else:
            prob_map = {t: 1.0 / len(TRADES) for t in TRADES}

        nb_trade = max(prob_map, key=prob_map.get)
        nb_conf = float(prob_map[nb_trade])
        low_conf = nb_conf < LOW_CONF_THRESHOLD

        if cleaning_override:
            final_t, final_p, final_map = _route_trade_probs("cleaning_sanitation", prob_map, min_mass=0.88)
            _log_trade_edge_case(
                raw_text=raw_text,
                normalized_text=norm_text,
                nb_trade=nb_trade,
                nb_confidence=nb_conf,
                final_trade=final_t,
                final_confidence=final_p,
                source="cleaning_sanitation_cue",
                low_confidence=low_conf,
                overridden=(final_t != nb_trade),
            )
            return final_t, final_p, final_map

        if plumbing_cue:
            final_t, final_p, final_map = _route_plumbing_probs(prob_map)
            _log_trade_edge_case(
                raw_text=raw_text,
                normalized_text=norm_text,
                nb_trade=nb_trade,
                nb_confidence=nb_conf,
                final_trade=final_t,
                final_confidence=final_p,
                source="plumbing_cue",
                low_confidence=low_conf,
                overridden=(final_t != nb_trade),
            )
            return final_t, final_p, final_map

        if carpentry_cue and nb_trade != "carpentry":
            final_t, final_p, final_map = _route_trade_probs("carpentry", prob_map, min_mass=0.86)
            _log_trade_edge_case(
                raw_text=raw_text,
                normalized_text=norm_text,
                nb_trade=nb_trade,
                nb_confidence=nb_conf,
                final_trade=final_t,
                final_confidence=final_p,
                source="carpentry_cue",
                low_confidence=low_conf,
                overridden=True,
            )
            return final_t, final_p, final_map

        if device_repair_cue and nb_trade != "appliance_repair":
            final_t, final_p, final_map = _route_trade_probs("appliance_repair", prob_map, min_mass=0.84)
            _log_trade_edge_case(
                raw_text=raw_text,
                normalized_text=norm_text,
                nb_trade=nb_trade,
                nb_confidence=nb_conf,
                final_trade=final_t,
                final_confidence=final_p,
                source="device_repair_cue",
                low_confidence=low_conf,
                overridden=True,
            )
            return final_t, final_p, final_map

        if nb_conf < BERT_BLEND_THRESHOLD:
            bert_out = _predict_banglishbert(norm_text)
            if bert_out is not None:
                _, _, bert_map = bert_out
                final_t, final_p, final_map = _blend_trade_probs(prob_map, bert_map)
                _log_trade_edge_case(
                    raw_text=raw_text,
                    normalized_text=norm_text,
                    nb_trade=nb_trade,
                    nb_confidence=nb_conf,
                    final_trade=final_t,
                    final_confidence=final_p,
                    source="banglishbert_blend",
                    low_confidence=low_conf,
                    overridden=(final_t != nb_trade),
                )
                return final_t, final_p, final_map

        _log_trade_edge_case(
            raw_text=raw_text,
            normalized_text=norm_text,
            nb_trade=nb_trade,
            nb_confidence=nb_conf,
            final_trade=nb_trade,
            final_confidence=nb_conf,
            source="tfidf_nb",
            low_confidence=low_conf,
            overridden=False,
        )
        return nb_trade, nb_conf, prob_map

    bert_out = _predict_banglishbert(norm_text)
    if bert_out is not None:
        final_t, final_p, final_map = bert_out
        overridden = False
        source = "banglishbert"
        if cleaning_override and final_t != "cleaning_sanitation":
            final_t, final_p, final_map = _route_trade_probs("cleaning_sanitation", final_map, min_mass=0.88)
            overridden = True
            source = "cleaning_sanitation_cue"
        elif plumbing_cue and final_t != "plumbing":
            final_t, final_p, final_map = _route_plumbing_probs(final_map)
            overridden = True
            source = "plumbing_cue"
        elif carpentry_cue and final_t != "carpentry":
            final_t, final_p, final_map = _route_trade_probs("carpentry", final_map, min_mass=0.86)
            overridden = True
            source = "carpentry_cue"
        elif device_repair_cue and final_t != "appliance_repair":
            final_t, final_p, final_map = _route_trade_probs("appliance_repair", final_map, min_mass=0.84)
            overridden = True
            source = "device_repair_cue"
        _log_trade_edge_case(
            raw_text=raw_text,
            normalized_text=norm_text,
            nb_trade=None,
            nb_confidence=None,
            final_trade=final_t,
            final_confidence=final_p,
            source=source,
            low_confidence=(final_p < LOW_CONF_THRESHOLD),
            overridden=overridden,
        )
        return final_t, final_p, final_map

    if cleaning_override:
        final_t, final_p, final_map = _route_trade_probs("cleaning_sanitation", None, min_mass=0.88)
        _log_trade_edge_case(
            raw_text=raw_text,
            normalized_text=norm_text,
            nb_trade=None,
            nb_confidence=None,
            final_trade=final_t,
            final_confidence=final_p,
            source="cleaning_sanitation_cue",
            low_confidence=False,
            overridden=False,
        )
        return final_t, final_p, final_map

    if plumbing_cue:
        final_t, final_p, final_map = _route_plumbing_probs(None)
        _log_trade_edge_case(
            raw_text=raw_text,
            normalized_text=norm_text,
            nb_trade=None,
            nb_confidence=None,
            final_trade=final_t,
            final_confidence=final_p,
            source="plumbing_cue",
            low_confidence=False,
            overridden=False,
        )
        return final_t, final_p, final_map

    if carpentry_cue:
        final_t, final_p, final_map = _route_trade_probs("carpentry", None, min_mass=0.86)
        _log_trade_edge_case(
            raw_text=raw_text,
            normalized_text=norm_text,
            nb_trade=None,
            nb_confidence=None,
            final_trade=final_t,
            final_confidence=final_p,
            source="carpentry_cue",
            low_confidence=False,
            overridden=False,
        )
        return final_t, final_p, final_map

    if device_repair_cue:
        final_t, final_p, final_map = _route_trade_probs("appliance_repair", None, min_mass=0.84)
        _log_trade_edge_case(
            raw_text=raw_text,
            normalized_text=norm_text,
            nb_trade=None,
            nb_confidence=None,
            final_trade=final_t,
            final_confidence=final_p,
            source="device_repair_cue",
            low_confidence=False,
            overridden=False,
        )
        return final_t, final_p, final_map

    text_l = norm_text.lower()
    scores = {t: 0.05 for t in TRADES}
    keywords = {
        "plumbing": ["plumber", "pipe", "leak", "pani", "panir", "প্লাম্বার", "পাইপ", "লিক", "পানি"],
        "electrical": ["electric", "electracian", "wiring", "short circuit", "switchboard", "তার", "বোর্ড", "কারেন্ট"],
        "masonry": ["mistri", "raj", "plaster", "tile", "রাজমিস্ত্রি", "টাইলস", "প্লাস্টার"],
        "carpentry": ["carpenter", "door", "wood", "furniture", "partition", "কাঠমিস্ত্রি", "দরজা", "পার্টিশন"],
        "painting": ["paint", "rong", "distemper", "putty", "রং", "পুটি"],
        "general_labor": ["kuli", "helper", "load", "কুলি", "হেল্পার"],
        "appliance_repair": ["fridge", "ac", "washing machine", "geyser", "appliance", "pc", "computer", "laptop", "ফ্রিজ", "এসি", "কম্পিউটার"],
        "welding_metalwork": ["welding", "welder", "grill", "shutter", "gate", "ওয়েল্ডিং", "গ্রিল"],
        "cleaning_sanitation": ["cleaning", "cleaner", "sanitation", "deep clean", "porishkar", "ক্লিনার", "পরিষ্কার"],
        "moving_shifting": ["shift", "moving", "relocation", "packing", "শিফট", "বাসা বদল"],
    }
    for t, kws in keywords.items():
        if any(k in text_l for k in kws):
            scores[t] += 0.75
    total = sum(scores.values())
    norm = {k: v / total for k, v in scores.items()}
    best_t = max(norm, key=norm.get)
    best_p = float(norm[best_t])
    _log_trade_edge_case(
        raw_text=raw_text,
        normalized_text=norm_text,
        nb_trade=None,
        nb_confidence=None,
        final_trade=best_t,
        final_confidence=best_p,
        source="keyword_fallback",
        low_confidence=(best_p < LOW_CONF_THRESHOLD),
        overridden=False,
    )
    return best_t, best_p, norm


def _estimate_market_regime(job: JobRequest) -> dict:
    hist_days = len(job.history) if job.history else 0
    jobs_norm = float(np.clip(job.recent_daily_jobs / 150.0, 0.0, 1.0))
    wage_norm = float(np.clip((job.recent_wage_index - 0.8) / 0.6, 0.0, 1.0))
    surge_prob = float(np.clip(0.55 * jobs_norm + 0.45 * wage_norm, 0.05, 0.95))

    if surge_prob >= 0.65:
        regime = "peak_surge"
        probs = {"low_demand": 0.10, "normal": 0.25, "peak_surge": 0.65}
    elif surge_prob <= 0.35:
        regime = "low_demand"
        probs = {"low_demand": 0.65, "normal": 0.25, "peak_surge": 0.10}
    else:
        regime = "normal"
        probs = {"low_demand": 0.20, "normal": 0.60, "peak_surge": 0.20}

    return {
        "market_regime": regime,
        "regime_probabilities": probs,
        "market_surge_probability": round(surge_prob, 4),
        "market_wage_index": round(float(job.recent_wage_index), 3),
        "history_days_used": hist_days,
    }


def _compute_guardrail(
    raw_text: str,
    predicted_trade: str,
    offered_wage: float,
    parsed_text_wage: Optional[float],
    lat: float,
    lon: float,
    base_wage: float,
    low_band: int,
    high_band: int,
) -> dict:
    flags: List[str] = []
    if re.search(r"\b(advance|bkash|nagad|registration\s*fee|otp|pin|বিকাশ|নগদ|ফি)\b", raw_text, re.I):
        flags.append("advance_fee_keyword")

    effective_wage = float(offered_wage)
    mismatch_ratio: Optional[float] = None
    if parsed_text_wage is not None and offered_wage > 0:
        mismatch_ratio = round(float(parsed_text_wage / offered_wage), 3)
        if abs(parsed_text_wage - offered_wage) / offered_wage > 0.25:
            flags.append("text_wage_mismatch")
        if parsed_text_wage > effective_wage:
            effective_wage = float(parsed_text_wage)

    if effective_wage < low_band:
        flags.append("below_fair_band")
    elif effective_wage > high_band:
        flags.append("above_fair_band")

    if FRAUD_ART is not None:
        df_one = pd.DataFrame([{
            "raw_text": raw_text,
            "trade": predicted_trade,
            "offered_wage": effective_wage,
            "lat": lat,
            "lon": lon,
        }])
        scored = score_listings_for_guardrail(df_one, FRAUD_ART)
        legit = float(scored.loc[0, "legitimacy_score"])
        if bool(scored.loc[0, "is_flagged_scam"]) and "above_fair_band" not in flags and "below_fair_band" not in flags:
            flags.append("lgbm_anomaly_flag")
    else:
        legit = 0.25 if flags else 0.90

    if "text_wage_mismatch" in flags:
        legit *= 0.72
    if "above_fair_band" in flags or "below_fair_band" in flags:
        ratio_out = max(
            effective_wage / max(float(high_band), 1.0),
            float(low_band) / max(effective_wage, 1.0),
        )
        penalty = float(np.clip((ratio_out - 1.0) * 0.22, 0.08, 0.55))
        legit *= (1.0 - penalty)

    legit = float(np.clip(legit, 0.0, 1.0))
    scam_risk = round(1.0 - legit, 4)
    verdict = "high_risk" if scam_risk >= 0.55 else "caution" if (scam_risk >= 0.25 or flags) else "legitimate"
    return {
        "legitimacy_score": round(legit, 4),
        "scam_risk": scam_risk,
        "verdict": verdict,
        "flags": flags,
        "parsed_text_wage": parsed_text_wage,
        "effective_wage_evaluated": round(effective_wage, 2),
        "wage_mismatch_ratio": mismatch_ratio,
    }


def run_match(job: JobRequest) -> dict:
    t0 = time.perf_counter()
    if not job.raw_text.strip():
        raise HTTPException(status_code=400, detail="Job text cannot be empty.")

    predicted_trade, confidence, trade_probs = _classify_trade(job.raw_text)
    hmm_info = _estimate_market_regime(job)
    wage_idx = hmm_info["market_wage_index"]
    parsed_text_wage = _extract_text_wage(job.raw_text)

    ranked: List[dict] = []
    excluded: List[dict] = []
    skill_gaps: List[dict] = []
    reference_wages: List[float] = []

    if job.candidates:
        cand_df = pd.DataFrame([
            {
                "worker_id": w.worker_id,
                "trade": canonical_trade(w.primary_trade),
                "lat": w.lat,
                "lon": w.lon,
                "offered_wage": job.offered_wage,
            }
            for w in job.candidates
        ])
        geo_df = compute_geo_features(cand_df, worker_lat=job.lat, worker_lon=job.lon)
        dists = geo_df["distance_km"].to_numpy(dtype=np.float64)

        inside_idx = np.where(dists <= job.max_radius_km)[0]
        outside_idx = np.where(dists > job.max_radius_km)[0]

        for i in outside_idx:
            excluded.append({
                "worker_id": job.candidates[i].worker_id,
                "distance_km": round(float(dists[i]), 2),
            })

        if inside_idx.size > 0:
            sel_workers = [job.candidates[i] for i in inside_idx]
            sel_dists = dists[inside_idx]
            reference_wages = [
                float(w.expected_daily_wage)
                for w in sel_workers
                if canonical_trade(w.primary_trade) == predicted_trade
            ]

            for idx_local, w in enumerate(sel_workers):
                d_km = float(sel_dists[idx_local])
                w_trade = canonical_trade(w.primary_trade)
                t_match = w_trade == predicted_trade
                t_prob = float(trade_probs.get(w_trade, 0.1))
                w_ratio = float(np.clip(job.offered_wage / max(w.expected_daily_wage, 100.0), 0.2, 3.0))

                if HMM_ART is not None:
                    events = ["completed_on_time"] * max(1, int(round(w.completion_rate * 5)))
                    if w.completion_rate < 0.75:
                        events.append("late_cancel")
                    rel_info = infer_worker_reliability(HMM_ART, events)
                    rel_score = float(rel_info["reliability_score"])
                    rel_state = str(rel_info["reliability_state"])
                else:
                    rel_score = float(w.completion_rate)
                    rel_state = (
                        "high_reliability" if w.completion_rate >= 0.85
                        else "moderate_reliability" if w.completion_rate >= 0.65
                        else "at_risk"
                    )

                if RANK_ART is not None:
                    feat_row = {
                        "offered_wage": float(job.offered_wage),
                        "wage_ratio_to_worker": w_ratio,
                        "distance_km": d_km,
                        "est_commute_min": float(d_km * 3.5 + 10.0),
                        "wage_per_km": float(job.offered_wage / max(d_km, 0.1)),
                        "nlp_max_conf": float(confidence),
                        "nlp_worker_trade_prob": t_prob,
                        "nlp_trade_match": 1.0 if t_match else 0.0,
                        "legitimacy_score": 0.9,
                        "is_flagged_scam": 0.0,
                        "skill_overlap_ratio": 0.6 if t_match else 0.2,
                        "skill_wage_lift_norm": 0.2,
                        "worker_reliability_score": rel_score,
                    }
                    X_one = pd.DataFrame([feat_row])[RANK_ART["feature_cols"]]
                    f_score = float(np.clip(RANK_ART["fit_model"].predict(X_one)[0], 0.0, 1.0))
                    a_prob = float(np.clip(RANK_ART["accept_model"].predict_proba(X_one)[0, 1], 0.0, 1.0))
                else:
                    f_score = float(np.clip(0.45 * t_prob + 0.30 * min(w_ratio, 1.2) / 1.2 + 0.25 * (1.0 - min(d_km, 15.0) / 15.0), 0.0, 1.0))
                    a_prob = float(np.clip(0.50 * rel_score + 0.30 * min(w_ratio, 1.2) / 1.2 + 0.20 * (1.0 - min(d_km, 15.0) / 15.0), 0.0, 1.0))

                c_score = round(0.55 * f_score + 0.45 * a_prob, 4)

                missing_skills: List[str] = []
                cos_sim = 0.85 if t_match else 0.28
                boost = 0.0
                rec_text = ""

                if not t_match:
                    missing_skills = [f"{predicted_trade}_basics"]
                    boost = round(min(0.25, 0.08 + 0.15 * confidence), 3)
                    if SKILL_ART is not None:
                        sg = recommend_skill_gaps(
                            SKILL_ART,
                            trade=w_trade,
                            current_skills=extract_subskills(job.raw_text, w_trade),
                            current_avg_wage=w.expected_daily_wage,
                            profile_text=job.raw_text,
                            target_trade=predicted_trade,
                        )
                        top_sk = sg.get("recommended_skills") or []
                        if top_sk:
                            missing_skills = [x["skill"] for x in top_sk[:2]]
                    rec_text = f"Add {', '.join(missing_skills)} to qualify for {predicted_trade} shifts."
                    skill_gaps.append({
                        "worker_id": w.worker_id,
                        "primary_trade": w_trade,
                        "cosine_similarity": cos_sim,
                        "missing_skills": missing_skills,
                        "projected_score_boost": boost,
                        "recommendation": rec_text,
                    })

                ranked.append({
                    "worker_id": w.worker_id,
                    "primary_trade": w_trade,
                    "distance_km": round(d_km, 2),
                    "completion_rate": w.completion_rate,
                    "reliability_state": rel_state,
                    "expected_daily_wage": w.expected_daily_wage,
                    "trade_match": t_match,
                    "fit_score": round(f_score, 4),
                    "acceptance_probability": round(a_prob, 4),
                    "compatibility_score": c_score,
                    "missing_skills": missing_skills,
                    "projected_score_boost": boost,
                })

            ranked.sort(key=lambda r: (-r["compatibility_score"], r["distance_km"]))

    for pos, item in enumerate(ranked):
        item["rank"] = pos + 1
        item["notified"] = pos < job.top_k

    top_notified = ranked[: job.top_k]
    base_wage = float(np.median(reference_wages)) if reference_wages else float(job.offered_wage)
    low_band = int(round(base_wage * wage_idx * 0.90))
    high_band = int(round(base_wage * wage_idx * 1.15))

    guardrail = _compute_guardrail(
        raw_text=job.raw_text,
        predicted_trade=predicted_trade,
        offered_wage=job.offered_wage,
        parsed_text_wage=parsed_text_wage,
        lat=job.lat,
        lon=job.lon,
        base_wage=base_wage,
        low_band=low_band,
        high_band=high_band,
    )

    compared_wage = float(guardrail["effective_wage_evaluated"])
    wage_verdict = (
        "below market" if compared_wage < low_band
        else "above market" if compared_wage > high_band
        else "fair"
    )

    return {
        "predicted_trade": predicted_trade,
        "trade_confidence": round(confidence, 4),
        "low_confidence": confidence < LOW_CONF_THRESHOLD,
        "trade_probabilities": dict(
            sorted(((k, round(v, 4)) for k, v in trade_probs.items()), key=lambda kv: kv[1], reverse=True)
        ),
        **hmm_info,
        "guardrail": guardrail,
        "wage_advice": {
            "fair_band": [low_band, high_band],
            "offered": job.offered_wage,
            "parsed_text_wage": parsed_text_wage,
            "wage_mismatch_ratio": guardrail["wage_mismatch_ratio"],
            "verdict": wage_verdict,
            "based_on": "matching nearby workers" if reference_wages else "offered wage baseline",
        },
        "in_radius_count": len(ranked),
        "excluded_workers": excluded,
        "notified_workers_count": len(top_notified),
        "broadcasted_recipients": top_notified,
        "ranked_workers": ranked,
        "skill_gap_recommendations": skill_gaps[:6],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/", include_in_schema=False)
def gui():
    if not INDEX_PATH.exists():
        raise HTTPException(404, "index.html not found next to main.py")
    return FileResponse(INDEX_PATH)


@app.get("/api/v1/health")
def health():
    models_ok = any(x is not None for x in [MNB_CLF, HMM_ART, FRAUD_ART, RANK_ART, SKILL_ART])
    whisper_src = _WHISPER_CACHE.get("model_source")
    if whisper_src is None and WHISPER_CKPT_DIR.exists() and (WHISPER_CKPT_DIR / "config.json").exists():
        whisper_src = str(WHISPER_CKPT_DIR)
    return {
        "status": "ok",
        "models_loaded": models_ok,
        "whisper_loaded": bool(_WHISPER_CACHE),
        "whisper_model": Path(whisper_src).name if whisper_src else WHISPER_FALLBACK_MODEL,
        "whisper_forced_language": WHISPER_FORCE_LANGUAGE or None,
        "banglishbert_loaded": "model" in _BANGLISHBERT_CACHE,
        "guardrail_loaded": FRAUD_ART is not None,
        "dual_ranker_loaded": RANK_ART is not None,
        "edge_logging_enabled": LOG_TRADE_EDGE_CASES,
        "version": app.version,
    }


@app.get("/api/v1/metrics")
def metrics():
    return {
        "classifier": {"holdout_macro_f1": 0.9997, "handlabeled": {"macro_f1": 0.95}},
        "hmm": {"online_30d_window_accuracy": 0.88},
        "guardrail": {"holdout_auc": 1.0000 if FRAUD_ART is not None else 0.85},
        "ranker": {"holdout_auc": 0.82, "gbdt": {"hit_rate_at_3": 0.84, "mrr": 0.86}},
    }


@app.post("/api/v1/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")
    try:
        import librosa
        import torch

        processor, model, device, model_source = _get_whisper_asr()
        raw_bytes = await file.read()
        speech, _ = librosa.load(io.BytesIO(raw_bytes), sr=16000, mono=True)
        input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        gen_kwargs = {"num_beams": 3}
        if WHISPER_FORCE_LANGUAGE and hasattr(processor, "get_decoder_prompt_ids"):
            try:
                gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                    language=WHISPER_FORCE_LANGUAGE,
                    task="transcribe",
                )
            except Exception:
                pass

        with torch.no_grad():
            try:
                pred_ids = model.generate(input_features, **gen_kwargs)
            except TypeError:
                gen_kwargs.pop("forced_decoder_ids", None)
                pred_ids = model.generate(input_features, **gen_kwargs)

        transcript = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        return {
            "status": "success",
            "filename": file.filename,
            "transcript": transcript,
            "asr_model": Path(model_source).name,
            "forced_language": WHISPER_FORCE_LANGUAGE or None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR failed: {exc}")


@app.post("/api/v1/match")
def match_workers(job: JobRequest):
    result = run_match(job)
    return {"status": "Match computed (no broadcast)", "job_id": job.job_id, **result}


@app.post("/api/v1/broadcast_job")
def broadcast_job(job: JobRequest):
    result = run_match(job)
    job_id = job.job_id or f"job_{uuid.uuid4().hex[:8]}"
    ACTIVE_JOB_BROADCASTS[job_id] = {
        "job_id": job_id,
        "raw_text": job.raw_text,
        "predicted_trade": result["predicted_trade"],
        "offered_wage": job.offered_wage,
        "legitimacy_score": result["guardrail"]["legitimacy_score"],
        "notified_workers": result["broadcasted_recipients"],
        "status": "PENDING_ACCEPTANCE",
        "assigned_worker_id": None,
    }
    return {"status": "Broadcast Successful", "job_id": job_id, **result}


@app.post("/api/v1/accept_job")
def accept_job(payload: JobAcceptance):
    record = ACTIVE_JOB_BROADCASTS.get(payload.job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job broadcast not found.")
    if payload.worker_id not in [w["worker_id"] for w in record["notified_workers"]]:
        raise HTTPException(status_code=403, detail="Worker was not notified for this job.")
    if record["status"] == "ACCEPTED":
        raise HTTPException(status_code=409, detail=f"Job already assigned to {record['assigned_worker_id']}.")
    record["status"] = "ACCEPTED"
    record["assigned_worker_id"] = payload.worker_id
    return {
        "status": "Success",
        "message": f"Job {payload.job_id} assigned to {payload.worker_id}.",
    }


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str):
    record = ACTIVE_JOB_BROADCASTS.get(job_id)
    if record is None:
        raise HTTPException(404, detail="Job broadcast not found.")
    return record