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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from step05_haversine_filter import compute_geo_features
from trade_taxonomy import TRADES, canonical_trade, normalize_text

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

# Small local LLM configuration (LLM-only mode for testing)
LOCAL_LLM_DIR = Path(os.getenv("LOCAL_LLM_DIR", str(CKPT_DIR / "local_llm")))
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "320"))
LLM_USE_4BIT = os.getenv("LLM_USE_4BIT", "1").strip().lower() in {"1", "true", "yes", "on"}

WHISPER_CKPT_DIR = CKPT_DIR / "whisper-banspeech-dhaka-slang-final"
WHISPER_FALLBACK_MODEL = os.getenv("WHISPER_FALLBACK_MODEL", "openai/whisper-small")
WHISPER_FORCE_LANGUAGE = os.getenv("WHISPER_FORCE_LANGUAGE", "").strip()

LOG_TRADE_EDGE_CASES = os.getenv("LOG_TRADE_EDGE_CASES", "1").strip().lower() in {"1", "true", "yes", "on"}
TRADE_EDGE_LOG_PATH = Path(os.getenv("TRADE_EDGE_LOG_PATH", str(BASE_DIR / "logs" / "trade_edge_cases.jsonl")))
_EDGE_LOG_LOCK = threading.Lock()

app = FastAPI(
    title="LABOR-HUB // Local Small LLM Test Backend",
    version="4.8.0-llm-only",
)

ACTIVE_JOB_BROADCASTS: Dict[str, dict] = {}
_WHISPER_CACHE: Dict[str, object] = {}
_LLM_CACHE: Dict[str, object] = {}


def _canonical_trade_name(raw_label: object) -> str:
    return canonical_trade(raw_label)


def _normalize_banglish(raw_text: str) -> str:
    cleaned = _transliterate_devanagari(raw_text)
    cleaned = _strip_diacritics(cleaned.lower())
    return normalize_text(cleaned)


def _get_local_llm():
    if _LLM_CACHE.get("failed"):
        raise RuntimeError(str(_LLM_CACHE.get("error", "Local LLM failed to load.")))
    if "model" in _LLM_CACHE and "tokenizer" in _LLM_CACHE:
        return _LLM_CACHE["tokenizer"], _LLM_CACHE["model"]

    if not LOCAL_LLM_DIR.exists() or not any(LOCAL_LLM_DIR.iterdir()):
        err = f"Local LLM checkpoint not found at {LOCAL_LLM_DIR}"
        _LLM_CACHE["failed"] = True
        _LLM_CACHE["error"] = err
        raise FileNotFoundError(err)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        has_cuda = torch.cuda.is_available()
        quant_cfg = None
        if has_cuda and LLM_USE_4BIT:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_LLM_DIR), local_files_only=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_LLM_DIR),
            local_files_only=True,
            quantization_config=quant_cfg,
            device_map="auto" if has_cuda else None,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
        )
        if not has_cuda:
            model = model.to("cpu")
        model.eval()

        _LLM_CACHE["tokenizer"] = tokenizer
        _LLM_CACHE["model"] = model
        return tokenizer, model
    except Exception as exc:
        _LLM_CACHE["failed"] = True
        _LLM_CACHE["error"] = str(exc)
        raise RuntimeError(f"Failed to load local LLM from {LOCAL_LLM_DIR}: {exc}") from exc


def _generate_json_from_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    import torch

    tokenizer, model = _get_local_llm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    inputs = tokenizer([prompt_text], return_tensors="pt")
    target_device = getattr(model, "device", None)
    if target_device is not None:
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][prompt_len:]
    raw_reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def _llm_analyze_job(
    raw_text: str,
    normalized_text: str,
    offered_wage: float,
    candidate_trades: List[str],
) -> Dict[str, Any]:
    system_prompt = (
        "You are an assistant for an informal-sector labor matching platform in Dhaka. "
        f"Choose 'trade' strictly from this allowed list: {list(TRADES)}. "
        "Respond ONLY with a single valid JSON object containing keys: "
        "'trade' (string from allowed list), "
        "'confidence' (float between 0.0 and 1.0), "
        "'trade_scores' (object mapping each allowed trade to a float >= 0), "
        "'fraud_risk' (float between 0.0 and 1.0), "
        "'fraud_flags' (list of short snake_case strings), "
        "'fraud_reason' (string), "
        "'skill_gaps_by_trade' (object mapping non-matching worker primary_trade to a list of 1-3 short missing skill strings)."
    )
    user_prompt = (
        f"Job text: {raw_text}\n"
        f"Normalized text: {normalized_text}\n"
        f"Offered daily wage (BDT): {offered_wage}\n"
        f"Candidate primary trades present: {sorted(set(candidate_trades))}"
    )

    data = _generate_json_from_llm(system_prompt, user_prompt)

    trade = _canonical_trade_name(data.get("trade", ""))
    if trade not in TRADES:
        trade = TRADES[0]

    try:
        conf = float(np.clip(float(data.get("confidence", 0.75)), 0.0, 1.0))
    except (TypeError, ValueError):
        conf = 0.75

    raw_scores = data.get("trade_scores") if isinstance(data.get("trade_scores"), dict) else {}
    prob_map: Dict[str, float] = {}
    for t in TRADES:
        try:
            val = max(0.0, float(raw_scores.get(t, 0.0)))
        except (TypeError, ValueError):
            val = 0.0
        prob_map[t] = val

    if prob_map.get(trade, 0.0) < conf:
        prob_map[trade] = conf

    rem_mass = max(0.0, 1.0 - prob_map[trade])
    other_trades = [t for t in TRADES if t != trade]
    other_sum = sum(prob_map[t] for t in other_trades)
    if other_sum > 0:
        for t in other_trades:
            prob_map[t] = rem_mass * (prob_map[t] / other_sum)
    else:
        share = rem_mass / max(len(other_trades), 1)
        for t in other_trades:
            prob_map[t] = share

    total = sum(prob_map.values())
    if total > 0:
        prob_map = {k: float(v / total) for k, v in prob_map.items()}
    else:
        prob_map = {t: (1.0 if t == trade else 0.0) for t in TRADES}

    try:
        fraud_risk = float(np.clip(float(data.get("fraud_risk", 0.10)), 0.0, 1.0))
    except (TypeError, ValueError):
        fraud_risk = 0.10

    raw_flags = data.get("fraud_flags")
    flags = [str(f).strip() for f in raw_flags if str(f).strip()] if isinstance(raw_flags, list) else []
    fraud_reason = str(data.get("fraud_reason") or "").strip()

    raw_gaps = data.get("skill_gaps_by_trade") if isinstance(data.get("skill_gaps_by_trade"), dict) else {}
    gaps_by_trade: Dict[str, List[str]] = {}
    for k, v in raw_gaps.items():
        ck = _canonical_trade_name(k)
        if isinstance(v, list):
            cleaned_skills = [str(s).strip() for s in v if str(s).strip()][:3]
            if cleaned_skills:
                gaps_by_trade[ck] = cleaned_skills

    return {
        "trade": trade,
        "confidence": float(prob_map[trade]),
        "trade_probabilities": prob_map,
        "fraud_risk": fraud_risk,
        "fraud_flags": flags,
        "fraud_reason": fraud_reason,
        "skill_gaps_by_trade": gaps_by_trade,
    }


def _log_trade_edge_case(
    raw_text: str,
    normalized_text: str,
    final_trade: str,
    final_confidence: float,
) -> None:
    if not LOG_TRADE_EDGE_CASES or final_confidence >= LOW_CONF_THRESHOLD:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "final_trade": final_trade,
        "final_confidence": round(float(final_confidence), 4),
        "source": "local_llm_only",
        "low_confidence": True,
    }
    try:
        with _EDGE_LOG_LOCK:
            TRADE_EDGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TRADE_EDGE_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


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


def _build_guardrail_from_llm(
    llm_out: Dict[str, Any],
    offered_wage: float,
    low_band: int,
    high_band: int,
) -> dict:
    scam_risk = float(np.clip(float(llm_out.get("fraud_risk", 0.1)), 0.0, 1.0))
    legit = round(1.0 - scam_risk, 4)
    flags = list(llm_out.get("fraud_flags") or [])

    if offered_wage < low_band and "below_fair_band" not in flags:
        flags.append("below_fair_band")
    elif offered_wage > high_band and "above_fair_band" not in flags:
        flags.append("above_fair_band")

    verdict = "high_risk" if scam_risk >= 0.55 else "caution" if (scam_risk >= 0.25 or flags) else "legitimate"
    return {
        "legitimacy_score": legit,
        "scam_risk": round(scam_risk, 4),
        "verdict": verdict,
        "flags": flags,
        "reason": llm_out.get("fraud_reason") or "",
        "parsed_text_wage": None,
        "effective_wage_evaluated": round(float(offered_wage), 2),
        "wage_mismatch_ratio": None,
    }


def run_match(job: JobRequest) -> dict:
    t0 = time.perf_counter()
    if not job.raw_text.strip():
        raise HTTPException(status_code=400, detail="Job text cannot be empty.")

    norm_text = _normalize_banglish(job.raw_text)
    candidate_trades = [_canonical_trade_name(w.primary_trade) for w in job.candidates]

    try:
        llm_out = _llm_analyze_job(
            raw_text=job.raw_text,
            normalized_text=norm_text,
            offered_wage=job.offered_wage,
            candidate_trades=candidate_trades,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Local LLM inference failed: {exc}") from exc

    predicted_trade = llm_out["trade"]
    confidence = llm_out["confidence"]
    trade_probs = llm_out["trade_probabilities"]
    gaps_by_trade = llm_out["skill_gaps_by_trade"]
    _log_trade_edge_case(job.raw_text, norm_text, predicted_trade, confidence)

    hmm_info = _estimate_market_regime(job)
    wage_idx = hmm_info["market_wage_index"]

    ranked: List[dict] = []
    excluded: List[dict] = []
    skill_gaps: List[dict] = []
    reference_wages: List[float] = []

    if job.candidates:
        cand_df = pd.DataFrame([
            {
                "worker_id": w.worker_id,
                "trade": _canonical_trade_name(w.primary_trade),
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
                if _canonical_trade_name(w.primary_trade) == predicted_trade
            ]

            for idx_local, w in enumerate(sel_workers):
                d_km = float(sel_dists[idx_local])
                w_trade = _canonical_trade_name(w.primary_trade)
                t_match = w_trade == predicted_trade
                t_prob = float(trade_probs.get(w_trade, 0.1))
                w_ratio = float(np.clip(job.offered_wage / max(w.expected_daily_wage, 100.0), 0.2, 3.0))

                rel_score = float(np.clip(w.completion_rate, 0.0, 1.0))
                rel_state = (
                    "high_reliability" if rel_score >= 0.85
                    else "moderate_reliability" if rel_score >= 0.65
                    else "at_risk"
                )

                f_score = float(np.clip(
                    0.45 * t_prob
                    + 0.30 * (min(w_ratio, 1.2) / 1.2)
                    + 0.25 * (1.0 - min(d_km, 15.0) / 15.0),
                    0.0,
                    1.0,
                ))
                a_prob = float(np.clip(
                    0.50 * rel_score
                    + 0.30 * (min(w_ratio, 1.2) / 1.2)
                    + 0.20 * (1.0 - min(d_km, 15.0) / 15.0),
                    0.0,
                    1.0,
                ))
                c_score = round(0.55 * f_score + 0.45 * a_prob, 4)

                missing_skills: List[str] = []
                cos_sim = 0.85 if t_match else 0.28
                boost = 0.0
                rec_text = ""

                if not t_match:
                    missing_skills = gaps_by_trade.get(w_trade) or [f"{predicted_trade}_basics"]
                    boost = round(min(0.25, 0.08 + 0.15 * confidence), 3)
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

    guardrail = _build_guardrail_from_llm(
        llm_out=llm_out,
        offered_wage=job.offered_wage,
        low_band=low_band,
        high_band=high_band,
    )

    wage_verdict = (
        "below market" if job.offered_wage < low_band
        else "above market" if job.offered_wage > high_band
        else "fair"
    )

    return {
        "mode": "local_llm_only",
        "local_llm_dir": str(LOCAL_LLM_DIR),
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
            "parsed_text_wage": None,
            "wage_mismatch_ratio": None,
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
    llm_loaded = "model" in _LLM_CACHE
    llm_present = LOCAL_LLM_DIR.exists() and any(LOCAL_LLM_DIR.iterdir())
    whisper_src = _WHISPER_CACHE.get("model_source")
    if whisper_src is None and WHISPER_CKPT_DIR.exists() and (WHISPER_CKPT_DIR / "config.json").exists():
        whisper_src = str(WHISPER_CKPT_DIR)
    return {
        "status": "ok",
        "mode": "local_llm_only",
        "local_llm_present": llm_present,
        "local_llm_loaded": llm_loaded,
        "local_llm_dir": str(LOCAL_LLM_DIR),
        "trained_task_models_used": False,
        "whisper_loaded": bool(_WHISPER_CACHE),
        "whisper_model": Path(whisper_src).name if whisper_src else WHISPER_FALLBACK_MODEL,
        "whisper_forced_language": WHISPER_FORCE_LANGUAGE or None,
        "edge_logging_enabled": LOG_TRADE_EDGE_CASES,
        "version": app.version,
    }


@app.get("/api/v1/metrics")
def metrics():
    return {
        "mode": "local_llm_only",
        "local_llm_dir": str(LOCAL_LLM_DIR),
        "note": "Running solely with local small LLM reasoning + Haversine geo filtering; dedicated ML checkpoints are bypassed.",
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