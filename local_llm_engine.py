import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = Path(os.getenv("LOCAL_LLM_DIR", "checkpoints/local_llm"))

_TOKENIZER = None
_MODEL = None
_LOAD_ERROR: str | None = None


def is_available() -> bool:
    return MODEL_DIR.exists() and any(MODEL_DIR.iterdir())


def load_local_llm() -> tuple[Any, Any]:
    global _TOKENIZER, _MODEL, _LOAD_ERROR
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    if not is_available():
        _LOAD_ERROR = f"Local LLM directory not found at {MODEL_DIR}"
        raise FileNotFoundError(_LOAD_ERROR)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    _TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto",
    )
    return _TOKENIZER, _MODEL


def analyze_request_with_llm(
    text: str,
    trades: list[str],
    worker_skills: list[str] | None = None,
) -> dict[str, Any]:
    tokenizer, model = load_local_llm()
    skills_context = ", ".join(worker_skills or [])

    system_prompt = (
        "You are a job-matching assistant for Dhaka informal sector workers. "
        f"Select 'trade' strictly from: {trades}. "
        "Return ONLY valid JSON with keys: "
        "'normalized_text' (string), 'trade' (string), 'confidence' (float 0 to 1), "
        "'fraud_risk' (boolean), 'fraud_reason' (string), 'skill_gaps' (list of strings)."
    )
    user_prompt = f"Job request: {text}\nWorker skills: {skills_context or 'none'}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    raw_reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
    parsed: dict[str, Any] = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}

    chosen_trade = parsed.get("trade")
    if chosen_trade not in trades:
        chosen_trade = trades[0] if trades else "general"

    confidence = parsed.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.5

    skill_gaps = parsed.get("skill_gaps", [])
    if not isinstance(skill_gaps, list):
        skill_gaps = []

    return {
        "normalized_text": str(parsed.get("normalized_text") or text),
        "trade": chosen_trade,
        "confidence": confidence,
        "fraud_risk": bool(parsed.get("fraud_risk", False)),
        "fraud_reason": str(parsed.get("fraud_reason") or ""),
        "skill_gaps": [str(s) for s in skill_gaps],
    }