# LABOR-HUB — Informal Sector Skill-Matching & ML Telemetry Backend

An end-to-end FastAPI backend and web interface for transcribing Dhaka-slang voice and text job requests, normalizing scripts and trades, evaluating wage/fraud guardrails, ranking nearby workers, and recommending skill gaps.

## Overview & Pipeline

1. **Speech & Text Ingestion (`/api/v1/transcribe-audio`, `/api/v1/match`, `/api/v1/broadcast_job`)**:
   - Transcribes uploaded audio (`16 kHz` mono via `librosa` and a fine-tuned Whisper checkpoint) or accepts raw job text.
   - Normalizes Devanagari transliteration, diacritics, and Banglish variants via `_normalize_banglish` and `trade_taxonomy.py`.
2. **Trade Classification**:
   - Uses TF-IDF + Multinomial Naive Bayes (`checkpoints/tfidf_nb_trade/`) with optional probability blending from `BanglishBERT` (`checkpoints/banglishbert_trade/`) when confidence is below `BERT_BLEND_THRESHOLD`, paired with domain cues for plumbing, carpentry/partition, device repair, and cleaning.
   - Logs low-confidence or rule-overridden cases to `logs/trade_edge_cases.jsonl` when `LOG_TRADE_EDGE_CASES` is enabled.
3. **Market Regime & Fraud Guardrail**:
   - Estimates demand/wage regime (`low_demand`, `normal`, `peak_surge`) from recent daily jobs and wage index.
   - Parses text-mentioned wages, flags advance-fee keywords or wage mismatches, compares against fair wage bands, and scores legitimacy with the LightGBM guardrail model (`checkpoints/lgbm_fraud_guardrail.joblib`).
4. **Geo Filtering, Reliability, Ranking & Skill Gaps**:
   - Computes Haversine distances and filters candidates within `max_radius_km`.
   - Estimates worker reliability via an HMM artifact (`checkpoints/hmm_worker_reliability.joblib`) with completion-rate fallback.
   - Ranks candidates using the GBDT dual-objective ranker (`checkpoints/gbdt_dual_objective_ranker.joblib`) for fit and acceptance probability, and generates skill-gap recommendations for non-matching trades.

## Project Structure

```text
LBproject/
├── main.py                              # FastAPI app, classification, guardrail, ranking & broadcast logic
├── index.html                           # Web UI served at /
├── trade_taxonomy.py                    # Trade definitions, canonicalization & text normalization
├── step04_hmm_worker_reliability.py     # HMM worker reliability inference
├── step05_haversine_filter.py           # Distance & radius filtering
├── step07_dual_objective_ranker.py      # Dual-objective ranking helpers
├── step08_skill_gap_recommender.py      # Skill extraction & gap recommendations
├── train_step03_classifiers.py          # Trade classifier training script
├── train_step06_fraud_guardrail.py      # Fraud guardrail training & scoring script
├── build_2m_dataset.py                  # Dataset building script
├── generate_dhaka_trade_dataset.py      # Synthetic/augmented Dhaka trade dataset generator
├── dataset_loader.py                    # Dataset loading utilities
├── test_all_cases.py                    # End-to-end test & benchmark suite
├── test_step05_full.py                  # Geo/ranking tests
├── checkpoints/                         # Trained model artifacts & checkpoints
│   ├── tfidf_nb_trade/
│   ├── gbdt_dual_objective_ranker.joblib
│   ├── lgbm_fraud_guardrail.joblib
│   └── skill_gap_recommender.joblib
└── docs/
    └── screenshots/                     # UI & API screenshots for documentation



How to Run
1. Set Up the Environment
Bash
git clone [https://github.com/FahiimReza/LBproject.git](https://github.com/FahiimReza/LBproject.git)
cd LBproject
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
pip install -r requirements.txt
2. Start the Server
Bash
uvicorn main:app --reload
Web UI: http://127.0.0.1:8000/

Interactive API Docs (Swagger): http://127.0.0.1:8000/docs

3. Check Health & Run Tests
Bash
curl [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
python test_all_cases.py
API Endpoints
GET /api/v1/health — Reports loaded models, ASR/BERT status, edge logging flag, and app version.

GET /api/v1/metrics — Returns reported evaluation metrics across classifier, HMM, guardrail, and ranker components.

POST /api/v1/transcribe-audio — Accepts an audio file (.wav, .mp3, .m4a, .flac, .ogg, .webm) and returns the Whisper transcript.

POST /api/v1/match — Classifies trade, computes wage advice and guardrail risk, ranks nearby workers, and returns skill gaps without broadcasting.

POST /api/v1/broadcast_job — Runs matching and creates a pending broadcast record in memory.

POST /api/v1/accept_job — Assigns a pending broadcast job to a notified worker (job_id, worker_id).

GET /api/v1/jobs/{job_id} — Fetches current broadcast status and assigned worker.




## Model Checkpoints & Weights

All trained model checkpoints and artifacts can be downloaded from the Google Drive folder:
**[LABOR-HUB Model Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1hf0jAxRsr2JDW-1XxtsTCmHlNZLyTpAt?usp=drive_link)**

Download the contents from Google Drive and place them inside your local `checkpoints/` directory so they match this layout:

```text
checkpoints/
├── whisper-banspeech-dhaka-slang-final/
├── banglishbert_trade/
├── tfidf_nb_trade/
├── gbdt_dual_objective_ranker.joblib
├── lgbm_fraud_guardrail.joblib
└── skill_gap_recommender.joblib
