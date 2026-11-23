import glob
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

CONFIG = {
    "csv_path": "dhaka_trade_dataset_1M.csv",
    "fallback_glob": "dhaka_trade_dataset_part*.csv",
    "artifact_path": "./checkpoints/hmm_worker_reliability.joblib",
    "n_workers": 5_000,
    "min_seq_len": 6,
    "max_seq_len": 16,
    "em_iters": 15,
    "seed": 42,
}

# Ordered from lowest reliability (0) to highest reliability (2)
STATE_NAMES = ["at_risk", "moderate_reliability", "high_reliability"]
OBS_NAMES = ["completed_on_time", "completed_late", "late_cancel", "no_show"]
OBS_TO_ID = {name: idx for idx, name in enumerate(OBS_NAMES)}


def load_trade_wage_context(cfg: Dict) -> Dict[str, float]:
    """Load trade median wages from the existing dataset to ground synthetic shift history."""
    if os.path.exists(cfg["csv_path"]):
        df = pd.read_csv(cfg["csv_path"], usecols=["trade", "offered_wage"])
    else:
        parts = sorted(glob.glob(cfg["fallback_glob"]))
        if not parts:
            return {"default": 1200.0}
        df = pd.concat(
            [pd.read_csv(p, usecols=["trade", "offered_wage"]) for p in parts],
            ignore_index=True,
        )
    df["trade"] = df["trade"].astype(str).str.strip().str.lower()
    df["offered_wage"] = pd.to_numeric(df["offered_wage"], errors="coerce")
    df = df.dropna(subset=["offered_wage"])
    medians = df.groupby("trade")["offered_wage"].median().to_dict()
    medians["default"] = float(df["offered_wage"].median()) if len(df) else 1200.0
    return medians


def generate_worker_sequences(
    n_workers: int,
    min_len: int,
    max_len: int,
    trade_medians: Dict[str, float],
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Simulate sequential worker reliability trajectories grounded in Dhaka trade wages."""
    rng = np.random.default_rng(seed)
    trades = [k for k in trade_medians.keys() if k != "default"] or [
        "carpentry",
        "electrical",
        "general_labor",
        "masonry",
        "painting",
        "plumbing",
    ]

    # True generative transition matrix (rows: at_risk, moderate, high)
    true_trans = np.array(
        [
            [0.65, 0.25, 0.10],
            [0.15, 0.60, 0.25],
            [0.03, 0.17, 0.80],
        ],
        dtype=np.float64,
    )

    # Emission probabilities over [completed_on_time, completed_late, late_cancel, no_show]
    true_emit = np.array(
        [
            [0.25, 0.30, 0.25, 0.20],  # at_risk
            [0.65, 0.22, 0.09, 0.04],  # moderate_reliability
            [0.90, 0.07, 0.02, 0.01],  # high_reliability
        ],
        dtype=np.float64,
    )

    sequences: List[np.ndarray] = []
    worker_ids: List[str] = []
    worker_trades: List[str] = []

    for w_idx in range(n_workers):
        tr = str(rng.choice(trades))
        seq_len = int(rng.integers(min_len, max_len + 1))
        state = int(rng.choice([0, 1, 2], p=[0.20, 0.45, 0.35]))
        obs_seq = np.zeros(seq_len, dtype=np.int32)

        for t in range(seq_len):
            obs_seq[t] = int(rng.choice(len(OBS_NAMES), p=true_emit[state]))
            state = int(rng.choice(len(STATE_NAMES), p=true_trans[state]))

        sequences.append(obs_seq)
        worker_ids.append(f"worker_{w_idx:05d}")
        worker_trades.append(tr)

    return sequences, worker_ids, worker_trades


def _forward_backward(
    obs_seq: np.ndarray, pi: np.ndarray, A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward-backward pass to avoid numerical underflow."""
    T = len(obs_seq)
    N = len(pi)
    alpha = np.zeros((T, N), dtype=np.float64)
    scales = np.zeros(T, dtype=np.float64)

    emit_0 = np.clip(B[:, obs_seq[0]], 1e-12, None)
    alpha[0] = pi * emit_0
    scales[0] = np.clip(alpha[0].sum(), 1e-12, None)
    alpha[0] /= scales[0]

    for t in range(1, T):
        emit_t = np.clip(B[:, obs_seq[t]], 1e-12, None)
        pred = alpha[t - 1] @ A
        alpha[t] = pred * emit_t
        scales[t] = np.clip(alpha[t].sum(), 1e-12, None)
        alpha[t] /= scales[t]

    beta = np.zeros((T, N), dtype=np.float64)
    beta[T - 1] = 1.0 / scales[T - 1]
    for t in range(T - 2, -1, -1):
        emit_next = np.clip(B[:, obs_seq[t + 1]], 1e-12, None)
        beta[t] = (A @ (emit_next * beta[t + 1])) / scales[t]

    gamma = alpha * beta
    gamma_sums = np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)
    gamma /= gamma_sums

    log_likelihood = float(np.sum(np.log(scales)))
    return alpha, gamma, log_likelihood


def fit_hmm_baum_welch(
    sequences: List[np.ndarray], n_states: int = 3, n_obs: int = 4, n_iters: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit HMM parameters (pi, A, B) via Baum-Welch EM with Dirichlet smoothing."""
    pi = np.array([0.20, 0.45, 0.35], dtype=np.float64)
    A = np.array(
        [
            [0.65, 0.25, 0.10],
            [0.15, 0.60, 0.25],
            [0.05, 0.20, 0.75],
        ],
        dtype=np.float64,
    )
    B = np.array(
        [
            [0.25, 0.30, 0.25, 0.20],
            [0.65, 0.22, 0.09, 0.04],
            [0.88, 0.08, 0.03, 0.01],
        ],
        dtype=np.float64,
    )

    for _ in range(n_iters):
        pi_num = np.full(n_states, 1e-3, dtype=np.float64)
        A_num = np.full((n_states, n_states), 1e-3, dtype=np.float64)
        B_num = np.full((n_states, n_obs), 1e-3, dtype=np.float64)

        for seq in sequences:
            if len(seq) == 0:
                continue
            alpha, gamma, _ = _forward_backward(seq, pi, A, B)
            pi_num += gamma[0]

            for o_val in range(n_obs):
                mask = (seq == o_val).astype(np.float64)
                B_num[:, o_val] += gamma.T @ mask

            for t in range(len(seq) - 1):
                emit_next = np.clip(B[:, seq[t + 1]], 1e-12, None)
                xi_t = np.outer(alpha[t], emit_next) * A
                s = xi_t.sum()
                if s > 1e-12:
                    A_num += xi_t / s

        pi = pi_num / np.clip(pi_num.sum(), 1e-12, None)
        A = A_num / np.clip(A_num.sum(axis=1, keepdims=True), 1e-12, None)
        B = B_num / np.clip(B_num.sum(axis=1, keepdims=True), 1e-12, None)

    return pi, A, B


def _align_states_by_reliability(
    pi: np.ndarray, A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure state indices are ordered from lowest (0) to highest (2) reliability."""
    # Expected on-time probability per state as a clean ordering criterion
    on_time_score = B[:, OBS_TO_ID["completed_on_time"]] - B[:, OBS_TO_ID["no_show"]]
    order = np.argsort(on_time_score)
    pi_sorted = pi[order]
    A_sorted = A[np.ix_(order, order)]
    B_sorted = B[order]
    return pi_sorted, A_sorted, B_sorted


def infer_worker_reliability(
    artifact: Dict, events: List[str]
) -> Dict[str, object]:
    """Infer reliability_score [0, 1] and state from a worker's event sequence."""
    obs_ids = [
        OBS_TO_ID[ev.strip().lower()]
        for ev in events
        if ev.strip().lower() in OBS_TO_ID
    ]
    if not obs_ids:
        obs_arr = np.array([OBS_TO_ID["completed_on_time"]], dtype=np.int32)
    else:
        obs_arr = np.array(obs_ids, dtype=np.int32)

    pi = artifact["pi"]
    A = artifact["A"]
    B = artifact["B"]

    alpha, _, log_lik = _forward_backward(obs_arr, pi, A, B)
    posterior = alpha[-1]
    posterior = posterior / np.clip(posterior.sum(), 1e-12, None)

    # Expected reliability score in [0.0, 1.0] using weights [0.15, 0.55, 0.95]
    weights = np.array([0.15, 0.55, 0.95], dtype=np.float64)
    score = float(np.dot(posterior, weights))
    best_state = int(np.argmax(posterior))

    return {
        "reliability_score": round(score, 4),
        "reliability_state": STATE_NAMES[best_state],
        "state_probabilities": {
            STATE_NAMES[i]: round(float(posterior[i]), 4) for i in range(len(STATE_NAMES))
        },
        "sequence_log_likelihood": round(log_lik, 4),
    }


def train_and_save() -> Dict:
    trade_medians = load_trade_wage_context(CONFIG)
    sequences, worker_ids, worker_trades = generate_worker_sequences(
        n_workers=CONFIG["n_workers"],
        min_len=CONFIG["min_seq_len"],
        max_len=CONFIG["max_seq_len"],
        trade_medians=trade_medians,
        seed=CONFIG["seed"],
    )

    pi_raw, A_raw, B_raw = fit_hmm_baum_welch(
        sequences,
        n_states=len(STATE_NAMES),
        n_obs=len(OBS_NAMES),
        n_iters=CONFIG["em_iters"],
    )
    pi, A, B = _align_states_by_reliability(pi_raw, A_raw, B_raw)

    artifact = {
        "pi": pi,
        "A": A,
        "B": B,
        "state_names": STATE_NAMES,
        "obs_names": OBS_NAMES,
        "obs_to_id": OBS_TO_ID,
        "trade_medians": trade_medians,
    }

    os.makedirs(os.path.dirname(CONFIG["artifact_path"]), exist_ok=True)
    joblib.dump(artifact, CONFIG["artifact_path"])

    scores = [infer_worker_reliability(artifact, list(map(OBS_NAMES.__getitem__, seq)))["reliability_score"] for seq in sequences[:500]]
    print(f"Trained Step 04 HMM on {len(sequences):,} worker sequences -> {CONFIG['artifact_path']}")
    print(f"Sample 500 workers reliability_score | mean: {np.mean(scores):.4f} | min: {np.min(scores):.4f} | max: {np.max(scores):.4f}")
    print("\nLearned Transition Matrix (rows=from, cols=to):")
    print(pd.DataFrame(A, index=STATE_NAMES, columns=STATE_NAMES).round(4).to_string())
    return artifact


if __name__ == "__main__":
    art = train_and_save()
    reliable_demo = infer_worker_reliability(
        art, ["completed_on_time", "completed_on_time", "completed_on_time", "completed_late", "completed_on_time"]
    )
    risky_demo = infer_worker_reliability(
        art, ["completed_on_time", "late_cancel", "no_show", "late_cancel"]
    )
    print("\nReliable worker example:", reliable_demo)
    print("At-risk worker example: ", risky_demo)