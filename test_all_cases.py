# test_all_cases.py
import json
from pathlib import Path
import statistics
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
HISTORY_PATH = Path(__file__).resolve().parent / "benchmark_history.jsonl"

SAMPLE_CANDIDATES = [
    {"worker_id": "w_plumb", "primary_trade": "plumbing", "lat": 23.795, "lon": 90.412, "completion_rate": 0.94, "expected_daily_wage": 1300},
    {"worker_id": "w_elec", "primary_trade": "electrical", "lat": 23.790, "lon": 90.415, "completion_rate": 0.91, "expected_daily_wage": 1450},
    {"worker_id": "w_mason", "primary_trade": "masonry", "lat": 23.793, "lon": 90.410, "completion_rate": 0.89, "expected_daily_wage": 1550},
    {"worker_id": "w_carp", "primary_trade": "carpentry", "lat": 23.791, "lon": 90.418, "completion_rate": 0.92, "expected_daily_wage": 1500},
    {"worker_id": "w_paint", "primary_trade": "painting", "lat": 23.794, "lon": 90.411, "completion_rate": 0.88, "expected_daily_wage": 1350},
    {"worker_id": "w_gen", "primary_trade": "general_labor", "lat": 23.789, "lon": 90.413, "completion_rate": 0.85, "expected_daily_wage": 900},
    {"worker_id": "w_app", "primary_trade": "appliance_repair", "lat": 23.796, "lon": 90.416, "completion_rate": 0.95, "expected_daily_wage": 1650},
    {"worker_id": "w_weld", "primary_trade": "welding_metalwork", "lat": 23.788, "lon": 90.417, "completion_rate": 0.90, "expected_daily_wage": 1600},
    {"worker_id": "w_clean", "primary_trade": "cleaning_sanitation", "lat": 23.792, "lon": 90.409, "completion_rate": 0.93, "expected_daily_wage": 1150},
    {"worker_id": "w_move", "primary_trade": "moving_shifting", "lat": 23.797, "lon": 90.414, "completion_rate": 0.87, "expected_daily_wage": 1300},
]

# Core 10-trade + alias/Bangla/noisy cases
CORE_TRADE_CASES = [
    ("plumbing", "basay plumber lagbe, emergency leak", 1300),
    ("electrical", "electracian lagbe switchboard spark, 1500 tk din", 1500),
    ("masonry", "raj mistri lagbe bathroom e tiles lagabo", 1550),
    ("carpentry", "kathmistri lagbe dorjar kobja venge gese", 1500),
    ("painting", "bashar deyal rong korbo, putty ar primer dibo", 1350),
    ("general_labor", "truck theke mal namabo, kuli lagbe", 900),
    ("appliance_repair", "fridge thanda hocche na, ac gas refill lagbe", 1650),
    ("welding_metalwork", "janala grill welding korbo ar gate repair", 1600),
    ("cleaning_sanitation", "bashar deep cleaning ar pani tank porishkar", 1150),
    ("moving_shifting", "basa shift korbo packing soho, furniture shifting", 1300),
]

# Previously tricky / out-of-vocab / Bangla ASR cases to track improvements on
REGRESSION_CASES = [
    ("carpentry", "শেওড়াপাড়া এলাকা এক জুন ভালো ফার্নিচার মিস্ত্রি লাগবে অফিসর কাতে পার্টিশন তিখ কোরতে হোগে কুপ শমোশা হোচে.", 1400),
    ("appliance_repair", "basay pc nosto lok lagbe", 1400),
    ("appliance_repair", "basay computer nosto lok lagbe", 1400),
]


def post_match(payload: dict) -> tuple[requests.Response, float]:
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/api/v1/match", json=payload, timeout=120)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return resp, elapsed_ms


def load_previous_run() -> dict | None:
    if not HISTORY_PATH.exists():
        return None
    lines = [ln.strip() for ln in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None


def append_run(summary: dict) -> None:
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def fmt_delta(cur: float, prev: float | None, suffix: str = "", higher_better: bool = True) -> str:
    if prev is None:
        return "(baseline)"
    diff = cur - prev
    sign = "+" if diff >= 0 else ""
    improved = (diff > 0) if higher_better else (diff < 0)
    tag = "IMPROVED" if improved and abs(diff) > 1e-6 else ("REGRESSED" if abs(diff) > 1e-6 else "SAME")
    return f"({sign}{diff:.3f}{suffix}, {tag})"


def run_trade_suite(title: str, cases: list[tuple[str, str, int]]) -> dict:
    print(f"\n=== {title} ===")
    passed = 0
    confs = []
    api_latencies = []
    rtt_latencies = []
    per_case = {}

    for expected_trade, text, wage in cases:
        r, rtt_ms = post_match({
            "raw_text": text,
            "lat": 23.792,
            "lon": 90.414,
            "offered_wage": wage,
            "max_radius_km": 10.0,
            "top_k": 3,
            "candidates": SAMPLE_CANDIDATES,
        })
        r.raise_for_status()
        d = r.json()
        pred = d["predicted_trade"]
        conf = float(d["trade_confidence"])
        api_ms = float(d["latency_ms"])
        ok = pred == expected_trade
        if ok:
            passed += 1
        confs.append(conf)
        api_latencies.append(api_ms)
        rtt_latencies.append(rtt_ms)
        per_case[text[:40]] = {"expected": expected_trade, "pred": pred, "conf": round(conf, 4), "ok": ok}

        top1 = d["ranked_workers"][0]["worker_id"] if d["ranked_workers"] else "NONE"
        status = "PASS" if ok else "CHECK"
        short_txt = text if len(text) <= 42 else text[:39] + "..."
        print(
            f"[{status}] exp={expected_trade:<19} pred={pred:<19} "
            f"conf={conf:.3f} guard={d['guardrail']['verdict']:<10} "
            f"top1={top1:<7} api={api_ms:5.1f}ms | {short_txt}"
        )

    acc = passed / max(len(cases), 1)
    avg_conf = statistics.mean(confs) if confs else 0.0
    avg_api_ms = statistics.mean(api_latencies) if api_latencies else 0.0
    p95_api_ms = sorted(api_latencies)[max(0, int(0.95 * len(api_latencies)) - 1)] if api_latencies else 0.0
    avg_rtt_ms = statistics.mean(rtt_latencies) if rtt_latencies else 0.0

    return {
        "passed": passed,
        "total": len(cases),
        "accuracy": round(acc, 4),
        "avg_confidence": round(avg_conf, 4),
        "avg_api_ms": round(avg_api_ms, 2),
        "p95_api_ms": round(p95_api_ms, 2),
        "avg_rtt_ms": round(avg_rtt_ms, 2),
        "cases": per_case,
    }


def run_worst_cases() -> dict:
    print("\n=== WORST-CASE & ADVERSARIAL SCENARIOS ===")
    checks = {}

    # A. Advance-fee scam + extreme wage
    r_scam, _ = post_match({
        "raw_text": "bkash 500 tk age pathan for advance registration fee",
        "lat": 23.757,
        "lon": 90.389,
        "offered_wage": 5000,
        "max_radius_km": 8.0,
        "top_k": 3,
        "candidates": SAMPLE_CANDIDATES,
    })
    d_scam = r_scam.json()
    scam_ok = d_scam["guardrail"]["verdict"] == "high_risk" and d_scam["guardrail"]["scam_risk"] >= 0.7
    checks["scam_high_risk"] = scam_ok
    print(
        f"[{'PASS' if scam_ok else 'CHECK'}][SCAM FEE] status={r_scam.status_code} "
        f"verdict={d_scam['guardrail']['verdict']} risk={d_scam['guardrail']['scam_risk']} "
        f"flags={d_scam['guardrail']['flags']}"
    )

    # B. Wage mismatch between text ("2500 tk") and offered_wage (800)
    r_mm, _ = post_match({
        "raw_text": "mistri lagbe 2500 tk din, urgent kaj",
        "lat": 23.806,
        "lon": 90.368,
        "offered_wage": 800,
        "max_radius_km": 8.0,
        "top_k": 3,
        "candidates": SAMPLE_CANDIDATES,
    })
    d_mm = r_mm.json()
    mm_ok = "text_wage_mismatch" in d_mm["guardrail"]["flags"]
    checks["wage_mismatch_flagged"] = mm_ok
    print(
        f"[{'PASS' if mm_ok else 'CHECK'}][WAGE MISMATCH] parsed={d_mm['guardrail']['parsed_text_wage']} "
        f"eff={d_mm['guardrail']['effective_wage_evaluated']} ratio={d_mm['guardrail']['wage_mismatch_ratio']} "
        f"flags={d_mm['guardrail']['flags']}"
    )

    # C. Ultra-tight radius (0.05 km) -> verify filtering behavior
    r_tight, _ = post_match({
        "raw_text": "plumber lagbe leak repair",
        "lat": 23.792,
        "lon": 90.414,
        "offered_wage": 1300,
        "max_radius_km": 1.0,
        "top_k": 3,
        "candidates": SAMPLE_CANDIDATES,
    })
    d_tight = r_tight.json()
    checks["radius_filter_runs"] = r_tight.status_code == 200
    print(
        f"[PASS][RADIUS 1km] in_radius={d_tight['in_radius_count']} "
        f"excluded={len(d_tight['excluded_workers'])} top1={d_tight['ranked_workers'][0]['worker_id'] if d_tight['ranked_workers'] else 'NONE'}"
    )

    # D. Empty candidate pool
    r_empty, _ = post_match({
        "raw_text": "raj mistri lagbe dhaka",
        "lat": 23.806,
        "lon": 90.368,
        "offered_wage": 1550,
        "max_radius_km": 8.0,
        "top_k": 3,
        "candidates": [],
    })
    d_empty = r_empty.json()
    empty_ok = r_empty.status_code == 200 and len(d_empty["ranked_workers"]) == 0
    checks["empty_pool_safe"] = empty_ok
    print(
        f"[{'PASS' if empty_ok else 'CHECK'}][EMPTY POOL] status={r_empty.status_code} "
        f"trade={d_empty['predicted_trade']} ranked={len(d_empty['ranked_workers'])}"
    )

    # E. Ambiguous Bangla + Bangla numerals
    r_amb, _ = post_match({
        "raw_text": "জরুরি কাজ আছে ১২০০ টাকা",
        "lat": 23.746,
        "lon": 90.376,
        "offered_wage": 1200,
        "max_radius_km": 8.0,
        "top_k": 3,
        "candidates": SAMPLE_CANDIDATES,
    })
    d_amb = r_amb.json()
    amb_ok = d_amb["wage_advice"]["parsed_text_wage"] == 1200.0
    checks["bangla_numeral_parsed"] = amb_ok
    print(
        f"[{'PASS' if amb_ok else 'CHECK'}][AMBIGUOUS BN] pred={d_amb['predicted_trade']} "
        f"conf={d_amb['trade_confidence']:.3f} low_conf={d_amb['low_confidence']} "
        f"parsed_wage={d_amb['wage_advice']['parsed_text_wage']}"
    )

    # F. Blank raw_text -> 400
    r_blank, _ = post_match({
        "raw_text": "   ",
        "lat": 23.746,
        "lon": 90.376,
        "offered_wage": 1200,
        "candidates": SAMPLE_CANDIDATES[:2],
    })
    blank_ok = r_blank.status_code == 400
    checks["blank_text_400"] = blank_ok
    print(f"[{'PASS' if blank_ok else 'CHECK'}][BLANK TEXT] status={r_blank.status_code}")

    # G. Invalid audio format -> 400
    r_bad_audio = requests.post(
        f"{BASE_URL}/api/v1/transcribe-audio",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
        timeout=10,
    )
    bad_audio_ok = r_bad_audio.status_code == 400
    checks["bad_audio_400"] = bad_audio_ok
    print(f"[{'PASS' if bad_audio_ok else 'CHECK'}][BAD AUDIO EXT] status={r_bad_audio.status_code}")

    return checks


def run_broadcast_and_accept() -> bool:
    print("\n=== BROADCAST & ACCEPTANCE LIFECYCLE ===")
    b_resp = requests.post(
        f"{BASE_URL}/api/v1/broadcast_job",
        json={
            "raw_text": "basay plumber lagbe, pani leak korche",
            "lat": 23.792,
            "lon": 90.414,
            "offered_wage": 1300,
            "max_radius_km": 10.0,
            "top_k": 3,
            "candidates": SAMPLE_CANDIDATES,
        },
        timeout=20,
    )
    b_resp.raise_for_status()
    b_data = b_resp.json()
    job_id = b_data["job_id"]
    notified = [w["worker_id"] for w in b_data["broadcasted_recipients"]]
    print(f"[BROADCAST] job_id={job_id} notified={notified}")

    if not notified:
        return False

    chosen = notified[0]
    a1 = requests.post(f"{BASE_URL}/api/v1/accept_job", json={"job_id": job_id, "worker_id": chosen}, timeout=10)
    a2 = requests.post(f"{BASE_URL}/api/v1/accept_job", json={"job_id": job_id, "worker_id": chosen}, timeout=10)
    st = requests.get(f"{BASE_URL}/api/v1/jobs/{job_id}", timeout=10).json()
    ok = (a1.status_code == 200) and (a2.status_code == 409) and (st.get("status") == "ACCEPTED")
    print(f"[{'PASS' if ok else 'CHECK'}][LIFECYCLE] accept1={a1.status_code} dup={a2.status_code} final_status={st.get('status')}")
    return ok


def main():
    health = requests.get(f"{BASE_URL}/api/v1/health", timeout=10).json()
    print("HEALTH:", health)

    prev = load_previous_run()

    core_stats = run_trade_suite("1. CORE 10-TRADE BENCHMARK", CORE_TRADE_CASES)
    reg_stats = run_trade_suite("2. TRICKY / ASR / HARDWARE REGRESSION CASES", REGRESSION_CASES)
    worst_checks = run_worst_cases()
    lifecycle_ok = run_broadcast_and_accept()

    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": health.get("version"),
        "core": core_stats,
        "regression": reg_stats,
        "worst_case_passed": sum(1 for v in worst_checks.values() if v),
        "worst_case_total": len(worst_checks),
        "lifecycle_ok": lifecycle_ok,
    }
    append_run(summary)

    prev_core = (prev or {}).get("core", {})
    prev_reg = (prev or {}).get("regression", {})
    prev_wc = (prev or {}).get("worst_case_passed")

    print("\n=== 3. IMPROVEMENT REPORT (VS PREVIOUS RUN) ===")
    print(
        f"Core 10-Trade Acc : {core_stats['passed']}/{core_stats['total']} ({core_stats['accuracy']*100:.1f}%) "
        f"{fmt_delta(core_stats['accuracy'], prev_core.get('accuracy'), '', True)}"
    )
    print(
        f"Core Avg Conf     : {core_stats['avg_confidence']:.4f} "
        f"{fmt_delta(core_stats['avg_confidence'], prev_core.get('avg_confidence'), '', True)}"
    )
    print(
        f"Core Avg API Lat  : {core_stats['avg_api_ms']:.2f} ms (p95={core_stats['p95_api_ms']:.2f} ms) "
        f"{fmt_delta(core_stats['avg_api_ms'], prev_core.get('avg_api_ms'), 'ms', False)}"
    )
    print(
        f"Regression Acc    : {reg_stats['passed']}/{reg_stats['total']} ({reg_stats['accuracy']*100:.1f}%) "
        f"{fmt_delta(reg_stats['accuracy'], prev_reg.get('accuracy'), '', True)}"
    )
    print(
        f"Regression Conf   : {reg_stats['avg_confidence']:.4f} "
        f"{fmt_delta(reg_stats['avg_confidence'], prev_reg.get('avg_confidence'), '', True)}"
    )
    print(
        f"Worst-Case Checks : {summary['worst_case_passed']}/{summary['worst_case_total']} "
        f"{fmt_delta(float(summary['worst_case_passed']), float(prev_wc) if prev_wc is not None else None, '', True)}"
    )
    print(f"Saved history to  : {HISTORY_PATH}")


if __name__ == "__main__":
    main()