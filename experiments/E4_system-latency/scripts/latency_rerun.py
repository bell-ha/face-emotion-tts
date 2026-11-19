"""E4 재측정(2026-11-19, 현재 app/ 코드 = E5 개선안 반영 후).

latency.py와 같은 프로토콜에 다음을 더했다.
  1) /set_emotion 왕복 300회 (이벤트 버퍼 약 2000개)                       ← 기존과 같음
  2) /emotion/resolve_one 왕복 300회 — 확률 요약이 없는 상태(지속 시간 방식 폴백) ← 기존과 같은 조건
  2b) /set_emotion_probs 왕복 300회 (확률 요약 버퍼를 최대치 3000개=10분으로 채운 뒤)   ← 추가
  2c) /emotion/resolve_one 왕복 300회 — 확률 요약이 쌓인 상태(질량 방식)              ← 추가
  3) 문장 하나 합성(/tts) 길이 3종 × 30회, 4) 1·3·5문장 × 30회              ← 기존과 같음
콜드 스타트 1회와 예열 3회는 통계에서 뺀다. 시드 42.
바뀐 점(코드 측): 현재 서버는 단계 시간을 X-STAGE-Timing 헤더로 준다(이전 X-FaceVox-Timing; 둘 다 읽음).
환경 측: .venv의 edge-tts 7.2.8은 2026-11-19 현재 서버가 403으로 거부하므로, 요청 내용은 같고 인증 토큰만
바뀐 edge-tts 7.3.1(data/pylib_edge_tts_7.3.1/)을 서버 프로세스의 PYTHONPATH 앞에 둔다.
실패한 요청(HTTP 200 아님/예외)은 최대 3회 재시도하고, 재시도 수를 원자료 attempts 열과 env.json에 남긴다.

실행: E4_SUBDIR=run_2026-11-19 .venv/bin/python experiments/E4_system-latency/scripts/latency_rerun.py
      E4_SUBDIR=run_2026-11-19/dp_single E4_MODE=DP E4_ONLY_SINGLE=1 .venv/bin/python .../latency_rerun.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "E4_system-latency" / "results" / os.getenv("E4_SUBDIR", "run_2026-11-19")
MODE = os.getenv("E4_MODE", "D")
INTENSITY = os.getenv("E4_INTENSITY", "1.0")
ONLY_SINGLE = os.getenv("E4_ONLY_SINGLE", "0") == "1"
SEED = 42
N_FAST = int(os.getenv("N_FAST", "300"))
N_SYN = int(os.getenv("N_SYN", "30"))
PORT = int(os.getenv("E4_PORT", "5057"))
BASE = f"http://127.0.0.1:{PORT}"
EMOS = ["행복", "슬픔", "분노", "놀람", "두려움", "중립"]
LABELS7 = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EDGE_LIB = ROOT / "data" / "pylib_edge_tts_7.3.1"
FAIL = {"tts_retries": 0, "fast_non200": 0}

sys.path.insert(0, str(Path(__file__).parent))
from latency import SENTENCES  # noqa: E402  (같은 문장 세트)


def timing_of(r):
    h = r.headers.get("X-STAGE-Timing") or r.headers.get("X-FaceVox-Timing") or "{}"
    return json.loads(h)


def start_server():
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", PORT)) != 0, f"port {PORT} busy"
    env = dict(os.environ, TTS_BACKEND="edge", FACEVOX_PORT=str(PORT), FACEVOX_QUIET="1",
               FACEVOX_PROSODY_MODE=MODE, FACEVOX_INTENSITY=INTENSITY)
    env.pop("OPENAI_API_KEY", None)
    env.pop("FACEVOX_W_NEUTRAL", None)
    if EDGE_LIB.exists():
        env["PYTHONPATH"] = str(EDGE_LIB) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log = open(OUT / "server_log.txt", "w")
    p = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT / "app" / "tts", env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    for _ in range(150):
        try:
            if requests.get(BASE + "/get_config", timeout=0.5).ok:
                return p
        except Exception:
            pass
        time.sleep(0.2)
    p.kill()
    raise RuntimeError("server did not start")


def post_tts(payload, timeout):
    """실패 시 최대 3회 재시도. 반환 (response, client_ms, attempts)."""
    for att in range(1, 5):
        t0 = time.perf_counter()
        try:
            r = requests.post(BASE + "/tts", json=payload, timeout=timeout)
            dt = (time.perf_counter() - t0) * 1000
            if r.ok:
                return r, dt, att
        except Exception:
            r, dt = None, None
        if att < 4:
            FAIL["tts_retries"] += 1
            time.sleep(2)
    return r, dt, att


def rand_probs(rng):
    return rng.dirichlet(np.ones(7)).round(5).tolist()


def run_fast(rows_fast):
    rng = np.random.default_rng(SEED)
    now = int(time.time() * 1000)
    for i in range(1700):
        requests.post(BASE + "/set_emotion", json={"t": now - (1700 - i) * 150,
                      "label": random.choice(["Happy", "Sad", "Angry", "Neutral", "Fear", "Surprise"]),
                      "tag": "", "confidence": round(random.uniform(0.3, 1.0), 3)})
    s = requests.Session()

    def timed(stage, i, path, payload, **extra):
        t0 = time.perf_counter()
        r = s.post(BASE + path, json=payload)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            FAIL["fast_non200"] += 1
        row = {"stage": stage, "i": i, "ms": ms, "status": r.status_code}
        if "resolve" in stage:
            row["tag"] = r.json().get("tag", "")
        rows_fast.append(row)

    # 1) /set_emotion
    for i in range(N_FAST):
        timed("set_emotion", i, "/set_emotion", {"t": int(time.time() * 1000), "label": random.choice(
            ["Happy", "Sad", "Angry", "Neutral"]), "tag": "", "confidence": round(random.uniform(0.3, 1.0), 3)})
    n_hist = len(requests.get(BASE + "/debug/emo_history?limit=5000").json()["events"])
    tagging0 = requests.get(BASE + "/debug/backend").json().get("tagging")
    # 2) /emotion/resolve_one — 확률 요약 없음(지속 시간 방식, 기존 측정과 같은 조건)
    for i in range(N_FAST):
        until = int(time.time() * 1000)
        timed("resolve_one", i, "/emotion/resolve_one", {"since_ms": until - random.randint(2000, 10000),
                                                         "until_ms": until})
    # 2b) 확률 요약 버퍼를 3000개(0.2 s × 3000 = 10분)로 채운 뒤 /set_emotion_probs
    now = int(time.time() * 1000)
    for j in range(3000):
        t1 = now - (3000 - j - 1) * 200
        requests.post(BASE + "/set_emotion_probs", json={"t0": t1 - 200, "t1": t1, "n": int(rng.integers(3, 7)),
                      "probs": rand_probs(rng), "labels": LABELS7, "classifier": "cnn"})
    for i in range(N_FAST):
        t1 = int(time.time() * 1000)
        timed("set_emotion_probs", i, "/set_emotion_probs", {"t0": t1 - 200, "t1": t1, "n": int(rng.integers(3, 7)),
              "probs": rand_probs(rng), "labels": LABELS7, "classifier": "cnn"})
    dbg = requests.get(BASE + "/debug/backend").json()
    # 2c) /emotion/resolve_one — 질량 방식
    for i in range(N_FAST):
        until = int(time.time() * 1000)
        timed("resolve_one_mass", i, "/emotion/resolve_one", {"since_ms": until - random.randint(2000, 10000),
                                                              "until_ms": until})
    info = {"emo_history_size": n_hist, "tagging_before_probs": tagging0, "tagging_after_probs": dbg.get("tagging"),
            "w_neutral": dbg.get("w_neutral"), "classifier": dbg.get("classifier"), "prob_history_size": 3000}
    print("fast stages done", info, flush=True)
    return info


def run_single(rows_syn):
    from io import BytesIO
    from pydub import AudioSegment
    for rep in range(N_SYN):
        for L in SENTENCES:
            text = SENTENCES[L][rep % 3]
            emo = EMOS[rep % 6]
            r, dt, att = post_tts({"text": text, "emotion": emo}, 60)
            ok = r is not None and r.ok
            tm = timing_of(r) if ok else {}
            seg = (tm.get("segments") or [{}])[0]
            audio_sec = len(AudioSegment.from_file(BytesIO(r.content), format="mp3")) / 1000 if ok else None
            rows_syn.append({"length": L, "rep": rep, "emotion": emo, "chars": len(text),
                             "status": r.status_code if r is not None else -1, "attempts": att,
                             "client_ms": dt, "server_ms": tm.get("server_total_ms"),
                             "network_ms": seg.get("network_ms"), "decode_ms": seg.get("decode_ms"),
                             "dsp_ms": seg.get("dsp_ms"), "encode_ms": seg.get("encode_ms"),
                             "synth_total_ms": seg.get("total_ms"), "export_ms": tm.get("merge_export_ms"),
                             "audio_sec": audio_sec, "t_wall": datetime.now().isoformat(timespec="seconds")})
            print(f"syn {L:6s} rep{rep:02d} {dt or -1:7.0f} ms net={seg.get('network_ms', 0) or 0:6.0f} "
                  f"dsp={seg.get('dsp_ms', 0) or 0:5.0f} att={att}", flush=True)


def run_multi(rows_multi):
    pool = SENTENCES["medium"]
    for rep in range(N_SYN):
        for n in (1, 3, 5):
            text = " ".join(pool[(rep + j) % 3] + f"<{EMOS[(rep + j) % 6]}>" for j in range(n))
            r, dt, att = post_tts({"text": text, "emotion": "중립"}, 120)
            ok = r is not None and r.ok
            tm = timing_of(r) if ok else {}
            segs = tm.get("segments") or []
            rows_multi.append({"n_sent": n, "rep": rep, "status": r.status_code if r is not None else -1,
                               "attempts": att, "client_ms": dt, "server_ms": tm.get("server_total_ms"),
                               "network_ms_sum": sum(x.get("network_ms", 0) for x in segs),
                               "dsp_ms_sum": sum(x.get("dsp_ms", 0) for x in segs),
                               "decode_ms_sum": sum(x.get("decode_ms", 0) for x in segs),
                               "merge_export_ms": tm.get("merge_export_ms"), "audio_sec": tm.get("audio_sec"),
                               "t_wall": datetime.now().isoformat(timespec="seconds")})
            print(f"multi n={n} rep{rep:02d} {dt or -1:7.0f} ms audio={tm.get('audio_sec')} att={att}", flush=True)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    import pandas as pd
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "raw_synth_sentence.csv").exists(), "결과 덮어쓰기 금지"
    started = datetime.now().isoformat(timespec="seconds")
    proc = start_server()
    sess_info = requests.get(BASE + "/debug/backend").json()
    print("backend:", sess_info, flush=True)
    rows_fast, rows_syn, rows_multi = [], [], []
    fast_info = None
    try:
        t0 = time.perf_counter()
        r = requests.post(BASE + "/tts", json={"text": SENTENCES["short"][0], "emotion": "중립"}, timeout=60)
        cold = {"client_ms": (time.perf_counter() - t0) * 1000, "status": r.status_code, **timing_of(r)}
        print(f"cold first /tts: {cold['client_ms']:.0f} ms status {r.status_code}", flush=True)
        for i in range(3):
            post_tts({"text": SENTENCES["short"][1], "emotion": EMOS[i]}, 60)
        t_fast = datetime.now().isoformat(timespec="seconds")
        if not ONLY_SINGLE:
            fast_info = run_fast(rows_fast)
        t_single = datetime.now().isoformat(timespec="seconds")
        run_single(rows_syn)
        t_multi = datetime.now().isoformat(timespec="seconds")
        if not ONLY_SINGLE:
            run_multi(rows_multi)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    if rows_fast:
        pd.DataFrame(rows_fast).to_csv(OUT / "raw_fast.csv", index=False)
    pd.DataFrame(rows_syn).to_csv(OUT / "raw_synth_sentence.csv", index=False)
    if rows_multi:
        pd.DataFrame(rows_multi).to_csv(OUT / "raw_multi_sentence.csv", index=False)
    env = {
        "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
        "phase_start": {"fast": t_fast, "single": t_single, "multi": t_multi},
        "machine": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
        "backend": sess_info, "prosody_mode": MODE, "intensity": float(INTENSITY),
        "n_fast": 0 if ONLY_SINGLE else N_FAST, "n_syn": N_SYN, "fast_info": fast_info,
        "cold_first_tts": cold, "seed": SEED, "sentences": SENTENCES, "failures": FAIL,
        "edge_tts_lib": str(EDGE_LIB) if EDGE_LIB.exists() else "venv",
        "md5": {"app.py": md5(ROOT / "app/tts/app.py"), "backends.py": md5(ROOT / "app/tts/backends.py"),
                "prosody.py": md5(ROOT / "app/tts/prosody.py")},
        "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "note": "edge-tts는 Microsoft 온라인 서비스이므로 network_ms에는 인터넷 왕복과 원격 합성 시간이 함께 포함됨",
    }
    (OUT / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2))
    print("saved.", FAIL)


if __name__ == "__main__":
    main()
