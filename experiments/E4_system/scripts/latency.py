"""E4 단계별 지연 측정.

Flask 앱(tts/app.py, edge 백엔드 + E3 prosody D 모드)을 별도 프로세스로 띄우고 HTTP로 측정한다.
  1) /set_emotion 왕복           (N_FAST회)
  2) /emotion/resolve_one 왕복   (N_FAST회, 이벤트 버퍼 가득 찬 상태 = 2000개)
  3) 문장 하나 합성(/tts, 태그 없음) × 길이 3종 × N_SYN회
     서버가 X-FaceVox-Timing 헤더로 단계 시간(네트워크 edge-tts / 디코딩 / DSP / 인코딩)을 돌려준다.
  4) 다문장 /tts(태그 포함) 1·3·5문장 × N_SYN회 (종단 간)

실행: .venv/bin/python experiments/E4_system/scripts/latency.py
"""
from __future__ import annotations

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

import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "E4_system" / "results" / os.getenv("E4_SUBDIR", "")
MODE = os.getenv("E4_MODE", "D")            # 운율 모드: D(논문 주 조건) | DP(D-psola) | C
INTENSITY = os.getenv("E4_INTENSITY", "1.0")
ONLY_SINGLE = os.getenv("E4_ONLY_SINGLE", "0") == "1"   # 문장 하나 합성만 측정
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FAST = int(os.getenv("N_FAST", "300"))
N_SYN = int(os.getenv("N_SYN", "30"))
PORT = int(os.getenv("E4_PORT", "5057"))
BASE = f"http://127.0.0.1:{PORT}"
EMOS = ["행복", "슬픔", "분노", "놀람", "두려움", "중립"]

# 감정 중립 내용의 한국어 문장(길이 3종 × 3개)
SENTENCES = {
    "short": ["오늘은 화요일입니다.", "창문이 열려 있어요.", "버스가 도착했습니다."],
    "medium": ["내일 아침 아홉 시에 회의실에서 팀 회의가 있습니다.",
               "도서관은 매주 월요일마다 정기 휴관을 합니다.",
               "그는 가방에서 노트를 꺼내 책상 위에 올려 두었어요."],
    "long": ["지난주에 주문한 책이 오늘 오후에 도착했는데, 생각보다 두꺼워서 이번 주말 동안 천천히 읽어 볼 생각입니다.",
             "역 앞 사거리에서 왼쪽으로 돌아 두 블록을 걸어가면, 오래된 빵집 옆에 작은 우체국 건물이 하나 있습니다.",
             "회의가 끝난 뒤에 우리는 다음 달 일정표를 다시 확인하고, 각자 맡은 일을 메모장에 정리해 두었습니다."],
}


def free_port_ok(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def start_server():
    assert free_port_ok(PORT), f"port {PORT} busy"
    env = dict(os.environ, TTS_BACKEND="edge", FACEVOX_PORT=str(PORT), FACEVOX_QUIET="1",
               FACEVOX_PROSODY_MODE=MODE, FACEVOX_INTENSITY=INTENSITY)
    env.pop("OPENAI_API_KEY", None)
    log = open(OUT / "server_log.txt", "w")
    p = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT / "tts", env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(100):
        try:
            if requests.get(BASE + "/get_config", timeout=0.5).ok:
                return p
        except Exception:
            pass
        time.sleep(0.2)
    p.kill()
    raise RuntimeError("server did not start")


def post(path, payload, **kw):
    t0 = time.perf_counter()
    r = requests.post(BASE + path, json=payload, **kw)
    dt = (time.perf_counter() - t0) * 1000
    return r, dt


def run_fast(rows_fast):
    """1) /set_emotion, 2) /emotion/resolve_one 왕복. 반환: resolve 중 버퍼 크기."""
    # 버퍼를 최대치(2000)에 가깝게 미리 채움
    now = int(time.time() * 1000)
    for i in range(1700):
        requests.post(BASE + "/set_emotion", json={"t": now - (1700 - i) * 150,
                      "label": random.choice(["Happy", "Sad", "Angry", "Neutral", "Fear", "Surprise"]),
                      "tag": "", "confidence": round(random.uniform(0.3, 1.0), 3)})
    s = requests.Session()
    for i in range(N_FAST):
        t0 = time.perf_counter()
        r = s.post(BASE + "/set_emotion", json={"t": int(time.time() * 1000), "label": random.choice(
            ["Happy", "Sad", "Angry", "Neutral"]), "tag": "", "confidence": round(random.uniform(0.3, 1.0), 3)})
        rows_fast.append({"stage": "set_emotion", "i": i, "ms": (time.perf_counter() - t0) * 1000,
                          "status": r.status_code})
    n_hist = len(requests.get(BASE + "/debug/emo_history?limit=5000").json()["events"])
    for i in range(N_FAST):   # 문장 작성 시간 2~10초 구간
        until = int(time.time() * 1000)
        since = until - random.randint(2000, 10000)
        t0 = time.perf_counter()
        r = s.post(BASE + "/emotion/resolve_one", json={"since_ms": since, "until_ms": until})
        rows_fast.append({"stage": "resolve_one", "i": i, "ms": (time.perf_counter() - t0) * 1000,
                          "status": r.status_code, "tag": r.json().get("tag", "")})
    print("fast stages done; history size", n_hist)
    return n_hist


def run_single(rows_syn):
    """3) 문장 하나 합성: 반복마다 길이 3종을 번갈아, 문장·감정 순환."""
    from io import BytesIO
    from pydub import AudioSegment
    for rep in range(N_SYN):
        for L in SENTENCES:
            text = SENTENCES[L][rep % 3]
            emo = EMOS[rep % 6]
            r, dt = post("/tts", {"text": text, "emotion": emo}, timeout=60)
            tm = json.loads(r.headers.get("X-FaceVox-Timing", "{}"))
            seg = (tm.get("segments") or [{}])[0]
            audio_sec = len(AudioSegment.from_file(BytesIO(r.content), format="mp3")) / 1000 if r.ok else None
            rows_syn.append({"length": L, "rep": rep, "emotion": emo, "chars": len(text), "status": r.status_code,
                             "client_ms": dt, "server_ms": tm.get("server_total_ms"),
                             "network_ms": seg.get("network_ms"), "decode_ms": seg.get("decode_ms"),
                             "dsp_ms": seg.get("dsp_ms"), "encode_ms": seg.get("encode_ms"),
                             "synth_total_ms": seg.get("total_ms"), "export_ms": tm.get("merge_export_ms"),
                             "audio_sec": audio_sec})
            print(f"syn {L:6s} rep{rep:02d} {dt:7.0f} ms net={seg.get('network_ms', 0):6.0f} dsp={seg.get('dsp_ms', 0):5.0f}")


def run_multi(rows_multi):
    """4) 다문장(태그 포함) 1·3·5문장 종단 간."""
    pool = SENTENCES["medium"]
    for rep in range(N_SYN):
        for n in (1, 3, 5):
            text = " ".join(pool[(rep + j) % 3] + f"<{EMOS[(rep + j) % 6]}>" for j in range(n))
            r, dt = post("/tts", {"text": text, "emotion": "중립"}, timeout=120)
            tm = json.loads(r.headers.get("X-FaceVox-Timing", "{}"))
            segs = tm.get("segments") or []
            rows_multi.append({"n_sent": n, "rep": rep, "status": r.status_code, "client_ms": dt,
                               "server_ms": tm.get("server_total_ms"),
                               "network_ms_sum": sum(x.get("network_ms", 0) for x in segs),
                               "dsp_ms_sum": sum(x.get("dsp_ms", 0) for x in segs),
                               "decode_ms_sum": sum(x.get("decode_ms", 0) for x in segs),
                               "merge_export_ms": tm.get("merge_export_ms"),
                               "audio_sec": tm.get("audio_sec")})
            print(f"multi n={n} rep{rep:02d} {dt:7.0f} ms audio={tm.get('audio_sec')}")


def main():
    import pandas as pd
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    proc = start_server()
    sess_info = requests.get(BASE + "/debug/backend").json()
    print("backend:", sess_info)
    rows_fast, rows_syn, rows_multi = [], [], []
    n_hist = None
    try:
        # cold start: 서버 기동 직후 첫 합성 1회(librosa 초기화 포함) — 통계에서 제외
        r, dt = post("/tts", {"text": SENTENCES["short"][0], "emotion": "중립"})
        cold = {"client_ms": dt, **json.loads(r.headers.get("X-FaceVox-Timing", "{}"))}
        print(f"cold first /tts: {dt:.0f} ms")
        for i in range(3):   # 예열 — 통계에서 제외
            post("/tts", {"text": SENTENCES["short"][1], "emotion": EMOS[i]})
        if not ONLY_SINGLE:
            n_hist = run_fast(rows_fast)
        run_single(rows_syn)
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
        "machine": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
        "backend": sess_info, "prosody_mode": MODE, "intensity": float(INTENSITY),
        "n_fast": N_FAST if not ONLY_SINGLE else 0, "n_syn": N_SYN, "history_size_during_resolve": n_hist,
        "cold_first_tts": cold, "seed": SEED, "sentences": SENTENCES,
        "note": "edge-tts는 Microsoft 온라인 서비스이므로 network_ms에는 인터넷 왕복과 원격 합성 시간이 함께 포함됨",
    }
    (OUT / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2))
    print("saved.")


if __name__ == "__main__":
    main()
