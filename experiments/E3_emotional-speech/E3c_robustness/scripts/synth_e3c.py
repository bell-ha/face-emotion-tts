"""E3c 합성: (A) SunHi 5회 독립 재합성, (B) 남성 화자 InJoon + D-psola(튜닝 없음).

실험 A  run1..run5: held-out 8문장 × 6감정.
    B  : edge-tts SunHi 중립(문장당 1개 = 모든 감정에 같은 음성)
    C  : edge-tts + SSML 프리셋 6감정(중립도 따로 요청, E3와 같음)
    D  : C 출력 + D_PRESETS PSOLA 후처리 (k=1)
    DP : 이 run의 B 중립 + DP_PRESETS PSOLA 후처리 (k=1)
    run 하나 = edge-tts 요청 56개(B 8 + C 48). run끼리는 순서대로, run 안에서는 동시 4개(E3와 같음).
실험 B  rep1..rep2: InJoon 중립 8문장(edge-tts 요청 8개) → DP_PRESETS를 k ∈ {0.5, 1.0}로 그대로 적용.

오디오는 data/E3c_audio/(.gitignore)에 저장한다. 기존 audio/, results/는 건드리지 않는다.
실패한 요청은 최대 6회까지 재시도하고 모든 실패를 results/synth_log_*.json에 남긴다.

실행: ../../../.venv/bin/python scripts/synth_e3c.py a   # 실험 A
      ../../../.venv/bin/python scripts/synth_e3c.py b   # 실험 B
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parents[1]
E3 = HERE.parent
sys.path.insert(0, str(E3 / "scripts"))
import common as C  # noqa: E402
# 2026-11-19 기준 .venv의 edge-tts 7.2.8은 서버가 403(WSServerHandshakeError)으로 거부한다.
# 요청 내용(SSML·출력 형식, communicate.py)은 같고 인증 토큰(drm.py)만 바뀐 7.3.1을
# data/pylib_edge_tts_7.3.1/에 따로 설치해 우선 사용한다(.venv는 건드리지 않음).
_EDGE = C.PROJ / "data" / "pylib_edge_tts_7.3.1"
if _EDGE.exists():
    sys.path.insert(0, str(_EDGE))
sys.path.insert(0, str(C.PROJ / "app" / "tts"))
import prosody as P  # noqa: E402

RES = HERE / "results"
BIG = C.PROJ / "data" / "E3c_audio"
N_RUNS = 5
N_REPS_B = 2
KS_B = [0.5, 1.0]
INJOON = "ko-KR-InJoonNeural"
MAX_TRIES = 10
np.random.seed(C.SEED)


def kname(k):
    return f"k{int(round(k * 100)):03d}"


class Log:
    def __init__(self):
        self.requests = 0
        self.failures = 0
        self.errors = []
        self.blocks = []


async def fetch(text, cp, voice, path: Path, sem, log: Log):
    if path.exists():
        return
    async with sem:
        for i in range(MAX_TRIES):
            log.requests += 1
            try:
                mp3 = await P._edge_async(text, cp["rate"], cp["pitch"], cp["volume"], voice)
                break
            except Exception as e:  # noqa: BLE001
                log.failures += 1
                log.errors.append({"file": str(path.relative_to(BIG)), "try": i + 1, "error": repr(e)[:200],
                                   "time": datetime.now().isoformat(timespec="seconds")})
                await asyncio.sleep(min(3 * (i + 1), 30))
        else:
            raise RuntimeError(f"edge-tts failed {MAX_TRIES} times: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, P.mp3_to_array(mp3), P.SR, subtype="PCM_16")


async def gather(jobs, log):
    sem = asyncio.Semaphore(4)
    await asyncio.gather(*[fetch(*j, sem, log) for j in jobs])


def dsp(src: Path, dst: Path, params: dict):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(src)
    sf.write(dst, P.apply_dsp(y, sr, params), sr, subtype="PCM_16")


def synth_a(log):
    for r in range(1, N_RUNS + 1):
        rd = BIG / f"run{r}"
        jobs = []
        for sid in C.TEST:
            text = C.SENTENCES[sid]
            jobs.append((text, P.C_PRESETS["neutral"], P.VOICE, rd / "B" / f"{sid}_neutral.wav"))
            for emo in C.EMOS:
                jobs.append((text, P.C_PRESETS[emo], P.VOICE, rd / "C" / f"{sid}_{emo}.wav"))
        t0 = datetime.now().isoformat(timespec="seconds")
        f0 = log.failures
        asyncio.run(gather(jobs, log))
        t1 = datetime.now().isoformat(timespec="seconds")
        for sid in C.TEST:
            for emo in C.EMOS:
                dsp(rd / "C" / f"{sid}_{emo}.wav", rd / "D" / f"{sid}_{emo}.wav", P.D_PRESETS[emo])
                dsp(rd / "B" / f"{sid}_neutral.wav", rd / "DP" / f"{sid}_{emo}.wav", P.DP_PRESETS[emo])
        log.blocks.append({"block": f"run{r}", "edge_requests_planned": len(jobs), "edge_start": t0, "edge_end": t1,
                           "failures_in_block": log.failures - f0})
        print(f"run{r}: edge {t0} → {t1}, failures {log.failures - f0}", flush=True)
        time.sleep(5)


def synth_b(log):
    for rep in range(1, N_REPS_B + 1):
        rd = BIG / "injoon" / f"rep{rep}"
        jobs = [(C.SENTENCES[sid], P.C_PRESETS["neutral"], INJOON, rd / "B" / f"{sid}_neutral.wav") for sid in C.TEST]
        t0 = datetime.now().isoformat(timespec="seconds")
        f0 = log.failures
        asyncio.run(gather(jobs, log))
        t1 = datetime.now().isoformat(timespec="seconds")
        for k in KS_B:
            for sid in C.TEST:
                for emo in C.EMOS:
                    dsp(rd / "B" / f"{sid}_neutral.wav", rd / f"DP_{kname(k)}" / f"{sid}_{emo}.wav",
                        P.scale_dsp_params(P.DP_PRESETS[emo], k))
        log.blocks.append({"block": f"injoon_rep{rep}", "edge_requests_planned": len(jobs), "edge_start": t0,
                           "edge_end": t1, "failures_in_block": log.failures - f0})
        print(f"injoon rep{rep}: edge {t0} → {t1}, failures {log.failures - f0}", flush=True)
        time.sleep(5)


def _write_log(which, started, log, status):
    out = {"status": status, "experiment": which, "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
           "edge_tts_version": __import__("edge_tts").__version__, "voice": P.VOICE if which == "a" else INJOON,
           "edge_requests_total_incl_retries": log.requests, "failures": log.failures,
           "blocks": log.blocks, "errors": log.errors, "seed": C.SEED,
           "note": "edge-tts는 Microsoft 온라인 서비스(인터넷 필요). PSOLA 후처리는 결정적(기식 잡음 시드 42)."}
    stamp = started.replace(":", "").replace("-", "")
    (RES / f"synth_log_{which}_{stamp}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "errors"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    which = sys.argv[1]
    started = datetime.now().isoformat(timespec="seconds")
    LOG = Log()
    status = "ok"
    try:
        (synth_a if which == "a" else synth_b)(LOG)
    except Exception as e:  # noqa: BLE001
        status = f"aborted: {e!r}"
        raise
    finally:
        _write_log(which, started, LOG, status)
