"""dev 문장 4개(S03~S06)로만 C/D 프리셋 크기를 격자 탐색하여 고정한다.

원칙: 각 감정의 '방향'(높/낮, 빠름/느림, 확대/축소)은 졸업논문 프리셋 서술과 운율 문헌
(Murray & Arnott 1993; Scherer 2003; Juslin & Laukka 2003)에서 정하고,
'크기'만 dev 문장에서 고른다. held-out 문장(TEST)은 이 스크립트에서 전혀 쓰지 않는다.

목적함수 = dev 6감정 제한 argmax 정답 수 + 0.01 × 평균 로그 마진
  로그 마진 = log10 P(target) - max_{other in 6} log10 P(other), [-10, 10]로 절단
(emotion2vec 출력이 거의 one-hot이라 정답 수만으로는 동점이 많아 마진으로 동점 해소)
격자는 1차 dev 탐색 후 한 차례 수정했다(경계값 선택 → 한 단계 확장, 두려움 0/4 → 기식 잡음·
강한 떨림 추가, 중립 D 탐색 추가). 수정 역시 dev 결과만 보고 했다.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as C  # noqa: E402
import prosody as P  # noqa: E402

CACHE = C.PROJ / "data" / "E3_dev_tuning_cache"   # 대용량(143MB) 캐시 → data/
EMO_T = [e for e in C.EMOS if e != "neutral"]

C_GRID = {
    "happy":    {"rate": ["+0%", "+10%", "+20%"], "pitch": ["+10Hz", "+20Hz", "+35Hz", "+50Hz", "+65Hz"], "volume": ["+0%", "+20%"]},
    "sad":      {"rate": ["-10%", "-20%", "-30%", "-40%"], "pitch": ["-5Hz", "-15Hz", "-30Hz", "-45Hz", "-60Hz"], "volume": ["+0%", "-30%"]},
    # 분노: 논문 서술(낮은 피치)과 문헌(높은 피치·넓은 범위)이 상충 → 양방향 모두 탐색
    "angry":    {"rate": ["+0%", "+10%", "+20%", "+30%"], "pitch": ["-20Hz", "-5Hz", "+10Hz", "+25Hz"], "volume": ["+20%", "+40%"]},
    "surprise": {"rate": ["+0%", "+10%", "+20%", "+30%"], "pitch": ["+20Hz", "+35Hz", "+50Hz", "+70Hz", "+90Hz"], "volume": ["+0%", "+20%"]},
    # 두려움: 논문 서술(낮은 피치, 느림)과 문헌(높은 피치, 빠름)이 상충 → 양방향 모두 탐색
    "fear":     {"rate": ["-15%", "+0%", "+15%", "+30%"], "pitch": ["-35Hz", "-20Hz", "+0Hz", "+20Hz", "+40Hz"], "volume": ["+0%", "-30%"]},
}

D_GRID = {
    "happy":    {"f0_range": [1.0, 1.3, 1.6, 2.0], "f0_shift": [0.0, 1.0, 2.0], "contour": [("none", 0.0)],
                 "tremor": [(6.0, 0.0)], "tempo": [1.0, 1.1], "tilt_db": [0.0, 3.0, 6.0], "gain_db": [2.0]},
    "sad":      {"f0_range": [0.4, 0.6, 0.8, 1.0], "f0_shift": [0.0, -1.0, -2.0], "contour": [("decline", 0.0), ("decline", 3.0)],
                 "tremor": [(6.0, 0.0)], "tempo": [1.0, 0.9], "tilt_db": [0.0, -3.0, -6.0], "gain_db": [-4.0]},
    "angry":    {"f0_range": [1.0, 1.3, 1.6, 2.0], "f0_shift": [-1.0, 0.0, 1.0], "contour": [("none", 0.0)],
                 "tremor": [(6.0, 0.0)], "tempo": [1.0, 1.1], "tilt_db": [0.0, 3.0, 6.0], "gain_db": [4.0]},
    "surprise": {"f0_range": [1.0, 1.4, 1.8], "f0_shift": [0.0, 1.0, 2.0], "contour": [("onset_rise", 0.0), ("onset_rise", 3.0), ("onset_rise", 6.0)],
                 "tremor": [(6.0, 0.0)], "tempo": [1.0, 1.1], "tilt_db": [0.0, 3.0], "gain_db": [2.0]},
    "fear":     {"f0_range": [0.6, 1.0, 1.4], "f0_shift": [0.0, 2.0], "contour": [("none", 0.0)],
                 "tremor": [(6.0, 0.0), (6.0, 0.6), (8.0, 1.0)], "tempo": [1.0, 1.15], "tilt_db": [0.0, -3.0], "gain_db": [-4.0],
                 "breath_snr": [None, 20.0, 12.0]},
    # 중립: 졸업논문 서술 '억양 과장 금지' → 범위 축소·약간 낮춤 방향만 탐색 (C 중립 = B 그대로)
    "neutral":  {"f0_range": [1.0, 0.8, 0.6], "f0_shift": [0.0, -1.0, -2.0], "contour": [("none", 0.0)],
                 "tremor": [(6.0, 0.0)], "tempo": [1.0, 0.95], "tilt_db": [0.0, -3.0], "gain_db": [0.0]},
}


def score(target: str, ours: dict) -> tuple[int, float]:
    lp = {k: np.log10(max(v, 1e-30)) for k, v in ours.items()}
    other = max(v for k, v in lp.items() if k != target)
    margin = float(np.clip(lp[target] - other, -10, 10))
    pred = max(ours, key=ours.get)
    return int(pred == target), margin


async def _synth_all(jobs, conc=4):
    sem = asyncio.Semaphore(conc)

    async def one(text, cp, path):
        if path.exists():
            return
        async with sem:
            for attempt in range(4):
                try:
                    mp3 = await P._edge_async(text, cp["rate"], cp["pitch"], cp["volume"], P.VOICE)
                    break
                except Exception:
                    await asyncio.sleep(2 + attempt * 2)
            else:
                raise RuntimeError(f"edge-tts failed: {path}")
        y = P.mp3_to_array(mp3)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, y, P.SR, subtype="PCM_16")

    await asyncio.gather(*[one(*j) for j in jobs])


def cfg_name(cp):
    return f"r{cp['rate']}_p{cp['pitch']}_v{cp['volume']}".replace("%", "").replace("+", "p").replace("-", "m")


def tune_c():
    jobs, rows = [], []
    for emo in EMO_T + ["neutral"]:
        grid = C_GRID.get(emo, {"rate": ["+0%"], "pitch": ["+0Hz"], "volume": ["+0%"]})
        for r, p, v in itertools.product(grid["rate"], grid["pitch"], grid["volume"]):
            cp = {"rate": r, "pitch": p, "volume": v}
            for sid in C.DEV:
                jobs.append((C.SENTENCES[sid], cp, CACHE / "C" / emo / cfg_name(cp) / f"{sid}.wav"))
    t = time.time()
    asyncio.run(_synth_all(jobs))
    print(f"[C] synthesized {len(jobs)} files in {time.time()-t:.0f}s", flush=True)
    for emo in EMO_T:
        grid = C_GRID[emo]
        for r, p, v in itertools.product(grid["rate"], grid["pitch"], grid["volume"]):
            cp = {"rate": r, "pitch": p, "volume": v}
            acc, mg = [], []
            for sid in C.DEV:
                s = C.ser_emotion2vec(CACHE / "C" / emo / cfg_name(cp) / f"{sid}.wav")
                a, m = score(emo, s["ours"])
                acc.append(a); mg.append(m)
            rows.append({"emotion": emo, **cp, "dev_correct": sum(acc), "dev_margin": np.mean(mg),
                         "objective": sum(acc) + 0.01 * np.mean(mg)})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "tuning_C_grid.csv", index=False)
    best = {}
    for emo in EMO_T:
        d = df[df.emotion == emo].sort_values("objective", ascending=False).iloc[0]
        best[emo] = {"rate": d.rate, "pitch": d.pitch, "volume": d.volume}
        print(f"[C] {emo}: {best[emo]} correct={d.dev_correct}/4 margin={d.dev_margin:.2f}", flush=True)
    best["neutral"] = {"rate": "+0%", "pitch": "+0Hz", "volume": "+0%"}
    return best


def tune_d(cbest):
    rows = []
    src = {emo: {sid: sf.read(CACHE / "C" / emo / cfg_name(cbest[emo]) / f"{sid}.wav")[0]
                 for sid in C.DEV} for emo in EMO_T}
    tmp = CACHE / "D_tmp.wav"
    t = time.time()
    src["neutral"] = {sid: sf.read(CACHE / "C" / "neutral" / cfg_name(cbest["neutral"]) / f"{sid}.wav")[0]
                      for sid in C.DEV}
    for emo in EMO_T + ["neutral"]:
        g = {"breath_snr": [None], **D_GRID[emo]}
        keys = list(g)
        for combo in itertools.product(*[g[k] for k in keys]):
            p = dict(zip(keys, combo))
            acc, mg = [], []
            for sid in C.DEV:
                z = P.apply_dsp(src[emo][sid], P.SR, p)
                sf.write(tmp, z, P.SR, subtype="PCM_16")
                s = C.ser_emotion2vec(tmp)
                a, m = score(emo, s["ours"])
                acc.append(a); mg.append(m)
            rows.append({"emotion": emo, **{k: str(v) for k, v in p.items()},
                         "dev_correct": sum(acc), "dev_margin": np.mean(mg),
                         "objective": sum(acc) + 0.01 * np.mean(mg), "_p": json.dumps(p)})
        print(f"[D] {emo} grid done ({time.time()-t:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns="_p").to_csv(C.RESULTS / "tuning_D_grid.csv", index=False)
    best = {}
    for emo in EMO_T + ["neutral"]:
        d = df[df.emotion == emo].sort_values("objective", ascending=False).iloc[0]
        p = json.loads(d._p)
        p["contour"] = tuple(p["contour"]); p["tremor"] = tuple(p["tremor"])
        best[emo] = p
        print(f"[D] {emo}: {p} correct={d.dev_correct}/4 margin={d.dev_margin:.2f}", flush=True)
    return best


if __name__ == "__main__":
    cbest = tune_c()
    dbest = tune_d(cbest)
    out = {"C_PRESETS": cbest, "D_PRESETS": dbest, "dev_sentences": C.DEV,
           "objective": "dev correct(6-class restricted argmax) + 0.01*mean clipped log10 margin"}
    (C.RESULTS / "presets_frozen.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("saved presets_frozen.json")
