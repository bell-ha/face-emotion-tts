"""E3c 측정: 파일마다 음향 특징, SER 2종, ECAPA 화자 유사도, SQUIM (E3·E3b와 같은 함수, scripts/common.py).

실행: ../../../.venv/bin/python scripts/measure_e3c.py a   # 실험 A → results/per_file_a.csv
      ../../../.venv/bin/python scripts/measure_e3c.py b   # 실험 B → results/per_file_b.csv
화자 유사도 기준:
  A: spk_cos_B   = E3 audio/B 같은 문장 중립(원 SunHi, 2026-10-12 합성)과의 코사인 (E3·E3b와 같은 기준)
     spk_cos_own = 같은 run의 B 중립과의 코사인
  B: spk_cos_ref    = E3 audio/spk_ref 같은 문장 InJoon 중립(2026-10-12 합성)과의 코사인 (E3b와 같은 방식의 고정 기준)
     spk_cos_own    = 같은 rep의 InJoon 중립과의 코사인
     spk_cos_sunhi  = E3 audio/B 같은 문장 SunHi 중립과의 코사인 (참고)
SQUIM-MOS 비매칭 참조는 E3와 같이 audio/B/S05_neutral.wav(SunHi)를 쓴다(평가 문장에 S05 없음).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "scripts"))
import common as C  # noqa: E402

RES = HERE / "results"
BIG = C.PROJ / "data" / "E3c_audio"
NMR = C.AUDIO / "B" / "S05_neutral.wav"


def manifest(which):
    rows = []
    if which == "a":
        for r in range(1, 6):
            for cond in ("B", "C", "D", "DP"):
                for sid in C.TEST:
                    for emo in C.EMOS:
                        f = BIG / f"run{r}" / cond / (f"{sid}_neutral.wav" if cond == "B" else f"{sid}_{emo}.wav")
                        rows.append(dict(run=r, cond=cond, sid=sid, emotion=emo, path=str(f)))
    else:
        for rep in (1, 2):
            for cond in ("IB", "DP_k050", "DP_k100"):
                for sid in C.TEST:
                    for emo in C.EMOS:
                        sub = "B" if cond == "IB" else cond
                        f = BIG / "injoon" / f"rep{rep}" / sub / (f"{sid}_neutral.wav" if cond == "IB" else f"{sid}_{emo}.wav")
                        rows.append(dict(run=rep, cond=cond, sid=sid, emotion=emo, path=str(f)))
    return pd.DataFrame(rows)


def main(which):
    m = manifest(which)
    assert all(Path(p).exists() for p in m.path), "missing audio"
    emb = {}

    def E(p):
        p = str(p)
        if p not in emb:
            emb[p] = C.spk_embed(p)
        return emb[p]

    out = []
    uniq = m.drop_duplicates("path")
    for i, r in enumerate(uniq.itertuples(index=False)):
        text = C.SENTENCES[r.sid]
        d = {"path": r.path}
        a = C.acoustic_features(r.path, text)
        d.update({x: a[x] for x in ("f0_mean_hz", "f0_sd_st", "int_mean_db", "rate_syl_per_s", "dur_s", "hf_ratio_db")})
        s = C.ser_emotion2vec(r.path)
        d["e2v_pred9"], d["e2v_pred6"] = s["pred9"], s["pred6"]
        for k, v in s["raw"].items():
            d[f"e2v_raw_{k}"] = v
        w = C.ser_w2v(r.path)
        d["w2v_pred"] = w["pred"]
        for k, v in w["probs"].items():
            d[f"w2v_{k}"] = v
        d.update(C.squim(r.path, NMR))
        e = E(r.path)
        if which == "a":
            d["spk_cos_B"] = float(e @ E(C.AUDIO / "B" / f"{r.sid}_neutral.wav"))
            d["spk_cos_own"] = float(e @ E(BIG / f"run{r.run}" / "B" / f"{r.sid}_neutral.wav"))
        else:
            d["spk_cos_ref"] = float(e @ E(C.AUDIO / "spk_ref" / f"{r.sid}_InJoon_neutral.wav"))
            d["spk_cos_own"] = float(e @ E(BIG / "injoon" / f"rep{r.run}" / "B" / f"{r.sid}_neutral.wav"))
            d["spk_cos_sunhi"] = float(e @ E(C.AUDIO / "B" / f"{r.sid}_neutral.wav"))
        out.append(d)
        if i % 40 == 0:
            print(f"  {which}: {i}/{len(uniq)}", flush=True)
    pf = m.merge(pd.DataFrame(out), on="path", how="left")
    pf["correct9"] = (pf.e2v_pred9 == pf.emotion).astype(int)
    pf["correct6"] = (pf.e2v_pred6 == pf.emotion).astype(int)
    pf["w2v_correct"] = (pf.w2v_pred == pf.emotion).astype(int)
    pf.to_csv(RES / f"per_file_{which}.csv", index=False)
    print("saved", len(pf), "rows")


if __name__ == "__main__":
    main(sys.argv[1])
