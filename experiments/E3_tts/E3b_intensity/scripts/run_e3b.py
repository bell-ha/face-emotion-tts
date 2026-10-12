"""E3b: 감정 강도 k 스케일링 × {D, Dp(D-psola)} — 합성, 지표, 요약, 운용점 선택.

k ∈ {0, 0.25, 0.5, 0.75, 1.0}. 모든 12문장 합성, 운용점 선택은 dev(S03~S06)만, 보고는 held-out 8문장.
운용점 규칙(사전 고정): k>0 후보 중 dev 화자 유사도(감정 음성 vs B 중립, 평균) ≥ 0.60 그리고
dev SQUIM-MOS(6감정 평균) ≥ 4.0 을 만족하는 것 가운데 dev emotion2vec 정확도 최대.
동점이면 화자 유사도가 높은 쪽. 만족 후보가 없으면 화자 유사도 기준을 0.50으로 낮춰 재시도(그래도 없으면 '없음' 보고).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import stats
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parents[1]
E3 = HERE.parent
sys.path.insert(0, str(E3 / "scripts"))
import common as C  # noqa: E402
import prosody as P  # noqa: E402

RES = HERE / "results"
BIG = C.PROJ / "data" / "E3b_audio"            # 전체 오디오(약 70 MB) → data/ (.gitignore)
EDGE = C.PROJ / "data" / "E3b_edge_cache"
KS = [0.0, 0.25, 0.5, 0.75, 1.0]
VARS = ["D", "DP"]
RNG = np.random.default_rng(C.SEED)
SPK_TH, MOS_TH, SPK_TH_FALLBACK = 0.60, 4.0, 0.50


def kname(k):
    return f"k{int(round(k * 100)):03d}"


def cfg(cp):
    return f"r{cp['rate']}_p{cp['pitch']}_v{cp['volume']}".replace("%", "").replace("+", "p").replace("-", "m")


async def edge_jobs(jobs):
    sem = asyncio.Semaphore(4)

    async def one(text, cp, path):
        if path.exists():
            return
        async with sem:
            for i in range(4):
                try:
                    mp3 = await P._edge_async(text, cp["rate"], cp["pitch"], cp["volume"], P.VOICE)
                    break
                except Exception:
                    await asyncio.sleep(2 + 2 * i)
            else:
                raise RuntimeError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, P.mp3_to_array(mp3), P.SR, subtype="PCM_16")
    await asyncio.gather(*[one(*j) for j in jobs])


def synth():
    jobs, plan = [], []
    for v in VARS:
        for k in KS:
            for sid, text in C.SENTENCES.items():
                for emo in C.EMOS:
                    cp = P.scale_c_preset(P.C_PRESETS[emo], k) if v == "D" else P.C_PRESETS["neutral"]
                    src = EDGE / cfg(cp) / f"{sid}.wav"
                    jobs.append((text, cp, src))
                    plan.append((v, k, sid, emo, src))
    uniq = {j[2]: j for j in jobs}
    asyncio.run(edge_jobs(list(uniq.values())))
    for v, k, sid, emo, src in plan:
        out = BIG / f"{v}_{kname(k)}" / f"{sid}_{emo}.wav"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        y, sr = sf.read(src)
        base = (P.D_PRESETS if v == "D" else P.DP_PRESETS)[emo]
        sf.write(out, P.apply_dsp(y, sr, P.scale_dsp_params(base, k)), sr, subtype="PCM_16")
    # 예시 오디오(S01, S02)만 실험 폴더에 복사
    for d in BIG.iterdir():
        dst = HERE / "audio" / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in d.glob("S0[12]_*.wav"):
            shutil.copy(f, dst / f.name)


def measure():
    cache = RES / "per_file_raw.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    rows = []
    nmr_d, nmr_a = C.AUDIO / "B" / "S05_neutral.wav", C.AUDIO / "B" / "S06_neutral.wav"
    bemb = {sid: C.spk_embed(C.AUDIO / "B" / f"{sid}_neutral.wav") for sid in C.SENTENCES}
    i = 0
    for v in VARS:
        for k in KS:
            for sid, text in C.SENTENCES.items():
                for emo in C.EMOS:
                    f = BIG / f"{v}_{kname(k)}" / f"{sid}_{emo}.wav"
                    r = dict(variant=v, k=k, sid=sid, emotion=emo, split="dev" if sid in C.DEV else "test",
                             path=str(f))
                    a = C.acoustic_features(f, text)
                    r.update({x: a[x] for x in ("f0_mean_hz", "f0_sd_st", "int_mean_db", "rate_syl_per_s", "dur_s")})
                    s = C.ser_emotion2vec(f)
                    r["e2v_pred9"], r["e2v_pred6"] = s["pred9"], s["pred6"]
                    r["e2v_p_target6"] = s["ours"][emo] / max(sum(s["ours"].values()), 1e-12)
                    w = C.ser_w2v(f)
                    r["w2v_pred"] = w["pred"]
                    r["spk_cos_B"] = float(np.dot(C.spk_embed(f), bemb[sid]))
                    r.update(C.squim(f, nmr_a if sid == "S05" else nmr_d))
                    rows.append(r)
                    i += 1
                    if i % 60 == 0:
                        print(f"  measured {i}/{2 * len(KS) * 72}", flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df


def boot_mean(sub, col, n=5000):
    sids = sub.sid.unique()
    g = {s: sub[sub.sid == s][col].values.astype(float) for s in sids}
    vals = [np.concatenate([g[s] for s in RNG.choice(sids, len(sids))]).mean() for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def mcnemar(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return n01, n10, (1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue))


def main():
    synth()
    d = measure()
    d["correct9"] = (d.e2v_pred9 == d.emotion).astype(int)
    d["w2v_correct"] = (d.w2v_pred == d.emotion).astype(int)
    d.to_csv(RES / "per_file.csv", index=False)
    rows = []
    for split in ("dev", "test"):
        for v in VARS:
            for k in KS:
                s = d[(d.variant == v) & (d.k == k) & (d.split == split)]
                emo_only = s[s.emotion != "neutral"]
                w = s[s.emotion.isin(["happy", "sad", "angry", "neutral"])]
                ci = boot_mean(s, "correct9")
                cis = boot_mean(emo_only, "spk_cos_B")
                cim = boot_mean(s, "squim_mos")
                rows.append(dict(
                    split=split, variant=v, k=k, n=len(s),
                    acc9=s.correct9.mean(), acc9_lo=ci[0], acc9_hi=ci[1],
                    macroF1=f1_score(s.emotion, s.e2v_pred9, labels=C.EMOS, average="macro", zero_division=0),
                    p_target_mean=s.e2v_p_target6.mean(),
                    w2v4_acc=w.w2v_correct.mean(),
                    spk_cos=emo_only.spk_cos_B.mean(), spk_lo=cis[0], spk_hi=cis[1],
                    spk_cos_min=emo_only.spk_cos_B.min(),
                    mos=s.squim_mos.mean(), mos_lo=cim[0], mos_hi=cim[1],
                    pesq=s.squim_pesq.mean(),
                ))
    sm = pd.DataFrame(rows)
    sm.to_csv(RES / "summary.csv", index=False)
    rec = d[d.split == "test"].groupby(["variant", "k", "emotion"]).correct9.mean().unstack()
    rec.to_csv(RES / "recall_by_emotion_test.csv")
    spe = d[(d.split == "test") & (d.emotion != "neutral")].groupby(["variant", "k", "emotion"]).spk_cos_B.mean().unstack()
    spe.to_csv(RES / "spk_by_emotion_test.csv")

    # 운용점 선택 (dev만)
    dev = sm[(sm.split == "dev") & (sm.k > 0)].copy()
    chosen, rule_used = None, None
    for th in (SPK_TH, SPK_TH_FALLBACK):
        ok = dev[(dev.spk_cos >= th) & (dev.mos >= MOS_TH)]
        if len(ok):
            chosen = ok.sort_values(["acc9", "spk_cos"], ascending=False).iloc[0]
            rule_used = f"spk_cos>={th} & mos>={MOS_TH}"
            break
    op = {"rule": "max dev acc9 s.t. dev spk_cos(emotional vs B neutral) >= 0.60 and dev SQUIM-MOS >= 4.0; "
                  "tie -> higher spk; fallback spk >= 0.50",
          "rule_used": rule_used,
          "dev_candidates": dev[["variant", "k", "acc9", "spk_cos", "mos"]].round(4).to_dict("records")}
    if chosen is not None:
        t = sm[(sm.split == "test") & (sm.variant == chosen.variant) & (sm.k == chosen.k)].iloc[0]
        op["chosen"] = {"variant": chosen.variant, "k": float(chosen.k),
                        "dev": chosen[["acc9", "spk_cos", "mos"]].astype(float).round(4).to_dict(),
                        "test": t.drop(["split", "variant"]).astype(float).round(4).to_dict()}

    # 검정 (test)
    tests = []
    T = d[d.split == "test"]
    def pair(v1, k1, v2, k2):
        a = T[(T.variant == v1) & (T.k == k1)].set_index(["sid", "emotion"]).sort_index()
        b = T[(T.variant == v2) & (T.k == k2)].set_index(["sid", "emotion"]).sort_index()
        n01, n10, p = mcnemar(a.correct9, b.correct9)
        ae, be = a[a.index.get_level_values(1) != "neutral"], b[b.index.get_level_values(1) != "neutral"]
        ws = stats.wilcoxon(ae.spk_cos_B, be.spk_cos_B) if (ae.spk_cos_B != be.spk_cos_B).any() else None
        wm = stats.wilcoxon(a.squim_mos, b.squim_mos) if (a.squim_mos != b.squim_mos).any() else None
        tests.append(dict(a=f"{v1}@{k1}", b=f"{v2}@{k2}", acc_a=a.correct9.mean(), acc_b=b.correct9.mean(),
                          a_only=n01, b_only=n10, mcnemar_p=p,
                          spk_a=ae.spk_cos_B.mean(), spk_b=be.spk_cos_B.mean(),
                          spk_wilcoxon_p=None if ws is None else float(ws.pvalue),
                          mos_a=a.squim_mos.mean(), mos_b=b.squim_mos.mean(),
                          mos_wilcoxon_p=None if wm is None else float(wm.pvalue)))
    for k in KS[1:]:
        pair("DP", k, "D", k)
    if chosen is not None:
        pair(chosen.variant, float(chosen.k), "D", 1.0)
    pd.DataFrame(tests).to_csv(RES / "stat_tests.csv", index=False)
    (RES / "operating_point.json").write_text(json.dumps(op, indent=1, ensure_ascii=False))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(sm.round(3).to_string()); print(rec.round(2)); print(spe.round(2))
    print(json.dumps(op, indent=1, ensure_ascii=False)); print(pd.DataFrame(tests).round(4).to_string())


if __name__ == "__main__":
    main()
