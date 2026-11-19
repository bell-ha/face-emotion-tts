"""E3c 분석: (A) 합성 반복 변동, (B) 남성 화자 InJoon 일반화.

실행: ../../../.venv/bin/python scripts/analyze_e3c.py a
      ../../../.venv/bin/python scripts/analyze_e3c.py b
run 0 = 기존 결과: B·C·D는 E3(results/per_file.csv), DP(k=1)는 E3b(E3b_intensity/results/per_file.csv).
E3b에서 다시 합성한 D(k=1)는 'run 0b'로 따로 둔다(D만 해당).
"""
from __future__ import annotations

import itertools
import json
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

RES = HERE / "results"
BIG = C.PROJ / "data" / "E3c_audio"
RNG = np.random.default_rng(C.SEED)
NB = 10000
W4 = ["happy", "sad", "angry", "neutral"]


def run0():
    e3 = pd.read_csv(E3 / "results" / "per_file.csv")
    e3 = e3[(e3.split == "test") & e3.cond.isin(["B", "C", "D"])].copy()
    e3 = e3.rename(columns={"spk_cos_B_neutral": "spk_cos_B"})
    e3["run"] = 0
    e3b = pd.read_csv(E3 / "E3b_intensity" / "results" / "per_file.csv")
    e3b = e3b[(e3b.split == "test") & (e3b.k == 1.0)].copy()
    e3b["cond"] = e3b.variant
    e3b["run"] = np.where(e3b.cond == "D", "0b", 0)
    e3b = e3b.rename(columns={})
    cols = ["run", "cond", "sid", "emotion", "e2v_pred9", "e2v_pred6", "w2v_pred", "spk_cos_B", "squim_mos"]
    e3b["e2v_pred6"] = e3b["e2v_pred6"]
    d = pd.concat([e3[cols], e3b[cols]], ignore_index=True)
    d["correct9"] = (d.e2v_pred9 == d.emotion).astype(int)
    d["correct6"] = (d.e2v_pred6 == d.emotion).astype(int)
    d["w2v_correct"] = (d.w2v_pred == d.emotion).astype(int)
    d["run"] = d.run.astype(str)
    return d


def metrics(s, spk_col="spk_cos_B"):
    w = s[s.emotion.isin(W4)]
    emo = s[s.emotion != "neutral"]
    return dict(n=len(s), acc9=s.correct9.mean(), acc6=s.correct6.mean(),
                macroF1=f1_score(s.emotion, s.e2v_pred9, labels=C.EMOS, average="macro", zero_division=0),
                w2v4_acc=w.w2v_correct.mean(),
                spk_cos=emo[spk_col].mean() if spk_col in s else np.nan,
                mos=s.squim_mos.mean())


def boot_cluster(s, col, keys, n=NB):
    """keys 열 조합을 하나의 클러스터로 재표집한 평균의 95% CI."""
    g = [grp[col].values.astype(float) for _, grp in s.groupby(keys)]
    idx = RNG.integers(0, len(g), size=(n, len(g)))
    sums = np.array([x.sum() for x in g]); lens = np.array([len(x) for x in g])
    vals = sums[idx].sum(1) / lens[idx].sum(1)
    return np.percentile(vals, [2.5, 97.5])


def boot_two_stage(s, col, n=NB):
    """문장과 run을 각각 독립적으로 재표집(교차 클러스터)한 평균의 95% CI (보수적)."""
    sids, runs = sorted(s.sid.unique()), sorted(s.run.unique())
    tab = s.groupby(["sid", "run"])[col].agg(["sum", "count"])
    S = np.array([[tab.loc[(a, b), "sum"] for b in runs] for a in sids])
    N = np.array([[tab.loc[(a, b), "count"] for b in runs] for a in sids])
    vals = np.empty(n)
    for i in range(n):
        si = RNG.integers(0, len(sids), len(sids)); ri = RNG.integers(0, len(runs), len(runs))
        vals[i] = S[np.ix_(si, ri)].sum() / N[np.ix_(si, ri)].sum()
    return np.percentile(vals, [2.5, 97.5])


def mcnemar(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return n01, n10, (1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue))


def wave_consistency():
    """run마다 B 중립을 E3 audio/B(run 0)와 비교: 길이 차, 파형 상관, 차이 신호 크기(dB)."""
    rows = []
    for r in range(1, 6):
        for sid in C.TEST:
            a, _ = sf.read(C.AUDIO / "B" / f"{sid}_neutral.wav")
            b, _ = sf.read(BIG / f"run{r}" / "B" / f"{sid}_neutral.wav")
            n = min(len(a), len(b))
            corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
            resid = 10 * np.log10(np.sum((a[:n] - b[:n]) ** 2) / np.sum(a[:n] ** 2) + 1e-20)
            rows.append(dict(run=r, sid=sid, len_diff_samples=len(b) - len(a), corr=corr, resid_db=resid))
    # run끼리(1..5) 짝
    for r1, r2 in itertools.combinations(range(1, 6), 2):
        for sid in C.TEST:
            a, _ = sf.read(BIG / f"run{r1}" / "B" / f"{sid}_neutral.wav")
            b, _ = sf.read(BIG / f"run{r2}" / "B" / f"{sid}_neutral.wav")
            n = min(len(a), len(b))
            rows.append(dict(run=f"{r1}-{r2}", sid=sid, len_diff_samples=len(b) - len(a),
                             corr=float(np.corrcoef(a[:n], b[:n])[0, 1]),
                             resid_db=10 * np.log10(np.sum((a[:n] - b[:n]) ** 2) / np.sum(a[:n] ** 2) + 1e-20)))
    return pd.DataFrame(rows)


def analyze_a():
    new = pd.read_csv(RES / "per_file_a.csv")
    new["run"] = new.run.astype(str)
    old = run0()
    d = pd.concat([old, new[old.columns.tolist() + ["spk_cos_own"]]], ignore_index=True)
    d.to_csv(RES / "per_item_a_all_runs.csv", index=False)
    CONDS = ["B", "C", "D", "DP"]
    NEW = [str(r) for r in range(1, 6)]

    # 1) run별 지표
    rows = []
    for cond in CONDS:
        for run in ["0", "0b"] + NEW:
            s = d[(d.cond == cond) & (d.run == run)]
            if s.empty:
                continue
            rows.append(dict(cond=cond, run=run, **metrics(s)))
    runs = pd.DataFrame(rows)
    runs.to_csv(RES / "runs_a.csv", index=False)

    # 2) run 간 요약 (새 5회 / 기존 포함 전체)
    summ = []
    for cond in CONDS:
        for scope, sel in (("new5", NEW), ("all", None)):
            r = runs[runs.cond == cond] if sel is None else runs[(runs.cond == cond) & runs.run.isin(sel)]
            for met in ("acc9", "acc6", "macroF1", "w2v4_acc", "spk_cos", "mos"):
                v = r[met].values.astype(float)
                summ.append(dict(cond=cond, scope=scope, n_runs=len(v), metric=met, mean=v.mean(),
                                 sd=v.std(ddof=1), min=v.min(), max=v.max(),
                                 values=";".join(f"{r_}:{x:.4f}" for r_, x in zip(r.run, v))))
    summ = pd.DataFrame(summ)
    summ.to_csv(RES / "summary_across_runs_a.csv", index=False)

    # 3) 5회 합친 정확도와 부트스트랩 CI
    pooled = []
    for cond in CONDS:
        s = d[(d.cond == cond) & d.run.isin(NEW)]
        w = s[s.emotion.isin(W4)]
        for col, sub in (("correct9", s), ("w2v_correct", w)):
            ci1 = boot_cluster(sub, col, ["sid", "run"])
            ci2 = boot_two_stage(sub, col)
            ci3 = boot_cluster(sub, col, ["sid"])
            pooled.append(dict(cond=cond, metric="e2v_acc9" if col == "correct9" else "w2v4_acc", n=len(sub),
                               mean=sub[col].mean(), ci_sent_x_run_lo=ci1[0], ci_sent_x_run_hi=ci1[1],
                               ci_two_stage_lo=ci2[0], ci_two_stage_hi=ci2[1],
                               ci_sent_only_lo=ci3[0], ci_sent_only_hi=ci3[1]))
        e = s[s.emotion != "neutral"]
        for col in ("spk_cos_B", "squim_mos"):
            sub = e if col == "spk_cos_B" else s
            ci1 = boot_cluster(sub, col, ["sid", "run"]); ci2 = boot_two_stage(sub, col)
            ci3 = boot_cluster(sub, col, ["sid"])
            pooled.append(dict(cond=cond, metric=col, n=len(sub), mean=sub[col].mean(),
                               ci_sent_x_run_lo=ci1[0], ci_sent_x_run_hi=ci1[1],
                               ci_two_stage_lo=ci2[0], ci_two_stage_hi=ci2[1],
                               ci_sent_only_lo=ci3[0], ci_sent_only_hi=ci3[1]))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(RES / "pooled_new5_a.csv", index=False)

    # 4) 문항 안정성: (문장, 감정) 문항이 5회 중 몇 번 정답이었나
    stab = []
    for cond in ("C", "D", "DP"):
        s = d[(d.cond == cond) & d.run.isin(NEW)]
        k = s.groupby(["sid", "emotion"]).correct9.sum()
        preds = s.groupby(["sid", "emotion"]).e2v_pred9.nunique()
        pair_agree = []
        for (sid, emo), g in s.groupby(["sid", "emotion"]):
            p = g.sort_values("run").e2v_pred9.values
            pair_agree += [p[i] == p[j] for i, j in itertools.combinations(range(len(p)), 2)]
        stab.append(dict(cond=cond, n_items=len(k), always_correct=int((k == 5).sum()), never_correct=int((k == 0).sum()),
                         mixed=int(((k > 0) & (k < 5)).sum()), items_with_pred_change=int((preds > 1).sum()),
                         mean_pairwise_pred_agreement=float(np.mean(pair_agree)),
                         dist_correct_count=";".join(f"{i}:{int((k == i).sum())}" for i in range(6))))
    stab = pd.DataFrame(stab)
    stab.to_csv(RES / "item_stability_a.csv", index=False)

    # 5) run별 짝지은 비교
    pairs = []
    for a_, b_ in (("D", "C"), ("C", "B"), ("D", "B"), ("DP", "D"), ("DP", "B")):
        for run in ["0"] + NEW:
            A = d[(d.cond == a_) & (d.run == run)].set_index(["sid", "emotion"]).sort_index()
            B = d[(d.cond == b_) & (d.run == run)].set_index(["sid", "emotion"]).sort_index()
            if A.empty or B.empty:
                continue
            n01, n10, p = mcnemar(A.correct9, B.correct9)
            pairs.append(dict(a=a_, b=b_, run=run, acc_a=A.correct9.mean(), acc_b=B.correct9.mean(),
                              diff=A.correct9.mean() - B.correct9.mean(), a_only=n01, b_only=n10, mcnemar_p=p))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(RES / "paired_by_run_a.csv", index=False)

    # 6) 감정별 재현율 (run별, 5회 합)
    rec = d[d.run.isin(["0", "0b"] + NEW)].groupby(["cond", "run", "emotion"]).correct9.sum().unstack()[C.EMOS]
    rec.to_csv(RES / "recall_by_run_a.csv")
    rec5 = d[d.run.isin(NEW)].groupby(["cond", "emotion"]).correct9.mean().unstack()[C.EMOS]
    rec5.to_csv(RES / "recall_pooled_new5_a.csv")

    # 7) 77.1%, 72.9%의 위치
    dvals = runs[(runs.cond == "D") & runs.run.isin(NEW)].acc9.values
    pos = {}
    for name, x in (("E3_run0_77.1", runs[(runs.cond == "D") & (runs.run == "0")].acc9.iloc[0]),
                    ("E3b_run0b_72.9", runs[(runs.cond == "D") & (runs.run == "0b")].acc9.iloc[0])):
        pos[name] = dict(value=float(x), n_new_runs_below=int((dvals < x).sum()), n_new_runs_equal=int((dvals == x).sum()),
                         n_new_runs_above=int((dvals > x).sum()), z_vs_new5=float((x - dvals.mean()) / dvals.std(ddof=1))
                         if dvals.std(ddof=1) > 0 else None)
    wc = wave_consistency()
    wc.to_csv(RES / "waveform_consistency_B_a.csv", index=False)
    vs0 = wc[~wc.run.astype(str).str.contains("-")]
    btw = wc[wc.run.astype(str).str.contains("-")]
    js = {"D_position": pos, "D_new5_acc9": dvals.tolist(),
          "waveform_B_vs_run0": dict(corr_min=vs0["corr"].min(), corr_max=vs0["corr"].max(), resid_db_min=vs0.resid_db.min(),
                                     resid_db_max=vs0.resid_db.max(), resid_db_median=vs0.resid_db.median(),
                                     n_len_diff_nonzero=int((vs0.len_diff_samples != 0).sum()), n=len(vs0)),
          "waveform_B_between_new_runs": dict(corr_min=btw["corr"].min(), corr_max=btw["corr"].max(),
                                              resid_db_min=btw.resid_db.min(), resid_db_max=btw.resid_db.max(),
                                              resid_db_median=btw.resid_db.median(),
                                              n_identical=int((btw.resid_db < -150).sum()), n=len(btw))}
    (RES / "summary_a.json").write_text(json.dumps(js, indent=1, ensure_ascii=False, default=float))

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    print(runs.round(4).to_string()); print(summ[summ.metric.isin(["acc9", "w2v4_acc", "spk_cos", "mos"])].round(4).drop(columns="values").to_string())
    print(pooled.round(4).to_string()); print(stab.to_string()); print(pairs.round(4).to_string())
    print(rec.to_string()); print(rec5.round(3).to_string()); print(json.dumps(js, indent=1, default=float))


def analyze_b():
    b = pd.read_csv(RES / "per_file_b.csv")
    b["run"] = b.run.astype(str)
    CONDS = ["IB", "DP_k050", "DP_k100"]
    # 중립 대비 F0·속도 변화(같은 rep의 InJoon 중립 기준)
    base = b[b.cond == "IB"].drop_duplicates(["run", "sid"]).set_index(["run", "sid"])
    b["d_f0_st"] = [12 * np.log2(r.f0_mean_hz / base.loc[(r.run, r.sid), "f0_mean_hz"]) for r in b.itertuples()]
    b["d_rate_pct"] = [100 * (r.rate_syl_per_s / base.loc[(r.run, r.sid), "rate_syl_per_s"] - 1) for r in b.itertuples()]
    rows = []
    for cond in CONDS:
        for run in ["1", "2", "pooled"]:
            s = b[b.cond == cond] if run == "pooled" else b[(b.cond == cond) & (b.run == run)]
            m = metrics(s, "spk_cos_ref")
            emo = s[s.emotion != "neutral"]
            m.update(spk_cos_own=emo.spk_cos_own.mean(), spk_cos_sunhi=emo.spk_cos_sunhi.mean())
            if run == "pooled":
                ci = boot_cluster(s, "correct9", ["sid"])
                cis = boot_cluster(emo, "spk_cos_ref", ["sid"])
                cim = boot_cluster(s, "squim_mos", ["sid"])
                m.update(acc9_lo=ci[0], acc9_hi=ci[1], spk_lo=cis[0], spk_hi=cis[1], mos_lo=cim[0], mos_hi=cim[1])
            rows.append(dict(speaker="InJoon", cond=cond, run=run, **m))
    # SunHi 비교값: E3b(run 0) DP k=0.5, 1.0 / E3c 실험 A DP k=1 run1..5
    e3b = pd.read_csv(E3 / "E3b_intensity" / "results" / "per_file.csv")
    for k in (0.0, 0.5, 1.0):
        s = e3b[(e3b.split == "test") & (e3b.variant == "DP") & (e3b.k == k)].copy()
        s["correct6"] = (s.e2v_pred6 == s.emotion).astype(int)
        m = metrics(s, "spk_cos_B")
        ci = boot_cluster(s, "correct9", ["sid"])
        rows.append(dict(speaker="SunHi(E3b)", cond=f"DP_k{int(k * 100):03d}" if k else "k0(B+PSOLA)", run="0",
                         acc9_lo=ci[0], acc9_hi=ci[1], **m))
    pa = RES / "per_file_a.csv"
    if pa.exists():
        a = pd.read_csv(pa)
        for cond, nm in (("DP", "DP_k100"), ("B", "B")):
            s = a[a.cond == cond]
            m = metrics(s, "spk_cos_B")
            ci = boot_cluster(s, "correct9", ["sid", "run"])
            rows.append(dict(speaker="SunHi(E3c-A, 5 runs)", cond=nm, run="pooled", acc9_lo=ci[0], acc9_hi=ci[1], **m))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "summary_b.csv", index=False)
    rec = b.groupby(["cond", "run", "emotion"]).correct9.sum().unstack()[C.EMOS]
    rec.to_csv(RES / "recall_b.csv")
    # SunHi vs InJoon 감정별 재현율 비교(DP k=1, k=0.5)
    s_rec = e3b[(e3b.split == "test") & (e3b.variant == "DP") & e3b.k.isin([0.5, 1.0])].groupby(["k", "emotion"]).correct9.sum().unstack()[C.EMOS]
    ac = b.groupby(["cond", "emotion"]).agg(d_f0_st=("d_f0_st", "mean"), d_rate_pct=("d_rate_pct", "mean"),
                                            f0_hz=("f0_mean_hz", "mean"), spk_ref=("spk_cos_ref", "mean"),
                                            mos=("squim_mos", "mean")).round(3)
    ac.to_csv(RES / "acoustic_by_emotion_b.csv")
    # rep 간 일치
    agree = {}
    for cond in CONDS[1:]:
        p1 = b[(b.cond == cond) & (b.run == "1")].set_index(["sid", "emotion"]).sort_index()
        p2 = b[(b.cond == cond) & (b.run == "2")].set_index(["sid", "emotion"]).sort_index()
        agree[cond] = dict(pred_agreement=float((p1.e2v_pred9 == p2.e2v_pred9).mean()),
                           correct_changed=int((p1.correct9 != p2.correct9).sum()))
    # InJoon 중립 F0
    f0 = base.f0_mean_hz
    sunhi_b = pd.read_csv(E3 / "results" / "per_file.csv")
    sunhi_b = sunhi_b[(sunhi_b.cond == "B") & (sunhi_b.split == "test")].drop_duplicates("sid")
    js = {"rep_agreement": agree, "InJoon_neutral_f0_mean_hz": float(f0.mean()), "InJoon_neutral_f0_sd_hz": float(f0.std()),
          "SunHi_neutral_f0_mean_hz_test": float(sunhi_b.f0_mean_hz.mean()),
          "SunHi_DP_recall_E3b": s_rec.to_dict(orient="index")}
    (RES / "summary_b.json").write_text(json.dumps(js, indent=1, ensure_ascii=False, default=float))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.round(4).to_string()); print(rec.to_string()); print(s_rec.to_string()); print(ac.to_string())
    print(json.dumps(js, indent=1, default=float))


if __name__ == "__main__":
    analyze_a() if sys.argv[1] == "a" else analyze_b()
