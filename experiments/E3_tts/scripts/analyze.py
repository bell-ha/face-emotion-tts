"""E3 분석: 음향 특징, SER(emotion2vec + 보조 wav2vec2-ER), 화자 일관성, SQUIM, 통계 검정."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

RNG = np.random.default_rng(C.SEED)
R = C.RESULTS
CONDS = ["A", "B", "C", "D"]
REFS = ["E_lex_B", "E_lex_D"]


def manifest() -> pd.DataFrame:
    rows = []
    for cond in CONDS:
        for sid, text in C.SENTENCES.items():
            if cond == "A" and sid not in ("S01", "S02"):
                continue
            split = "dev" if sid in C.DEV else "test"
            for emo in C.EMOS:
                f = C.AUDIO / cond / (f"{sid}_neutral.wav" if cond == "B" else f"{sid}_{emo}.wav")
                rows.append(dict(cond=cond, sid=sid, split=split, emotion=emo, text=text, path=str(f)))
    for cond in REFS:
        for emo, sents in C.LEXICAL.items():
            for i, text in enumerate(sents):
                rows.append(dict(cond=cond, sid=f"L{i+1}{emo[:2]}", split="ref", emotion=emo, text=text,
                                 path=str(C.AUDIO / cond / f"L{i+1}_{emo}.wav")))
    return pd.DataFrame(rows)


def per_file(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df.drop_duplicates("path")[["path", "text"]]
    out = []
    nmr_default = C.AUDIO / "B" / "S05_neutral.wav"
    nmr_alt = C.AUDIO / "B" / "S06_neutral.wav"
    for i, (path, text) in enumerate(uniq.itertuples(index=False)):
        r = {"path": path}
        r.update(C.acoustic_features(path, text))
        s = C.ser_emotion2vec(path)
        r["e2v_pred9"], r["e2v_pred6"] = s["pred9"], s["pred6"]
        for k, v in s["raw"].items():
            r[f"e2v_raw_{k}"] = v
        w = C.ser_w2v(path)
        r["w2v_pred"] = w["pred"]
        for k, v in w["probs"].items():
            r[f"w2v_{k}"] = v
        nmr = nmr_alt if "S05" in Path(path).name else nmr_default
        try:
            r.update(C.squim(path, nmr))
        except Exception as e:  # noqa: BLE001
            r["squim_err"] = str(e)
        r["_emb"] = C.spk_embed(path).tolist()
        out.append(r)
        if i % 50 == 0:
            print(f"  file {i}/{len(uniq)}", flush=True)
    return pd.DataFrame(out)


def cluster_boot_acc(sub: pd.DataFrame, col: str, n=10000):
    """문장 단위 클러스터 부트스트랩 95% CI."""
    sids = sub.sid.unique()
    g = {s: sub[sub.sid == s][col].values for s in sids}
    accs = []
    for _ in range(n):
        pick = RNG.choice(sids, size=len(sids), replace=True)
        v = np.concatenate([g[s] for s in pick])
        accs.append(v.mean())
    return np.percentile(accs, [2.5, 97.5])


def cluster_boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, n=10000):
    m = a.merge(b, on=["sid", "emotion"], suffixes=("_a", "_b"))
    sids = m.sid.unique()
    g = {s: (m[m.sid == s][f"{col}_a"].values.astype(float) - m[m.sid == s][f"{col}_b"].values.astype(float))
         for s in sids}
    d = []
    for _ in range(n):
        pick = RNG.choice(sids, size=len(sids), replace=True)
        d.append(np.concatenate([g[s] for s in pick]).mean())
    return float(np.mean(np.concatenate(list(g.values())))), np.percentile(d, [2.5, 97.5])


def mcnemar(a, b):
    """정확(이항) McNemar. a,b: 0/1 배열(짝지음)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n01, n, 0.5).pvalue)
    return n01, n10, p


def main():
    df = manifest()
    cache = R / "per_file_raw.pkl"
    if cache.exists():
        pf = pd.read_pickle(cache)
    else:
        print(f"analyzing {df.path.nunique()} unique files", flush=True)
        pf = per_file(df)
        pf.to_pickle(cache)
    d = df.merge(pf, on="path", how="left")
    d["correct9"] = (d.e2v_pred9 == d.emotion).astype(int)
    d["correct6"] = (d.e2v_pred6 == d.emotion).astype(int)
    d["w2v_correct"] = (d.w2v_pred == d.emotion).astype(int)

    # ── 화자 일관성: 같은 조건·같은 문장의 중립 음성, 그리고 B 중립(원 화자)과의 코사인
    emb = {p: np.array(e) for p, e in zip(pf.path, pf._emb)}
    def cos(p, q):
        return float(np.dot(emb[p], emb[q])) if (p in emb and q in emb) else np.nan
    own, vsB = [], []
    for r in d.itertuples():
        if r.split == "ref":
            own.append(np.nan); vsB.append(np.nan); continue
        neu_own = d[(d.cond == r.cond) & (d.sid == r.sid) & (d.emotion == "neutral")].path.iloc[0]
        neu_B = str(C.AUDIO / "B" / f"{r.sid}_neutral.wav")
        own.append(cos(r.path, neu_own)); vsB.append(cos(r.path, neu_B))
    d["spk_cos_own_neutral"] = own
    d["spk_cos_B_neutral"] = vsB
    # 척도 참고: 같은 화자·다른 문장(B 중립끼리), 다른 화자(InJoon, OpenAI shimmer)
    ref_rows = []
    bpaths = [str(C.AUDIO / "B" / f"{s}_neutral.wav") for s in C.SENTENCES]
    for p in bpaths:
        emb[p] = C.spk_embed(p) if p not in emb else emb[p]
    same_spk_diff_sent = [float(np.dot(emb[p], emb[q])) for i, p in enumerate(bpaths) for q in bpaths[i + 1:]]
    inj = []
    for s in C.SENTENCES:
        q = str(C.AUDIO / "spk_ref" / f"{s}_InJoon_neutral.wav")
        inj.append(float(np.dot(emb[str(C.AUDIO / 'B' / f'{s}_neutral.wav')], C.spk_embed(q))))
    shim = [float(np.dot(emb[str(C.AUDIO / 'B' / f'{s}_neutral.wav')], C.spk_embed(C.AUDIO / 'A' / f'{s}_neutral.wav')))
            for s in ("S01", "S02")]
    spk_scale = {
        "SunHi_vs_SunHi_diff_sentence": [float(np.mean(same_spk_diff_sent)), float(np.std(same_spk_diff_sent)), len(same_spk_diff_sent)],
        "SunHi_vs_InJoon(male)_same_sentence": [float(np.mean(inj)), float(np.std(inj)), len(inj)],
        "SunHi_vs_OpenAI_shimmer_same_sentence": [float(np.mean(shim)), float(np.std(shim)), len(shim)],
    }

    # ── 중립 대비 변화량(같은 조건·같은 문장)
    for col, how in (("f0_mean_hz", "st"), ("rate_syl_per_s", "pct"), ("int_mean_db", "diff"),
                     ("f0_sd_st", "diff"), ("hf_ratio_db", "diff")):
        base = d[d.emotion == "neutral"].set_index(["cond", "sid"])[col]
        b = d.apply(lambda r: base.get((r.cond, r.sid), np.nan), axis=1)
        if how == "st":
            d["d_f0_st"] = 12 * np.log2(d[col] / b)
        elif how == "pct":
            d["d_rate_pct"] = 100 * (d[col] / b - 1)
        else:
            d[f"d_{col}"] = d[col] - b

    d.drop(columns=["_emb"]).to_csv(R / "per_file.csv", index=False)

    # ── 조건별 SER 요약
    summ, conf_json = [], {}
    labels7 = C.EMOS + ["nomatch"]
    for split in ("test", "dev", "all"):
        for cond in CONDS + REFS:
            if cond in REFS and split != "all":
                continue
            sub = d[d.cond == cond] if split == "all" else d[(d.cond == cond) & (d.split == split)]
            if sub.empty:
                continue
            ci9 = cluster_boot_acc(sub, "correct9")
            ci6 = cluster_boot_acc(sub, "correct6")
            w = sub[sub.emotion.isin(["happy", "sad", "angry", "neutral"])]
            summ.append(dict(
                split=split, cond=cond, n=len(sub), n_sent=sub.sid.nunique(),
                acc9=sub.correct9.mean(), acc9_lo=ci9[0], acc9_hi=ci9[1],
                macroF1_9=f1_score(sub.emotion, sub.e2v_pred9, labels=C.EMOS, average="macro", zero_division=0),
                acc6=sub.correct6.mean(), acc6_lo=ci6[0], acc6_hi=ci6[1],
                macroF1_6=f1_score(sub.emotion, sub.e2v_pred6, labels=C.EMOS, average="macro", zero_division=0),
                n_pred_distinct=sub.e2v_pred9.nunique(),
                w2v4_acc=w.w2v_correct.mean(), w2v4_n=len(w),
                w2v4_macroF1=f1_score(w.emotion, w.w2v_pred, labels=["happy", "sad", "angry", "neutral"],
                                      average="macro", zero_division=0),
            ))
            conf_json[f"{split}/{cond}/e2v9"] = confusion_matrix(sub.emotion, sub.e2v_pred9, labels=labels7)[:6].tolist()
            conf_json[f"{split}/{cond}/e2v6"] = confusion_matrix(sub.emotion, sub.e2v_pred6, labels=C.EMOS).tolist()
            conf_json[f"{split}/{cond}/w2v4"] = confusion_matrix(
                w.emotion, w.w2v_pred, labels=["happy", "sad", "angry", "neutral"]).tolist()
    summ = pd.DataFrame(summ)
    summ.to_csv(R / "ser_summary.csv", index=False)
    (R / "confusion_matrices.json").write_text(json.dumps(
        {"row_labels": C.EMOS, "col_labels_e2v9": labels7, "col_labels_e2v6": C.EMOS,
         "labels_w2v4": ["happy", "sad", "angry", "neutral"], "matrices": conf_json}, indent=1))

    # 감정별 재현율
    rec = d[d.split.isin(["test", "ref"])].groupby(["cond", "emotion"]).agg(
        recall9=("correct9", "mean"), recall6=("correct6", "mean"), n=("correct9", "size")).reset_index()
    rec.to_csv(R / "ser_recall_by_emotion.csv", index=False)

    # ── 짝지은 비교 (held-out test)
    tests = []
    for split, pairs in (("test", [("D", "C"), ("C", "B"), ("D", "B")]),
                         ("test_S01S02", [("A", "B"), ("C", "A"), ("D", "A")]),
                         ("dev", [("D", "C"), ("C", "B"), ("D", "B")])):
        for a, b in pairs:
            if split == "test_S01S02":
                da = d[(d.cond == a) & d.sid.isin(["S01", "S02"])]
                db = d[(d.cond == b) & d.sid.isin(["S01", "S02"])]
            else:
                da = d[(d.cond == a) & (d.split == split)]
                db = d[(d.cond == b) & (d.split == split)]
            for col in ("correct9", "correct6"):
                m = da.merge(db, on=["sid", "emotion"], suffixes=("_a", "_b"))
                n01, n10, p = mcnemar(m[f"{col}_a"], m[f"{col}_b"])
                diff, ci = cluster_boot_diff(da, db, col)
                tests.append(dict(split=split, a=a, b=b, metric=col, n_pairs=len(m),
                                  acc_a=m[f"{col}_a"].mean(), acc_b=m[f"{col}_b"].mean(),
                                  diff=diff, diff_lo=ci[0], diff_hi=ci[1],
                                  a_only=n01, b_only=n10, mcnemar_p=p))
    # 우연 수준(1/6) 대비 이항검정
    for r in summ[summ.split == "test"].itertuples():
        k = int(round(r.acc9 * r.n))
        tests.append(dict(split="test", a=r.cond, b="chance(1/6)", metric="correct9", n_pairs=r.n,
                          acc_a=r.acc9, acc_b=1 / 6,
                          mcnemar_p=float(stats.binomtest(k, r.n, 1 / 6, alternative="greater").pvalue)))
    pd.DataFrame(tests).to_csv(R / "stat_tests.csv", index=False)

    # ── 음향 특징 요약 (조건 × 감정, test 문장)
    feats = ["f0_mean_hz", "f0_sd_st", "f0_range_st", "int_mean_db", "rate_syl_per_s", "dur_s",
             "hf_ratio_db", "d_f0_st", "d_rate_pct", "d_int_mean_db", "d_f0_sd_st", "d_hf_ratio_db"]
    t = d[d.split == "test"]
    t = t[~((t.cond == "B") & (t.emotion != "neutral"))]   # B는 감정 무관 동일 음성 → 중립 1회만
    ac = t.groupby(["cond", "emotion"])[feats].agg(["mean", "std"])
    ac.columns = [f"{a}_{b}" for a, b in ac.columns]
    ac.reset_index().to_csv(R / "acoustic_summary.csv", index=False)
    # Kruskal-Wallis: 감정 간 차이 (조건별, test 문장)
    kw = []
    for cond in ["A", "C", "D"]:
        sub = d[(d.cond == cond) & (d.split == "test")]
        for f in ["f0_mean_hz", "f0_sd_st", "int_mean_db", "rate_syl_per_s", "hf_ratio_db"]:
            groups = [sub[sub.emotion == e][f].dropna().values for e in C.EMOS]
            H, p = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            eps2 = (H - len(groups) + 1) / (n - len(groups))   # epsilon-squared 효과크기
            kw.append(dict(cond=cond, feature=f, H=H, p=p, n=n, epsilon2=eps2))
    pd.DataFrame(kw).to_csv(R / "kruskal_wallis.csv", index=False)

    # ── 화자 일관성 & SQUIM 요약
    sp = d[(d.split == "test") & (d.emotion != "neutral") & (d.cond != "B")]
    spk = sp.groupby("cond")[["spk_cos_own_neutral", "spk_cos_B_neutral"]].agg(["mean", "std", "min", "count"])
    spk.columns = [f"{a}_{b}" for a, b in spk.columns]
    spk_e = sp.groupby(["cond", "emotion"])["spk_cos_B_neutral"].agg(["mean", "std"]).reset_index()
    spk.reset_index().to_csv(R / "speaker_similarity.csv", index=False)
    spk_e.to_csv(R / "speaker_similarity_by_emotion.csv", index=False)
    (R / "speaker_scale_reference.json").write_text(json.dumps(spk_scale, indent=1))
    sq_cols = [c for c in d.columns if c.startswith("squim_") and c != "squim_err"]
    sq = d[d.split.isin(["test", "ref"])].drop_duplicates("path").groupby("cond")[sq_cols].agg(["mean", "std", "count"])
    sq.columns = [f"{a}_{b}" for a, b in sq.columns]
    sq.reset_index().to_csv(R / "squim_summary.csv", index=False)
    sqe = d[d.split == "test"].drop_duplicates("path").groupby(["cond", "emotion"])["squim_mos"].agg(["mean", "std"]).reset_index()
    sqe.to_csv(R / "squim_by_emotion.csv", index=False)
    # Wilcoxon: D vs C SQUIM-MOS (짝지음, test)
    m = d[(d.cond == "D") & (d.split == "test")].merge(d[(d.cond == "C") & (d.split == "test")],
                                                       on=["sid", "emotion"], suffixes=("_D", "_C"))
    w = stats.wilcoxon(m.squim_mos_D, m.squim_mos_C)
    m2 = d[(d.cond == "D") & (d.split == "test") & (d.emotion != "neutral")].merge(
        d[(d.cond == "C") & (d.split == "test") & (d.emotion != "neutral")], on=["sid", "emotion"], suffixes=("_D", "_C"))
    w2 = stats.wilcoxon(m2.spk_cos_B_neutral_D, m2.spk_cos_B_neutral_C)
    extra = {"wilcoxon_squim_mos_D_vs_C": {"stat": float(w.statistic), "p": float(w.pvalue), "n": len(m),
                                           "median_diff": float(np.median(m.squim_mos_D - m.squim_mos_C))},
             "wilcoxon_spkcosB_D_vs_C": {"stat": float(w2.statistic), "p": float(w2.pvalue), "n": len(m2),
                                         "median_diff": float(np.median(m2.spk_cos_B_neutral_D - m2.spk_cos_B_neutral_C))}}
    (R / "extra_tests.json").write_text(json.dumps(extra, indent=1))

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(summ.round(3).to_string())
    print(pd.DataFrame(tests).round(4).to_string())
    print(spk.round(3).to_string()); print(spk_scale)
    print(sq.round(3).to_string()); print(extra)
    print(pd.DataFrame(kw).round(4).to_string())


if __name__ == "__main__":
    main()
