"""E5 Part A (+ Part B 2x2): baseline (thesis code) vs improved stabilization / sentence tagging.

Protocol
  dev actors  : 01-04  (ALL hyperparameters tuned here only)
  test actors : 05-24  (reported)
  streams     : per actor, the 52 non-calm clips concatenated in a random order (seed 42 + actor),
                one stabilizer instance per stream (state carried across clips), 29.97 fps video time.
  sentence    : one clip = one sentence window [first frame, last frame] (as in E2).
  "" tag      : scored as Neutral (default emotion), as in E2.
Classifiers  : cnn (thesis FER2013 CNN), afew / mtl (HSEmotion ONNX, AffectNet; Contempt dropped,
               softmax over the remaining 7 logits).
Usage: e5_analyze.py [cnn|afew|mtl ...]   (default: cnn and the HSEmotion model chosen on dev)
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e5_methods import (EMOTION_LABELS, KO6, LABEL_TO_KO, EmotionServer, ImprovedStabilizer,
                        ProbSummaryServer, Stabilizer, SummaryClient)

SEED = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRB = os.path.join(ROOT, "data", "ravdess", "probs_e5")
RES = os.path.join(ROOT, "experiments", "E5_improved", "results")
DEV = [1, 2, 3, 4]
SKIP_S, SETTLE_S = 0.5, 0.5
RAV_EMO = {1: "Neutral", 3: "Happy", 4: "Sad", 5: "Angry", 6: "Fear", 7: "Disgust", 8: "Surprise"}
HSE_TO_OURS = [0, 2, 3, 4, 6, 7, 5]  # HSE logits (Anger,Contempt,Disgust,Fear,Happiness,Neutral,Sadness,Surprise) -> EMOTION_LABELS order
CT_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
MR_GRID = [0.2, 0.3, 0.4, 0.5, 0.6]
WN_GRID = [round(x, 2) for x in np.arange(0.10, 1.001, 0.05)]
RMIN_GRID = [0.0, 0.3, 0.4, 0.5]
LAT_BUDGET_S = 0.30  # dev constraint for stabilizer selection: median first-hit latency <= 0.30 s
N_BOOT = 2000


def idx_to_ko(i):
    if i == -2:
        return "없음"
    if i == -1:
        return "중립"
    return LABEL_TO_KO[EMOTION_LABELS[i]]


IDX_KO = np.array([idx_to_ko(i) for i in range(7)] + ["없음", "중립"])  # index -2 -> 7 ("없음"), -1 -> 8


def ko_of(L):
    L = np.asarray(L)
    return IDX_KO[np.where(L >= 0, L, np.where(L == -2, 7, 8))]


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def load_clips(clf):
    clips = []
    for f in sorted(glob.glob(os.path.join(PRB, "*.npz"))):
        b = os.path.basename(f)[:-4]
        p = [int(x) for x in b.split("-")]
        if p[2] == 2:
            continue
        d = np.load(f)
        face = d["face"]
        if clf == "cnn":
            P = d["cnn"]
        else:
            L = d[clf][:, :8][:, HSE_TO_OURS]
            P = np.full(L.shape, np.nan, np.float32)
            P[face] = softmax(L[face]).astype(np.float32)
        emo = RAV_EMO[p[2]]
        clips.append({"clip": b, "actor": p[6], "emotion": emo, "emo_idx": EMOTION_LABELS.index(emo),
                      "ko": LABEL_TO_KO[emo], "intensity": p[3], "t": d["t"], "probs": P, "face": face})
    return clips


def streams(clips):
    out = {}
    for a in sorted(set(c["actor"] for c in clips)):
        cs = [c for c in clips if c["actor"] == a]
        order = np.random.default_rng(SEED + a).permutation(len(cs))
        out[a] = [cs[i] for i in order]
    return out


def make_stab(cfg):
    if cfg["rule"] == "code":
        return Stabilizer()
    return ImprovedStabilizer(rule=cfg["rule"], delta=cfg.get("delta", 0.0))


def run_sequence(seq, cfg, with_tags=True):
    st = make_stab(cfg)
    srv, cli, psrv = EmotionServer(), SummaryClient(0.2), ProbSummaryServer()
    n_ev, n_sum = 0, 0
    t_off, out = 0.0, []
    for c in seq:
        t, P, F = c["t"], c["probs"], c["face"]
        n = len(t)
        emitted = np.empty(n, int)
        e0 = len(st.events)
        for k in range(n):
            tt = t_off + t[k]
            pr = P[k] if F[k] else None
            r = st.step(tt, pr)
            emitted[k] = r["emitted"]
            if with_tags:
                cli.step(tt, pr)
                while n_ev < len(st.events):
                    srv.set_emotion(*st.events[n_ev]); n_ev += 1
                while n_sum < len(cli.out):
                    psrv.add(cli.out[n_sum]); n_sum += 1
        t0, t1 = int((t_off + t[0]) * 1000), int((t_off + t[-1]) * 1000)
        rec = {"clip": c, "t": t, "emitted": emitted, "n_events": len(st.events) - e0, "t0": t0, "t1": t1}
        if with_tags:
            rec["durs"] = {ct: srv._segment_durations_ko(t0, t1, ct) for ct in CT_GRID}
            rec["mass6"] = psrv.mass6(t0, t1)
        out.append(rec)
        t_off += t[-1] + ((t[1] - t[0]) if n > 1 else 1 / 30)
    return out


def tag_baseline(durs, t0, t1, ct, mr):
    d = durs[ct]
    if not d:
        return ""
    emo, best = max(d.items(), key=lambda kv: kv[1])
    if emo == "중립" or best / max(1, t1 - t0) < mr:
        return ""
    return f"<{emo}>"


def tag_mass(m, wn, rmin):
    if m.sum() <= 0:
        return ""
    m = m.copy(); m[5] *= wn
    k = int(np.argmax(m))
    if k == 5 or m[k] / m.sum() < rmin:
        return ""
    return f"<{KO6[k]}>"


def clip_rows(recs):
    rows = []
    for r in recs:
        c, t, L = r["clip"], r["t"], r["emitted"]
        dur = t[-1] - t[0] + (t[1] - t[0])
        m = t >= SKIP_S
        ko = ko_of(L)
        vk, ck = np.unique(ko[m], return_counts=True)
        rows.append({"clip": c["clip"], "actor": c["actor"], "emotion": c["emotion"], "ko": c["ko"],
                     "intensity": c["intensity"], "dur_s": dur, "n_events": r["n_events"],
                     "switch_per_s": int((L[1:] != L[:-1]).sum()) / dur,
                     "frame_acc_ko": float((ko[m] == c["ko"]).mean()),
                     "clip_acc_ko": bool(vk[np.argmax(ck)] == c["ko"]),
                     "_rec": r})
    return rows


def evaluate_stab(clips, cfg, with_tags=True):
    rows = []
    for a, seq in streams(clips).items():
        rows += clip_rows(run_sequence(seq, cfg, with_tags))
    return pd.DataFrame(rows)


def tag_eval(df, kind, p1, p2):
    if kind == "baseline":
        tags = [tag_baseline(r["durs"], r["t0"], r["t1"], p1, p2) for r in df._rec]
    else:
        tags = [tag_mass(r["mass6"], p1, p2) for r in df._rec]
    tko = np.array([s.strip("<>") if s else "중립" for s in tags])
    corr = tko == df.ko.values
    return np.array(tags), tko, corr


def tag_metrics(df, tags, tko, corr):
    empty = tags == ""
    rec = [corr[df.ko.values == k].mean() for k in KO6 if (df.ko.values == k).any()]
    return {"tag_acc": float(corr.mean()), "tag_bal_acc": float(np.mean(rec)), "empty_rate": float(empty.mean()),
            "precision_nonempty": float(corr[~empty].mean()) if (~empty).any() else float("nan"),
            "n_nonempty": int((~empty).sum()),
            "tag_acc_emotional": float(corr[df.ko.values != "중립"].mean())}


# ------------------------------------------------------------- transitions (E2 protocol)
def trim(c):
    m = c["t"] >= SKIP_S
    d = dict(c); d["t"] = c["t"][m] - c["t"][m][0]; d["probs"] = c["probs"][m]; d["face"] = c["face"][m]
    return d


def transition_pairs(clips, n_per_actor=150):
    rng = np.random.default_rng(SEED)
    raw_ok = {}
    for c in clips:
        m = c["t"] >= SKIP_S
        lab = np.where(c["face"][m], np.nan_to_num(c["probs"][m], nan=0).argmax(1), 6)
        raw_ok[c["clip"]] = (ko_of(lab) == c["ko"]).mean() > 0.5
    by = {}
    for c in clips:
        by.setdefault(c["actor"], []).append(c)
    pairs = []
    for a, cs in sorted(by.items()):
        cand = [(i, j) for i in range(len(cs)) for j in range(len(cs))
                if i != j and cs[i]["ko"] != cs[j]["ko"] and raw_ok[cs[j]["clip"]]]
        if not cand:
            continue
        sel = rng.choice(len(cand), size=min(n_per_actor, len(cand)), replace=False)
        pairs += [(a, trim(cs[cand[s][0]]), trim(cs[cand[s][1]])) for s in sel]
    return pairs


def transition_latency(pairs, cfg):
    rows = []
    for a, A, B in pairs:
        res = run_sequence([A, B], cfg, with_tags=False)
        rb = res[1]; tb = rb["t"] - rb["t"][0]
        ko = ko_of(rb["emitted"]); prevA = ko_of(res[0]["emitted"][-1:])[0]
        hit = np.where(ko == B["ko"])[0]
        eq = ko == B["ko"]
        mlen = max(1, int(round(SETTLE_S / np.median(np.diff(tb)))))
        st = np.nan
        for k in hit:
            if k + mlen <= len(eq) and eq[k:k + mlen].all():
                st = float(tb[k]); break
        rows.append({"actor": a, "A": A["clip"], "B": B["clip"], "already": prevA == B["ko"],
                     "lat_s": float(tb[hit[0]]) if len(hit) else np.nan, "settle_s": st})
    return pd.DataFrame(rows)


def lat_summary(tl):
    x = tl.loc[~tl.already, "lat_s"]; y = tl.loc[~tl.already, "settle_s"]
    return {"n_valid": int(len(x)), "reached_rate": float(x.notna().mean()), "median_s": float(np.nanmedian(x)),
            "q1_s": float(np.nanpercentile(x, 25)), "q3_s": float(np.nanpercentile(x, 75)),
            "mean_s": float(np.nanmean(x)), "settle_median_s": float(np.nanmedian(y)),
            "settle_q1_s": float(np.nanpercentile(y, 25)), "settle_q3_s": float(np.nanpercentile(y, 75)),
            "settle_reached_rate": float(y.notna().mean())}


# ------------------------------------------------------------- statistics
def cluster_boot(values_by_actor_fn, actors, n=N_BOOT, seed=SEED):
    """values_by_actor_fn(list_of_actors) -> scalar. Resample actors with replacement."""
    rng = np.random.default_rng(seed)
    acts = np.array(actors)
    return np.array([values_by_actor_fn(list(rng.choice(acts, len(acts), replace=True))) for _ in range(n)])


def ci_mean(x_by_clip, actor_of_clip, n=N_BOOT):
    x = np.asarray(x_by_clip, float); a = np.asarray(actor_of_clip)
    acts = sorted(set(a)); idx = {k: np.where(a == k)[0] for k in acts}
    bs = cluster_boot(lambda s: x[np.concatenate([idx[k] for k in s])].mean(), acts, n)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mcnemar(b_corr, i_corr):
    b01 = int((~b_corr & i_corr).sum()); b10 = int((b_corr & ~i_corr).sum())
    p = stats.binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
    return {"improved_only": b01, "baseline_only": b10, "p_exact": float(p)}


# ------------------------------------------------------------- main per classifier
def analyze(clf, log):
    t_start = time.time()
    clips = load_clips(clf)
    dev = [c for c in clips if c["actor"] in DEV]
    test = [c for c in clips if c["actor"] not in DEV]
    log(f"[{clf}] clips {len(clips)} dev {len(dev)} test {len(test)} "
        f"test actors {sorted(set(c['actor'] for c in test))}")
    out = {"classifier": clf, "n_dev_clips": len(dev), "n_test_clips": len(test),
           "test_actors": sorted(set(int(c["actor"]) for c in test))}

    # ---------------- DEV: tagging hyperparameters
    base_dev = evaluate_stab(dev, {"rule": "code"})
    grid_b = []
    for ct in CT_GRID:
        for mr in MR_GRID:
            grid_b.append({"conf_thresh": ct, "min_ratio": mr, **tag_metrics(base_dev, *tag_eval(base_dev, "baseline", ct, mr))})
    grid_b = pd.DataFrame(grid_b)
    grid_m = []
    for wn in WN_GRID:
        for rm in RMIN_GRID:
            grid_m.append({"w_neutral": wn, "r_min": rm, **tag_metrics(base_dev, *tag_eval(base_dev, "mass", wn, rm))})
    grid_m = pd.DataFrame(grid_m)
    grid_b.to_csv(os.path.join(RES, f"dev_grid_tag_baseline_{clf}.csv"), index=False)
    grid_m.to_csv(os.path.join(RES, f"dev_grid_tag_mass_{clf}.csv"), index=False)
    # selection: max balanced accuracy on dev; ties -> closest to thesis default / simplest (w=1, r=0)
    gb = grid_b.assign(_d=(grid_b.conf_thresh - 0.5).abs() + (grid_b.min_ratio - 0.4).abs())
    gb = gb.sort_values(["tag_bal_acc", "_d"], ascending=[False, True]).iloc[0]
    gm = grid_m.assign(_d=(1 - grid_m.w_neutral) + grid_m.r_min)
    gm = gm.sort_values(["tag_bal_acc", "_d"], ascending=[False, True]).iloc[0]
    sel_tag = {"baseline_tuned": {"conf_thresh": float(gb.conf_thresh), "min_ratio": float(gb.min_ratio),
                                  "dev_bal_acc": float(gb.tag_bal_acc)},
               "mass_tuned": {"w_neutral": float(gm.w_neutral), "r_min": float(gm.r_min), "dev_bal_acc": float(gm.tag_bal_acc)}}
    out["selected_tag_params"] = sel_tag
    log(f"[{clf}] dev tag selection {sel_tag}")

    # ---------------- DEV: stabilizer rule
    dev_pairs = transition_pairs(dev)
    cands = [{"rule": "code"}, {"rule": "gate"}] + \
            [{"rule": r, "delta": d} for r in ["hyst", "gate_hyst"] for d in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
    srows = []
    for cfg in cands:
        d = base_dev if cfg["rule"] == "code" else evaluate_stab(dev, cfg, with_tags=False)
        ls = lat_summary(transition_latency(dev_pairs, cfg))
        srows.append({"rule": cfg["rule"], "delta": cfg.get("delta", np.nan),
                      "switch_per_s": d.switch_per_s.mean(), "clip_acc_ko": d.clip_acc_ko.mean(),
                      "frame_acc_ko": d.frame_acc_ko.mean(), "events_per_clip": d.n_events.mean(),
                      "lat_median_s": ls["median_s"], "lat_mean_s": ls["mean_s"], "settle_median_s": ls["settle_median_s"]})
    srows = pd.DataFrame(srows)
    srows.to_csv(os.path.join(RES, f"dev_grid_stabilizer_{clf}.csv"), index=False)
    base_clip = float(srows.loc[srows.rule == "code", "clip_acc_ko"].iloc[0])
    ok = srows[(srows.rule != "code") & (srows.lat_median_s <= LAT_BUDGET_S + 1e-9) &
               (srows.clip_acc_ko >= base_clip - 0.02)]
    best = ok.sort_values(["switch_per_s", "lat_median_s"]).iloc[0]
    stab_cfg = {"rule": best.rule} | ({} if np.isnan(best.delta) else {"delta": float(best.delta)})
    out["selected_stabilizer"] = stab_cfg
    log(f"[{clf}] dev stabilizer selection {stab_cfg}\n{srows.round(3).to_string()}")

    # ---------------- TEST
    base = evaluate_stab(test, {"rule": "code"})
    imp = evaluate_stab(test, stab_cfg, with_tags=False)
    gate = evaluate_stab(test, {"rule": "gate"}, with_tags=False) if stab_cfg["rule"] != "gate" else imp
    tags = {
        "B0_thesis(0.5,0.4)": tag_eval(base, "baseline", 0.5, 0.4),
        "B1_thesis_tuned": tag_eval(base, "baseline", sel_tag["baseline_tuned"]["conf_thresh"],
                                    sel_tag["baseline_tuned"]["min_ratio"]),
        "T1_mass(w=1)": tag_eval(base, "mass", 1.0, 0.0),
        "T2_mass_tuned": tag_eval(base, "mass", sel_tag["mass_tuned"]["w_neutral"], sel_tag["mass_tuned"]["r_min"]),
    }
    test_pairs = transition_pairs(test)
    tl_b = transition_latency(test_pairs, {"rule": "code"})
    tl_i = transition_latency(test_pairs, stab_cfg)
    tl_g = transition_latency(test_pairs, {"rule": "gate"}) if stab_cfg["rule"] != "gate" else tl_i
    tl_b.assign(variant="baseline").to_csv(os.path.join(RES, f"test_transition_baseline_{clf}.csv"), index=False)
    tl_i.assign(variant="improved").to_csv(os.path.join(RES, f"test_transition_improved_{clf}.csv"), index=False)

    act = base.actor.values
    res = {"stabilizer": {}, "tagging": {}, "latency": {}}
    for name, d in [("baseline", base), ("improved", imp), ("gate_only", gate)]:
        res["stabilizer"][name] = {k: ci_mean(d[k].astype(float).values, d.actor.values)
                                   for k in ["switch_per_s", "clip_acc_ko", "frame_acc_ko", "n_events"]}
    for name, tl in [("baseline", tl_b), ("improved", tl_i), ("gate_only", tl_g)]:
        res["latency"][name] = lat_summary(tl)
    # paired: flicker & clip accuracy (same clips, same order)
    assert (base["clip"].values == imp["clip"].values).all()
    dsw = imp.switch_per_s.values - base.switch_per_s.values
    res["stabilizer"]["diff_switch_improved_minus_baseline"] = ci_mean(dsw, act)
    res["stabilizer"]["wilcoxon_switch_p"] = float(stats.wilcoxon(imp.switch_per_s, base.switch_per_s).pvalue)
    res["stabilizer"]["diff_clip_acc"] = ci_mean(imp.clip_acc_ko.astype(float).values - base.clip_acc_ko.astype(float).values, act)
    res["stabilizer"]["mcnemar_clip_acc"] = mcnemar(base.clip_acc_ko.values, imp.clip_acc_ko.values)
    both = (~tl_b.already) & (~tl_i.already)
    dl = (tl_i.lat_s - tl_b.lat_s)[both].dropna()
    res["latency"]["excess_improved_minus_baseline_median_s"] = float(dl.median())
    res["latency"]["excess_iqr_s"] = [float(dl.quantile(.25)), float(dl.quantile(.75))]
    res["latency"]["wilcoxon_p"] = float(stats.wilcoxon(dl).pvalue) if (dl != 0).any() else 1.0

    per_clip = base[["clip", "actor", "emotion", "ko", "intensity", "dur_s"]].copy()
    per_clip["switch_base"] = base.switch_per_s.values; per_clip["switch_imp"] = imp.switch_per_s.values
    per_clip["clipacc_base"] = base.clip_acc_ko.values; per_clip["clipacc_imp"] = imp.clip_acc_ko.values
    per_clip["events_base"] = base.n_events.values; per_clip["events_imp"] = imp.n_events.values
    for name, (tg, tko, corr) in tags.items():
        m = tag_metrics(base, tg, tko, corr)
        res["tagging"][name] = m | {
            "tag_acc_ci": ci_mean(corr.astype(float), act),
            "tag_bal_acc_ci": _bal_ci(corr, base.ko.values, act),
            "empty_rate_ci": ci_mean((tg == "").astype(float), act),
            "precision_ci": ci_mean(corr[tg != ""].astype(float), act[tg != ""]) if (tg != "").any() else None,
            "per_class_recall": {k: float(corr[base.ko.values == k].mean()) for k in KO6},
        }
        per_clip[f"tag_{name}"] = tko; per_clip[f"tagempty_{name}"] = tg == ""; per_clip[f"tagcorr_{name}"] = corr
    b0 = tags["B0_thesis(0.5,0.4)"][2]
    for name in ["B1_thesis_tuned", "T1_mass(w=1)", "T2_mass_tuned"]:
        ci = tags[name][2]
        res["tagging"][name]["vs_B0_mcnemar"] = mcnemar(b0, ci)
        res["tagging"][name]["vs_B0_diff_acc_ci"] = ci_mean(ci.astype(float) - b0.astype(float), act)
        res["tagging"][name]["vs_B0_diff_balacc_ci"] = _bal_diff_ci(b0, ci, base.ko.values, act)
    t1 = tags["B1_thesis_tuned"][2]; t2 = tags["T2_mass_tuned"][2]
    res["tagging"]["T2_mass_tuned"]["vs_B1_mcnemar"] = mcnemar(t1, t2)
    res["tagging"]["T2_mass_tuned"]["vs_B1_diff_acc_ci"] = ci_mean(t2.astype(float) - t1.astype(float), act)
    res["tagging"]["T2_mass_tuned"]["vs_B1_diff_balacc_ci"] = _bal_diff_ci(t1, t2, base.ko.values, act)
    # confusion of tags (6x6) for B0 and T2
    for name in ["B0_thesis(0.5,0.4)", "T2_mass_tuned"]:
        tko = tags[name][1]
        res["tagging"][name]["confusion6"] = [[int(((base.ko.values == a) & (tko == b)).sum()) for b in KO6] for a in KO6]
    per_clip.to_csv(os.path.join(RES, f"test_per_clip_{clf}.csv"), index=False)
    # test-set sweeps (for the trade-off figure only; selection was done on dev)
    sw = [{"method": "thesis", "p1": ct, "p2": mr, **tag_metrics(base, *tag_eval(base, "baseline", ct, mr))}
          for ct in CT_GRID for mr in MR_GRID]
    sw += [{"method": "mass", "p1": wn, "p2": 0.0, **tag_metrics(base, *tag_eval(base, "mass", wn, 0.0))}
           for wn in WN_GRID]
    pd.DataFrame(sw).to_csv(os.path.join(RES, f"test_sweep_{clf}.csv"), index=False)
    out["test"] = res
    log(f"[{clf}] done in {time.time()-t_start:.0f}s")
    return out


def _bal(corr, y, idx):
    c, yy = corr[idx], y[idx]
    return np.mean([c[yy == k].mean() for k in KO6 if (yy == k).any()])


def _bal_ci(corr, y, act):
    acts = sorted(set(act)); ix = {k: np.where(act == k)[0] for k in acts}
    bs = cluster_boot(lambda s: _bal(corr, y, np.concatenate([ix[k] for k in s])), acts)
    return float(_bal(corr, y, np.arange(len(y)))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _bal_diff_ci(c0, c1, y, act):
    acts = sorted(set(act)); ix = {k: np.where(act == k)[0] for k in acts}
    f = lambda idx: _bal(c1, y, idx) - _bal(c0, y, idx)
    bs = cluster_boot(lambda s: f(np.concatenate([ix[k] for k in s])), acts)
    return float(f(np.arange(len(y)))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def choose_hse_on_dev(log):
    accs = {}
    for clf in ["afew", "mtl"]:
        cl = [c for c in load_clips(clf) if c["actor"] in DEV]
        corr = []
        for c in cl:
            m = c["t"] >= SKIP_S
            lab = np.where(c["face"][m], np.nan_to_num(c["probs"][m], nan=0).argmax(1), 6)
            corr.append((ko_of(lab) == c["ko"]).mean())
        accs[clf] = float(np.mean(corr))
    log(f"HSEmotion model selection on dev (raw 6-class frame acc, mean over clips): {accs}")
    return max(accs, key=accs.get), accs


def main():
    os.makedirs(RES, exist_ok=True)
    logf = open(os.path.join(RES, "e5_analyze_log.txt"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    clfs = sys.argv[1:]
    summary_path = os.path.join(RES, "e5_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    if not clfs:
        hse, accs = choose_hse_on_dev(log)
        summary["hse_dev_selection"] = {"dev_raw_frame_acc_ko": accs, "chosen": hse}
        clfs = ["cnn", hse]
    for clf in clfs:
        summary[clf] = analyze(clf, log)
        json.dump(summary, open(summary_path, "w"), ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
