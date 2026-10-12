"""E2-b: Temporal stabilization analysis on cached RAVDESS per-frame probabilities.

Variants (frame-level label sequences):
  raw    : argmax of per-frame CNN probs
  ema    : argmax of EMA probs (EMA_ALPHA=0.8)
  maj    : majority of raw argmax over last WIN_SEC (fps-adaptive window, history maxlen 30)
  final  : code's "EMA+majority" rule (final_idx = maj if maj==ema else ema)  -> identical to ema
  gated  : (NOT in code; reference) agreement-gated: output changes only when maj==ema
  emitted: full pipeline (+hold-time 100 ms, conf>=0.3, send-interval 0.2 s) -> last emitted label
No-face frames are treated as 'Neutral' (as in face_emotion.py) for raw/ema/maj/final.

Modes:
  stream   : per actor, all clips concatenated in random order (seed 42), stabilizer state carried
             over (= continuous live use). Main mode.
  isolated : stabilizer reset at each clip.
Frame-level accuracy excludes the first 0.5 s of each clip (neutral lead-in of RAVDESS + adaptation).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "E1_fer", "scripts"))
from stabilizer import EMOTION_LABELS, LABEL_TO_KO, EmotionServer, Stabilizer
from fer_common import ROOT, savefig, setup_korean_font

SEED = 42
PRB = os.path.join(ROOT, "data", "ravdess", "probs")
OUT = os.path.join(ROOT, "experiments", "E2_stabilization")
RES, FIG = os.path.join(OUT, "results"), os.path.join(OUT, "figures")
SKIP_S = 0.5
SETTLE_S = 0.5  # transition counted as settled when label stays at new emotion this long
VARIANTS = ["raw", "ema", "maj", "final", "gated", "emitted"]
VNAME = {"raw": "Raw argmax", "ema": "EMA", "maj": "다수결", "final": "EMA+다수결(코드)",
         "gated": "합의 게이트(참고)", "emitted": "전체(+유지·간격)"}
RAV_EMO = {1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad", 5: "Angry", 6: "Fear", 7: "Disgust", 8: "Surprise"}
KO6 = ['분노', '두려움', '행복', '슬픔', '놀람', '중립']


def idx_to_ko(i):
    if i == -2:
        return "없음"
    if i == -1:
        return "중립"
    return LABEL_TO_KO[EMOTION_LABELS[i]]


def load_clips():
    clips = []
    for f in sorted(glob.glob(os.path.join(PRB, "*.npz"))):
        b = os.path.basename(f)[:-4]
        p = [int(x) for x in b.split("-")]
        emo = RAV_EMO[p[2]]
        if emo == "Calm":
            continue  # calm dropped (no FER2013 counterpart)
        d = np.load(f)
        clips.append({"clip": b, "actor": p[6], "emotion": emo, "emo_idx": EMOTION_LABELS.index(emo),
                      "ko": LABEL_TO_KO[emo], "intensity": p[3], "statement": p[4], "rep": p[5],
                      "t": d["t"], "probs": d["probs"], "face": d["face"]})
    return clips


def run_sequence(segments, stab_kw=None, server=None, tag_kw=None, subsample=1):
    """segments: list of clips, run back-to-back through ONE stabilizer (continuous time).
    Returns per-clip dict of per-frame variant labels (+ times) and sentence tags."""
    st = Stabilizer(**(stab_kw or {}))
    out = []
    t_off = 0.0
    gated_prev = None
    for c in segments:
        t = c["t"][::subsample]
        P = c["probs"][::subsample]
        F = c["face"][::subsample]
        n = len(t)
        lab = {v: np.empty(n, int) for v in VARIANTS}
        n_ev0 = len(st.events)
        for k in range(n):
            tt = t_off + t[k]
            r = st.step(tt, P[k] if F[k] else None)
            for v in ["raw", "ema", "maj", "final"]:
                lab[v][k] = r[v] if r[v] >= 0 else 6  # no face -> Neutral
            if r["raw"] >= 0 and (gated_prev is None or r["maj"] == r["ema"]):
                gated_prev = r["ema"]  # change output only when EMA and majority agree
            lab["gated"][k] = gated_prev if gated_prev is not None else 6
            lab["emitted"][k] = r["emitted"]
            # forward newly emitted events to the server
            while server is not None and len(st.events) > getattr(server, "_n", 0):
                e = st.events[server._n]
                server.set_emotion(*e)
                server._n += 1
        t0_ms, t1_ms = int((t_off + t[0]) * 1000), int((t_off + t[-1]) * 1000)
        tag = None
        if server is not None:
            tag = server.pick_dominant_ko(t0_ms, t1_ms, **(tag_kw or {}))
        out.append({"clip": c, "t": t, "t_abs": t_off + t, "lab": lab, "n_events": len(st.events) - n_ev0,
                    "events": st.events[n_ev0:], "tag": tag, "t0_ms": t0_ms, "t1_ms": t1_ms})
        dt = (t[1] - t[0]) if n > 1 else 1 / 30
        t_off += t[-1] + dt
    return out


def clip_metrics(r):
    c, t, lab = r["clip"], r["t"], r["lab"]
    dur = t[-1] - t[0] + (t[1] - t[0])
    m = t >= SKIP_S
    row = {"clip": c["clip"], "actor": c["actor"], "emotion": c["emotion"], "ko": c["ko"],
           "intensity": c["intensity"], "dur_s": dur, "face_rate": float(c["face"].mean()),
           "n_events": r["n_events"], "tag": r["tag"]}
    for v in VARIANTS:
        L = lab[v]
        sw = int((L[1:] != L[:-1]).sum())
        ko = np.array([idx_to_ko(i) for i in L])
        row[f"{v}_switch_per_s"] = sw / dur
        row[f"{v}_switch_per_s_ko"] = int((ko[1:] != ko[:-1]).sum()) / dur
        row[f"{v}_acc7"] = float((L[m] == c["emo_idx"]).mean())
        row[f"{v}_acc_ko"] = float((ko[m] == c["ko"]).mean())
        vals, cnt = np.unique(L[m], return_counts=True)
        dom = int(vals[np.argmax(cnt)])
        row[f"{v}_dom7"] = dom
        row[f"{v}_clip7"] = dom == c["emo_idx"]
        vk, ck = np.unique(ko[m], return_counts=True)
        row[f"{v}_clip_ko"] = vk[np.argmax(ck)] == c["ko"]
    return row


def per_actor_streams(clips, rng_seed=SEED):
    rng = np.random.default_rng(rng_seed)
    streams = {}
    for a in sorted(set(c["actor"] for c in clips)):
        cs = [c for c in clips if c["actor"] == a]
        order = rng.permutation(len(cs))
        streams[a] = [cs[i] for i in order]
    return streams


def evaluate(clips, mode="stream", stab_kw=None, tag_kw=None, subsample=1):
    rows = []
    if mode == "stream":
        for a, seq in per_actor_streams(clips).items():
            srv = EmotionServer(); srv._n = 0
            for r in run_sequence(seq, stab_kw, srv, tag_kw, subsample):
                rows.append(clip_metrics(r))
    else:
        for c in clips:
            srv = EmotionServer(); srv._n = 0
            for r in run_sequence([c], stab_kw, srv, tag_kw, subsample):
                rows.append(clip_metrics(r))
    df = pd.DataFrame(rows)
    df["tag_ko"] = df.tag.map(lambda s: s.strip("<>") if s else "중립")
    df["tag_correct"] = df.tag_ko == df.ko
    df["tag_empty"] = df.tag == ""
    return df


def summarize(df):
    s = {"n_clips": int(len(df)), "face_rate": float(df.face_rate.mean())}
    for v in VARIANTS:
        s[v] = {"switch_per_s": float(df[f"{v}_switch_per_s"].mean()),
                "switch_per_s_ko": float(df[f"{v}_switch_per_s_ko"].mean()),
                "frame_acc7": float(df[f"{v}_acc7"].mean()),
                "frame_acc_ko": float(df[f"{v}_acc_ko"].mean()),
                "clip_acc7": float(df[f"{v}_clip7"].mean()),
                "clip_acc_ko": float(df[f"{v}_clip_ko"].mean())}
    s["events_per_clip_mean"] = float(df.n_events.mean())
    s["events_per_clip_median"] = float(df.n_events.median())
    s["events_per_s"] = float(df.n_events.sum() / df.dur_s.sum())
    emo = df[df.ko != "중립"]
    s["tag"] = {"accuracy": float(df.tag_correct.mean()), "empty_rate": float(df.tag_empty.mean()),
                "accuracy_emotional_clips": float(emo.tag_correct.mean()),
                "empty_rate_emotional_clips": float(emo.tag_empty.mean()),
                "accuracy_neutral_clips": float(df[df.ko == "중립"].tag_correct.mean()),
                "precision_nonempty": float(df[~df.tag_empty].tag_correct.mean()) if (~df.tag_empty).any() else None,
                "n_nonempty": int((~df.tag_empty).sum())}
    return s


# ---------------------------------------------------------------- transitions
def transition_latency(clips, n_per_actor=150, stab_kw=None):
    rng = np.random.default_rng(SEED)
    raw_ok = {}
    for c in clips:
        m = c["t"] >= SKIP_S
        P = c["probs"][m]
        lab = np.where(c["face"][m], np.nan_to_num(P, nan=0).argmax(1), 6)
        ko = np.array([idx_to_ko(i) for i in lab])
        raw_ok[c["clip"]] = (ko == c["ko"]).mean() > 0.5
    rows = []
    by_actor = {}
    for c in clips:
        by_actor.setdefault(c["actor"], []).append(c)
    for a, cs in sorted(by_actor.items()):
        cand = [(i, j) for i in range(len(cs)) for j in range(len(cs))
                if i != j and cs[i]["ko"] != cs[j]["ko"] and raw_ok[cs[j]["clip"]]]
        if not cand:
            continue
        sel = rng.choice(len(cand), size=min(n_per_actor, len(cand)), replace=False)
        for s in sel:
            A, B = cs[cand[s][0]], cs[cand[s][1]]
            segA = trim(A); segB = trim(B)
            res = run_sequence([segA, segB], stab_kw)
            rb = res[1]
            tb = rb["t"] - rb["t"][0]
            row = {"actor": a, "A": A["clip"], "B": B["clip"], "A_ko": A["ko"], "B_ko": B["ko"]}
            for v in VARIANTS:
                ko = np.array([idx_to_ko(i) for i in rb["lab"][v]])
                prevA = idx_to_ko(res[0]["lab"][v][-1])
                hit = np.where(ko == B["ko"])[0]
                row[f"{v}_already"] = prevA == B["ko"]
                row[f"{v}_lat_s"] = float(tb[hit[0]]) if len(hit) else np.nan
                # settled latency: first time from which the label stays = B for >= SETTLE_S
                eq = ko == B["ko"]
                m = max(1, int(round(SETTLE_S / np.median(np.diff(tb)))))
                st = np.nan
                for k in hit:
                    if k + m <= len(eq) and eq[k:k + m].all():
                        st = float(tb[k]); break
                row[f"{v}_settle_s"] = st
            rows.append(row)
    return pd.DataFrame(rows)


def trim(c):
    m = c["t"] >= SKIP_S
    d = dict(c)
    d["t"] = c["t"][m] - c["t"][m][0]
    d["probs"] = c["probs"][m]
    d["face"] = c["face"][m]
    return d


# ---------------------------------------------------------------- main
def main():
    clips = load_clips()
    print("clips", len(clips), "actors", sorted(set(c["actor"] for c in clips)))
    plt = setup_korean_font()
    summary = {"n_clips": len(clips), "actors": sorted(set(int(c["actor"]) for c in clips)),
               "n_frames": int(sum(len(c["t"]) for c in clips)),
               "emotion_counts": pd.Series([c["emotion"] for c in clips]).value_counts().to_dict()}

    # --- main evaluation (stream, 30 fps) + isolated + 15 fps
    res = {}
    for name, kw in [("stream_30fps", dict(mode="stream")), ("isolated_30fps", dict(mode="isolated")),
                     ("stream_15fps", dict(mode="stream", subsample=2))]:
        df = evaluate(clips, **kw)
        df.to_csv(os.path.join(RES, f"clip_metrics_{name}.csv"), index=False)
        res[name] = df
        summary[name] = summarize(df)
        print(name, json.dumps({v: summary[name][v] for v in VARIANTS}, ensure_ascii=False)[:900])
        print(name, "tag", summary[name]["tag"], "events/clip", summary[name]["events_per_clip_mean"])

    # main table CSV
    rows = []
    for name in res:
        for v in VARIANTS:
            rows.append({"condition": name, "variant": v, **summary[name][v]})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "variant_summary.csv"), index=False)

    # per-emotion raw accuracy (domain shift check) + confusion (raw frames, ko6, t>=0.5)
    df = res["stream_30fps"]
    summary["per_emotion_stream_30fps"] = df.groupby("emotion")[[f"{v}_acc7" for v in ["raw", "ema", "emitted"]] +
                                                               ["raw_clip7", "tag_correct", "tag_empty"]].mean().to_dict()
    y, p = [], []
    for c in clips:
        m = c["t"] >= SKIP_S
        lab = np.where(c["face"][m], np.nan_to_num(c["probs"][m], nan=0).argmax(1), 6)
        y += [c["emo_idx"]] * int(m.sum()); p += list(lab)
    from sklearn.metrics import confusion_matrix
    cm7 = confusion_matrix(y, p, labels=list(range(7)))
    summary["raw_frame_confusion7"] = cm7.tolist()

    # --- transition latency
    tl = transition_latency(clips)
    tl.to_csv(os.path.join(RES, "transition_latency.csv"), index=False)
    lat = {"n_pairs": int(len(tl))}
    for v in VARIANTS:
        x = tl.loc[~tl[f"{v}_already"], f"{v}_lat_s"]
        lat[v] = {"n_valid": int(len(x)), "reached_rate": float(x.notna().mean()),
                  "median_s": float(np.nanmedian(x)), "q1_s": float(np.nanpercentile(x, 25)),
                  "q3_s": float(np.nanpercentile(x, 75)), "mean_s": float(np.nanmean(x))}
        y = tl.loc[~tl[f"{v}_already"], f"{v}_settle_s"]
        lat[v].update({"settle_reached_rate": float(y.notna().mean()), "settle_median_s": float(np.nanmedian(y)),
                       "settle_q1_s": float(np.nanpercentile(y, 25)), "settle_q3_s": float(np.nanpercentile(y, 75))})
        both = tl[(~tl[f"{v}_already"]) & (~tl["raw_already"])]
        d = (both[f"{v}_lat_s"] - both["raw_lat_s"]).dropna()
        lat[v]["excess_over_raw_median_s"] = float(d.median()) if len(d) else None
        lat[v]["excess_over_raw_iqr_s"] = [float(d.quantile(.25)), float(d.quantile(.75))] if len(d) else None
    summary["transition_latency"] = lat
    print("latency", json.dumps(lat, ensure_ascii=False)[:1500])

    # --- sensitivity: tag params (conf_thresh x min_ratio) on stream 30fps
    cts = [0.3, 0.4, 0.5, 0.6, 0.7]; mrs = [0.2, 0.3, 0.4, 0.5, 0.6]
    sens = []
    for ct in cts:
        for mr in mrs:
            d = evaluate(clips, mode="stream", tag_kw=dict(conf_thresh=ct, min_ratio=mr))
            sens.append({"conf_thresh": ct, "min_ratio": mr, "tag_acc": d.tag_correct.mean(),
                         "empty_rate": d.tag_empty.mean(),
                         "tag_acc_emotional": d[d.ko != "중립"].tag_correct.mean()})
    sens = pd.DataFrame(sens); sens.to_csv(os.path.join(RES, "sensitivity_tag_params.csv"), index=False)

    # --- sensitivity: EMA_ALPHA and MIN_HOLD_MS
    sens2 = []
    for a in [0.2, 0.4, 0.6, 0.8, 1.0]:
        d = evaluate(clips, mode="stream", stab_kw=dict(EMA_ALPHA=a))
        sens2.append({"param": "EMA_ALPHA", "value": a, **flat(summarize(d))})
    for h in [0, 100, 200, 300, 500, 800]:
        d = evaluate(clips, mode="stream", stab_kw=dict(MIN_HOLD_MS=h))
        sens2.append({"param": "MIN_HOLD_MS", "value": h, **flat(summarize(d))})
    sens2 = pd.DataFrame(sens2); sens2.to_csv(os.path.join(RES, "sensitivity_stab_params.csv"), index=False)

    with open(os.path.join(RES, "e2_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    figures(plt, clips, res, summary, tl, sens, sens2, cm7)


def flat(s):
    return {"switch_raw": s["raw"]["switch_per_s"], "switch_ema": s["ema"]["switch_per_s"],
            "switch_emitted": s["emitted"]["switch_per_s"], "events_per_clip": s["events_per_clip_mean"],
            "frame_acc_ko_ema": s["ema"]["frame_acc_ko"], "frame_acc_ko_emitted": s["emitted"]["frame_acc_ko"],
            "tag_acc": s["tag"]["accuracy"], "tag_empty": s["tag"]["empty_rate"]}


def panel(ax, txt):
    ax.text(-0.3, 1.04, txt, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")


def figures(plt, clips, res, summary, tl, sens, sens2, cm7):
    """Journal rules: all-Korean text, no in-figure titles, grayscale-distinguishable,
    width 160 mm (6.3 in) full / 76 mm (3.0 in) column, printed text 7-9 pt (>=6 pt for cell numbers)."""
    KO7 = EMOTION_LABELS  # English class names in figures
    KO2EN = {'분노': 'Angry', '두려움': 'Fear', '행복': 'Happy', '슬픔': 'Sad', '놀람': 'Surprise', '중립': 'Neutral'}
    vs = ["raw", "maj", "ema", "gated", "emitted"]  # final == ema, shown once (noted)
    # * 지수평활 = 코드의 'EMA+다수결' 최종 규칙과 동일, † 합의 게이트 = 코드에 없는 참고 변형 (캡션에 설명)
    labels = ["Raw", "Majority", "EMA*", "Agreement\ngate†", "Full\npipeline"]
    hatches = ["", "\\\\", "///", "xx", "..."]
    grays = ["0.2", "0.45", "0.65", "0.8", "0.95"]
    S = summary["stream_30fps"]

    # Fig 1: flicker & accuracy
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3))
    metrics = [("switch_per_s", "Label switches per second"), ("frame_acc_ko", "Frame accuracy (6-class)"),
               ("clip_acc_ko", "Clip accuracy (6-class)")]
    for ax, (k, yl), pl in zip(axes, metrics, ["(a)", "(b)", "(c)"]):
        vals = [S[v][k] for v in vs]
        bars = ax.bar(range(len(vs)), vals, color=grays, edgecolor="black", linewidth=0.6)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        for i, val in enumerate(vals):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=6, rotation=40, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel(yl); ax.spines[["top", "right"]].set_visible(False)
        if "acc" in k:
            ax.set_ylim(0, 1)
        panel(ax, pl)
    fig.tight_layout(w_pad=0.8)
    savefig(fig, os.path.join(FIG, "fig_e2_flicker_accuracy")); plt.close(fig)

    # Fig 2: transition latency boxplots (first hit / settled)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, key, rk, pl in zip(axes, ["lat_s", "settle_s"], ["reached_rate", "settle_reached_rate"], ["(a)", "(b)"]):
        data = [tl.loc[~tl[f"{v}_already"], f"{v}_{key}"].dropna().values for v in vs]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color="black", lw=1.2), boxprops=dict(lw=0.6),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        for patch, g, h in zip(bp["boxes"], grays, hatches):
            patch.set_facecolor(g); patch.set_hatch(h)
        for i, d in enumerate(data):
            ax.plot(i + 1, d.mean(), marker="D", color="black", mfc="white", ms=3)
        ax.set_xticks(range(1, len(vs) + 1)); ax.set_xticklabels(labels, fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, pl)
    axes[0].set_ylabel("Latency after emotion switch (s)")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG, "fig_e2_transition_latency")); plt.close(fig)

    # Fig 3: tag sensitivity heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for ax, k, pl in zip(axes, ["tag_acc", "empty_rate"], ["(a) Tag accuracy", "(b) Empty-tag rate"]):
        M = sens.pivot(index="conf_thresh", columns="min_ratio", values=k)
        im = ax.imshow(M.values, cmap="Greys", vmin=0, vmax=1, origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M.values[i, j] > 0.5 else "black")
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
        ax.set_xlabel("min_ratio")
        if k == "tag_acc":
            ax.set_ylabel("conf_thresh")
        j0 = list(M.columns).index(0.4); i0 = list(M.index).index(0.5)
        ax.add_patch(plt.Rectangle((j0 - .5, i0 - .5), 1, 1, fill=False, lw=1.5, ec="black", ls="--"))
        ax.text(0.0, 1.03, pl, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cb.ax.tick_params(labelsize=6.5); cb.set_label("Proportion", fontsize=7)
    savefig(fig, os.path.join(FIG, "fig_e2_tag_sensitivity")); plt.close(fig)

    # Fig 4: EMA_ALPHA / hold sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for ax, prm, xl, pl in zip(axes, ["EMA_ALPHA", "MIN_HOLD_MS"],
                               ["EMA_ALPHA (1.0 = no smoothing)", "MIN_HOLD_MS (ms)"], ["(a)", "(b)"]):
        d = sens2[sens2.param == prm]
        ax2 = ax.twinx()
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)  # left-axis lines on top
        span = d.value.max() - d.value.min()
        dx = 0.03 * span  # small x-offset so overlapping markers stay visible
        l1, = ax.plot(d.value - dx, d.events_per_clip, "k-o", ms=3, lw=1, label="Emitted events per clip")
        l2, = ax.plot(d.value - dx, d.switch_ema, "k--s", ms=3, lw=1, mfc="white", label="EMA label switches (/s)")
        l3, = ax2.plot(d.value + dx, d.tag_acc, "-^", ms=3, lw=1, color="0.5", label="Sentence tag accuracy")
        l4, = ax2.plot(d.value + dx, d.frame_acc_ko_emitted, ":D", ms=2.6, lw=1, color="0.5", mfc="white",
                       label="Frame accuracy (full pipeline)")
        ax2.set_ylim(0, 1)
        ax.set_xlabel(xl); ax.set_ylabel("Count"); ax2.set_ylabel("Accuracy")
        ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
        ax.text(-0.14, 1.02, pl, transform=ax.transAxes, fontsize=8, va="bottom")
    fig.legend(handles=[l1, l2, l3, l4], fontsize=6.5, frameon=False, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, os.path.join(FIG, "fig_e2_stab_sensitivity")); plt.close(fig)

    # Fig 5: RAVDESS raw confusion (7 cls) - column width
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    cmn = cm7 / np.maximum(cm7.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(KO7, fontsize=6.5, rotation=45, ha="right"); ax.set_yticklabels(KO7, fontsize=6.5)
    ax.set_xlabel("Predicted (per-frame raw output)"); ax.set_ylabel("True (RAVDESS)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(labelsize=6.5)
    savefig(fig, os.path.join(FIG, "fig_e2_ravdess_confusion_raw")); plt.close(fig)

    # Fig 6: example timeline from one actor stream (first 4 clips)
    a = sorted(set(c["actor"] for c in clips))[0]
    seq = per_actor_streams(clips)[a][:4]
    srv = EmotionServer(); srv._n = 0
    rr = run_sequence(seq, None, srv)
    fig, axes = plt.subplots(4, 1, figsize=(6.3, 3.9), sharex=True)
    for r in rr:
        tt = r["t_abs"]
        for ax, v, nm in zip(axes, ["raw", "ema", "maj", "emitted"],
                             ["Raw", "EMA*", "Majority", "Emitted\n(full)"]):
            L = r["lab"][v].astype(float); L[L < 0] = np.nan
            ax.plot([tt[0], tt[-1]], [r["clip"]["emo_idx"]] * 2, color="0.75", lw=4, solid_capstyle="butt")
            ax.step(tt, L, where="post", color="black", lw=0.8)
            ax.set_ylabel(nm, fontsize=7, rotation=0, ha="right", va="center")
            ax.set_yticks(range(7)); ax.set_yticklabels(KO7, fontsize=5.5)
            ax.axvline(tt[0], color="0.3", ls=":", lw=0.6)
        for e in r["events"]:
            axes[3].plot(e[0] / 1000, EMOTION_LABELS.index(e[1]), "kv", ms=3)
        import matplotlib.transforms as mtrans
        axes[0].text((tt[0] + tt[-1]) / 2, 1.03, f"True: {KO7[r['clip']['emo_idx']]}\nTag: {('<' + KO2EN[r['tag'].strip('<>')] + '>') if r['tag'] else '(none)'}",
                     ha="center", va="bottom", fontsize=6, linespacing=1.1,
                     transform=mtrans.blended_transform_factory(axes[0].transData, axes[0].transAxes))
    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_ylim(-0.5, 6.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(h_pad=0.3)
    savefig(fig, os.path.join(FIG, "fig_e2_timeline_example")); plt.close(fig)


if __name__ == "__main__":
    main()
