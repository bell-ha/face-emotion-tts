"""E5 figures (journal rules: English text only, no in-figure titles, grayscale + hatching,
160 mm full width / 76 mm column, Arial, 300 dpi PNG + PDF)."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "E1_face-recognition", "scripts"))
from fer_common import EMOTION_LABELS, ROOT, savefig, setup_korean_font

OUT = os.path.join(ROOT, "experiments", "E5_improvements")
RES, FIG = os.path.join(OUT, "results"), os.path.join(OUT, "figures")
S = json.load(open(os.path.join(RES, "e5_summary.json")))
H = json.load(open(os.path.join(RES, "hse_eval.json")))
HSE = S["hse_dev_selection"]["chosen"]
plt = setup_korean_font()
GR = ["0.35", "0.35", "0.8", "0.8"]
HA = ["", "////", "", "////"]
LAB = ["FER2013-CNN\nbaseline", "FER2013-CNN\nproposed", "HSEmotion\nbaseline", "HSEmotion\nproposed"]


def panel(ax, s, x=-0.28):
    ax.text(x, 1.03, s, transform=ax.transAxes, fontsize=8, va="bottom")


def bars(ax, vals, los, his, yl, ylim=None, fmt="{:.2f}"):
    x = np.arange(4)
    b = ax.bar(x, vals, color=GR, edgecolor="black", linewidth=0.6, width=0.7)
    for bb, h in zip(b, HA):
        bb.set_hatch(h)
    err = np.array([np.array(vals) - np.array(los), np.array(his) - np.array(vals)])
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    for i, v in enumerate(vals):
        ax.text(i, his[i] + (ylim[1] if ylim else max(his)) * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=5.5, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(yl, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)


# ---------------- Fig 1: 2x2 (classifier x method) on test actors
fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.35))
items = []
for clf in ["cnn", HSE]:
    T = S[clf]["test"]
    items.append((T, "B0_thesis(0.5,0.4)", "baseline"))
    items.append((T, "T2_mass_tuned", "improved"))
specs = [("tag_acc_ci", "Sentence-tag accuracy", (0, 1.08), "tag"),
         ("empty_rate_ci", "Empty-tag rate", (0, 1.08), "tag"),
         ("switch_per_s", "Label switches per second", None, "stab"),
         ("clip_acc_ko", "Clip accuracy (6-class)", (0, 1.08), "stab")]
for ax, (k, yl, yl_, kind), pl in zip(axes, specs, "abcd"):
    v, lo, hi = [], [], []
    for T, tagname, stab in items:
        m = T["tagging"][tagname][k] if kind == "tag" else T["stabilizer"][stab][k]
        v.append(m[0]); lo.append(m[1]); hi.append(m[2])
    bars(ax, v, lo, hi, yl, yl_)
    panel(ax, f"({pl})", -0.36)
fig.tight_layout(w_pad=0.6)
savefig(fig, os.path.join(FIG, "fig_e5_2x2")); plt.close(fig)

# ---------------- Fig 2: tag accuracy vs empty-tag rate (test sweeps; selected points from dev)
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
for ax, clf, name, pl in zip(axes, ["cnn", HSE], ["FER2013-CNN", "HSEmotion"], "ab"):
    sw = pd.read_csv(os.path.join(RES, f"test_sweep_{clf}.csv"))
    th = sw[sw.method == "thesis"]; ms = sw[sw.method == "mass"].sort_values("p1")
    ax.scatter(th.empty_rate, th.tag_acc, s=10, marker="o", facecolor="white", edgecolor="0.4", lw=0.6,
               label="Thesis rule, conf_thresh × min_ratio grid")
    ax.plot(ms.empty_rate, ms.tag_acc, "-", color="black", lw=1, label="Probability mass, w_neutral sweep")
    T = S[clf]["test"]["tagging"]
    pts = [("B0_thesis(0.5,0.4)", "s", "black", "Thesis default (0.5, 0.4)"),
           ("B1_thesis_tuned", "D", "0.6", "Thesis rule, dev-tuned"),
           ("T1_mass(w=1)", "^", "white", "Probability mass, w_neutral = 1"),
           ("T2_mass_tuned", "*", "black", "Probability mass, dev-tuned w_neutral")]
    for key, mk, fc, lab in pts:
        ax.scatter(T[key]["empty_rate"], T[key]["tag_acc"], marker=mk, s=55 if mk == "*" else 28,
                   facecolor=fc, edgecolor="black", lw=0.8, zorder=5, label=lab)
    ax.set_xlabel("Empty-tag rate"); ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.97, 0.04, name, transform=ax.transAxes, ha="right", fontsize=7)
    panel(ax, f"({pl})", -0.18)
axes[0].set_ylabel("Sentence-tag accuracy")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.13, 1, 1))
savefig(fig, os.path.join(FIG, "fig_e5_tag_tradeoff")); plt.close(fig)

# ---------------- Fig 3: raw per-frame confusion on RAVDESS test actors, CNN vs HSEmotion
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
for ax, key, pl in zip(axes, ["cnn_test", f"{HSE}_test"], "ab"):
    cm = np.array(H["ravdess"][key]["confusion7"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(EMOTION_LABELS, fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_LABELS, fontsize=6)
    ax.set_xlabel("Predicted (per-frame)", fontsize=7)
    if pl == "a":
        ax.set_ylabel("True (RAVDESS)", fontsize=7)
    panel(ax, f"({pl})", -0.3)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.ax.tick_params(labelsize=6)
savefig(fig, os.path.join(FIG, "fig_e5_confusion_cnn_vs_hse")); plt.close(fig)

# ---------------- Fig 4: per-class tag recall (column width), 4 configurations
fig, ax = plt.subplots(figsize=(3.0, 2.3))
classes = ['분노', '두려움', '행복', '슬픔', '놀람', '중립']
en = ["Angry*", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
w = 0.2
for i, (T, tagname, _) in enumerate(items):
    r = [T["tagging"][tagname]["per_class_recall"][c] for c in classes]
    b = ax.bar(np.arange(6) + (i - 1.5) * w, r, w, color=GR[i], edgecolor="black", linewidth=0.5,
               label=LAB[i].replace("\n", " "))
    for bb in b:
        bb.set_hatch(HA[i])
ax.set_xticks(range(6)); ax.set_xticklabels(en, fontsize=6.5)
ax.set_ylabel("Sentence-tag recall", fontsize=7); ax.set_ylim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
savefig(fig, os.path.join(FIG, "fig_e5_tag_recall_per_class")); plt.close(fig)
print("figures done")
