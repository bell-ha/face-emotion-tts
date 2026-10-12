"""E3 그림 — 학회 그림 규정 준수.

- 그림 안 글자는 영문으로 통일(코디네이터 지시), 그림 제목 없음(패널 기호만)
- 2단 편집: 전체 폭 160 mm(6.3 in) 또는 단 폭 76 mm(3.0 in), 글자 7~8 pt
- 흑백 인쇄 구분: 명도 + 해칭 + 선 모양
- Arial, 300 dpi PNG + PDF
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "axes.unicode_minus": False, "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.titlesize": 8, "hatch.linewidth": 0.6,
                     "savefig.dpi": 300, "figure.dpi": 100, "axes.linewidth": 0.6})
FULL, COL = 160 / 25.4, 76 / 25.4
R, F = C.RESULTS, C.FIGS
EN = {"happy": "Happy", "sad": "Sad", "angry": "Angry", "surprise": "Surprise", "fear": "Fear", "neutral": "Neutral"}
KO = [EN[e] for e in C.EMOS]
COND_KO = {"A": "A: Speed only", "B": "B: Neutral", "C": "C: Preset", "D": "D: Preset+DSP",
           "E_lex_B": "Ref: Emotional text+neutral", "E_lex_D": "Ref: Emotional text+D"}
GRAYS = {"A": "0.85", "B": "1.0", "C": "0.6", "D": "0.25", "E_lex_B": "0.92", "E_lex_D": "0.45"}
HATCH = {"A": "xxxx", "B": "", "C": "////", "D": "", "E_lex_B": "....", "E_lex_D": "\\\\\\\\"}
PANEL = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def save(fig, name):
    fig.savefig(F / f"{name}.png", bbox_inches="tight")
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_confusion():
    cj = json.loads((R / "confusion_matrices.json").read_text())["matrices"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 0.78))
    for k, (ax, cond) in enumerate(zip(axes.flat, ["A", "B", "C", "D"])):
        m = np.array(cj[f"test/{cond}/e2v9"], dtype=float)
        rs = m.sum(1, keepdims=True)
        p = np.divide(m, rs, out=np.zeros_like(m), where=rs > 0)
        ax.imshow(p, cmap="Greys", vmin=0, vmax=1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j] > 0:
                    ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=7,
                            color="white" if p[i, j] > 0.5 else "black")
        ax.set_xticks(range(7)); ax.set_xticklabels(KO + ["Other"], rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(6)); ax.set_yticklabels(KO)
        ax.set_title(f"{PANEL[k]} {COND_KO[cond]}", loc="left")
        ax.set_xlabel("Predicted (emotion2vec)")
        ax.set_ylabel("Target emotion")
    fig.tight_layout()
    save(fig, "fig_e3_confusion")


def fig_accuracy():
    s = pd.read_csv(R / "ser_summary.csv")
    rows = [("A", "test"), ("B", "test"), ("C", "test"), ("D", "test"), ("E_lex_B", "all"), ("E_lex_D", "all")]
    fig, ax = plt.subplots(figsize=(COL, COL * 0.95))
    for i, (c, sp) in enumerate(rows):
        r = s[(s.split == sp) & (s.cond == c)].iloc[0]
        ax.bar(i, r.acc9 * 100, color=GRAYS[c], edgecolor="black", hatch=HATCH[c], lw=0.6)
        ax.errorbar(i, r.acc9 * 100, yerr=[[100 * (r.acc9 - r.acc9_lo)], [100 * (r.acc9_hi - r.acc9)]],
                    color="black", capsize=2, lw=0.7)
        ytxt = max(r.acc9_hi * 100, 100 / 6) + 3.5
        ax.text(i, ytxt, f"{r.acc9 * 100:.0f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(100 / 6, ls="--", color="black", lw=0.7, label="Chance (1/6)")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["A\nSpeed\nonly", "B\nNeutral", "C\nPreset", "D\nPreset\n+DSP",
                        "R1\nText\n+B", "R2\nText\n+D"], fontsize=6.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("SER accuracy (%)")
    fig.tight_layout()
    save(fig, "fig_e3_ser_accuracy")


def fig_acoustic():
    d = pd.read_csv(R / "per_file.csv")
    d = d[d.split == "test"]
    feats = [("d_f0_st", "ΔMean F0 (st)"), ("f0_sd_st", "F0 SD (st)"),
             ("d_rate_pct", "ΔSpeech rate (%)"), ("d_int_mean_db", "ΔMean intensity (dB)")]
    conds = ["A", "C", "D"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 0.62))
    w = 0.27
    for k, (ax, (f, lab)) in enumerate(zip(axes.flat, feats)):
        for j, c in enumerate(conds):
            g = d[d.cond == c].groupby("emotion")[f].agg(["mean", "std"]).reindex(C.EMOS)
            x = np.arange(6) + (j - 1) * w
            ax.bar(x, g["mean"], w, yerr=g["std"].fillna(0), color=GRAYS[c], edgecolor="black",
                   hatch=HATCH[c], capsize=1.5, label=COND_KO[c], lw=0.6, error_kw={"lw": 0.6})
        if f == "f0_sd_st":
            b = d[d.cond == "B"].drop_duplicates("sid")[f]
            ax.axhline(b.mean(), ls=":", color="black", lw=0.8, label="B: Neutral (mean)")
        else:
            ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(range(6)); ax.set_xticklabels(KO)
        ax.set_ylabel(lab)
        ax.set_title(PANEL[k], loc="left")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_e3_acoustic")


def clean_f0(f0, win=31, max_dev_st=6.0, min_run=5):
    """표시용 F0 정리: 0(무성)→NaN, 이동 중앙값(31프레임=310 ms)에서 6반음 넘게 벗어난 프레임(옥타브 오류) 제거,
    5프레임(50 ms) 미만의 고립 유성 구간 제거. 그림 표시에만 적용(통계 특징에는 미적용)."""
    f0 = f0.astype(float)
    f0[f0 <= 0] = np.nan
    st = 12 * np.log2(f0 / 100.0)
    s = pd.Series(st)
    med = s.rolling(win, center=True, min_periods=3).median()
    st[np.abs(s - med).values > max_dev_st] = np.nan
    v = ~np.isnan(st)
    i = 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            if j - i < min_run:
                st[i:j] = np.nan
            i = j
        else:
            i += 1
    return 100.0 * 2 ** (st / 12)


def fig_f0_contours(sid="S02"):
    import parselmouth
    styles = {"B": dict(color="0.55", ls=":", lw=1.1), "C": dict(color="0.35", ls="--", lw=0.9),
              "D": dict(color="black", ls="-", lw=1.0)}
    fig, axes = plt.subplots(2, 3, figsize=(FULL, FULL * 0.55), sharey=True, sharex=True)
    for k, (ax, emo) in enumerate(zip(axes.flat, C.EMOS)):
        for cond in ("B", "C", "D"):
            p = C.AUDIO / cond / (f"{sid}_neutral.wav" if cond == "B" else f"{sid}_{emo}.wav")
            pt = parselmouth.Sound(str(p)).to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
            f0 = clean_f0(pt.selected_array["frequency"].copy())
            ax.plot(pt.xs(), f0, label=COND_KO[cond], **styles[cond])
        ax.set_title(f"{PANEL[k]} {EN[emo]}", loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("F0 (Hz)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_e3_f0_contours")


def fig_quality():
    d = pd.read_csv(R / "per_file.csv")
    d = d[d.split == "test"].drop_duplicates("path")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, FULL * 0.36))
    e = d[(d.emotion != "neutral") & (d.cond != "B")]
    cs = ["A", "C", "D"]
    bp = axes[0].boxplot([e[e.cond == c].spk_cos_own_neutral.dropna().values for c in cs],
                         patch_artist=True, widths=0.5, flierprops={"markersize": 3})
    for patch, c in zip(bp["boxes"], cs):
        patch.set_facecolor(GRAYS[c]); patch.set_hatch(HATCH[c])
    for med in bp["medians"]:
        med.set_color("black")
    sc = json.loads((R / "speaker_scale_reference.json").read_text())
    for key, ls, txt in (("SunHi_vs_InJoon(male)_same_sentence", "--", "Different speaker (ref.)"),
                         ("SunHi_vs_SunHi_diff_sentence", ":", "Same speaker, diff. sentence (ref.)")):
        v = sc[key][0]
        axes[0].axhline(v, ls=ls, color="black", lw=0.7, label=txt)
    axes[0].legend(loc="center left", bbox_to_anchor=(0.01, 0.56), ncol=1, fontsize=6.5, frameon=False)
    axes[0].set_xticklabels([COND_KO[c].replace(": ", "\n") for c in cs])
    axes[0].set_ylabel("Speaker cosine similarity")
    axes[0].set_title(PANEL[0], loc="left")
    cs = ["A", "B", "C", "D"]
    bp = axes[1].boxplot([d[d.cond == c].squim_mos.dropna().values for c in cs],
                         patch_artist=True, widths=0.5, flierprops={"markersize": 3})
    for patch, c in zip(bp["boxes"], cs):
        patch.set_facecolor(GRAYS[c]); patch.set_hatch(HATCH[c])
    for med in bp["medians"]:
        med.set_color("black")
    axes[1].set_xticklabels([COND_KO[c].replace(": ", "\n") for c in cs])
    axes[1].set_ylabel("SQUIM MOS (est.)")
    axes[1].set_title(PANEL[1], loc="left")
    fig.tight_layout()
    save(fig, "fig_e3_speaker_quality")


if __name__ == "__main__":
    fig_confusion(); fig_accuracy(); fig_acoustic(); fig_f0_contours(); fig_quality()
    print(sorted(p.name for p in F.iterdir()))
