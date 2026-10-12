"""E3b 그림: 정확도–화자 유사도 / 정확도–MOS 절충 곡선 (영문 전용, 제목 없음, 흑백 구분)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titlesize": 8,
                     "savefig.dpi": 300, "axes.linewidth": 0.6})
FULL, COL = 160 / 25.4, 76 / 25.4
STY = {"D": dict(color="0.55", ls="--", marker="s", mfc="white", label="D: Preset+DSP"),
       "DP": dict(color="black", ls="-", marker="o", mfc="black", label="D-psola: Neutral+PSOLA")}
# 점 라벨 위치(겹침 방지용 오프셋, pt)
OFF = {"spk_cos": {"D": (-5, -11), "DP": (6, 3)}, "mos": {"D": (6, 3), "DP": (-6, -11)}}
OVR = {("spk_cos", "D", 1.0): (-6, -9), ("spk_cos", "D", 0.75): (7, 3), ("spk_cos", "D", 0.25): (-6, -8),
       ("spk_cos", "DP", 0.0): (0, -13),
       ("mos", "D", 1.0): (-6, -8), ("mos", "D", 0.75): (-7, -8), ("mos", "D", 0.5): (8, 6),
       ("mos", "D", 0.25): (9, -2), ("mos", "DP", 0.0): (0, -13), ("mos", "DP", 0.25): (5, 11),
       ("mos", "DP", 0.5): (7, 6)}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    sm = pd.read_csv(RES / "summary.csv")
    t = sm[sm.split == "test"]
    op = json.loads((RES / "operating_point.json").read_text()).get("chosen")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, FULL * 0.42), sharey=True)
    for ax, (x, lo, hi, xlab) in zip(axes, [("spk_cos", "spk_lo", "spk_hi", "Speaker cosine similarity to neutral"),
                                             ("mos", "mos_lo", "mos_hi", "SQUIM MOS (est.)")]):
        for v in ("D", "DP"):
            s = t[t.variant == v].sort_values("k")
            st = STY[v]
            ax.errorbar(s[x], s.acc9 * 100, xerr=[s[x] - s[lo], s[hi] - s[x]],
                        yerr=[100 * (s.acc9 - s.acc9_lo), 100 * (s.acc9_hi - s.acc9)],
                        color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"], mec=st["color"],
                        ms=4, lw=0.9, elinewidth=0.5, capsize=1.5, label=st["label"])
            for _, r in s.iterrows():
                if r.k == 0 and v == "D":
                    continue          # k=0은 두 변형이 같은 점 → 한 번만 표기
                o = OVR.get((x, v, float(r.k)), OFF[x][v])
                ax.annotate(f"k={r.k:g}", (r[x], r.acc9 * 100), textcoords="offset points",
                            xytext=o, fontsize=6.5, va="center" if o[1] > -10 else "top",
                            ha="center" if o[0] == 0 else ("left" if o[0] > 0 else "right"),
                            color="0.4" if v == "D" else "black",
                            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
        if op is not None:
            r = t[(t.variant == op["variant"]) & (t.k == op["k"])].iloc[0]
            ax.scatter([r[x]], [r.acc9 * 100], s=110, facecolors="none", edgecolors="black", lw=1.0,
                       zorder=5, label="Recommended (dev rule)")
        ax.axhline(100 / 6, color="black", lw=0.6, ls=":", label="Chance (1/6)")
        ax.set_xlabel(xlab)
    axes[0].set_ylabel("SER accuracy (%)")
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    axes[0].set_ylim(0, 100)
    axes[0].set_xlim(0.14, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig_e3b_tradeoff")


if __name__ == "__main__":
    main()
