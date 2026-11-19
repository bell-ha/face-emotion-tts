# -*- coding: utf-8 -*-
"""그림 5(안정화 단계별 라벨 시간 변화)의 한 단 폭(76 mm) 판.

e2_analyze.py 의 fig_e2_timeline_example 은 전체 폭(160 mm) 기준 글자 크기라, 원고에서 한 단 폭으로 줄이면
글자가 3–4 pt 가 된다. 같은 데이터(배우 1 스트림의 처음 4클립)로 76 mm 폭에 맞춰 다시 그린다.
결과 파일(results/)은 건드리지 않고 figures/fig_e2_timeline_example_col.{png,pdf} 만 만든다.

    .venv/bin/python experiments/E2_emotion-stabilization/scripts/e2_fig_timeline_col.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_analyze as E  # noqa: E402
from stabilizer import EMOTION_LABELS, EmotionServer  # noqa: E402
from fer_common import savefig, setup_korean_font  # noqa: E402


def main():
    clips = E.load_clips()
    plt = setup_korean_font()
    import matplotlib.transforms as mtrans
    KO7 = EMOTION_LABELS
    KO2EN = {'분노': 'Angry', '두려움': 'Fear', '행복': 'Happy', '슬픔': 'Sad', '놀람': 'Surprise', '중립': 'Neutral'}
    a = sorted(set(c["actor"] for c in clips))[0]
    seq = E.per_actor_streams(clips)[a][:4]
    srv = EmotionServer(); srv._n = 0
    rr = E.run_sequence(seq, None, srv)
    fig, axes = plt.subplots(4, 1, figsize=(3.0, 3.1), sharex=True)
    for r in rr:
        tt = r["t_abs"]
        for ax, v, nm in zip(axes, ["raw", "ema", "maj", "emitted"],
                             ["Raw", "EMA*", "Majority", "Emitted"]):
            L = r["lab"][v].astype(float); L[L < 0] = np.nan
            ax.plot([tt[0], tt[-1]], [r["clip"]["emo_idx"]] * 2, color="0.75", lw=3.5, solid_capstyle="butt")
            ax.step(tt, L, where="post", color="black", lw=0.7)
            ax.set_ylabel(nm, fontsize=6.5, rotation=0, ha="right", va="center", labelpad=2)
            ax.set_yticks(range(7)); ax.set_yticklabels(KO7, fontsize=5)
            ax.tick_params(axis="both", length=2, pad=1.5, labelsize=5)
            ax.axvline(tt[0], color="0.3", ls=":", lw=0.5)
        for e in r["events"]:
            axes[3].plot(e[0] / 1000, EMOTION_LABELS.index(e[1]), "kv", ms=2.5)
        tag = ('<' + KO2EN[r['tag'].strip('<>')] + '>') if r['tag'] else '(none)'
        axes[0].text((tt[0] + tt[-1]) / 2, 1.04, f"True: {KO7[r['clip']['emo_idx']]}\nTag: {tag}",
                     ha="center", va="bottom", fontsize=5.5, linespacing=1.05,
                     transform=mtrans.blended_transform_factory(axes[0].transData, axes[0].transAxes))
    axes[-1].set_xlabel("Time (s)", fontsize=6.5, labelpad=1)
    for ax in axes:
        ax.set_ylim(-0.5, 6.5); ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.6)
    fig.tight_layout(h_pad=0.25, pad=0.3)
    savefig(fig, os.path.join(E.FIG, "fig_e2_timeline_example_col")); plt.close(fig)
    print("saved", os.path.join(E.FIG, "fig_e2_timeline_example_col"))


if __name__ == "__main__":
    main()
