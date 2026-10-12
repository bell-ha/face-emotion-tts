"""latency.py 결과 요약표와 그림 생성."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E4 = Path(__file__).resolve().parents[1]
R, F = E4 / "results", E4 / "figures"
# 학회 그림 규정: 그림 안 글자는 영문, 제목 없음, 단 폭 76 mm(3.0 in) / 전체 폭 160 mm(6.3 in), 인쇄 7–9 pt
plt.rcParams.update({"font.family": "Arial", "axes.unicode_minus": False, "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "savefig.dpi": 300, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "hatch.linewidth": 0.5})


def stats(x):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    return {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1) if len(x) > 1 else 0.0,
            "median": np.median(x), "p95": np.percentile(x, 95), "min": x.min(), "max": x.max()}


fast = pd.read_csv(R / "raw_fast.csv")
syn = pd.read_csv(R / "raw_synth_sentence.csv")
multi = pd.read_csv(R / "raw_multi_sentence.csv")
assert (syn.status == 200).all() and (multi.status == 200).all(), "failed requests present"

rows = []
for st in ["set_emotion", "resolve_one"]:
    rows.append({"group": "HTTP 왕복", "item": st, "unit": "ms", **stats(fast[fast.stage == st].ms)})
LEN_KO = {"short": "짧은 문장", "medium": "중간 문장", "long": "긴 문장"}
for L in ["short", "medium", "long"]:
    d = syn[syn.length == L]
    for col, name in [("network_ms", "edge-tts 네트워크"), ("decode_ms", "MP3 디코딩"), ("dsp_ms", "DSP(PSOLA)"),
                      ("encode_ms", "WAV 인코딩"), ("export_ms", "MP3 내보내기"), ("client_ms", "/tts 왕복(클라이언트)")]:
        rows.append({"group": f"문장 합성-{LEN_KO[L]}({d.chars.mean():.0f}자)", "item": name, "unit": "ms", **stats(d[col])})
    rows.append({"group": f"문장 합성-{LEN_KO[L]}({d.chars.mean():.0f}자)", "item": "음성 길이", "unit": "s", **stats(d.audio_sec)})
    rows.append({"group": f"문장 합성-{LEN_KO[L]}({d.chars.mean():.0f}자)", "item": "RTF(왕복/음성길이)", "unit": "-",
                 **stats(d.client_ms / 1000 / d.audio_sec)})
# D-psola(Dp) 문장 하나 합성 (추가 조건)
dp_path = R / "dp_single" / "raw_synth_sentence.csv"
if dp_path.exists():
    dps = pd.read_csv(dp_path)
    assert (dps.status == 200).all()
    for L in ["short", "medium", "long"]:
        d = dps[dps.length == L]
        g = f"문장 합성(Dp)-{LEN_KO[L]}({d.chars.mean():.0f}자)"
        for col, name in [("network_ms", "edge-tts 네트워크"), ("decode_ms", "MP3 디코딩"), ("dsp_ms", "DSP(PSOLA)"),
                          ("export_ms", "MP3 내보내기"), ("client_ms", "/tts 왕복(클라이언트)")]:
            rows.append({"group": g, "item": name, "unit": "ms", **stats(d[col])})
        rows.append({"group": g, "item": "음성 길이", "unit": "s", **stats(d.audio_sec)})
        rows.append({"group": g, "item": "RTF(왕복/음성길이)", "unit": "-", **stats(d.client_ms / 1000 / d.audio_sec)})
for n in [1, 3, 5]:
    d = multi[multi.n_sent == n]
    g = f"다문장 /tts-{n}문장"
    rows.append({"group": g, "item": "종단 간 왕복(클라이언트)", "unit": "ms", **stats(d.client_ms)})
    rows.append({"group": g, "item": "edge-tts 네트워크 합계", "unit": "ms", **stats(d.network_ms_sum)})
    rows.append({"group": g, "item": "DSP 합계", "unit": "ms", **stats(d.dsp_ms_sum)})
    rows.append({"group": g, "item": "병합·MP3 내보내기", "unit": "ms", **stats(d.merge_export_ms)})
    rows.append({"group": g, "item": "음성 길이", "unit": "s", **stats(d.audio_sec)})
    rows.append({"group": g, "item": "RTF", "unit": "-", **stats(d.client_ms / 1000 / d.audio_sec)})
summ = pd.DataFrame(rows)
summ.to_csv(R / "summary.csv", index=False, float_format="%.3f")
# 네트워크 비중
share = {L: float((syn[syn.length == L].network_ms / syn[syn.length == L].client_ms).mean()) for L in LEN_KO}
share_multi = {int(n): float((multi[multi.n_sent == n].network_ms_sum / multi[multi.n_sent == n].client_ms).mean())
               for n in [1, 3, 5]}
(R / "summary.json").write_text(json.dumps({"network_share_single": share, "network_share_multi": share_multi},
                                           ensure_ascii=False, indent=2))
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(summ[["group", "item", "unit", "n", "mean", "median", "p95", "sd"]].round(2).to_string(index=False))
print(share, share_multi)

# ── 그림 1 (전체 폭 160 mm): (a) 문장 하나 합성 단계 분해, (b) 다문장 종단 간
LEN_EN = {"short": "Short", "medium": "Medium", "long": "Long"}
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
comps = [("network_ms", "edge-tts (network)", "#4a4a4a", ""), ("decode_ms", "MP3 decoding", "#9a9a9a", "////"),
         ("dsp_ms", "Prosody DSP (PSOLA)", "#d0d0d0", "xxx"), ("export_ms", "MP3 export", "#ffffff", "...")]
xs = np.arange(3)
bottom = np.zeros(3)
for col, name, c, h in comps:
    vals = np.array([syn[syn.length == L][col].mean() for L in LEN_KO])
    ax.bar(xs, vals, 0.5, bottom=bottom, color=c, hatch=h, edgecolor="black", linewidth=0.5, label=name)
    bottom += vals
rt = [syn[syn.length == L].client_ms for L in LEN_KO]
ax.errorbar(xs + 0.36, [r.median() for r in rt],
            yerr=[[r.median() - r.quantile(0.05) for r in rt], [r.quantile(0.95) - r.median() for r in rt]],
            fmt="o", color="black", ms=3.5, capsize=2, elinewidth=0.8, label="Round trip, median (5–95th pct.)")
ax.set_xticks(xs, [f"{LEN_EN[L]}\n({syn[syn.length == L].chars.mean():.0f} chars)" for L in LEN_KO])
ax.set_xlim(-0.45, 2.65)
ax.set_ylabel("Time (ms)")
ax.set_ylim(0, 1500)
ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)
ax.grid(axis="y", lw=0.3, alpha=0.6)
ax.set_axisbelow(True)
ax.text(-0.15, 1.02, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

ax = axes[1]
ns = [1, 3, 5]
data = [multi[multi.n_sent == n].client_ms / 1000 for n in ns]
bp = ax.boxplot(data, positions=ns, widths=0.7, patch_artist=True, showfliers=True,
                flierprops={"marker": "x", "ms": 3, "mew": 0.6},
                boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
for b in bp["boxes"]:
    b.set(facecolor="#e6e6e6", edgecolor="black")
for m in bp["medians"]:
    m.set(color="black", lw=1.2)
aud = [multi[multi.n_sent == n].audio_sec.mean() for n in ns]
ax.plot(ns, aud, "s--", color="black", ms=4, mfc="white", lw=0.9, label="Generated audio length (mean)")
from matplotlib.patches import Patch
h1, l1 = ax.get_legend_handles_labels()
ax.legend(handles=[Patch(facecolor="#e6e6e6", edgecolor="black", lw=0.6, label="End-to-end /tts round trip")] + h1,
          frameon=False, loc="upper left", borderaxespad=0.2)
ax.set_xticks(ns, [f"{n} sentence" + ("s" if n > 1 else "") for n in ns])
ax.set_xlim(0, 6)
ax.set_ylim(0, 27)
ax.set_ylabel("Time (s)")
ax.grid(axis="y", lw=0.3, alpha=0.6)
ax.set_axisbelow(True)
ax.text(-0.15, 1.02, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
fig.tight_layout(w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(F / f"latency_breakdown.{ext}", dpi=300)
plt.close(fig)

# ── 그림 2 (단 폭 76 mm): 표정 이벤트 경로 HTTP 왕복
fig, ax = plt.subplots(figsize=(3.0, 1.6))
d = [fast[fast.stage == s].ms for s in ["set_emotion", "resolve_one"]]
bp = ax.boxplot(d, orientation="horizontal", widths=0.5, patch_artist=True,
                flierprops={"marker": "x", "ms": 2.5, "mew": 0.5},
                boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
for b, h in zip(bp["boxes"], ["", "////"]):
    b.set(facecolor="#e6e6e6", edgecolor="black", hatch=h)
for m in bp["medians"]:
    m.set(color="black", lw=1.2)
ax.set_yticks([1, 2], ["/set_emotion", "/emotion/\nresolve_one"])
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
ax.set_xlim(0.5, 50)
ax.set_xlabel("HTTP round-trip time (ms, log scale)")
ax.grid(axis="x", lw=0.3, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(F / f"latency_http.{ext}", dpi=300)
print("figures saved")
