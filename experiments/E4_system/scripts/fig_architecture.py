"""시스템 구조도 (논문 그림, 전체 폭 160 mm, 그림 안 글자 영문, 제목 없음)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Patch
from matplotlib.lines import Line2D

F = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "svg.fonttype": "none"})

W, H = 6.3, 3.5
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# 그룹 색(흑백에서도 명도 차이): 표정 클라이언트 / 서버 / 브라우저
C_FACE, C_SRV, C_WEB = "#cddbea", "#f6efe3", "#b9d9c3"
EDGE = "#333333"
FS, FS_S = 7, 6.5


def box(cx, cy, w, h, text, fc, ls="-", fs=FS, bold_first=False, lw=0.7):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                       fc=fc, ec=EDGE, lw=lw, ls=ls)
    ax.add_patch(p)
    if bold_first:
        first, rest = text.split("\n", 1)
        ax.text(cx, cy + h / 2 - 0.07, first, ha="center", va="top", fontsize=fs, fontweight="bold")
        ax.text(cx, cy - 0.07, rest, ha="center", va="center", fontsize=fs, linespacing=1.15)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.15)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def arrow(x0, y0, x1, y1, cs="arc3", lw=0.9, color=EDGE):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=lw, color=color,
                                shrinkA=0, shrinkB=0, connectionstyle=cs))


def label(x, y, t, ha="center", va="center", fs=FS_S, style="italic", bg=True):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, style=style, linespacing=1.1, color="#222222",
            bbox=dict(fc="white", ec="none", pad=0.6) if bg else None)


Y1, Y2, Y3 = 2.62, 1.75, 0.82        # 윗줄, 가운데(서버 판정), 아랫줄
h1 = 0.62
# ── 윗줄: 표정 인식 클라이언트
b1 = box(1.55, Y1, 0.58, h1, "Webcam\n(or video file)", C_FACE)
b2 = box(2.36, Y1, 0.86, h1, "MediaPipe\nface detection", C_FACE)
b3 = box(3.32, Y1, 0.88, h1, "CNN classifier\n(FER2013,\n7 classes)", C_FACE)
b4 = box(4.31, Y1, 0.92, h1, "Temporal\nstabilization\n(EMA + majority\n+ hold-time)", C_FACE)
for a, b in [(b1, b2), (b2, b3), (b3, b4)]:
    arrow(a[2], Y1, b[0], Y1)
# ── 오른쪽 열: 서버 → 편집기
XR, WR = 5.76, 0.94
b5 = box(XR, Y1, WR, h1, "Event buffer\n(label, confidence,\ntimestamp)", C_SRV)
arrow(b4[2], Y1, b5[0], Y1)
label((b4[2] + b5[0]) / 2, Y1 + h1 / 2 + 0.16, "HTTP POST\n/set_emotion", fs=FS_S)
b6 = box(XR, Y2, WR, 0.54, "Sentence-level\ndominant emotion\n(longest-held span)", C_SRV)
arrow(XR, b5[1], XR, b6[3])
label(XR - WR / 2 - 0.06, (b5[1] + b6[3]) / 2, "span of the sentence being typed\n(since, until) at . ! ?", ha="right", fs=FS_S)
b7 = box(XR, Y3, WR, 0.66, "Web editor\nemotion tag\n<Happy> inserted", C_WEB, bold_first=True)
arrow(XR, b6[1], XR, b7[3])
label(XR - WR / 2 - 0.06, (b6[1] + b7[3]) / 2, "POST /emotion/resolve_one → tag", ha="right", fs=FS_S)

# ── 아랫줄 (오른쪽 → 왼쪽): 합성
h3 = 0.80
b8 = box(4.78, Y3, 0.62, h3, "POST /tts\nper-sentence\nsynthesis", C_SRV)
arrow(b7[0], Y3, b8[2], Y3)
# TTS backend: 바깥 상자 + 두 선택지(점선 = 외부 온라인 서비스)
bx, bw = 3.68, 1.18
outer = box(bx, Y3, bw, h3, "", C_SRV)
ax.text(bx, Y3 + h3 / 2 - 0.06, "TTS backend", ha="center", va="top", fontsize=FS, fontweight="bold")
box(bx, Y3 + 0.08, bw - 0.12, 0.22, "OpenAI gpt-4o-mini-tts\n+ instructions", "white", ls=(0, (3, 1.5)), fs=6)
ax.text(bx, Y3 - 0.09, "or", ha="center", va="center", fontsize=6, style="italic")
box(bx, Y3 - 0.26, bw - 0.12, 0.22, "edge-tts\n+ prosody preset", "white", ls=(0, (3, 1.5)), fs=6)
arrow(b8[0], Y3, outer[2], Y3)
b10 = box(2.39, Y3, 1.0, h3, "PSOLA prosody\npost-processing:\nF0 mean, range,\ncontour, tempo,\nintensity (scaled by k)", C_SRV)
arrow(outer[0], Y3, b10[2], Y3)
b11 = box(1.36, Y3, 0.66, h3, "Merged\naudio\n(0.25 s\npauses)", C_SRV)
arrow(b10[0], Y3, b11[2], Y3)

# ── 창작자 (루프)
cx, cy = 0.45, 1.62
ax.add_patch(Circle((cx, cy + 0.26), 0.1, fc="#666666", ec="none"))
ax.add_patch(FancyBboxPatch((cx - 0.17, cy - 0.12), 0.34, 0.24, boxstyle="round,pad=0,rounding_size=0.11",
                            fc="#666666", ec="none"))
ax.text(cx, cy - 0.23, "Creator", ha="center", va="top", fontsize=7.5, fontweight="bold")
# 표정 → 웹캠
arrow(cx, cy + 0.42, b1[0], Y1, cs="angle,angleA=90,angleB=180,rad=0")
label(cx + 0.02, Y1 + 0.17, "makes\nexpressions", ha="left", fs=FS_S, bg=False)
# 듣기 ← 합성 음성
arrow(b11[0], Y3, cx, cy - 0.45, cs="angle,angleA=180,angleB=-90,rad=0")
label(cx + 0.02, Y3 - 0.2, "listens", ha="left", fs=FS_S, bg=False)
# 대본 작성·강도 조절 → 편집기 (아래 레일)
yr = 0.17
ax.plot([cx - 0.25, cx - 0.25, XR], [cy - 0.1, yr, yr], color=EDGE, lw=0.9, solid_capstyle="butt")
ax.plot([cx - 0.25, cx - 0.05], [cy - 0.1, cy - 0.1], color=EDGE, lw=0.9)
arrow(XR, yr, XR, b7[1])
label(3.1, yr, "writes the script (. ! ? triggers tagging), adjusts emotion intensity k, edits tags", fs=FS_S)

# ── 범례
lx, ly = 1.25, 3.36
items = [Patch(fc=C_FACE, ec=EDGE, lw=0.7, label="Face client (local)"),
         Patch(fc=C_SRV, ec=EDGE, lw=0.7, label="Flask server (local)"),
         Patch(fc=C_WEB, ec=EDGE, lw=0.7, label="Web editor (browser)"),
         Patch(fc="white", ec=EDGE, lw=0.7, ls=(0, (3, 1.5)), label="External online TTS service")]
ax.legend(handles=items, loc="upper left", bbox_to_anchor=(lx / W, (ly + 0.12) / H), ncol=4, frameon=False,
          fontsize=FS_S, handlelength=1.4, handleheight=0.9, columnspacing=1.2, borderaxespad=0)

for ext in ("png", "pdf", "svg"):
    fig.savefig(F / f"fig_architecture.{ext}", dpi=300)
print("saved")
