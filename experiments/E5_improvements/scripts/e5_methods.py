"""E5 methods: improved (training-free) stabilization and sentence tagging.

Baseline (thesis code, faithful) = experiments/E2_emotion-stabilization/scripts/stabilizer.py
  Stabilizer (EMA 0.8 + dead majority rule + hold 100 ms + conf>=0.3 + send interval 0.2 s)
  EmotionServer.pick_dominant_ko(conf_thresh=0.5, min_ratio=0.4)  (duration of emitted events)

Improved stabilizer (ImprovedStabilizer):
  rule="gate"      : output label changes only when majority == EMA argmax (else keep previous output)
  rule="hyst"      : switch from current output c to EMA argmax k only if p_k - p_c >= delta
  rule="gate_hyst" : both conditions
  The hold-time / conf / send-interval logic after the rule is unchanged from the thesis code.
  Each emitted event additionally carries the current EMA probability vector.

Improved tagging (ProbSummaryServer):
  The face client sends, every SUMMARY_SEC (= MIN_SEND_INTERVAL, 0.2 s), the mean per-frame
  probability vector over the frames of that interval (face frames only). The server integrates
  the probability mass over the sentence window [t0, t1] (overlap-weighted), merges 7->6 classes
  (Disgust -> Angry, as LABEL_TO_KO), multiplies the Neutral mass by w_neutral (<1 = neutral
  suppression, compensates the CNN's neutral bias), and returns argmax; "" (= default/neutral)
  only if Neutral wins or the winning share < r_min.
"""
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "E2_emotion-stabilization", "scripts"))
from stabilizer import DEFAULTS, EMOTION_LABELS, LABEL_TO_KO, EmotionServer, Stabilizer  # noqa: F401

KO6 = ['분노', '두려움', '행복', '슬픔', '놀람', '중립']
KO2EN = {'분노': 'Angry', '두려움': 'Fear', '행복': 'Happy', '슬픔': 'Sad', '놀람': 'Surprise', '중립': 'Neutral'}
# 7 (EMOTION_LABELS order) -> 6 (KO6 order); Disgust merged into Angry
M76 = np.zeros((7, 6), np.float32)
for i, lab in enumerate(EMOTION_LABELS):
    M76[i, KO6.index(LABEL_TO_KO[lab])] = 1.0
NEUTRAL6 = KO6.index('중립')


class ImprovedStabilizer(Stabilizer):
    def __init__(self, rule="gate", delta=0.0, **kw):
        super().__init__(**kw)
        self.rule, self.delta = rule, delta
        self.out_idx = None
        self.event_probs = []  # EMA prob vector attached to each emitted event

    def _maybe_send(self, label, conf, now):
        n = len(self.events)
        super()._maybe_send(label, conf, now)
        if len(self.events) > n:
            self.event_probs.append(self.prob_ema[0].copy())

    def step(self, t, probs):
        c = self.c
        if self.t_prev is None:
            self.t_prev = t - 1.0 / 30.0
        fps = 1.0 / max(1e-6, (t - self.t_prev))
        self.t_prev = t
        self.fps_smooth = self.fps_smooth * 0.9 + fps * 0.1
        out = {"raw": -1, "ema": -1, "maj": -1, "final": -1}
        if probs is not None:
            p = probs[None, :]
            self.prob_ema = p if self.prob_ema is None else c["EMA_ALPHA"] * p + (1 - c["EMA_ALPHA"]) * self.prob_ema
            raw_idx = int(np.argmax(p))
            self.history.append(raw_idx)
            window_len = max(1, int(c["WIN_SEC"] * max(1.0, self.fps_smooth)))
            recent = list(self.history)[-window_len:]
            vals, counts = np.unique(recent, return_counts=True)
            maj = int(vals[np.argmax(counts)])
            k = int(np.argmax(self.prob_ema))
            pe = self.prob_ema[0]
            if self.out_idx is None:
                self.out_idx = k
            else:
                cur = self.out_idx
                ok = k != cur
                if self.rule in ("gate", "gate_hyst"):
                    ok = ok and (maj == k)
                if self.rule in ("hyst", "gate_hyst"):
                    ok = ok and (pe[k] - pe[cur] >= self.delta)
                if ok:
                    self.out_idx = k
            final_idx = self.out_idx
            current_label = EMOTION_LABELS[final_idx]
            current_conf = float(pe[final_idx])
            out.update(raw=raw_idx, ema=k, maj=maj, final=final_idx)
        else:
            current_label, current_conf = "Neutral", 0.0
        # ---- unchanged thesis logic: hold-time, confidence threshold, send interval
        now_ms = int(t * 1000)
        if current_label == self.stable_label:
            if self.stable_start_ms is None:
                self.stable_start_ms = now_ms
            if current_conf > self.stable_conf:
                self.stable_conf = current_conf
        else:
            self.stable_label, self.stable_conf, self.stable_start_ms = current_label, current_conf, now_ms
        held_ms = now_ms - (self.stable_start_ms if self.stable_start_ms is not None else now_ms)
        if held_ms >= c["MIN_HOLD_MS"] and self.stable_conf >= c["CONF_THRESHOLD"]:
            self._maybe_send(self.stable_label, self.stable_conf, t)
        out["emitted"] = EMOTION_LABELS.index(self.last_sent_label) if self.last_sent_label is not None else -2
        return out


class SummaryClient:
    """Client-side accumulator: every period seconds, emits (t_start_ms, t_end_ms, mean_probs7, n_frames)."""

    def __init__(self, period=0.2):
        self.period = period
        self.t_start = None
        self.acc = np.zeros(7, np.float64)
        self.n = 0
        self.out = []

    def step(self, t, probs):
        if self.t_start is None:
            self.t_start = t
        if probs is not None:
            self.acc += probs; self.n += 1
        if t - self.t_start >= self.period - 1e-9:
            self.flush(t)

    def flush(self, t):
        if self.n > 0:
            self.out.append((int(self.t_start * 1000), int(t * 1000), (self.acc / self.n).astype(np.float32), self.n))
        self.t_start, self.acc, self.n = t, np.zeros(7, np.float64), 0


class ProbSummaryServer:
    def __init__(self, maxlen=2000):
        self.H = deque(maxlen=maxlen)

    def add(self, s):
        self.H.append(s)

    def mass6(self, t0, t1):
        m = np.zeros(6, np.float64)
        for (a, b, p, n) in self.H:
            ov = min(b, t1) - max(a, t0)
            if ov > 0 and b > a:
                m += (ov / (b - a)) * n * (p @ M76)  # frame-count weighted mass inside the window
        return m

    def pick(self, t0, t1, w_neutral=1.0, r_min=0.0):
        m = self.mass6(t0, t1)
        if m.sum() <= 0:
            return "", m
        m = m.copy(); m[NEUTRAL6] *= w_neutral
        k = int(np.argmax(m))
        if k == NEUTRAL6 or m[k] / m.sum() < r_min:
            return "", m
        return f"<{KO6[k]}>", m
