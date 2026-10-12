"""Faithful offline re-implementation of the temporal stabilization in app/face/face_emotion.py
and the sentence-level dominant-emotion selection in app/tts/app.py.

The real-time loop uses wall-clock time (time.time()); here the video timestamps replace it.
Everything else (EMA, fps_smooth-adaptive majority window, final-label rule, hold-time,
send-interval, confidence threshold, history maxlen) follows the original code line by line.
"""
from collections import deque

import numpy as np

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABEL_TO_KO = {"Happy": "행복", "Sad": "슬픔", "Angry": "분노", "Disgust": "분노",
               "Surprise": "놀람", "Fear": "두려움", "Neutral": "중립", "no face": "중립"}

DEFAULTS = dict(EMA_ALPHA=0.8, WIN_SEC=0.35, CONF_THRESHOLD=0.3, MIN_HOLD_MS=100,
                MIN_SEND_INTERVAL=0.2, HIST_MAXLEN=30)


class Stabilizer:
    """One instance = one running face_emotion.py process."""

    def __init__(self, **kw):
        c = dict(DEFAULTS); c.update(kw)
        self.c = c
        self.prob_ema = None
        self.history = deque(maxlen=c["HIST_MAXLEN"])
        self.fps_smooth = 0.0
        self.t_prev = None
        self.stable_label = None
        self.stable_conf = 0.0
        self.stable_start_ms = None
        self.last_sent_label = None
        self.last_sent_time = -1e18  # original: 0.0 epoch -> effectively "long ago"
        self.events = []  # (t_ms, label, conf)

    def _maybe_send(self, label, conf, now):
        if (now - self.last_sent_time) < self.c["MIN_SEND_INTERVAL"]:
            return
        if label == self.last_sent_label:
            return
        self.events.append((int(now * 1000), label, float(conf)))
        self.last_sent_label = label
        self.last_sent_time = now

    def step(self, t, probs):
        """t: timestamp (s). probs: (7,) array or None (no face).
        Returns dict with per-frame labels of each variant (indices; -1 = no face)."""
        c = self.c
        if self.t_prev is None:
            # original: t_prev = time.time() at start; first frame arrives ~1 frame later
            self.t_prev = t - 1.0 / 30.0
        fps = 1.0 / max(1e-6, (t - self.t_prev))
        self.t_prev = t
        self.fps_smooth = self.fps_smooth * 0.9 + fps * 0.1

        out = {"raw": -1, "ema": -1, "maj": -1, "final": -1}
        if probs is not None:
            p = probs[None, :]
            if self.prob_ema is None:
                self.prob_ema = p
            else:
                self.prob_ema = c["EMA_ALPHA"] * p + (1.0 - c["EMA_ALPHA"]) * self.prob_ema
            raw_idx = int(np.argmax(p))
            self.history.append(raw_idx)
            window_len = max(1, int(c["WIN_SEC"] * max(1.0, self.fps_smooth)))
            recent = list(self.history)[-window_len:]
            vals, counts = np.unique(recent, return_counts=True)
            majority_idx = int(vals[np.argmax(counts)])
            ema_label_idx = int(np.argmax(self.prob_ema))
            final_idx = majority_idx if majority_idx == ema_label_idx else ema_label_idx
            current_label = EMOTION_LABELS[final_idx]
            current_conf = float(self.prob_ema[0, final_idx])
            out.update(raw=raw_idx, ema=ema_label_idx, maj=majority_idx, final=final_idx)
        else:
            current_label = "Neutral"
            current_conf = 0.0

        now_ms = int(t * 1000)
        if current_label == self.stable_label:
            if self.stable_start_ms is None:
                self.stable_start_ms = now_ms
            if current_conf > self.stable_conf:
                self.stable_conf = current_conf
        else:
            self.stable_label = current_label
            self.stable_conf = current_conf
            self.stable_start_ms = now_ms
        held_ms = now_ms - (self.stable_start_ms if self.stable_start_ms is not None else now_ms)
        if (held_ms >= c["MIN_HOLD_MS"] and self.stable_conf >= c["CONF_THRESHOLD"]
                and self.stable_label not in (None, "no face")):
            self._maybe_send(self.stable_label, self.stable_conf, t)
        out["emitted"] = (EMOTION_LABELS.index(self.last_sent_label)
                          if self.last_sent_label is not None else -2)  # -2 = nothing emitted yet
        return out


class EmotionServer:
    """Faithful copy of the EMO_HISTORY buffer and _segment_durations_ko/_pick_dominant_ko."""

    def __init__(self):
        self.EMO_HISTORY = deque(maxlen=2000)

    def set_emotion(self, t_ms, label, conf):
        self.EMO_HISTORY.append({"t": int(t_ms), "emo": LABEL_TO_KO.get(label, "중립"),
                                 "label": label, "tag": "", "confidence": float(conf)})

    def _segment_durations_ko(self, t0, t1, conf_thresh=0.5):
        if not self.EMO_HISTORY or t1 <= t0:
            return {}
        hist = list(self.EMO_HISTORY)
        prev = None
        for e in hist:
            if e["t"] <= t0:
                prev = e
            else:
                break
        if prev is None:
            prev = hist[0]
        prev_t = t0
        prev_emo = prev.get("emo", "중립")
        prev_conf = prev.get("confidence", 0.0)
        durs = {}
        for e in hist:
            if e["t"] <= t0:
                continue
            if e["t"] >= t1:
                break
            dt = e["t"] - prev_t
            if dt > 0 and prev_conf >= conf_thresh:
                durs[prev_emo] = durs.get(prev_emo, 0) + dt
            prev_t = e["t"]
            prev_emo = e.get("emo", "중립")
            prev_conf = e.get("confidence", 0.0)
        dt_last = t1 - prev_t
        if dt_last > 0 and prev_conf >= conf_thresh:
            durs[prev_emo] = durs.get(prev_emo, 0) + dt_last
        return durs

    def pick_dominant_ko(self, t0, t1, conf_thresh=0.5, min_ratio=0.4):
        durs = self._segment_durations_ko(t0, t1, conf_thresh)
        if not durs:
            return ""
        emo, best_ms = max(durs.items(), key=lambda kv: kv[1])
        total = max(1, t1 - t0)
        if emo == "중립" or (best_ms / total) < min_ratio:
            return ""
        return f"<{emo}>"
