"""face_emotion.py의 계산 부분(카메라·TensorFlow·MediaPipe 없이 테스트 가능한 순수 로직).

- Stabilizer   : 프레임별 7확률 → 안정화된 라벨 전환 이벤트(/set_emotion)
                 EMA 0.8, 다수결 창 0.35 s, 유지 100 ms, 신뢰도 0.3, 전송 간격 0.2 s는 졸업논문 코드와 같고,
                 "현재 후보 라벨"을 정하는 규칙만 E5에서 고른 규칙으로 바꿨다.
                   thesis    : EMA argmax (졸업논문 코드. 다수결 규칙은 결과에 영향이 없었다)
                   gate      : 다수결 == EMA argmax 일 때만 바꿈
                   hyst      : p_EMA[k] - p_EMA[현재] >= delta 일 때만 바꿈
                   gate_hyst : 두 조건 모두
                 (= experiments/E5_improvements/scripts/e5_methods.py: ImprovedStabilizer)
- ProbSummary  : 0.2 s마다 그 구간 얼굴 프레임들의 원시 확률 평균과 프레임 수(/set_emotion_probs)
                 (= e5_methods.SummaryClient)
- HSEmotion    : 공개 사전학습 분류기(ONNX) 전처리와 8→7 클래스 변환 (= e5_extract.hse_preprocess,
                 e5_analyze.load_clips)
"""
from collections import deque

import numpy as np

# 모델 출력 순서 (FER2013)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 분류기별 기본 설정 (E5 dev 배우 01–04에서 고른 값). 서버의 중립 억제 가중치는 app.py가 이름으로 고른다.
CLASSIFIER_PRESETS = {
    "cnn": {"rule": "hyst", "delta": 0.5},             # FER2013-CNN (face_emotion.h5), 서버 W_NEUTRAL 0.25
    "hsemotion": {"rule": "gate_hyst", "delta": 0.5},  # HSEmotion enet_b0_8_va_mtl, 서버 W_NEUTRAL 0.9
}
RULES = ("thesis", "gate", "hyst", "gate_hyst")


class Stabilizer:
    """한 프레임씩 step(t, probs)를 부르면 화면 표시용 상태와 보낼 이벤트(있으면)를 돌려준다."""

    def __init__(self, rule="hyst", delta=0.5, ema_alpha=0.8, win_sec=0.35, conf_threshold=0.3,
                 min_hold_ms=100, min_send_interval=0.2, hist_maxlen=30):
        if rule not in RULES:
            raise ValueError(f"unknown rule: {rule}")
        self.rule, self.delta = rule, float(delta)
        self.ema_alpha, self.win_sec = ema_alpha, win_sec
        self.conf_threshold, self.min_hold_ms, self.min_send_interval = conf_threshold, min_hold_ms, min_send_interval
        self.prob_ema = None                       # (1, 7)
        self.history = deque(maxlen=hist_maxlen)   # 30 fps 기준 1 s, 최근 win_sec만 사용
        self.fps_smooth = 0.0
        self.t_prev = None
        self.out_idx = None                        # 현재 출력 라벨(히스테리시스 상태)
        self.stable_label = None
        self.stable_conf = 0.0
        self.stable_start_ms = None
        self.held_ms = 0
        self.last_sent_label = None
        self.last_sent_time = float("-inf")

    def _choose(self, k, majority_idx):
        if self.rule == "thesis" or self.out_idx is None:
            self.out_idx = k
            return k
        pe = self.prob_ema[0]
        cur = self.out_idx
        ok = k != cur
        if self.rule in ("gate", "gate_hyst"):
            ok = ok and (majority_idx == k)
        if self.rule in ("hyst", "gate_hyst"):
            ok = ok and (pe[k] - pe[cur] >= self.delta)
        if ok:
            self.out_idx = k
        return self.out_idx

    def step(self, t, probs, none_label="Neutral"):
        """t: 초(epoch). probs: (7,) 또는 (1,7) 확률, 얼굴이 없으면 None.
        반환: {"label", "conf", "final_idx", "send"}; send는 보낼 (t_ms, label, conf) 또는 None."""
        if self.t_prev is None:
            self.t_prev = t - 1.0 / 30.0
        fps = 1.0 / max(1e-6, (t - self.t_prev))
        self.t_prev = t
        self.fps_smooth = self.fps_smooth * 0.9 + fps * 0.1

        final_idx = -1
        if probs is not None:
            p = np.asarray(probs).reshape(1, -1)
            self.prob_ema = p if self.prob_ema is None else self.ema_alpha * p + (1.0 - self.ema_alpha) * self.prob_ema
            self.history.append(int(np.argmax(p)))
            window_len = max(1, int(self.win_sec * max(1.0, self.fps_smooth)))
            recent = list(self.history)[-window_len:]
            vals, counts = np.unique(recent, return_counts=True)
            majority_idx = int(vals[np.argmax(counts)])
            final_idx = self._choose(int(np.argmax(self.prob_ema)), majority_idx)
            label, conf = EMOTION_LABELS[final_idx], float(self.prob_ema[0, final_idx])
        else:
            label, conf = none_label, 0.0

        # ── 유지 시간·신뢰도·전송 간격 (졸업논문 코드 그대로)
        now_ms = int(t * 1000)
        if label == self.stable_label:
            if self.stable_start_ms is None:
                self.stable_start_ms = now_ms
            if conf > self.stable_conf:
                self.stable_conf = conf
        else:
            self.stable_label, self.stable_conf, self.stable_start_ms = label, conf, now_ms
        self.held_ms = now_ms - (self.stable_start_ms if self.stable_start_ms is not None else now_ms)
        send = None
        if (self.held_ms >= self.min_hold_ms and self.stable_conf >= self.conf_threshold
                and self.stable_label not in (None, "no face")):
            send = self.maybe_send(self.stable_label, self.stable_conf, t)
        return {"label": label, "conf": conf, "final_idx": final_idx, "send": send}

    def maybe_send(self, label, conf, now, force=False):
        """라벨이 바뀌었고 전송 간격이 지났으면 (t_ms, label, conf)를 돌려주고 상태를 갱신."""
        if not force:
            if (now - self.last_sent_time) < self.min_send_interval:
                return None
            if label == self.last_sent_label:
                return None
        self.last_sent_label, self.last_sent_time = label, now
        return (int(now * 1000), label, float(conf))


class ProbSummary:
    """period 초마다 얼굴 프레임 원시 확률의 평균을 /set_emotion_probs payload로 만든다."""

    def __init__(self, period=0.2, classifier="cnn", labels=EMOTION_LABELS):
        self.period, self.classifier, self.labels = period, classifier, list(labels)
        self.t_start = None
        self.acc = np.zeros(len(self.labels), np.float64)
        self.n = 0

    def step(self, t, probs):
        """t: 초(epoch), probs: (7,)/(1,7) 또는 None. 구간이 끝났고 얼굴 프레임이 있었으면 payload, 아니면 None."""
        if self.t_start is None:
            self.t_start = t
        if probs is not None:
            self.acc += np.asarray(probs, np.float64).reshape(-1)
            self.n += 1
        if t - self.t_start >= self.period - 1e-9:
            return self.flush(t)
        return None

    def flush(self, t):
        out = None
        if self.n > 0:
            out = {"t0": int(self.t_start * 1000), "t1": int(t * 1000), "n": self.n,
                   "probs": (self.acc / self.n).astype(np.float32).tolist(),
                   "labels": self.labels, "classifier": self.classifier}
        self.t_start, self.acc, self.n = t, np.zeros(len(self.labels), np.float64), 0
        return out


# ───────────────────────── HSEmotion (선택) ─────────────────────────
HSE_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
HSE_STD = np.array([0.229, 0.224, 0.225], np.float32)
# HSE 로짓 (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise) → EMOTION_LABELS 순서
HSE_TO_OURS = [0, 2, 3, 4, 6, 7, 5]


def hse_preprocess(face_bgr, size=224):
    """hsemotion-onnx preprocess와 같음: BGR→RGB, 224×224 bilinear, /255, ImageNet 정규화, NCHW."""
    import cv2
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    x = (x - HSE_MEAN) / HSE_STD
    return x.transpose(2, 0, 1)[None]


def hse_logits_to_probs(logits):
    """(N, 8 또는 10) 로짓 → (N, 7) 확률. Contempt는 버리고 남은 7개에 softmax (valence/arousal 무시)."""
    L = np.asarray(logits, np.float32)[:, :8][:, HSE_TO_OURS]
    e = np.exp(L - L.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


class HSEmotionClassifier:
    """onnxruntime CPU로 enet_b0_8_va_mtl.onnx 실행. __call__(face_bgr) → (1, 7) 확률."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, face_bgr):
        logits = self.sess.run(None, {self.input_name: hse_preprocess(face_bgr)})[0]
        return hse_logits_to_probs(logits)
