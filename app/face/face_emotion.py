# face_emotion.py (label transitions + probability summaries to the STAGE server)
# - /set_emotion       : stable label transitions (UI display / fallback tagging), LABEL ONLY
#                        EMA + hysteresis (E5) + hold-time; 'tag' is always "" (app.py maps label -> Korean)
# - /set_emotion_probs : every 0.2 s, mean raw 7-class probabilities of that interval (sentence tagging, E5)
# - Classifier: FER2013-CNN (face_emotion.h5, default) or HSEmotion ONNX (--classifier hsemotion)
# Pure logic (stabilizer, summaries, HSEmotion pre/post-processing) lives in emotion_core.py.
#
# Usage:
#   python face_emotion.py                      # webcam 0 (OS별 기본 백엔드 자동 선택)
#   python face_emotion.py --camera 1           # 다른 카메라
#   python face_emotion.py --video clip.mp4     # 웹캠 대신 영상 파일 (데모/테스트)
#   python face_emotion.py --video clip.mp4 --realtime --headless --server http://127.0.0.1:5000
#   python face_emotion.py --classifier hsemotion   # data/models/hsemotion/enet_b0_8_va_mtl.onnx 필요

import argparse
import os
import platform
import sys
import time
import cv2
import numpy as np
import mediapipe as mp

from emotion_core import CLASSIFIER_PRESETS, RULES, HSEmotionClassifier, ProbSummary, Stabilizer

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HSE_ONNX = os.path.join(_HERE, "..", "..", "data", "models", "hsemotion", "enet_b0_8_va_mtl.onnx")

# =========================
# CLI
# =========================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="STAGE real-time facial emotion sender")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: CAM_INDEX=0)")
    ap.add_argument("--video", type=str, default=None, help="video file path instead of webcam")
    ap.add_argument("--realtime", action="store_true",
                    help="with --video: throttle playback to the file's native FPS")
    ap.add_argument("--headless", action="store_true", help="no preview window (cv2.imshow off)")
    ap.add_argument("--server", type=str, default=None, help="Flask server URL (overrides discovery)")
    ap.add_argument("--no-send", action="store_true", help="do not POST to the server (local test)")
    ap.add_argument("--classifier", choices=sorted(CLASSIFIER_PRESETS), default="cnn",
                    help="cnn = FER2013 CNN (face_emotion.h5, default); hsemotion = HSEmotion ONNX")
    ap.add_argument("--hse-model", type=str, default=DEFAULT_HSE_ONNX,
                    help="HSEmotion ONNX path (default: data/models/hsemotion/enet_b0_8_va_mtl.onnx)")
    ap.add_argument("--rule", choices=RULES, default=None,
                    help="label rule (default per classifier: cnn=hyst, hsemotion=gate_hyst; thesis = original code)")
    ap.add_argument("--delta", type=float, default=None, help="hysteresis delta (default 0.5)")
    return ap.parse_args(argv)

ARGS = parse_args()

# =========================
# Server discovery & sender
# =========================
DEFAULT_CANDIDATES = [
    os.getenv("ORPHEUS_URL", ""),              # highest priority if set
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://host.docker.internal:5000",
]

try:
    import requests
    HAVE_REQ = True
except Exception:
    HAVE_REQ = False
    import json as _json
    import urllib.request as _ur

def _try_ping(url: str, timeout=0.3) -> bool:
    try:
        if HAVE_REQ:
            r = requests.get(url + "/get_config", timeout=timeout)
            return r.ok
        else:
            req = _ur.Request(url + "/get_config")
            with _ur.urlopen(req, timeout=timeout) as _:
                return True
    except Exception:
        return False

def choose_server_url() -> str:
    tried = []
    for cand in DEFAULT_CANDIDATES:
        if not cand:
            continue
        ok = _try_ping(cand, timeout=0.3)
        tried.append((cand, ok))
        if ok:
            return cand
    # fallback: first non-empty candidate
    for cand, _ in tried:
        if cand:
            return cand
    return "http://127.0.0.1:5005"

ORPHEUS_URL = ARGS.server.rstrip("/") if ARGS.server else choose_server_url()

def post_json(url, payload, timeout=0.3):
    if ARGS.no_send:
        return
    try:
        if HAVE_REQ:
            requests.post(url, json=payload, timeout=timeout)
        else:
            data = _json.dumps(payload).encode("utf-8")
            req = _ur.Request(url, data=data, headers={"Content-Type": "application/json"})
            _ur.urlopen(req, timeout=timeout).read()
    except Exception:
        # suppress network errors (best-effort sender)
        pass

# ================
# Config
# ================
CAM_INDEX = 0                # 0: 기본/드로이드캠
FRAME_W, FRAME_H = 640, 480
WIN_SEC = 0.35              # majority window(sec)
EMA_ALPHA = 0.8             # prob EMA
CONF_THRESHOLD = 0.3        # 최소 신뢰도(테스트 편의상 살짝 낮춤)
MODEL_PATH = "face_emotion.h5"

# Transition sending
MIN_HOLD_MS = 100           # 같은 label이 이 시간 이상 유지될 때만 전송
MIN_SEND_INTERVAL = 0.2     # 연속 전송 최소 간격(sec)
VERBOSE_LOG = True          # 전송/연결/디버그 로그

SUMMARY_SEC = 0.2           # 확률 요약 전송 주기(sec) = MIN_SEND_INTERVAL

# Label rule (E5): CNN = hyst δ=0.5, HSEmotion = gate_hyst δ=0.5
_preset = CLASSIFIER_PRESETS[ARGS.classifier]
RULE = ARGS.rule or _preset["rule"]
HYST_DELTA = _preset["delta"] if ARGS.delta is None else ARGS.delta

# =========================
# Utils
# =========================
def safe_crop(frame, x, y, w, h):
    H, W = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return frame[y1:y2, x1:x2]

def preprocess_face(face_bgr):
    """Model input: 48x48 grayscale, scaled to [0,1]."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1,48,48,1)
    return arr

# =========================
# Load model → classify(face_bgr) returns (1, 7) probs in EMOTION_LABELS order
# =========================
if ARGS.classifier == "hsemotion":
    if not os.path.exists(ARGS.hse_model):
        sys.exit(f"HSEmotion ONNX 파일이 없습니다: {ARGS.hse_model} (experiments/E5_improvements/README.md 8절 참고)")
    try:
        classify = HSEmotionClassifier(ARGS.hse_model)
    except ImportError:
        sys.exit("HSEmotion에는 onnxruntime이 필요합니다: pip install onnxruntime")
else:
    from tensorflow.keras.models import load_model
    _model_file = MODEL_PATH
    if not os.path.exists(_model_file):
        # 다른 폴더에서 실행해도 스크립트 옆의 모델을 찾도록
        _model_file = os.path.join(_HERE, MODEL_PATH)
    if not os.path.exists(_model_file):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    model = load_model(_model_file)

    def classify(face_bgr):
        # model.predict 대신 직접 호출 (E1: 프레임당 약 26 ms → 4 ms, 출력 동일)
        return model(preprocess_face(face_bgr), training=False).numpy()

# =========================
# MediaPipe detect
# =========================
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# =========================
# Video capture
# =========================
def camera_backend():
    """OS별 카메라 백엔드: Windows=MSMF, macOS=AVFoundation, 그 외=OpenCV 기본."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_MSMF
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY

if ARGS.video:
    if not os.path.exists(ARGS.video):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {ARGS.video}")
    cap = cv2.VideoCapture(ARGS.video)
    VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
else:
    cam_index = CAM_INDEX if ARGS.camera is None else ARGS.camera
    cap = cv2.VideoCapture(cam_index, camera_backend())
    if not cap.isOpened():  # 지정 백엔드 실패 시 기본 백엔드로 재시도
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit(f"카메라 {cam_index}를 열 수 없습니다. (macOS: 시스템 설정 > 개인정보 보호 > 카메라 권한 확인)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    VIDEO_FPS = None

# =========================
# State
# =========================
stab = Stabilizer(rule=RULE, delta=HYST_DELTA, ema_alpha=EMA_ALPHA, win_sec=WIN_SEC,
                  conf_threshold=CONF_THRESHOLD, min_hold_ms=MIN_HOLD_MS, min_send_interval=MIN_SEND_INTERVAL)
summary = ProbSummary(period=SUMMARY_SEC, classifier=ARGS.classifier)

if VERBOSE_LOG:
    print(f"[info] Orpheus server: {ORPHEUS_URL}")
    print(f"[info] Source: {'video ' + ARGS.video if ARGS.video else 'camera'}")
    print(f"[info] Classifier: {ARGS.classifier}  rule: {RULE}  delta: {HYST_DELTA}")
    if not ARGS.headless:
        print("[info] Press 'e' to force-send current stable label, 'q' to quit.")

def send_transition(event):
    """event = (t_ms, label, conf) from Stabilizer → POST /set_emotion."""
    if event is None:
        return
    t_ms, label, conf = event
    payload = {
        "t": t_ms,              # epoch ms
        "label": label,         # e.g., "Happy"
        "tag": "",              # ← always empty (no <laugh>/<gasp> etc.)
        "confidence": float(conf)
    }
    post_json(f"{ORPHEUS_URL}/set_emotion", payload, timeout=0.3)
    if VERBOSE_LOG:
        print(f"[send] {payload}")

def send_summary(payload):
    """0.2 s probability summary → POST /set_emotion_probs (sentence tagging)."""
    if payload is not None:
        post_json(f"{ORPHEUS_URL}/set_emotion_probs", payload, timeout=0.3)

while True:
    t_frame_start = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    # Detect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    probs = None                # (1, 7) raw probabilities of this frame, None = no face
    none_label = "Neutral"      # no face → neutral candidate (never sent: conf 0)
    box = None

    if results.detections:
        # pick largest face
        h, w = frame.shape[:2]
        best = None
        best_area = -1
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w); y = int(bbox.ymin * h)
            bw = int(bbox.width * w); bh = int(bbox.height * h)
            area = max(0, bw) * max(0, bh)
            if area > best_area:
                best_area = area
                best = (x, y, bw, bh)
        x, y, bw, bh = best
        roi = safe_crop(frame, x, y, bw, bh)

        if roi is not None:
            x1, y1 = max(0, x), max(0, y)
            box = (x1, y1, x1 + roi.shape[1], y1 + roi.shape[0])
            probs = classify(roi)   # (1, 7)
        else:
            none_label = "no face"  # face too small to crop → never sent

    # =========================
    # Stabilization (EMA + label rule + hold-time + send interval) & probability summary
    # =========================
    t_now = time.time()
    st = stab.step(t_now, probs, none_label=none_label)
    send_transition(st["send"])
    send_summary(summary.step(t_now, probs))

    if box is not None:
        # draw bbox & label
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        disp = f"{st['label']} ({st['conf']:.2f})"
        cv2.putText(frame, disp, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # =========================
    # HUD (debug)
    # =========================
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {stab.fps_smooth:.1f}  [{ARGS.classifier}/{RULE}]", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    hud2 = f"Stable: {stab.stable_label or 'None'}  conf:{stab.stable_conf:.2f}  held:{stab.held_ms}ms"
    cv2.putText(frame, hud2, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2)

    if ARGS.video and ARGS.realtime and VIDEO_FPS:
        # 영상 파일을 원래 속도로 재생 (타임스탬프가 실제 시간과 맞도록)
        wait = (1.0 / VIDEO_FPS) - (time.time() - t_frame_start)
        if wait > 0:
            time.sleep(wait)

    if ARGS.headless:
        continue

    cv2.imshow("Emotion (press e: force send, q: quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        # force send current stable state for quick end-to-end testing
        label_to_send = stab.stable_label or st["label"]
        send_transition(stab.maybe_send(label_to_send, max(stab.stable_conf, 0.0), time.time(), force=True))

cap.release()
if not ARGS.headless:
    cv2.destroyAllWindows()
