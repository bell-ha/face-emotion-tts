"""STAGE TTS 서버 (Flask).

- 얼굴 감정 이벤트 수신(/set_emotion), 문장 구간 지배 감정 계산(/emotion/resolve_one)
- 문장별 감정 태그(<행복> 등) 파싱 후 문장마다 감정 프리셋으로 합성하여 이어 붙임(/tts)
- TTS 백엔드: OPENAI_API_KEY가 있으면 gpt-4o-mini-tts(+instructions), 없으면 edge (무료 온라인 TTS + 로컬 운율 후처리)
  (TTS_BACKEND=openai|edge|auto 로 강제 가능, backends.py 참고)
"""
from flask import Flask, request, send_file, render_template, jsonify
from io import BytesIO
import json, os, re, string, time
from collections import deque
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
except Exception:
    pass

from backends import make_backend

# ───────────────────────────── 선택적 오디오 병합/후처리 ─────────────────────────────
try:
    from pydub import AudioSegment
    try:
        from pydub.utils import which
        FFMPEG_OK = bool(which("ffmpeg")) and bool(which("ffprobe"))
    except Exception:
        FFMPEG_OK = False
    PydubAvailable = True
except Exception:
    PydubAvailable = False
    FFMPEG_OK = False

# 출력 포맷 자동 결정: ffmpeg 있으면 mp3, 없으면 wav
OUTPUT_FORMAT = "mp3" if (PydubAvailable and FFMPEG_OK) else "wav"
MIMETYPE = "audio/mpeg" if OUTPUT_FORMAT == "mp3" else "audio/wav"
OUT_EXT = "mp3" if OUTPUT_FORMAT == "mp3" else "wav"

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))

# ───────────────────────── 얼굴 감정 이벤트(메모리 버퍼) ───────────────────────────
# face_emotion.py → POST /set_emotion 로 들어오는 이벤트 저장
# 예: {"t": 1723871123456, "label": "Happy", "tag": "<laugh>", "confidence": 0.82}
EMO_HISTORY = deque(maxlen=2000)

# 영문 라벨 → 한국어 6감정 (요청대로 Disgust→분노)
LABEL_TO_KO = {
    "Happy": "행복",
    "Sad": "슬픔",
    "Angry": "분노",
    "Disgust": "분노",
    "Surprise": "놀람",
    "Fear": "두려움",
    "Neutral": "중립",
    "no face": "중립",
}

# 원 태그 → 한국어 감정 (라벨 없을 때 폴백)
TAG_TO_KO = {
    "<laugh>": "행복",
    "<sigh>": "슬픔",
    "<gasp>": "놀람",   # Surprise/Fear 공통 태그였으나 기본값은 놀람으로
    "<groan>": "분노",
    "<cough>": "분노",  # Disgust 매핑
    "": "중립",
}

def _event_to_ko(label: str, tag: str) -> str:
    if label in LABEL_TO_KO:
        return LABEL_TO_KO[label]
    return TAG_TO_KO.get(tag or "", "중립")

def _segment_durations_ko(t0: int, t1: int, conf_thresh: float = 0.5) -> dict:
    """[t0,t1] 구간 내 한국어 감정별 누적 지속(ms) 계산"""
    if not EMO_HISTORY or t1 <= t0:
        return {}
    hist = list(EMO_HISTORY)

    # 구간 시작 직전 상태
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

    # 꼬리
    dt_last = t1 - prev_t
    if dt_last > 0 and prev_conf >= conf_thresh:
        durs[prev_emo] = durs.get(prev_emo, 0) + dt_last
    return durs

def _pick_dominant_ko(t0: int, t1: int, conf_thresh: float = 0.5, min_ratio: float = 0.4) -> str:
    durs = _segment_durations_ko(t0, t1, conf_thresh)
    if not durs:
        return ""
    emo, best_ms = max(durs.items(), key=lambda kv: kv[1])
    total = max(1, t1 - t0)
    if emo == "중립" or (best_ms / total) < min_ratio:
        return ""
    return f"<{emo}>"

# ───────────────────────────── 감정별 프리셋(여성 화자) ─────────────────────────────
# 허용 음성: alloy, echo, fable, onyx, nova, shimmer, coral, verse, ballad, ash, sage
EMOTION_PRESETS = {
    "행복": {
        "instruction": (
            "같은 여성 화자(한 사람)로 유지하되, 매우 행복한 감정이 또렷하게 들리도록 "
            "밝은 피치와 높은 에너지로, 리듬을 가볍고 경쾌하게 말해. "
            "호흡은 짧고 가볍게. 억양은 위쪽으로 열어. "
            "추가 감탄사나 웃음소리 없이, 오직 따옴표 안 문장만."
        ),
        "voice": "shimmer",
        "speed": 1.22,
    },
    "슬픔": {
        "instruction": (
            "같은 여성 화자(한 사람)로 유지하되, 깊은 슬픔이 분명하게 느껴지도록 "
            "낮은 피치와 낮은 에너지로, 속도는 느리게 말해. "
            "호흡은 길고 무겁되 소리를 추가하지는 마. 억양 폭을 줄여 평탄하게. "
            "추가 멘트 없이 오직 따옴표 안 문장만."
        ),
        "voice": "shimmer",
        "speed": 0.85,
    },
    "분노": {
        "instruction": (
            "같은 여성 화자(한 사람)로 유지하되, 분노가 명확하게 들리도록 "
            "낮은 피치와 강한 어택으로 선명하게 발음해. "
            "호흡은 짧고 단단하게, 문장 내 쉼을 줄이고 밀도 있게. "
            "추가 멘트 없이 오직 따옴표 안 문장만."
        ),
        "voice": "shimmer",
        "speed": 1.15,
    },
    "놀람": {
        "instruction": (
            "같은 여성 화자(한 사람)로 유지하되, 놀람이 즉시 느껴지도록 "
            "시작을 높은 피치와 강한 어택으로, 이후 살짝 안정시키며 말해. "
            "호흡은 짧고 반응적이되 소리 추가는 금지. "
            "추가 멘트 없이 오직 따옴표 안 문장만."
        ),
        "voice": "shimmer",
        "speed": 1.25,
    },
    "두려움": {
        "instruction": (
            "같은 여성 화자(한 사람)로 유지하되, 두려움이 분명히 느껴지도록 "
            "낮은 피치에 약간 위축된 톤으로 작게 말해. "
            "속도는 조금 느리게, 단어 사이 간격을 아주 약간 늘려 긴장을 만들되 "
            "추가 소리(한숨/떨림 효과)는 넣지 마. "
            "추가 멘트 없이 오직 따옴표 안 문장만."
        ),
        "voice": "shimmer",
        "speed": 0.92,
    },
    "중립": {
        "instruction": (
            "같은 여성 화자(한 사람)로 유지하되, 감정 개입 없이 중립적으로 "
            "중간 피치와 표준 속도로 담담하게 말해. "
            "억양 과장 금지. 추가 멘트 없이 오직 따옴표 안 문장만."
        ),
        "voice": "shimmer",
        "speed": 1.0,
    },
}
VALID_EMOTIONS = set(EMOTION_PRESETS.keys())


# ───────────────────────────── TTS 백엔드 ─────────────────────────────
BACKEND = make_backend(EMOTION_PRESETS, response_format=OUTPUT_FORMAT)
USE_REWRITE = (BACKEND.name == "openai") and os.getenv("FACEVOX_REWRITE", "1") != "0"


def preset_for_emotion(emotion: str):
    e = (emotion or "").strip()
    preset = EMOTION_PRESETS.get(e, EMOTION_PRESETS["중립"])
    print(f"[emotion] '{e or '중립'}' -> backend={BACKEND.name}")
    return preset["instruction"], preset["voice"], float(preset["speed"])

# ───────────────────────────── LLM 연출(부호/쉼만 조정, OpenAI 모드 전용) ─────────────
PUNCTS = set(string.punctuation) | {"…", "—", "“", "”", "‘", "’"}
def _normalize_letters(s: str) -> str:
    return "".join(ch for ch in (s or "") if (not ch.isspace()) and (ch not in PUNCTS))

def rewrite_with_instruction(text: str, instruction: str) -> str:
    """단어는 그대로, 부호/쉼/리듬만 조정(낭독용 텍스트만 반환). 키가 없으면 원문 그대로."""
    if not text or not USE_REWRITE:
        return text
    sys_msg = (
        "너는 내레이션 연출가다. 주어진 한국어 문장의 '단어'는 절대 바꾸지 말고 "
        "문장부호(쉼표, 마침표, 말줄임표 …, 대시 —, 줄바꿈)와 공백만 조정해 감정을 표현하라. "
        "단어 추가/삭제/치환 금지. 이모지/효과음/설명 금지. 결과 텍스트만 출력."
    )
    user = (
        "[감정 연출 지시]\n" + instruction + "\n\n"
        "[원문]\n" + text + "\n\n"
        "[요청]\n- 단어 유지\n- 부호/쉼/리듬만 조정\n- 결과 텍스트만 출력"
    )
    try:
        r = BACKEND.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        out = (r.choices[0].message.content or "").strip()
        if _normalize_letters(out) == _normalize_letters(text):
            return out
        return text
    except Exception as e:
        print("[rewrite error]", e)
        return text

# ───────────────────────────── 한국어 태그 파서 ─────────────────────────────
TAG_RE = re.compile(r"<\s*(행복|슬픔|분노|놀람|두려움|중립)\s*>")
def parse_tagged_script(raw: str):
    """
    "...문장1.<행복> 문장2.<슬픔> ..." → [("문장1.", "행복"), ("문장2.", "슬픔"), ...]
    마지막 꼬리 문장은 감정 None → 전역 감정 적용
    """
    s = (raw or "").strip()
    segs = []
    pos = 0
    for m in TAG_RE.finditer(s):
        before = s[pos:m.start()]
        emo = m.group(1)
        if before.strip():
            segs.append((before.strip(), emo))
        pos = m.end()
    tail = s[pos:]
    if tail.strip():
        segs.append((tail.strip(), None))
    return segs

# ───────────────────────────── 합성(문장 하나) ─────────────────────────────
def _intensity(data: dict):
    """요청의 감정 강도 k (선택, 0~1.5). 없으면 None → 백엔드 기본값(FACEVOX_INTENSITY, 기본 1.0)."""
    v = data.get("intensity")
    if v is None or v == "":
        return None
    try:
        return max(0.0, min(1.5, float(v)))
    except (TypeError, ValueError):
        return None


def tts_one(text: str, emotion: str, intensity=None):
    """문장 하나 합성 → SynthResult (audio bytes, fmt, timings)"""
    instruction, _, _ = preset_for_emotion(emotion)
    t0 = time.perf_counter()
    perf_text = rewrite_with_instruction(text, instruction)
    rewrite_ms = (time.perf_counter() - t0) * 1000
    res = (BACKEND.synthesize(perf_text, emotion) if intensity is None
           else BACKEND.synthesize(perf_text, emotion, intensity=intensity))
    res.timings["rewrite_ms"] = rewrite_ms
    return res

# ───────────────────────────── 라우트 ─────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# face_emotion.py가 서버 찾을 때 호출
@app.route("/get_config", methods=["GET"])
def get_config():
    return jsonify({"server": "openaiTTS", "ok": True, "backend": BACKEND.name,
                    "mode": getattr(BACKEND, "mode", BACKEND.name), "format": OUTPUT_FORMAT,
                    "emotions": list(EMOTION_PRESETS.keys())})

# 얼굴 감정 이벤트 수신 (라벨/태그→한국어 감정으로 표준화 저장)
@app.route("/set_emotion", methods=["POST"])
def set_emotion():
    data = request.get_json(silent=True) or {}
    t = int(data.get("t", int(time.time()*1000)))
    label = str(data.get("label", "Neutral"))
    tag = str(data.get("tag", ""))
    conf = float(data.get("confidence", 0.0))
    emo_ko = _event_to_ko(label, tag)
    EMO_HISTORY.append({
        "t": t,
        "emo": emo_ko,         # 한국어 6감정
        "label": label,        # 원 라벨
        "tag": tag,            # 원 태그(<laugh> 등)
        "confidence": conf
    })
    if os.getenv("FACEVOX_QUIET") != "1":
        print(f"[face] t={t} label={label} tag={tag} -> emo='{emo_ko}' conf={conf:.2f}")
    return jsonify({"status": "ok"})

# 문장 구간 지배 감정 → 한국어 태그("<행복>") 반환
@app.route("/emotion/resolve_one", methods=["POST"])
def resolve_one():
    data = request.get_json(silent=True) or {}
    try:
        since_ms = int(data.get("since_ms"))
        until_ms = int(data.get("until_ms"))
    except Exception:
        return jsonify({"tag": ""}), 400
    if until_ms <= since_ms:
        return jsonify({"tag": ""})
    # 필요시 더 자주 태그 나오도록 완화: conf_thresh=0.45, min_ratio=0.3
    tag = _pick_dominant_ko(since_ms, until_ms, conf_thresh=0.5, min_ratio=0.4)
    return jsonify({"tag": tag})

# ── 디버그: 문장 파싱 및 적용 예정 파라미터 미리보기
@app.post("/debug/segments")
def debug_segments():
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text") or "").strip()
    global_emotion = (data.get("emotion") or "중립").strip() or "중립"
    if not raw_text:
        return jsonify({"segments": [], "backend": BACKEND.name})

    segs = parse_tagged_script(raw_text)
    out = []
    for body, emo in segs:
        applied = (emo or global_emotion)
        instruction, voice, speed = preset_for_emotion(applied)
        perf_text = rewrite_with_instruction(body, instruction)
        desc = BACKEND.describe(applied, _intensity(data))
        out.append({
            "text": body,
            "emotion": applied,
            "tagged": emo is not None,
            "voice": desc.get("voice", voice),
            "speed": speed,
            "perf_text": perf_text,
            "params": desc,
        })
    return jsonify({"segments": out, "backend": BACKEND.name})

# ── 디버그: 최근 얼굴 감정 이벤트
@app.get("/debug/emo_history")
def debug_emo_history():
    try:
        limit = int(request.args.get("limit", "50"))
    except Exception:
        limit = 50
    items = list(EMO_HISTORY)[-limit:]
    return jsonify({"events": items})


def _audio_response(raw: bytes, timing: dict):
    resp = send_file(BytesIO(raw), mimetype=MIMETYPE, as_attachment=False,
                     download_name=f"tts.{OUT_EXT}", max_age=0)
    resp.headers["X-STAGE-Backend"] = BACKEND.name
    resp.headers["X-STAGE-Timing"] = json.dumps(timing)
    resp.headers["Access-Control-Expose-Headers"] = "X-STAGE-Backend, X-STAGE-Timing"
    return resp


def _error_response(e: Exception):
    try:
        from openai import BadRequestError, APIConnectionError, APITimeoutError, APIStatusError
        if isinstance(e, BadRequestError):
            return jsonify({"error": f"요청 오류: {e.message}"}), 400
        if isinstance(e, (APIConnectionError, APITimeoutError, APIStatusError)):
            return jsonify({"error": f"API 통신 오류: {str(e)}"}), 502
    except Exception:
        pass
    if isinstance(e, RuntimeError) and "edge-tts" in str(e):
        return jsonify({"error": f"TTS 통신 오류: {str(e)}"}), 502
    print("[tts error]", repr(e))
    return jsonify({"error": f"서버 내부 오류: {str(e)}"}), 500


# TTS: 태그가 있으면 문장별 합성/병합, 없으면 단일 감정
@app.route("/tts", methods=["POST"])
def tts():
    t_req = time.perf_counter()
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text") or "").strip()
    global_emotion = (data.get("emotion") or data.get("style") or "").strip() or "중립"
    if not raw_text:
        return jsonify({"error": "text가 비어 있습니다."}), 400

    segments = parse_tagged_script(raw_text)
    has_tags = any(emo is not None for _, emo in segments)

    # A) 문장별 태그 → 병합
    if has_tags:
        if not PydubAvailable:
            return jsonify({"error": "여러 문장 병합에는 pydub가 필요합니다. `pip install pydub` 후 재시도하세요."}), 400
        try:
            seg_timings = []
            combined = AudioSegment.silent(duration=0)
            pause = AudioSegment.silent(duration=250)  # 0.25초 쉼
            for body, emo in segments:
                applied_emo = emo or global_emotion
                print(f"[segment] text='{body[:20]}...' emo='{applied_emo}'")
                res = tts_one(body, applied_emo, _intensity(data))
                seg_timings.append(res.timings)
                seg = AudioSegment.from_file(BytesIO(res.audio), format=res.fmt)
                combined += seg + pause
            t_m = time.perf_counter()
            out = BytesIO()
            combined.export(out, format=OUTPUT_FORMAT, bitrate="192k" if OUTPUT_FORMAT == "mp3" else None)
            merge_ms = (time.perf_counter() - t_m) * 1000
            timing = {"segments": seg_timings, "merge_export_ms": merge_ms,
                      "server_total_ms": (time.perf_counter() - t_req) * 1000,
                      "audio_sec": len(combined) / 1000.0}
            return _audio_response(out.getvalue(), timing)
        except Exception as e:
            return _error_response(e)

    # B) 태그 없음 → 단일 감정
    try:
        res = tts_one(raw_text, global_emotion, _intensity(data))
        raw = res.audio
        merge_ms = 0.0
        if res.fmt != OUTPUT_FORMAT and PydubAvailable:
            t_m = time.perf_counter()
            out = BytesIO()
            AudioSegment.from_file(BytesIO(raw), format=res.fmt).export(out, format=OUTPUT_FORMAT)
            raw = out.getvalue()
            merge_ms = (time.perf_counter() - t_m) * 1000
        timing = {"segments": [res.timings], "merge_export_ms": merge_ms,
                  "server_total_ms": (time.perf_counter() - t_req) * 1000}
        return _audio_response(raw, timing)
    except Exception as e:
        return _error_response(e)


@app.get("/debug/backend")
def debug_backend():
    return jsonify({"backend": BACKEND.name, "mode": getattr(BACKEND, "mode", BACKEND.name),
                    "prosody_status": getattr(BACKEND, "prosody_status", None),
                    "rewrite": USE_REWRITE, "format": OUTPUT_FORMAT,
                    "pydub": PydubAvailable, "ffmpeg": FFMPEG_OK})


if __name__ == "__main__":
    print(f"[audio backend] pydub={PydubAvailable}, ffmpeg={FFMPEG_OK}, format={OUTPUT_FORMAT}")
    print(f"[tts backend] {BACKEND.name} ({getattr(BACKEND, 'mode', '')}) rewrite={USE_REWRITE}")
    app.run(host=os.getenv("FACEVOX_HOST", "127.0.0.1"), port=int(os.getenv("FACEVOX_PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "0") == "1", threaded=True)
