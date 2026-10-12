# tts/ — STAGE 서버와 웹 UI

Flask 서버. 얼굴 감정 이벤트를 받아 두고, 문장별 감정 태그가 붙은 대본을 문장마다 감정 프리셋으로 합성해 이어 붙인다.

## 파일

| 파일 | 설명 |
|---|---|
| `app.py` | 라우트, 감정 이벤트 버퍼, 지배 감정 판정(`_pick_dominant_ko`), 태그 파서(`parse_tagged_script`), 문장 병합 |
| `backends.py` | TTS 백엔드(`openai`, `edge`)와 선택 로직 |
| `templates/index.html` | 웹 UI (바닐라 JS, 빌드 불필요) |
| `requirements.txt` | 의존 패키지 |

## 백엔드

- **openai** (`OPENAI_API_KEY`가 있으면 자동): `gpt-4o-mini-tts`, 음성 `shimmer`. 감정 프리셋 지시문을 `instructions` 파라미터로, 감정별 `speed`를 함께 넘긴다. 그 전에 gpt-4o-mini가 단어는 그대로 두고 문장부호만 고쳐 준다(`FACEVOX_REWRITE=0`으로 끔).
  - 졸업논문 구현(2025)은 `instructions`를 넘기지 않아 실제로는 `speed`와 문장부호만 반영됐다. 지금 코드는 이를 고쳤다.
- **edge** (키가 없으면 자동): edge-tts `ko-KR-SunHiNeural`.
  - `experiments/E3_tts/scripts/prosody.py`가 있으면 그 프리셋(rate/pitch/volume)과 parselmouth PSOLA 후처리(F0 범위·이동·윤곽, 템포, 스펙트럼 기울기, 이득)를 쓴다(`FACEVOX_PROSODY_MODE=D`, 기본 = 논문 주 조건). `C`는 프리셋만, `DP`(D-psola)는 중립 edge-tts 위에 PSOLA로 모든 운율 변화를 준다(화자 보존형).
  - 감정 강도 k: 요청 JSON의 `intensity`(0–1.5, 기본 1.0 = 고정 프리셋, 0 = 중립). 웹 UI 슬라이더로 조절. 기본값은 `FACEVOX_INTENSITY`. OpenAI 백엔드에서는 쓰지 않는다.
  - 없으면 `backends.py`의 `EDGE_PRESETS`(rate/pitch/volume)만 쓴다.
  - edge-tts는 Microsoft 온라인 서비스다. 키는 필요 없지만 인터넷 연결이 필요하다.

`TTS_BACKEND=openai|edge|auto`로 강제할 수 있다. 현재 상태는 `GET /debug/backend`로 확인한다.

## 실행

```bash
pip install -r requirements.txt
python app.py                  # http://127.0.0.1:5000  (FACEVOX_PORT로 변경)
```

macOS에서는 AirPlay 수신 모드가 5000번 포트를 함께 쓰므로 `localhost` 대신 `127.0.0.1`로 접속한다.

## 웹 UI

1. **Start**를 누르고 대본을 입력한다. 문장 끝에서 `.` `!` `?`를 치면, 그 문장의 첫 글자를 친 시각부터 지금까지의 구간을 `/emotion/resolve_one`에 보내 지배 감정 태그를 받아 문장 끝에 넣는다. 표정 근거가 부족하면(중립이 가장 길거나, 1위 감정이 구간의 40% 미만, 신뢰도 0.5 미만) 기본 감정 태그를 넣는다.
2. 오른쪽 패널은 `/debug/emo_history`를 0.5초마다 읽어 현재 표정과 최근 20초 타임라인을 보여 준다.
3. **합성 계획 보기**(`/debug/segments`)는 문장별 감정과 운율 파라미터를, **음성 만들기**(`/tts`)는 합성 결과를 재생한다.

## API

| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| GET | `/` | – | 웹 UI |
| GET | `/get_config` | – | `{"server","ok","backend","mode","format","emotions"}` |
| POST | `/set_emotion` | `{"t": epoch_ms, "label": "Happy", "tag": "", "confidence": 0.9}` | `{"status":"ok"}` |
| POST | `/emotion/resolve_one` | `{"since_ms", "until_ms"}` | `{"tag": "<행복>"}` 또는 `{"tag": ""}` |
| POST | `/debug/segments` | `{"text", "emotion", "intensity"}` | `{"segments":[{"text","emotion","tagged","voice","speed","perf_text","params"}]}` |
| GET | `/debug/emo_history?limit=50` | – | `{"events":[...]}` |
| POST | `/tts` | `{"text": "문장.<행복> 문장.<슬픔>", "emotion": "중립", "intensity": 1.0}` (`intensity` 선택) | 오디오(MP3, ffmpeg 없으면 WAV). 헤더 `X-STAGE-Timing`에 단계별 시간(JSON) |
| GET | `/debug/backend` | – | 백엔드·모드·출력 포맷 |

감정 태그는 `<행복> <슬픔> <분노> <놀람> <두려움> <중립>` 6가지. FER 라벨 Disgust는 분노로 매핑한다. 문장 사이에는 0.25초 쉼을 넣는다.
