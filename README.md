# STAGE

**S**entence-wise **T**one **A**daptation **G**uided by facial **E**xpressions — 웹캠 표정으로 문장마다 감정을 연출하는 실시간 감정 음성 합성

Real-time facial expression-driven emotional TTS (MediaPipe + CNN/HSEmotion + OpenAI TTS / edge-tts)

웹캠으로 사용자의 표정을 실시간으로 인식하고, 그 감정을 문장 단위 태그로 바꿔 감정이 담긴 음성으로 합성하는 시스템입니다.

## 폴더 안내

| 폴더 | 무엇이 있나 |
|---|---|
| `app/` | **실제 프로그램.** 표정 인식(`face/`), 음성 합성 서버와 웹 화면(`tts/`), 자동 테스트(`tests/`) |
| `experiments/` | **논문 실험.** 폴더마다 README에 방법과 결과 수치가 있음 |
| `paper/` | **논문 파일.** 학사 졸업논문 최종본 PDF. 이를 확장한 학술지 원고는 투고 준비 중이라 올리지 않음([paper/README.md](paper/README.md)) |
| `docs/` | **메모.** 졸업논문 검토와 인용 수정안 |
| `assets/` | 졸업논문 때 만든 그림과 감정별 음성 샘플 |
| `data/` | 데이터셋, 모델, 캐시 같은 큰 파일 (git 제외, 다시 받거나 만들 수 있음) |

```
face-emotion-tts/
├── app/
│   ├── face/      # 표정 인식: MediaPipe 얼굴 검출 + CNN(FER2013, 7개 감정; 선택 HSEmotion) + EMA/히스테리시스/Hold-Time 안정화
│   ├── tts/       # Flask 서버 + 웹 UI: 감정 태깅, 합성(OpenAI 또는 edge-tts), 운율 후처리(prosody.py)
│   └── tests/     # pytest
├── experiments/
│   ├── E1_face-recognition/       # 표정 인식 정확도·속도
│   ├── E2_emotion-stabilization/  # 감정이 흔들리지 않게 잡는 안정화
│   ├── E3_emotional-speech/       # 감정 음성 표현력 (E3b: 감정 강도 조절, E3c: 재합성 변동·남성 화자)
│   ├── E4_system-latency/         # 전체 시스템 지연 시간, 웹 UI 캡처
│   └── E5_improvements/           # 개선한 안정화·문장 태깅, 다른 FER 모델 비교
├── paper/
│   ├── 1_thesis/        # 학사 졸업논문 최종본 PDF
│   └── README.md        # 논문 상태 안내
├── docs/
│   └── research/    # 졸업논문 검토, 인용 수정안
├── assets/
│   ├── figures/     # 졸업논문 그림 (학습 곡선, 감정별 캡처, 웹 UI 등)
│   └── audio/       # 감정별 합성 샘플
└── data/            # (git 제외) 데이터셋·모델·캐시
```

## 파이프라인

1. `app/face/face_emotion.py`가 웹캠(또는 영상 파일) 프레임에서 얼굴을 찾고 7개 감정을 분류
2. 감정이 안정적으로 바뀌면(EMA + 히스테리시스 δ=0.5 + Hold-time) `POST /set_emotion`으로 서버에 전송(화면 표시용), 0.2초마다 원시 확률 평균을 `POST /set_emotion_probs`로 전송
3. 웹 UI에서 문장 끝(`.` `!` `?`)을 입력하면, 그 문장 구간의 확률 질량(중립 억제 가중치 적용)이 가장 큰 감정 태그(`<행복>` 등)가 붙음(`/emotion/resolve_one`). 확률 요약이 없으면 `/set_emotion` 이벤트의 지속 시간으로 정하는 졸업논문 규칙으로 폴백한다. 근거: `experiments/E5_improvements`
4. `POST /tts`가 문장마다 감정 프리셋을 적용해 합성하고, pydub으로 이어 붙임

## TTS 백엔드

| 백엔드 | 조건 | 감정 표현 |
|---|---|---|
| `openai` | `OPENAI_API_KEY`가 있을 때(auto) | gpt-4o-mini-tts에 감정 지시문을 `instructions`로 전달 + `speed`. gpt-4o-mini 문장부호 연출 사용 |
| `edge` | 키가 없을 때(auto) | edge-tts `ko-KR-SunHiNeural` + rate/pitch/volume 프리셋. `app/tts/prosody.py`가 있으면 parselmouth(PSOLA) 운율 후처리까지 적용 |

`TTS_BACKEND=openai|edge|auto`로 강제할 수 있다. edge-tts는 Microsoft 온라인 서비스라 **API 키는 필요 없지만 인터넷 연결은 필요**하다. 그 밖의 환경변수: `FACEVOX_PORT`(기본 5000), `FACEVOX_PROSODY_MODE`(`D`=프리셋+PSOLA, `C`=프리셋만, `DP`=D-psola), `FACEVOX_INTENSITY`(감정 강도 k 기본값, 웹 UI 슬라이더로도 조절), `FACEVOX_REWRITE=0`(OpenAI 연출 끄기).

## 실행

Python 3.10–3.11. MP3 출력에는 ffmpeg가 필요하다(없으면 WAV로 내보냄). edge 백엔드는 **edge-tts 7.3.1 이상**이 필요하다(7.2.8은 2026-11부터 서비스 인증 실패로 합성되지 않음).

### macOS

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r app/tts/requirements.txt -r app/face/requirements.txt
brew install ffmpeg            # 선택

# 1) TTS 서버 — 키 없으면 자동으로 edge 백엔드
python app/tts/app.py              # → http://127.0.0.1:5000  (localhost 대신 127.0.0.1 사용)

# 2) 표정 인식 (다른 터미널). 카메라 권한: 시스템 설정 > 개인정보 보호 및 보안 > 카메라
python app/face/face_emotion.py
python app/face/face_emotion.py --video demo.mp4 --realtime   # 웹캠 대신 영상 파일
python app/face/face_emotion.py --classifier hsemotion          # 선택: HSEmotion(ONNX 가중치 필요, app/face/README.md)
```

> macOS 12 이상에서는 AirPlay 수신 모드가 5000번 포트를 쓰므로 `http://localhost:5000`이 AirPlay(403)로 연결될 수 있다. `127.0.0.1`로 접속하거나 `FACEVOX_PORT=5050`처럼 포트를 바꾸면 된다(이때 `face_emotion.py --server http://127.0.0.1:5050`).

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r app\tts\requirements.txt -r app\face\requirements.txt
winget install ffmpeg          # 선택
python app\tts\app.py
python app\face\face_emotion.py
```

### OpenAI 백엔드 사용(선택)

`app/tts/.env` 또는 저장소 루트 `.env`에 `OPENAI_API_KEY=sk-...`를 넣으면 자동으로 OpenAI 백엔드가 선택된다.

### 테스트

```bash
python -m pytest app/tests -q     # 63개: 태그 파서, 지속 시간·확률 질량 태깅, 백엔드 선택, Flask 엔드포인트(합성은 mock), E5 참조 구현과의 동등성
```

## 실험

`experiments/` 아래 E1(FER 평가), E2(시간적 안정화), E3(감정 음성 렌더링, E3b 감정 강도, E3c 재합성 변동·남성 화자), E4(시스템 지연·웹 UI), E5(개선안·FER 모델 교체). 각 폴더의 README에 목적, 절차, 실제 수치, 한계, 재현 명령이 있다. 사람 청취 평가는 후속으로 진행할 예정이다.

## 데이터와 모델의 이용 조건

| 항목 | 조건 |
|---|---|
| FER2013 | 연구용 공개 데이터(원 Kaggle 대회) |
| RAVDESS | CC BY-NC-SA 4.0. 원 영상과 프레임은 저장소에 싣지 않고 수치만 쓴다 |
| HSEmotion 가중치 | 코드는 Apache-2.0, 학습 데이터 AffectNet은 연구용 라이선스 |
| edge-tts | LGPL-3.0 오픈소스 도구. Microsoft 온라인 음성 서비스를 호출한다 |
| 표정 사진(`assets/figures`) | 제1저자 본인 사진 |

상업적으로 쓰려면 이용 약관이 허용하는 TTS와 데이터로 바꿔야 한다.

## 출처

이종하, 「실시간 표정 인식을 이용한 감정 기반 TTS 보컬 합성 시스템 연구」, 단국대학교 뮤직테크놀러지&컴퓨터음악작곡 전공 학사학위논문, 2025. (지도교수: 구자환)
