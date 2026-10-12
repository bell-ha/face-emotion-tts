# STAGE 재실험 계획 (2026-10-12)

목표 학술지: 한국엔터테인먼트산업학회논문지 (KCI 등재), 원고 마감 2026-10-23.
사용자는 실험을 직접 하지 않는다. 모든 실험·코드·그림·원고를 Claude들이 만든다.

## 제약

- **OpenAI API 키 없음.** gpt-4o-mini-tts는 새로 호출할 수 없다. 기존 합성 샘플 12개(`assets/audio/`, 2025-08-23, speed+문장부호만 적용, instructions 미적용)는 분석에만 쓴다.
- 키 없이 쓸 수 있는 TTS: `edge-tts` (Microsoft, 한국어 `ko-KR-SunHiNeural` 여성 등, rate/pitch/volume 조절 가능). 로컬 DSP 후처리: `praat-parselmouth`(PSOLA로 F0 평균·범위, 길이 조작), `librosa`.
- 사람 청취 평가는 할 수 없다. 대신 **자동 음성 감정 인식(SER) 모델 `emotion2vec_plus_large`** (funasr, hub="hf")를 대리 지표로 쓰고, 논문에 한계로 분명히 적는다.
- 사용자 얼굴 사진(`assets/figures/{angry,disgust,fear,happy,neutral,sad,surprise}.png`, `face_merge.png`)은 사용자 본인 얼굴이며 논문에 재사용 허락을 받았다.
- 상위 폴더(`~/Desktop`)의 원본 파일은 수정하지 않는다. 작업은 `face-emotion-tts/` 안에서만.

## 환경

- 가상환경: `face-emotion-tts/.venv` (Python 3.11). tensorflow 2.15.1, mediapipe 0.10.14, opencv, librosa, parselmouth, torch, funasr, edge-tts, datasets, scikit-learn, pandas, matplotlib 설치됨. 추가 패키지는 `.venv/bin/pip install`로 설치 가능.
- 하드웨어: Apple M1 Max, RAM 32GB, GPU 없음(MPS는 가능). 디스크 여유 약 180GB.
- FER 모델: `face/face_emotion.h5` (48×48 흑백 입력, Conv 6층 + Dense 512, 파라미터 66.7만 개, 7클래스 순서 `['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']`, 입력 /255 정규화).

## 공통 규칙 (모든 실험)

1. 결과물 위치: `face-emotion-tts/experiments/<실험ID>/` 아래에 `scripts/`, `results/`(CSV/JSON), `figures/`(PNG 300dpi + PDF), `README.md`.
2. `README.md`에는 목적, 데이터(출처·라이선스·개수), 절차, 하이퍼파라미터, **실제 수치 표**, 그림 목록과 캡션 초안(국문), 한계를 적는다.
3. **수치는 실제로 돌린 결과만 쓴다. 결과가 나쁘면 나쁜 대로 보고한다.** 추정·가공 금지. 난수 시드 고정(42).
4. 그림의 한글 폰트는 `AppleGothic` 또는 `Apple SD Gothic Neo`. 흑백 인쇄에서도 구분되게(마커/해칭).
5. 대용량 데이터는 `face-emotion-tts/data/`에 받는다(.gitignore 대상).
6. 끝나면 핵심 수치 5~10줄 요약을 보고한다.

## 실험

| ID | 내용 | 담당 |
|---|---|---|
| E1 | FER 모델 평가: FER2013 PublicTest/PrivateTest 정확도, macro-F1, 혼동행렬, 클래스별 지표. M1 CPU에서 프레임당 지연(MediaPipe 검출 + CNN). 사용자 얼굴 사진 7장 분류 결과(정성). | agent-fer |
| E2 | 시간적 안정화 평가: 실제 영상(RAVDESS 비디오 일부 배우)에서 프레임별 FER → raw / EMA / 다수결 / EMA+다수결 / +Hold-time 비교. 지표: 초당 라벨 전환 수(깜빡임), 클립 단위 정확도, 감정 전환 반응 지연. 문장 단위 지배 감정 선택(`_pick_dominant_ko`) 정확도 시뮬레이션. | agent-fer (E1 다음) |
| E3 | 감정 음성 렌더링 평가: (A) 기존 OpenAI 샘플(speed만), (B) edge-tts 중립, (C) edge-tts + 감정 프리셋(rate/pitch/volume), (D) C + parselmouth F0 범위·에너지 후처리. 감정 중립 내용의 한국어 문장 여러 개 × 6감정. 지표: F0 평균/표준편차, 강도, 발화 속도, emotion2vec 분류 정확도·혼동행렬, 화자 일관성(화자 임베딩 코사인 유사도). | agent-tts |
| E4 | 시스템: 웹 UI(`tts/templates/index.html`) 재작성, TTS 백엔드 추상화(OpenAI 키 있으면 OpenAI+instructions, 없으면 edge-tts+DSP), macOS 카메라 수정, 단계별 지연 측정(반복 30회 이상). | agent-system |
| L1 | 관련 연구·투고 규정: KOEN 투고 규정과 원고 양식, 관련 연구 실존 확인과 APA7 서지, 졸업논문 인용 오류 정리. | agent-lit |

## 논문 방향 (초안)

- 국문 제목(안): STAGE: 엔터테인먼트 콘텐츠 제작을 위한 실시간 표정 인식 기반 감정 음성 합성 시스템
- 핵심 주장: 학습 없이 기존 TTS에 붙일 수 있는 "표정 → 문장 단위 감정 태그 → 운율 제어" 파이프라인. 안정화 기법의 효과와, 속도만 바꾸는 방식(졸업논문 구현)이 감정 전달에 부족하다는 점, 명시적 운율 제어의 효과를 객관 지표로 보인다.

## 그림 규정 (학회 원문 확인, 2026-10-12)

- 본문은 국문. 1쪽만 전체 영문, 국문 요약은 참고문헌 뒤.
- 캡션: `<그림 1> 국문 제목` 다음 줄 `(Fig. 1 English Title)`. 그림 캡션은 그림 아래, 표 캡션은 표 위. 그림 가운데, 표 왼쪽 정렬.
- **그림 안 글자는 전부 영문** (사용자 결정, 2026-10-12). 감정 라벨은 Happy, Sad, Angry, Surprise, Fear, Neutral (Disgust). 캡션만 국문+영문 두 줄.
- 그림 안에 제목을 넣지 않는다 (캡션이 대신함).
- 컬러/흑백 규정은 없음. 인쇄판이 있으므로 컬러로 만들되 흑백 인쇄에서도 구분되게(마커, 해칭, 명도 차이).
- 2단 편집: 그림 폭은 단 폭 약 76 mm 또는 전체 폭 160 mm 기준으로 글자 크기(인쇄 시 7~9 pt)를 맞춘다.
