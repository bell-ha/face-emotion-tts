# 졸업논문 검토 메모

검토 대상은 `paper/1_thesis/thesis_final_2025-11-06.pdf`와 `app/face/`, `app/tts/` 코드이다. 학술지 원고를 쓸 때 고쳐야 할 부분을 정리했다.

## 논문 설명과 실제 코드가 다른 부분

1. **감정 지시문이 TTS에 들어가지 않음 (가장 중요)**
   - 논문은 감정 지시문을 TTS 프롬프트에 "직접 반영"한다고 설명한다.
   - 실제 코드(`app/tts/app.py`의 `tts_one`)는 `client.audio.speech.create`에 `instructions`를 넘기지 않는다. 지시문은 gpt-4o-mini가 문장부호만 고치는 데에만 쓰인다.
   - 결과적으로 TTS는 `speed`와 문장부호가 바뀐 텍스트만 받는다. 음성은 모든 감정에서 `shimmer`로 같다.
   - 해결: gpt-4o-mini-tts의 `instructions` 파라미터에 프리셋 지시문을 넣는다.
2. 5.1절의 "WebSocket 실시간 인터페이스"와 달리, 코드는 HTTP POST(`requests`)를 쓴다.
3. 논문은 결과를 WAV로 전송한다고 했지만, 코드는 ffmpeg가 있으면 MP3로 내보낸다.
4. 논문의 JSON 예시는 `"emotion":"happy"`인데, 코드가 실제로 보내는 필드는 `label`이다.

## 논문 내부에서 맞지 않는 부분

- 5.3절은 "기쁨과 슬픔은 전달이 잘 됐고 분노가 혼동됐다"고 하는데, 5.4절은 "놀람과 슬픔이 약했다"고 한다. 게다가 놀람은 청취 평가 대상(기쁨·슬픔·분노·중립)에 포함되지 않았다.
- 청취 평가 참가자 수와 점수, 6초→4초 측정의 반복 횟수, FER 모델의 테스트 정확도가 수치로 나와 있지 않다. 학습 곡선만 있다.
- 4.1절의 로그(Disgust 0.9999 → Happy, 2.6초)는 두 번의 전송 사이 시간이라 인식 지연 시간이 아니다. Disgust는 학습 데이터의 1.6%뿐이라 오분류일 가능성도 있다.

## 인용 오류

- 본문은 "Ten et al., 2023"인데, 참고문헌 목록에는 Tan et al. (2021)이 있다.
- 본문의 "AITimes, 2023"(ICASSP 2023 FastSpeech2)에 해당하는 항목이 참고문헌에 없다.
- 저자명이 "김혜란"과 "김혜련"으로 섞여 있다.
- "(유은정, 2020.)"처럼 연도 뒤에 마침표가 붙은 곳이 있다.
- 그림 1 캡션에 "Tractron" 오타가 있다.

## 추가해야 할 관련 연구 (얼굴 → 감정 음성)

| 연구 | 출처 | 우리 연구와의 차이 |
|---|---|---|
| FEIM-TTS | 서강대, ECCV Workshop 2024 (arXiv 2409.16203) | 정지된 얼굴 이미지로 오프라인 합성 |
| DEmoFace / Emotional Face-to-Speech | ICML 2025 (arXiv 2502.01046) | diffusion 모델을 학습해 얼굴에서 바로 음성 생성 |
| FaceSpeak | arXiv 2501.03181 | 인물 사진에서 음성 생성 |
| Face-TTS | Interspeech 계열 | 얼굴로 화자 정체성을 정함 |

**STAGE만의 차이점**: 실시간 웹캠 입력, 글을 쓰는 동안 문장 단위로 감정 태그가 자동으로 붙는 방식, 별도 학습 없이 프롬프트 기반 상용 TTS를 사용, 창작 도구로서의 활용.

## 코드 상태

- `app/tts/templates/index.html`이 없어서 웹 UI를 띄울 수 없다.
- TTS용 requirements 파일과 README에 나오는 `tts_test.py`도 없다.
- `cv2.CAP_MSMF`는 Windows 전용이다. macOS에서는 `cv2.CAP_AVFOUNDATION`을 쓰거나 백엔드를 지정하지 않아야 한다.
- `app/face/README.md`는 채팅 답변을 그대로 붙여 넣은 형태라 정리가 필요하다.
- API 키는 `.env`에서 읽는다. 코드에 직접 들어 있지는 않다.
