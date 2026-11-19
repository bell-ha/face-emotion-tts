# 감정별 합성 샘플

`app/tts/app.py`의 감정 프리셋(OpenAI `gpt-4o-mini-tts`, 음성 shimmer)으로 합성한 샘플이다(2025-08-23).

졸업논문 구현이라 감정별 `speed`와 문장부호만 반영됐고, 감정 지시문(`instructions`)은 전달되지 않았다. E3의 조건 A(`experiments/E3_emotional-speech/README.md`)가 이 샘플을 분석한 것이다.

| 파일 접두어 | 문장 |
|---|---|
| `greeting_*` | 안녕하세요 만나서 반갑습니다 |
| `rabbit_*` | 깊은 숲속 오두막집에서 작은 토끼가 조심스레 문을 열고 밖으로 나왔어요 |

| 접미어 | 감정 | speed |
|---|---|---|
| `happy` | 행복 | 1.22 |
| `sad` | 슬픔 | 0.85 |
| `angry` | 분노 | 1.15 |
| `surprise` | 놀람 | 1.25 |
| `fear` | 두려움 | 0.92 |
| `neutral` | 중립 | 1.0 |
