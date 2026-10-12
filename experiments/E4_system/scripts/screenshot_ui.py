"""웹 UI 스크린샷(논문 그림용).

Flask 앱(edge 백엔드)을 띄우고, headless Chromium(playwright)으로 대본을 한 글자씩 입력한다.
각 문장을 입력하기 직전에 가짜 표정 이벤트를 /set_emotion으로 보내므로, 문장 끝(. ! ?)에서
/emotion/resolve_one이 실제로 지배 감정을 판정해 태그를 삽입한다(태그를 직접 써 넣지 않음).
마지막에 '음성 만들기'로 실제 합성까지 수행한 뒤 캡처한다.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "experiments" / "E4_system" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PORT = int(os.getenv("E4_UI_PORT", "5058"))
BASE = f"http://127.0.0.1:{PORT}"
# UI 언어: en(논문 그림용, 기본) → web_ui*.png / ko → web_ui_ko*.png. 대본 텍스트는 항상 한국어(TTS가 한국어).
LANG = os.getenv("UI_LANG", "en")
SUFFIX = "" if LANG == "en" else f"_{LANG}"

SCRIPT = [  # (FER 라벨, 신뢰도, 문장) — 짧은 오디오북/1인 방송 대본
    ("Happy", 0.93, "오늘 들려드릴 이야기는 숲속 작은 토끼의 첫 모험이에요!"),
    ("Surprise", 0.81, " 토끼가 문을 열자 눈앞에 커다란 무지개 다리가 나타났어요!"),
    ("Fear", 0.74, " 그런데 다리 아래에서 낮게 으르렁거리는 소리가 들려왔어요."),
    ("Sad", 0.86, " 겁이 난 토끼는 집에 두고 온 엄마 생각에 눈물이 났지요."),
    ("Angry", 0.88, " 그때 심술궂은 여우가 길을 막고 비키지 않았어요!"),
    ("Happy", 0.90, " 하지만 친구 다람쥐가 달려와 손을 내밀어 주었어요"),  # 작성 중(문장부호 없음)
]


def send(label, conf, t=None):
    requests.post(BASE + "/set_emotion", json={"t": int(t or time.time() * 1000), "label": label,
                                                "tag": "", "confidence": conf}, timeout=2)


def main():
    env = dict(os.environ, TTS_BACKEND="edge", FACEVOX_PORT=str(PORT), FACEVOX_QUIET="1")
    env.pop("OPENAI_API_KEY", None)
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT / "tts", env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                if requests.get(BASE + "/get_config", timeout=0.5).ok:
                    break
            except Exception:
                time.sleep(0.2)
        # 시작 전 몇 초간 중립 상태
        now = time.time() * 1000
        send("Neutral", 0.78, now - 6000)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=2,
                                    locale="ko-KR")
            page.goto(BASE + ("/?lang=en" if LANG == "en" else "/"))
            page.wait_for_timeout(1500)
            page.click("#btnStart")
            page.click("#text")
            for i, (label, conf, sent) in enumerate(SCRIPT):
                send(label, conf)
                page.wait_for_timeout(350)
                page.keyboard.type(sent, delay=45)
                page.wait_for_timeout(700)
                if i == 2:
                    send("Neutral", 0.66)      # 문장 사이 잠깐 중립(실제 표정 흐름처럼)
                    page.wait_for_timeout(500)
            page.wait_for_timeout(600)
            page.screenshot(path=str(FIG / f"web_ui_typing{SUFFIX}.png"), full_page=True)
            page.click("#btnPlan")
            page.wait_for_timeout(800)
            page.click("#btnTts")
            page.wait_for_selector("#result a", timeout=120000)
            page.wait_for_timeout(800)
            page.screenshot(path=str(FIG / f"web_ui{SUFFIX}.png"), full_page=True)
            # 왼쪽 편집 영역만(본문용 확대 그림)
            page.locator("main > section").screenshot(path=str(FIG / f"web_ui_editor{SUFFIX}.png"))
            page.locator("main > aside").screenshot(path=str(FIG / f"web_ui_face_panel{SUFFIX}.png"))
            text = page.input_value("#text")
            (FIG / "web_ui_text.txt").write_text(text, encoding="utf-8")
            print(text)
            browser.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
