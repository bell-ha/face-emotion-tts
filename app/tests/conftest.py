import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tts"))
sys.path.insert(0, str(ROOT / "face"))   # emotion_core (TensorFlow·카메라 없이 import 가능)

# 테스트는 항상 키 없는 edge 백엔드로, E3 모듈은 끄고(네트워크·DSP 의존 제거) 돌린다.
os.environ["TTS_BACKEND"] = "edge"
os.environ["FACEVOX_DISABLE_E3"] = "1"
os.environ["FACEVOX_QUIET"] = "1"
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture(scope="session")
def appmod():
    import app as appmod  # noqa: E402
    return appmod


def _reset(appmod):
    appmod.EMO_HISTORY.clear()
    appmod.PROB_HISTORY.clear()
    appmod.PROB_STATE["classifier"] = None


@pytest.fixture()
def fresh(appmod, monkeypatch):
    monkeypatch.delenv("FACEVOX_W_NEUTRAL", raising=False)
    _reset(appmod)
    yield appmod
    _reset(appmod)
