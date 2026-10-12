"""조건 A~E 음성 생성. 프리셋은 prosody.py의 고정값(=results/presets_frozen.json)을 쓴다.

A        : 기존 OpenAI gpt-4o-mini-tts 샘플(speed만, S01/S02), 복사만
B        : edge-tts SunHi 중립(문장당 1개; 모든 감정에 동일 음성 = 통제 조건)
C        : edge-tts + 감정 프리셋(rate/pitch/volume)
D        : C 출력 + PSOLA 후처리 (중립에도 dev 튜닝된 범위 축소·약간 낮춤 후처리)
E_lex_B  : (참고) 어휘가 감정적인 문장 × 중립 운율
E_lex_D  : (참고) 어휘가 감정적인 문장 × D
spk_ref  : 화자 유사도 척도 참고용 — 남성 화자 ko-KR-InJoonNeural 중립
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
import prosody as P  # noqa: E402


def _norm(d):
    return json.loads(json.dumps(d))


def check_frozen():
    fz = json.loads((C.RESULTS / "presets_frozen.json").read_text())
    assert _norm(fz["C_PRESETS"]) == _norm(P.C_PRESETS), "prosody.C_PRESETS != frozen"
    assert _norm(fz["D_PRESETS"]) == _norm(P.D_PRESETS), "prosody.D_PRESETS != frozen"


async def edge_to(path: Path, text, cp, voice=P.VOICE, sem=None):
    if path.exists():
        return
    async with sem:
        for i in range(4):
            try:
                mp3 = await P._edge_async(text, cp["rate"], cp["pitch"], cp["volume"], voice)
                break
            except Exception:
                await asyncio.sleep(2 + 2 * i)
        else:
            raise RuntimeError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, P.mp3_to_array(mp3), P.SR, subtype="PCM_16")


async def main():
    sem = asyncio.Semaphore(4)
    A = C.AUDIO
    jobs = []
    neutral = P.C_PRESETS["neutral"]
    for sid, text in C.SENTENCES.items():
        jobs.append(edge_to(A / "B" / f"{sid}_neutral.wav", text, neutral, sem=sem))
        jobs.append(edge_to(A / "spk_ref" / f"{sid}_InJoon_neutral.wav", text, neutral,
                            voice="ko-KR-InJoonNeural", sem=sem))
        for emo in C.EMOS:
            jobs.append(edge_to(A / "C" / f"{sid}_{emo}.wav", text, P.C_PRESETS[emo], sem=sem))
    for emo, sents in C.LEXICAL.items():
        for i, text in enumerate(sents):
            jobs.append(edge_to(A / "E_lex_B" / f"L{i+1}_{emo}.wav", text, neutral, sem=sem))
            jobs.append(edge_to(A / "_E_lex_C" / f"L{i+1}_{emo}.wav", text, P.C_PRESETS[emo], sem=sem))
    await asyncio.gather(*jobs)


def dsp_pass(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("*.wav")):
        emo = f.stem.split("_")[-1]
        y, sr = sf.read(f)
        sf.write(dst_dir / f.name, P.apply_dsp(y, sr, P.D_PRESETS[emo]), sr, subtype="PCM_16")


if __name__ == "__main__":
    check_frozen()
    # A: 기존 샘플 복사
    adir = C.AUDIO / "A"
    adir.mkdir(parents=True, exist_ok=True)
    for pre, sid in (("greeting", "S01"), ("rabbit", "S02")):
        for emo in C.EMOS:
            shutil.copy(C.PROJ / "assets" / "audio" / f"{pre}_{emo}.wav", adir / f"{sid}_{emo}.wav")
    asyncio.run(main())
    dsp_pass(C.AUDIO / "C", C.AUDIO / "D")
    dsp_pass(C.AUDIO / "_E_lex_C", C.AUDIO / "E_lex_D")
    shutil.rmtree(C.AUDIO / "_E_lex_C")
    for d in sorted(C.AUDIO.iterdir()):
        if d.is_dir():
            print(d.name, len(list(d.glob("*.wav"))))
