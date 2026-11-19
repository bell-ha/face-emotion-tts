"""2026-11-19 재측정(latency_rerun.py) 요약과 기존(2026-10-12 02:35) 값과의 비교.

결과: results/run_2026-11-19/summary.csv, comparison.csv, summary.json (기존 results/summary.csv는 읽기만 함)
실행: .venv/bin/python experiments/E4_system-latency/scripts/analyze_rerun.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

E4 = Path(__file__).resolve().parents[1]
R0 = E4 / "results"
R = R0 / "run_2026-11-19"
LEN_KO = {"short": "짧은 문장", "medium": "중간 문장", "long": "긴 문장"}


def st(x):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    return {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1) if len(x) > 1 else 0.0,
            "median": np.median(x), "p95": np.percentile(x, 95), "min": x.min(), "max": x.max()}


def single_rows(syn, tag, with_encode=True):
    rows = []
    for L in LEN_KO:
        d = syn[syn.length == L]
        g = f"문장 합성{tag}-{LEN_KO[L]}({d.chars.mean():.0f}자)"
        cols = [("network_ms", "edge-tts 네트워크"), ("decode_ms", "MP3 디코딩"), ("dsp_ms", "DSP(PSOLA)")]
        if with_encode:
            cols.append(("encode_ms", "WAV 인코딩"))
        cols += [("export_ms", "MP3 내보내기"), ("client_ms", "/tts 왕복(클라이언트)")]
        for col, name in cols:
            rows.append({"group": g, "item": name, "unit": "ms", **st(d[col])})
        rows.append({"group": g, "item": "음성 길이", "unit": "s", **st(d.audio_sec)})
        rows.append({"group": g, "item": "RTF(왕복/음성길이)", "unit": "-", **st(d.client_ms / 1000 / d.audio_sec)})
    return rows


def main():
    fast = pd.read_csv(R / "raw_fast.csv")
    syn = pd.read_csv(R / "raw_synth_sentence.csv")
    multi = pd.read_csv(R / "raw_multi_sentence.csv")
    dps = pd.read_csv(R / "dp_single" / "raw_synth_sentence.csv")
    for nm, df in (("fast", fast), ("syn", syn), ("multi", multi), ("dp", dps)):
        assert (df.status == 200).all(), f"failed requests remain in {nm}"
    rows = []
    for stg in ["set_emotion", "resolve_one", "set_emotion_probs", "resolve_one_mass"]:
        rows.append({"group": "HTTP 왕복", "item": stg, "unit": "ms", **st(fast[fast.stage == stg].ms)})
    rows += single_rows(syn, "")
    rows += single_rows(dps, "(Dp)", with_encode=False)
    for n in [1, 3, 5]:
        d = multi[multi.n_sent == n]
        g = f"다문장 /tts-{n}문장"
        rows.append({"group": g, "item": "종단 간 왕복(클라이언트)", "unit": "ms", **st(d.client_ms)})
        rows.append({"group": g, "item": "edge-tts 네트워크 합계", "unit": "ms", **st(d.network_ms_sum)})
        rows.append({"group": g, "item": "DSP 합계", "unit": "ms", **st(d.dsp_ms_sum)})
        rows.append({"group": g, "item": "병합·MP3 내보내기", "unit": "ms", **st(d.merge_export_ms)})
        rows.append({"group": g, "item": "음성 길이", "unit": "s", **st(d.audio_sec)})
        rows.append({"group": g, "item": "RTF", "unit": "-", **st(d.client_ms / 1000 / d.audio_sec)})
    summ = pd.DataFrame(rows)
    summ.to_csv(R / "summary.csv", index=False, float_format="%.3f")

    # 기존 값과 비교 (같은 group·item). 원자료끼리 Mann–Whitney U(양측)도 붙인다.
    old = pd.read_csv(R0 / "summary.csv")
    cmp = summ.merge(old, on=["group", "item", "unit"], how="left", suffixes=("_new", "_old"))
    cmp["diff_mean"] = cmp.mean_new - cmp.mean_old
    cmp["diff_median"] = cmp.median_new - cmp.median_old
    cmp["diff_p95"] = cmp.p95_new - cmp.p95_old
    old_fast = pd.read_csv(R0 / "raw_fast.csv")
    old_syn = pd.read_csv(R0 / "raw_synth_sentence.csv")
    old_multi = pd.read_csv(R0 / "raw_multi_sentence.csv")
    old_dp = pd.read_csv(R0 / "dp_single" / "raw_synth_sentence.csv")
    pv = {}
    for stg in ["set_emotion", "resolve_one"]:
        pv[("HTTP 왕복", stg)] = ss.mannwhitneyu(fast[fast.stage == stg].ms, old_fast[old_fast.stage == stg].ms).pvalue
    for tag, new_, old_ in (("", syn, old_syn), ("(Dp)", dps, old_dp)):
        for L in LEN_KO:
            g = f"문장 합성{tag}-{LEN_KO[L]}({new_[new_.length == L].chars.mean():.0f}자)"
            for col, name in (("network_ms", "edge-tts 네트워크"), ("dsp_ms", "DSP(PSOLA)"),
                              ("client_ms", "/tts 왕복(클라이언트)")):
                pv[(g, name)] = ss.mannwhitneyu(new_[new_.length == L][col], old_[old_.length == L][col]).pvalue
    for n in [1, 3, 5]:
        pv[(f"다문장 /tts-{n}문장", "종단 간 왕복(클라이언트)")] = ss.mannwhitneyu(
            multi[multi.n_sent == n].client_ms, old_multi[old_multi.n_sent == n].client_ms).pvalue
        pv[(f"다문장 /tts-{n}문장", "edge-tts 네트워크 합계")] = ss.mannwhitneyu(
            multi[multi.n_sent == n].network_ms_sum, old_multi[old_multi.n_sent == n].network_ms_sum).pvalue
    cmp["mwu_p"] = [pv.get((g, i)) for g, i in zip(cmp.group, cmp.item)]
    cols = ["group", "item", "unit", "n_new", "mean_new", "median_new", "p95_new", "mean_old", "median_old", "p95_old",
            "diff_mean", "diff_median", "diff_p95", "mwu_p"]
    cmp[cols].to_csv(R / "comparison.csv", index=False, float_format="%.4g")

    share = {L: float((syn[syn.length == L].network_ms / syn[syn.length == L].client_ms).mean()) for L in LEN_KO}
    share_dp = {L: float((dps[dps.length == L].network_ms / dps[dps.length == L].client_ms).mean()) for L in LEN_KO}
    share_multi = {int(n): float((multi[multi.n_sent == n].network_ms_sum / multi[multi.n_sent == n].client_ms).mean())
                   for n in [1, 3, 5]}
    tags = fast[fast.stage == "resolve_one_mass"].tag.fillna("")
    tags0 = fast[fast.stage == "resolve_one"].tag.fillna("")
    env = json.loads((R / "env.json").read_text())
    envdp = json.loads((R / "dp_single" / "env.json").read_text())
    js = {"network_share_single": share, "network_share_single_dp": share_dp, "network_share_multi": share_multi,
          "resolve_mass_tag_counts": tags.replace("", "(none)").value_counts().to_dict(),
          "resolve_duration_tag_counts": tags0.replace("", "(none)").value_counts().to_dict(),
          "retries": {"D": env["failures"], "Dp": envdp["failures"]},
          "attempts_gt1": {"single": int((syn.attempts > 1).sum()), "multi": int((multi.attempts > 1).sum()),
                           "dp": int((dps.attempts > 1).sum())},
          "cold_start_ms": {"D": env["cold_first_tts"]["client_ms"], "Dp": envdp["cold_first_tts"]["client_ms"]},
          "cold_decode_ms": {"D": env["cold_first_tts"]["segments"][0].get("decode_ms"),
                             "Dp": envdp["cold_first_tts"]["segments"][0].get("decode_ms")}}
    (R / "summary.json").write_text(json.dumps(js, ensure_ascii=False, indent=2, default=float))
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_columns", 20):
        print(cmp[cols].round(3).to_string(index=False))
    print(json.dumps(js, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
