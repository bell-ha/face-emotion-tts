"""E5: fetch AV clips of additional RAVDESS actors (default 09-24) with the E2 partial-unzip trick.
Reuses experiments/E2_stabilization/scripts/e2_partial_unzip.py unchanged (only extends its SIZES table).
Zip sizes from the Zenodo API (record 1188976)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "E2_stabilization", "scripts"))
import e2_partial_unzip as pu

pu.SIZES.update({"09": 523489249, "10": 565580163, "11": 518634370, "12": 557928718,
                 "13": 501877549, "14": 552484059, "15": 525205576, "16": 562416513,
                 "17": 551688845, "18": 565655520, "19": 581121687, "20": 561349264,
                 "21": 590292896, "22": 560907466, "23": 545177865, "24": 595799165})
os.makedirs(pu.Z, exist_ok=True)
if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += [f"{a:02d}" for a in range(9, 25)]
    for a in sys.argv[1:]:  # e2 script expects the (partial) file to exist
        open(os.path.join(pu.Z, f"Video_Speech_Actor_{a}.zip"), "ab").close()
    pu.main()
