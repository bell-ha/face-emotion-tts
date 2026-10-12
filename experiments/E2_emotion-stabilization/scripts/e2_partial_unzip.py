"""Extract the audio-visual (modality 01) RAVDESS mp4s from partially downloaded Zenodo zips.

Each Video_Speech_Actor_XX.zip stores the 60 AV clips (01-*) first and the 60 video-only clips
(02-*) second. The video track is identical, so only the first half of each zip is needed.
The central directory is read from the remote file with an HTTP range request; entries whose
bytes are fully present in the local partial file are extracted. Then the download is stopped.
"""
import os
import struct
import subprocess
import sys
import time
import urllib.request
import zlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
Z = os.path.join(ROOT, "data", "ravdess", "zips")
VID = os.path.join(ROOT, "data", "ravdess", "videos")
URL = "https://zenodo.org/api/records/1188976/files/Video_Speech_Actor_{a}.zip/content"
SIZES = {"01": 552968353, "02": 570673024, "03": 567824665, "04": 546560009,
         "05": 562966425, "06": 568517859, "07": 565652404, "08": 562573079}


def central_dir(a):
    size = SIZES[a]
    req = urllib.request.Request(URL.format(a=a), headers={"Range": f"bytes={size-65536}-{size-1}"})
    d = urllib.request.urlopen(req, timeout=60).read()
    base = size - len(d)
    i = d.rfind(b"PK\x05\x06")
    n, cds, off = struct.unpack("<HII", d[i + 10:i + 20])
    cd = d[off - base:off - base + cds]
    p, ents = 0, []
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        meth = struct.unpack("<H", cd[p + 10:p + 12])[0]
        crc, csz, usz = struct.unpack("<III", cd[p + 16:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        lo = struct.unpack("<I", cd[p + 42:p + 46])[0]
        ents.append(dict(name=cd[p + 46:p + 46 + fl].decode(), lo=lo, csz=csz, usz=usz, meth=meth, crc=crc))
        p += 46 + fl + el + cl
    return ents


def try_extract(a, ents):
    f = os.path.join(Z, f"Video_Speech_Actor_{a}.zip")
    have = os.path.getsize(f) if os.path.exists(f) else 0
    want = [e for e in ents if os.path.basename(e["name"]).startswith("01-")]
    ok = 0
    with open(f, "rb") as fh:
        for e in want:
            dst = os.path.join(VID, f"Actor_{a}", os.path.basename(e["name"]))
            if os.path.exists(dst):
                ok += 1; continue
            fh.seek(e["lo"]); h = fh.read(30)
            if len(h) < 30:
                continue
            fl, el = struct.unpack("<HH", h[26:30])
            start = e["lo"] + 30 + fl + el
            if start + e["csz"] > have:
                continue
            fh.seek(start); data = fh.read(e["csz"])
            if e["meth"] == 8:
                data = zlib.decompress(data, -15)
            if zlib.crc32(data) & 0xFFFFFFFF != e["crc"]:
                print("CRC mismatch", e["name"]); continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as o:
                o.write(data)
            ok += 1
    return ok, len(want)


def main():
    actors = sys.argv[1:] or sorted(SIZES)
    cds = {a: central_dir(a) for a in actors}
    done = set()
    while len(done) < len(actors):
        for a in actors:
            if a in done:
                continue
            ok, n = try_extract(a, cds[a])
            if ok == n:
                done.add(a)
                subprocess.run(["pkill", "-f", f"Video_Speech_Actor_{a}.zip"])
                print(f"Actor_{a}: {ok}/{n} AV clips extracted, download stopped", flush=True)
            elif subprocess.run(["pgrep", "-f", f"Video_Speech_Actor_{a}.zip"],
                                capture_output=True).returncode != 0:
                # (re)start a resumable download for this actor
                f = os.path.join(Z, f"Video_Speech_Actor_{a}.zip")
                subprocess.Popen(["curl", "-sL", "--speed-limit", "1000", "--speed-time", "60", "-C", "-",
                                  "-o", f, URL.format(a=a)])
                print(f"Actor_{a}: (re)started download", flush=True)
        if len(done) < len(actors):
            time.sleep(20)


if __name__ == "__main__":
    main()
