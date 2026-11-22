#!/usr/bin/env python3
"""Higgsfield recordings in voice-src/ -> one audio sprite per hero (voice/<lang>.mp3) + voice/index.json with [start, duration].
Word lists were recorded 25 per take with long pauses; takes are cut on silence and the piece count must match the word count."""
import json, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np

root = Path(__file__).parent
src, out = root / "voice-src", root / "voice"
raw = src / "raw"; raw.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
FF = "/opt/homebrew/opt/ffmpeg-full/bin/"
SR = 24000
reqs = json.load(open(src / "reqs.json")); keys = reqs["keys"]
chunks = json.load(open(src / "chunks.json"))

def key_of(tag):
    return tag if tag.startswith("w-") else keys[int(tag)]

for line in open(src / "urls.txt"):
    tag, url = line.split()
    f = raw / f"{key_of(tag)}.mp3"
    if not f.exists():
        urllib.request.urlretrieve(url, f)

def pcm(f):
    b = subprocess.run([FF + "ffmpeg", "-v", "error", "-i", str(f), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(b, "<i2").astype(np.float32) / 32768

def speech_runs(x, gap=0.35):
    hop = int(SR * .01); n = len(x) // hop
    e = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    on = 20 * np.log10(e / e.max()) > -38
    runs, start, last = [], None, None
    for i, v in enumerate(on):
        if v:
            if start is None: start = i
            elif (i - last) * hop / SR > gap:
                runs.append((start, last)); start = i
            last = i
    if start is not None: runs.append((start, last))
    runs = [(a, b) for a, b in runs if (b - a) * hop / SR > .08]  # drop clicks
    return [(max(0, a * hop - int(.04 * SR)), min(len(x), (b + 1) * hop + int(.06 * SR))) for a, b in runs]

clips = {l: [] for l in ("en", "ru", "ua")}
bad = []
for f in sorted(raw.glob("*.mp3")):
    k = f.stem; kind, l, rest = k.split("-", 2)
    x = pcm(f)
    if kind == "w":
        words = chunks[l][int(rest)]
        runs = speech_runs(x)
        if len(runs) != len(words):
            bad.append(f"{k}: {len(runs)} pieces for {len(words)} words"); continue
        clips[l] += [("w:" + w, x[a:b]) for w, (a, b) in zip(words, runs)]
    else:
        runs = speech_runs(x, gap=10)
        a, b = runs[0][0], runs[-1][1]
        clips[l].append(("l:" + rest, x[a:b]))

index = {}
pad = np.zeros(int(.25 * SR), np.float32)
for l, items in clips.items():
    if not items: continue
    parts, pos, idx = [], 0, {}
    for name, c in items:
        idx[name] = [round(pos / SR, 3), round(len(c) / SR, 3)]
        parts += [c, pad]; pos += len(c) + len(pad)
    wav = (np.concatenate(parts) * 32767).clip(-32768, 32767).astype("<i2").tobytes()
    subprocess.run([FF + "ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-c:a", "libmp3lame", "-b:a", "56k", str(out / f"{l}.mp3")], input=wav, check=True)
    index[l] = idx
    print(l, len(idx), "clips", (out / f"{l}.mp3").stat().st_size // 1024, "KB")
json.dump(index, open(out / "index.json", "w"), ensure_ascii=False, separators=(",", ":"))
if bad: print("CHECK:", *bad, sep="\n  ")
