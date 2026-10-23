"""Compare streaming ASR models on recordings with known text and word timings: word error rate, delay from a
word's start in the audio to its first appearance in a partial result, and compute time."""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app" / "backend"))
from mudra_api.speech import StreamingASR  # noqa: E402

D = ROOT / "experiments" / "tools" / "meeting_tests" / "asr_compare"  # s1.wav + s1.txt + s1.json ...
MODELS = ROOT / "app" / "backend" / "var" / "asr_models"
norm = lambda t: re.sub(r"[^a-z' ]+", " ", t.lower()).split()  # noqa: E731


def wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = range(len(ref) + 1)
    d[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[len(ref), len(hyp)]


for model in sys.argv[1:]:
    asr = StreamingASR(MODELS / model)
    t = time.perf_counter(); asr.recognizer(); load = time.perf_counter() - t
    errors = words_total = 0
    delays = []
    compute = audio = 0.0
    misses = []
    for wav in sorted(D.glob("s*.wav"), key=lambda p: int(p.stem[1:])):
        ref = norm(wav.with_suffix(".txt").read_text(encoding="utf-8-sig"))
        timing = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8-sig"))
        onset = {}
        for w in timing:
            for k in norm(w["text"]):
                onset.setdefault(k, w["start_ms"] / 1000)
        pcm = np.frombuffer(wav.read_bytes()[44:], dtype="<i2")
        s = asr.session()
        finals, seen = [], {}
        t = time.perf_counter()
        for i in range(0, len(pcm), 1600):
            for e in s.accept(pcm[i:i + 1600].tobytes()):
                if e["event"] == "final":
                    finals.append(e["text"])
                for k in norm(e["text"]):
                    seen.setdefault(k, (i + 1600) / 16000)
        finals += [e["text"] for e in s.flush()]
        compute += time.perf_counter() - t
        audio += len(pcm) / 16000
        hyp = norm(" ".join(finals))
        e = wer(ref, hyp)
        errors += e
        words_total += len(ref)
        if e:
            misses.append(f"{wav.stem}: {' '.join(hyp)}")
        delays += [seen[k] - onset[k] for k in seen if k in onset]
    delays = np.array(delays)
    print(f"{model}\n  WER {errors}/{words_total} = {errors / words_total:.1%}; first-partial delay after word start: median {np.median(delays):.2f} s, p90 {np.percentile(delays, 90):.2f} s;"
          f" compute {compute:.2f} s for {audio:.0f} s audio ({audio / compute:.0f}x real time); load {load:.1f} s")
    for m in misses:
        print("   ", m)
