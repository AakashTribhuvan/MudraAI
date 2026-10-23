"""Fake webcam for Chrome (--use-file-for-fake-video-capture): held-out INCLUDE test videos of known words, joined
into one 640x480 25 fps Y4M with 1 s of a still frame (hands down) between signs. Writes the label order as JSON."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "tools" / "live_benchmark" / (sys.argv[3] if len(sys.argv) > 3 else "fakecam")
OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
W, H = 640, 480
FPS = int(sys.argv[2]) if len(sys.argv) > 2 else 25

rows = [r for r in csv.DictReader(open(ROOT / "data/metadata/splits/combined_words_v3/test.csv", encoding="utf-8"))
        if r["source"] == "include_video" and r.get("raw_path") and float(r.get("any_hand_rate") or 0) > 0.85]
seen, picks = set(), []
for r in rows:
    if r["label"] in seen:
        continue
    p = Path(r["raw_path"]) if Path(r["raw_path"]).is_absolute() else ROOT / r["raw_path"]
    if p.exists():
        picks.append((r["label"], p))
        seen.add(r["label"])
    if len(picks) == N:
        break


def frames(path):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out, t, k = [], 0.0, 0
    ok, f = cap.read()
    while ok:
        # resample to 25 fps and letterbox to 640x480
        while t <= k / src_fps:
            h, w = f.shape[:2]
            s = min(W / w, H / h)
            img = cv2.resize(f, (int(w * s), int(h * s)))
            canvas = np.zeros((H, W, 3), np.uint8)
            y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
            canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
            out.append(canvas)
            t += 1 / FPS
        ok, f = cap.read()
        k += 1
    return out


y4m = open(OUT / "signs.y4m", "wb")
y4m.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
timeline, t = [], 0.0


def write(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    y4m.write(b"FRAME\n" + yuv.tobytes())


for label, path in picks:
    fr = frames(path)
    still = fr[0]
    for _ in range(FPS):  # 1 s still (the video's first frame: hands down)
        write(still)
    t += 1
    timeline.append({"label": label, "start_s": round(t, 2), "end_s": round(t + len(fr) / FPS, 2), "video": str(path)})
    for img in fr:
        write(img)
    t += len(fr) / FPS
for _ in range(2 * FPS):
    write(still)
y4m.close()
(OUT / "signs.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
print(len(picks), "signs,", round(t + 2, 1), "s:", [p[0] for p in picks])
