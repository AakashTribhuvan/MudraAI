"""Scores a bench run: the server's sign events in the run's time window vs the fake webcam's word order."""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "experiments" / "tools" / "live_benchmark"
run = json.loads((D / f"bench_{sys.argv[1]}.json").read_text(encoding="utf-8"))
truth = [t["label"] for t in json.loads((D / os.environ.get("CAMDIR", "fakecam") / "signs.json").read_text(encoding="utf-8"))]
start = datetime.fromisoformat(run["startedAt"].replace("Z", "+00:00")).astimezone().replace(tzinfo=None) - timedelta(seconds=1)
end = datetime.fromisoformat(run["endedAt"].replace("Z", "+00:00")).astimezone().replace(tzinfo=None) + timedelta(seconds=2)
events = []
for line in open(ROOT / "experiments" / "logs" / "stream_events.jsonl", encoding="utf-8"):
    e = json.loads(line)
    t = datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S")
    if start <= t <= end:
        events.append(e)
signs = [e for e in events if e.get("event") == "sign_ended"]
diag = [e for e in events if (e.get("event") or e.get("type")) == "client_diag"][1:]
med = lambda k: sorted(e.get(k, 0) for e in diag)[len(diag) // 2] if diag else None  # noqa: E731
print(f"{run['label']}: client tracking {med('tracking_fps')} fps, camera {med('camera_fps')} fps, {med('inference_ms')} ms/frame ({diag[-1].get('delegate') if diag else '-'})")
committed = [e["sign"] for e in signs if e.get("status") == "ok"]
fps = sorted(e.get("fps", 0) for e in signs)
right = sum(1 for w in truth if w in committed)
wrong = [s for s in committed if s not in truth]
print(f"{run['label']}: {len(signs)} signs segmented, server fps median {fps[len(fps) // 2] if fps else 0}; "
      f"committed {len(committed)}: right {right}/{len(truth)}, wrong {len(wrong)} {wrong}; "
      f"unsure/missed: {[(e.get('status'), (e.get('top') or [[None]])[0][0]) for e in signs if e.get('status') != 'ok']}")

lat = run.get("lat") or {}
if lat.get("camAt") and lat.get("signs"):
    tl = json.loads((D / "fakecam" / "signs.json").read_text(encoding="utf-8"))
    ends = [x for x in lat["signs"] if x["ev"] == "sign_ended"]
    out = []
    for item in tl:
        end_ms = lat["camAt"] + item["end_s"] * 1000
        hit = [x for x in ends if x.get("sign") == item["label"] and x["t"] > end_ms - 3000]
        if hit:
            out.append(round(hit[0]["t"] - end_ms))
    out.sort()
    print(f"{run['label']}: sign end in the video -> word in the page: median {out[len(out) // 2] if out else None} ms, all {out}")
