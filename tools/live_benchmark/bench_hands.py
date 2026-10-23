"""Hand detection during a bench run: share of each sign's frames with a hand / with both hands (server log)."""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "experiments" / "tools" / "live_benchmark"
for label in sys.argv[1:]:
    run = json.loads((D / f"bench_{label}.json").read_text(encoding="utf-8"))
    start = datetime.fromisoformat(run["startedAt"].replace("Z", "+00:00")).astimezone().replace(tzinfo=None) - timedelta(seconds=1)
    end = datetime.fromisoformat(run["endedAt"].replace("Z", "+00:00")).astimezone().replace(tzinfo=None) + timedelta(seconds=2)
    signs = []
    for line in open(ROOT / "experiments" / "logs" / "stream_events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e.get("event") == "sign_ended" and start <= datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S") <= end:
            signs.append(e)
    n = max(len(signs), 1)
    print(f"{label}: {len(signs)} signs, any hand {sum(e['hand_rate'] for e in signs) / n:.1%}, "
          f"both hands {sum(e['both_hands_rate'] for e in signs) / n:.1%} of sign frames")
