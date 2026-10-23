"""Download exact Good / Goodmorning / Goodafternoon clips from ISL500."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DATASET = "ISL500/ISL-DATA"
REVISION = "main"
WORDS = {"good", "goodmorning", "goodafternoon"}
OUT = ROOT / "data" / "external" / "signline_intro_terms"
UA = "SignLineAcademicResearch/1.0"


def get_json(url: str) -> list[dict]:
    request = Request(url, headers={"User-Agent": UA})
    with urlopen(request, timeout=90) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    files: list[dict] = []
    for signer_number in range(1, 16):
        user = f"User{signer_number:03d}"
        url = (f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/Videos/{user}"
               "?recursive=true&expand=false")
        for entry in get_json(url):
            if entry.get("type") != "file" or Path(entry.get("path", "")).suffix.lower() != ".mp4":
                continue
            label = Path(entry["path"]).name.split("_", 1)[0].casefold()
            if label in WORDS:
                files.append({"signer": user, "word": label, "path": entry["path"], "size": entry.get("size", 0)})
    selected = sorted(files, key=lambda row: (row["word"], row["signer"], row["path"]))
    counts = {word: sum(item["word"] == word for item in selected) for word in sorted(WORDS)}
    if any(count != 15 for count in counts.values()):
        raise RuntimeError(f"Expected one clip for each of 15 signers per class; found {counts}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, item in enumerate(selected, start=1):
        destination = args.output / item["word"] / item["signer"] / Path(item["path"]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or (item["size"] and destination.stat().st_size != item["size"]):
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{quote(item['path'], safe='/')}?download=true"
            temporary = destination.with_suffix(destination.suffix + ".part")
            error: Exception | None = None
            for attempt in range(4):
                try:
                    request = Request(url, headers={"User-Agent": UA})
                    with urlopen(request, timeout=180) as response, temporary.open("wb") as target:
                        while chunk := response.read(1024 * 1024):
                            target.write(chunk)
                    if item["size"] and temporary.stat().st_size != item["size"]:
                        raise OSError(f"size mismatch: {temporary.stat().st_size} != {item['size']}")
                    temporary.replace(destination)
                    error = None
                    break
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    error = exc
                    temporary.unlink(missing_ok=True)
                    if attempt < 3:
                        time.sleep(2 * (attempt + 1))
            if error:
                raise RuntimeError(f"Failed to download {item['path']}: {error}")
        rows.append({**item, "local_path": destination.relative_to(args.output).as_posix(), "downloaded": destination.stat().st_size})
        if number % 10 == 0 or number == len(selected):
            print(f"Verified/downloaded {number}/{len(selected)} clips")

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"dataset": DATASET, "revision": REVISION, "counts": counts,
                      "total_bytes": sum(row["downloaded"] for row in rows),
                      "manifest": str(args.output / "manifest.csv")}, indent=2))


if __name__ == "__main__":
    main()
