"""Download reviewed, unique isolated ISL word clips for common communication vocabulary.

Videos and metadata are local-only under ignored data/external/. The dataset is multi-source;
each downloaded manifest row preserves its upstream source and stated license.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DATASET = "vidit031/isl-isolated-40words"
REVISION = "main"
TARGET_WORDS = (
    "hello", "goodbye", "thank you", "sorry", "please", "yes", "no", "help",
    "me", "you", "what", "where", "when", "today", "food", "water", "home",
    "friend", "school", "come", "go", "eat", "drink",
)
DEFAULT_OUTPUT = ROOT / "data" / "external" / "isl_common_word_videos_v1"
USER_AGENT = "MudraTrainingDataFetcher/1.0 (academic research)"


def get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def fetch_rows() -> list[dict]:
    endpoint = "https://datasets-server.huggingface.co/rows"
    rows: list[dict] = []
    for offset in range(0, 642, 100):
        url = f"{endpoint}?dataset={DATASET}&config=default&split=train&offset={offset}&length=100"
        rows.extend(item["row"] for item in get_json(url)["rows"])
    return rows


def fetch_clip(row: dict, output: Path) -> dict:
    video_path = PurePosixPath(row["video_path"])
    destination = output / video_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = row.get("sha256", "").lower()
    if destination.is_file() and expected:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected:
            return {**row, "local_path": video_path.as_posix(), "download_status": "already_present"}

    encoded_path = quote(video_path.as_posix(), safe="/")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{encoded_path}?download=true"
    temporary = destination.with_suffix(destination.suffix + ".part")
    error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if expected and digest != expected:
                raise ValueError(f"SHA-256 mismatch for {video_path}")
            temporary.replace(destination)
            return {**row, "local_path": video_path.as_posix(), "download_status": "downloaded"}
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {video_path}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = fetch_rows()
    wanted = set(TARGET_WORDS)
    selected = [row for row in rows if row.get("normalized_word") in wanted
                and row.get("review_status") == "accepted"
                and row.get("duplicate_status") == "unique"
                and row.get("dataset") != "ISLRTC_dictionary"]
    if not selected:
        raise RuntimeError("The dataset returned no accepted, unique clips for the selected vocabulary.")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "README.md").write_text(
        "# Common ISL word video clips\n\n"
        f"Source: Hugging Face `{DATASET}` at revision `{REVISION}`.\n"
        f"Downloaded {len(selected)} accepted, unique clips across "
        f"{len({row['normalized_word'] for row in selected})} words.\n\n"
        "This is an upstream multi-source aggregate, not a single-license dataset. See "
        "`manifest.csv` for each clip's source, license note, source URL, review status, "
        "duration, signer metadata, and SHA-256. Follow each upstream license and cite the "
        "original dataset before reuse or redistribution. Government open-data clips were "
        "excluded because their reuse terms need separate checking.\n\n"
        "Words in this first subset: " + ", ".join(sorted({row['normalized_word'] for row in selected})) + ".\n\n"
        "This set does not provide exact labels for `we`, `are`, `inventor(s)`, `from`, `and`, `this`, `is`, "
        "`K.S.E`, `Pune`, or `SignLine`. `me` is not relabeled as `we`; `hello` is not relabeled as "
        "`Namaste`; and `school` is not a substitute for `K.S.E Pune`. The clips have not yet been "
        "reviewed for ISL sign variants by an ISL-fluent signer.\n",
        encoding="utf-8",
    )

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_clip, row, args.output): row for row in selected}
        total = len(futures)
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if number % 25 == 0 or number == total:
                print(f"Downloaded/verified {number}/{total}")

    fields = ("normalized_word", "word", "dataset", "original_label", "signer", "split", "fps",
              "resolution", "duration", "license", "paper", "repository", "download_url",
              "video_path", "local_path", "sha256", "phash", "quality_score",
              "duplicate_status", "review_status", "download_status")
    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: (row["normalized_word"], row["dataset"], row["video_path"])))

    print(f"Saved {len(results)} clips to {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
