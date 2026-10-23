"""Downloads the streaming speech-recognition model the meeting features use (sherpa-onnx, k2-fsa, Apache-2.0) into
app/backend/var/asr_models/ (git-ignored). Keeps only the int8 model files, tokens.txt and the README.

    python tools/setup/download_asr_model.py                  # default model (2023-06-26, 310 MB download, 68 MB kept)
    python tools/setup/download_asr_model.py --model small    # 20M model (128 MB download, 42 MB kept)

Does nothing when the model is already there. Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "app" / "backend" / "var" / "asr_models"
RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
MODELS = {
    "default": "sherpa-onnx-streaming-zipformer-en-2023-06-26",  # settings.py default (fewer word errors)
    "small": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
}


def present(folder: Path) -> bool:
    return (folder / "tokens.txt").exists() and all(any(folder.glob(f"{p}-*.int8.onnx")) for p in ("encoder", "decoder", "joiner"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="default")
    args = ap.parse_args()
    name = MODELS[args.model]
    target = DEST / name
    if present(target):
        print(f"Speech model already present: {target}")
        return 0
    DEST.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE}{name}.tar.bz2"
    with tempfile.TemporaryDirectory(prefix="mudra_asr_") as tmp:
        archive = Path(tmp) / f"{name}.tar.bz2"
        print(f"Downloading {url} ...")
        with urllib.request.urlopen(url) as r, open(archive, "wb") as f:  # noqa: S310 (fixed https URL)
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done >> 20} / {total >> 20} MB", end="", flush=True)
        print()
        with tarfile.open(archive) as t:
            keep = [m for m in t.getmembers() if m.isfile() and (
                m.name.endswith(".int8.onnx") or m.name.endswith("tokens.txt") or m.name.lower().endswith("readme.md"))]
            t.extractall(DEST, members=keep, filter="data")  # data filter: no absolute paths or links
    if not present(target):
        print(f"The archive did not contain the expected files; see {target}", file=sys.stderr)
        shutil.rmtree(target, ignore_errors=True)
        return 1
    size = sum(p.stat().st_size for p in target.iterdir()) >> 20
    print(f"Speech model ready: {target} ({size} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
