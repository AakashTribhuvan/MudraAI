"""Build the "I don't know this sign" boundaries for an exported word model.

Every known word's training clips are embedded; their centroid and the spread of their similarity to it
are saved as <model>/known_boundaries.npz (aggregate vectors only, no clips). The live strictness is
``known_boundary_k`` in training/configs/streaming.json (None = off), chosen on validation.

    python training/scripts/build_known_boundaries.py --model app/backend/models/<model> \
        --split data/metadata/splits/combined_words_v4_app/train.csv
"""

from __future__ import annotations

import argparse

import numpy as np

from mudra_ml.config import project_path
from mudra_ml.data.manifest import read_manifest
from mudra_ml.fewshot import KnownWordBoundaries
from mudra_ml.preprocessing.sequence import load_sequence
from mudra_ml.streaming import ModelBundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()
    bundle = ModelBundle(project_path(args.model))
    rows = [r for r in read_manifest(project_path(args.split)) if r.label in bundle.classes]
    embs = []
    for r in rows:
        s = load_sequence(project_path(r.landmarks_path))
        embs.append(bundle.embed([s], int(s.meta.get("width", 0)), int(s.meta.get("height", 0)))[0])
    kb = KnownWordBoundaries(np.array([r.label for r in rows]), np.stack(embs), k=2.0)
    out = project_path(args.model) / "known_boundaries.npz"
    kb.save(out)
    print(f"{len(kb.classes)} words from {len(rows)} training clips -> {out}")


if __name__ == "__main__":
    main()
