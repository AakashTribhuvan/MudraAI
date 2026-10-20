""""How to sign it": one real training example per word, served as landmark frames.

The example is a training clip re-extracted with the same MediaPipe pipeline as the webcam
(team recordings first, then INCLUDE videos), with the best hand visibility. Only landmark
points are served, never video, so no signer's face or body image leaves the machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mudra_ml.data.manifest import read_manifest
from mudra_ml.preprocessing.sequence import fill_hand_gaps, load_sequence

PREFERRED_SOURCES = ("islwords", "include_video", "include")  # browser-like pipeline first; pose release as fallback
MAX_FPS = 15.0


class ReferenceClips:
    def __init__(self, split_csv: Path, sources: tuple[str, ...] = PREFERRED_SOURCES, world_dir: Path | None = None):
        # optional re-extraction with MediaPipe world landmarks (training/scripts/extract_reference_world.py):
        # same video, image + world (metres) landmarks; used by the avatar for real depth
        self.world_dir = Path(world_dir) if world_dir else None
        self.best: dict[str, object] = {}
        if Path(split_csv).exists():
            for r in read_manifest(Path(split_csv)):
                if r.source not in sources or r.quality_flag or not Path(r.landmarks_path).exists():
                    continue
                rank = (sources.index(r.source), -r.any_hand_rate)
                cur = self.best.get(r.label)
                if cur is None or rank < cur[0]:
                    self.best[r.label] = (rank, r)
        self.available = set(self.best)
        self._cache: dict[str, dict] = {}

    def get(self, sign: str) -> dict | None:
        if sign not in self.best:
            return None
        world = self.world_dir / f"{sign}.npz" if self.world_dir else None
        if sign not in self._cache and world is not None and world.exists():
            self._cache[sign] = self._from_world(sign, world)
        if sign not in self._cache:
            row = self.best[sign][1]
            seq = fill_hand_gaps(load_sequence(Path(row.landmarks_path)), 0.4)  # bridge motion-blur gaps
            fps = float(seq.meta.get("fps") or 25.0)
            step = max(1, round(fps / MAX_FPS))
            frames = []
            for i in range(0, seq.num_frames, step):
                frames.append({
                    "hands": [np.round(seq.hands[i, h], 4).tolist() if seq.hand_present[i, h] else None for h in range(2)],
                    "pose": np.round(seq.pose[i], 4).tolist() if seq.pose_present[i] else None,
                })
            self._cache[sign] = {"sign": sign, "fps": fps / step, "width": int(seq.meta.get("width", 0)),
                                 "height": int(seq.meta.get("height", 0)), "frames": frames}
        return self._cache[sign]

    @staticmethod
    def _from_world(sign: str, path: Path) -> dict:
        z = np.load(path, allow_pickle=False)
        fps = float(z["fps"]) or 25.0
        step = max(1, round(fps / MAX_FPS))
        r4 = lambda a: np.round(a, 4).tolist()
        frames = []
        for i in range(0, len(z["timestamps_ms"]), step):
            hp = z["hand_present"][i]
            pose_ok = bool(z["pose_present"][i])
            frames.append({
                "hands": [r4(z["hands"][i, h]) if hp[h] else None for h in range(2)],
                "pose": r4(z["pose"][i]) if pose_ok else None,
                "world": {"hands": [r4(z["hands_world"][i, h]) if hp[h] else None for h in range(2)],
                          "pose": r4(z["pose_world"][i]) if pose_ok else None},
            })
        return {"sign": sign, "fps": fps / step, "width": int(z["width"]), "height": int(z["height"]),
                "frames": frames, "world": True}
