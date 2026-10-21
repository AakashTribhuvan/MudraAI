"""The owner's own recordings become avatar examples for words without a training video."""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from mudra_ml.data.manifest import ManifestRow, write_manifest  # noqa: E402

from mudra_api.references import ReferenceClips  # noqa: E402


def row(tmp_path, label, sid, hand_rate):
    f = tmp_path / f"{sid}.npz"
    f.write_bytes(b"x")
    return ManifestRow(sample_id=sid, label=label, source="app", kind="landmarks", signer_id="app:owner",
                       group_id=f"app:{sid}", raw_path=str(f), landmarks_path=str(f), num_frames=40, fps=20.0,
                       duration_s=2.0, any_hand_rate=hand_rate, pose_rate=1.0, quality_flag="", split="")


def test_app_recordings_fill_only_words_without_a_training_example(tmp_path):
    refs = ReferenceClips(tmp_path / "missing_split.csv")
    refs.best["HELLO"] = ((0, -1.0), object())  # a word that has a training video
    refs.trained = {"HELLO"}
    manifest = tmp_path / "manifest_app.csv"
    write_manifest([row(tmp_path, "HELP", "a1", 0.5), row(tmp_path, "HELP", "a2", 0.8),
                    row(tmp_path, "HELLO", "a3", 0.9)], manifest)
    assert refs.add_app_recordings(manifest) == 1
    assert "HELP" in refs.available and refs.best["HELP"][1].sample_id == "a2"  # the take with the most hands
    assert not isinstance(refs.best["HELLO"][1], ManifestRow)  # the training example stays
    # a new, better take replaces the example after the next refresh
    write_manifest([row(tmp_path, "HELP", "a1", 0.5), row(tmp_path, "HELP", "a2", 0.8),
                    row(tmp_path, "HELP", "a4", 0.95)], manifest)
    refs.add_app_recordings(manifest)
    assert refs.best["HELP"][1].sample_id == "a4"


def test_missing_manifest_adds_nothing(tmp_path):
    refs = ReferenceClips(tmp_path / "missing_split.csv")
    assert refs.add_app_recordings(tmp_path / "none.csv") == 0 and refs.available == set()
