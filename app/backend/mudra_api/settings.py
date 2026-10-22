"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _origins() -> list[str]:
    raw = os.getenv("MUDRA_CORS_ORIGINS", "")
    extra = [o.strip() for o in raw.split(",") if o.strip()]
    return [*DEFAULT_ORIGINS, *extra]


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


@dataclass
class Settings:
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_MODEL_DIR", BACKEND_DIR / "models" / "isl_mudra_combined_v15b_bilstm"))
    )
    device: str = field(default_factory=lambda: os.getenv("MUDRA_DEVICE", "cpu"))
    # Meeting speech (mudra_api/speech.py): streaming ASR model (sherpa-onnx, downloaded locally, not in git); the
    # 2023-06-26 zipformer beat the 20M one on content words (it keeps the first word after silence), 2026-10-21
    asr_model_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "MUDRA_ASR_MODEL_DIR", BACKEND_DIR / "var" / "asr_models" / "sherpa-onnx-streaming-zipformer-en-2023-06-26")))
    cors_origins: list[str] = field(default_factory=_origins)
    # Overrides the validation-chosen threshold from the model card when set.
    uncertain_threshold: float | None = field(default_factory=lambda: _optional_float("MUDRA_UNCERTAIN_THRESHOLD"))
    # Below this share of frames with a visible hand, the API reports "no_hands" instead of guessing.
    min_hand_rate: float = field(default_factory=lambda: float(os.getenv("MUDRA_MIN_HAND_RATE", "0.3")))
    # One line per finished live sign (decision summary only), for diagnosing live use
    events_log: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_EVENTS_LOG", BACKEND_DIR.parents[1] / "experiments" / "logs" / "stream_events.jsonl"))
    )
    # Opt-in: save each live /stream session (landmarks + client diagnostics, no video) here as .npz,
    # to replay a real session offline (training/scripts/replay_session.py). Unset = nothing is saved.
    session_dir: Path | None = field(
        default_factory=lambda: Path(os.environ["MUDRA_SESSION_DIR"]) if os.getenv("MUDRA_SESSION_DIR") else None
    )
    # Training split whose clips are shown as "how to sign it" examples
    reference_split: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_REFERENCE_SPLIT", BACKEND_DIR.parents[1] / "data" / "metadata" / "splits" / "combined_words_v3" / "train.csv"))
    )
    # Optional world-landmark re-extraction of those examples (extract_reference_world.py), used by the avatar
    reference_world_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_REFERENCE_WORLD_DIR", BACKEND_DIR.parents[1] / "data" / "processed" / "landmarks" / "reference_world"))
    )
    # Signs recorded in the app (landmarks only): data/processed/landmarks/app + data/metadata/manifest_app.csv
    recordings_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_RECORDINGS_DIR", BACKEND_DIR.parents[1] / "data"))
    )
    # Optional experimental still-pose model (teammate's MitIndia branch); selectable, never the default
    intro_static_model: Path = field(
        default_factory=lambda: Path(os.getenv(
            "MUDRA_INTRO_STATIC_MODEL", BACKEND_DIR / "models" / "mudra_intro_static_pose_medoids_v1.npz"))
    )
    # Teammate's packaged video-word models (mudra_api/extra_models.py); selectable, never the default
    common_word_video_model_dir: Path = field(
        default_factory=lambda: Path(os.getenv(
            "MUDRA_COMMON_WORD_VIDEO_MODEL_DIR", BACKEND_DIR / "models" / "common_word_video_baseline"))
    )
    signline_intro_video_model: Path = field(
        default_factory=lambda: Path(os.getenv(
            "MUDRA_SIGNLINE_INTRO_MODEL", BACKEND_DIR / "models" / "signline_intro_video" / "model.joblib"))
    )
    static_top10_model: Path = field(
        default_factory=lambda: Path(os.getenv(
            "MUDRA_STATIC_TOP10_MODEL", BACKEND_DIR / "models" / "static_top10_v2" / "model.joblib"))
    )
    isl500_video_model: Path = field(
        default_factory=lambda: Path(os.getenv(
            "MUDRA_ISL500_VIDEO_MODEL", BACKEND_DIR / "models" / "isl500_video_pilot" / "model.joblib"))
    )
    # Sign library beyond the word model (mudra_api/library.py): fingerspelling letters and whole phrases
    letters_model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_LETTERS_MODEL_DIR", BACKEND_DIR / "models" / "isl_letters_v1"))
    )
    letters_landmarks: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_LETTERS_LANDMARKS", BACKEND_DIR.parents[1] / "data" / "processed" / "landmarks" / "letters"))
    )
    phrases_model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_PHRASES_MODEL_DIR", BACKEND_DIR / "models" / "isl_phrases_v1_bilstm"))
    )
    phrases_split: Path = field(
        default_factory=lambda: Path(os.getenv("MUDRA_PHRASES_SPLIT", BACKEND_DIR.parents[1] / "data" / "metadata" / "splits" / "phrases_v1" / "train.csv"))
    )
