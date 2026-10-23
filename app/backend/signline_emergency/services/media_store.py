"""Storage for gesture demonstration videos (and the copies attached to shared incidents).

Videos can show a person's face, so they are treated as sensitive: stored only after explicit recording
consent, under a random storage key (never the uploaded file name), encrypted at rest with AES-256-GCM (the
storage key is the associated data, so a file cannot be swapped under another name), served only to the owning
device or, for an incident copy shared with consent, to responders; deleted with their gesture or incident and by
the retention job (``retention.py``); never logged and never used for training.

The encryption key comes from ``SIGNLINE_EM_MEDIA_KEY`` (base64, 32 bytes) or is generated once into
``app/backend/var/emergency_media.key`` (git-ignored, like the database). Losing the key makes the videos
unreadable. ``LocalMediaStore`` keeps files on disk; another backend can implement save / read / delete.
"""

from __future__ import annotations

import base64
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ALLOWED_TYPES = {"video/webm": ".webm", "video/mp4": ".mp4"}
MAX_VIDEO_BYTES = 25 * 1024 * 1024
MAX_VIDEO_SECONDS = 30.0
MIN_VIDEO_SECONDS = 0.5
MAGIC = b"SLVE1"  # SignLine video, encrypted, format 1: MAGIC + 12-byte nonce + AES-GCM ciphertext and tag


class MediaError(ValueError):
    """Invalid upload (type, size, duration or content): the API answers 422."""


class MediaStorageError(RuntimeError):
    """The store could not save or read a file: the API answers 503."""


def base_type(mime: str | None) -> str:
    return (mime or "").split(";")[0].strip().lower()


def sniff_video(data: bytes) -> str | None:
    """The container a file really is, from its first bytes (a declared type alone is not trusted)."""
    if data[:4] == b"\x1a\x45\xdf\xa3":  # EBML header: WebM / Matroska
        return "video/webm"
    if len(data) >= 12 and data[4:8] == b"ftyp":  # ISO base media: MP4
        return "video/mp4"
    return None


def validate_video(data: bytes, declared_mime: str | None, duration_s: float | None) -> str:
    """Return the checked MIME type, or raise MediaError."""
    mime = base_type(declared_mime)
    if mime not in ALLOWED_TYPES:
        raise MediaError("only WebM or MP4 videos are accepted")
    if not data:
        raise MediaError("the video is empty")
    if len(data) > MAX_VIDEO_BYTES:
        raise MediaError(f"the video is larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
    if sniff_video(data) != mime:
        raise MediaError("the file is not a valid video of the declared type")
    if duration_s is None or not (MIN_VIDEO_SECONDS <= duration_s <= MAX_VIDEO_SECONDS):
        raise MediaError(f"the video must be between {MIN_VIDEO_SECONDS:g} and {MAX_VIDEO_SECONDS:g} seconds")
    return mime


def load_key(env_value: str | None, key_file: Path) -> bytes:
    """The 32-byte media key: from the environment, else from the key file (created on first use)."""
    if env_value:
        key = base64.urlsafe_b64decode(env_value.encode() + b"=" * (-len(env_value) % 4))
        if len(key) != 32:
            raise MediaStorageError("SIGNLINE_EM_MEDIA_KEY must be 32 bytes (base64)")
        return key
    try:
        if key_file.is_file():
            key = base64.urlsafe_b64decode(key_file.read_bytes().strip())
            if len(key) == 32:
                return key
            raise MediaStorageError(f"{key_file} does not hold a 32-byte key")
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(32)
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(base64.urlsafe_b64encode(key))
        return key
    except OSError as exc:
        raise MediaStorageError("the media encryption key could not be read or created") from exc


@dataclass
class LocalMediaStore:
    root: Path
    key: bytes

    def _path(self, storage_key: str) -> Path:
        name = Path(storage_key).name
        if name != storage_key or not name or name.startswith("."):
            raise MediaStorageError("invalid storage key")
        return self.root / name

    def save(self, data: bytes, mime: str, prefix: str = "gv") -> str:
        storage_key = f"{prefix}_{secrets.token_hex(16)}{ALLOWED_TYPES[mime]}"
        nonce = secrets.token_bytes(12)
        blob = MAGIC + nonce + AESGCM(self.key).encrypt(nonce, data, storage_key.encode())
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f".{storage_key}.part"
            tmp.write_bytes(blob)
            os.replace(tmp, self._path(storage_key))
        except OSError as exc:
            raise MediaStorageError("the video could not be stored") from exc
        return storage_key

    def read(self, storage_key: str) -> bytes:
        p = self._path(storage_key)
        try:
            blob = p.read_bytes()
        except OSError as exc:
            raise MediaStorageError("the stored video is missing") from exc
        if not blob.startswith(MAGIC):
            raise MediaStorageError("the stored video is not in the encrypted format")
        nonce, body = blob[len(MAGIC):len(MAGIC) + 12], blob[len(MAGIC) + 12:]
        try:
            return AESGCM(self.key).decrypt(nonce, body, storage_key.encode())
        except InvalidTag as exc:
            raise MediaStorageError("the stored video could not be decrypted (wrong key or damaged file)") from exc

    def copy(self, storage_key: str, prefix: str = "iv") -> str:
        """A separate encrypted copy (an incident keeps its own, so deleting the gesture does not remove it)."""
        data = self.read(storage_key)
        ext = Path(storage_key).suffix
        mime = next(m for m, e in ALLOWED_TYPES.items() if e == ext)
        return self.save(data, mime, prefix)

    def keys(self) -> list[str]:
        return [p.name for p in self.root.glob("*") if p.is_file() and not p.name.startswith(".")] if self.root.exists() else []

    def delete(self, storage_key: str | None) -> None:
        """Overwrite, then remove (best effort: SSDs and backups may still hold copies, but they are encrypted)."""
        if not storage_key:
            return
        try:
            p = self._path(storage_key)
            if p.is_file():
                with p.open("r+b") as f:
                    f.write(b"\0" * p.stat().st_size)
                p.unlink()
        except (OSError, MediaStorageError):
            pass


def get_media_store() -> LocalMediaStore:
    """The store configured in the settings (``SIGNLINE_EM_MEDIA_DIR`` and the media key)."""
    from ..config import get_settings

    s = get_settings()
    return LocalMediaStore(Path(s.media_dir), load_key(s.media_key, Path(s.media_key_file)))
