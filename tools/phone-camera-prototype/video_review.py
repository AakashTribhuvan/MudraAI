"""Local browser reviewer for timestamped sign segments in source videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import cv2


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_ROOT = ROOT / "training_data" / "Videos_Sentence_Level"
DEFAULT_MANIFEST = ROOT / "training_data" / "manifests" / "video_sign_segments.csv"
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
CONTENT_TYPES = {
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
ASSETS = {
    "/": ("video-review.html", "text/html; charset=utf-8"),
    "/video-review.html": ("video-review.html", "text/html; charset=utf-8"),
    "/video-review.css": ("video-review.css", "text/css; charset=utf-8"),
    "/video-review.js": ("video-review.js", "text/javascript; charset=utf-8"),
}
MANIFEST_FIELDS = (
    "annotation_id",
    "source_clip_id",
    "relative_path",
    "source_sha256",
    "phrase_folder",
    "verified_gloss",
    "start_s",
    "end_s",
    "phase_events_json",
    "signer_id",
    "session_id",
    "source_origin",
    "license_or_consent_ref",
    "provenance_status",
    "review_status",
    "reviewer",
    "review_notes",
)
REVIEW_STATUSES = {"accepted", "needs_review", "rejected"}
PROVENANCE_STATUSES = {"confirmed", "not_permitted", "unknown"}
MAX_REQUEST_BYTES = 16_384
WEB_ROOT = ROOT / "web"


def video_records(video_root: Path = DEFAULT_VIDEO_ROOT) -> list[dict[str, Any]]:
    root = video_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Sentence-video folder does not exist: {root}")

    records: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
            continue
        resolved = path.resolve()
        try:
            relative_path = resolved.relative_to(root)
        except ValueError:
            continue
        relative_text = relative_path.as_posix()
        clip_id = hashlib.sha256(relative_text.encode("utf-8")).hexdigest()[:20]
        records.append(
            {
                "clip_id": clip_id,
                "relative_path": relative_text,
                "filename": path.name,
                "phrase_folder": (
                    relative_path.parts[0]
                    if len(relative_path.parts) > 1
                    else "Uncategorized"
                ),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return sorted(
        records,
        key=lambda record: record["relative_path"].casefold(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path.name}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError(f"Video metadata is incomplete or invalid: {path.name}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"First frame could not be decoded: {path.name}")
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
            "duration_s": frame_count / fps,
            "source_sha256": sha256_file(path),
        }
    finally:
        capture.release()


def extract_frame_jpeg(path: Path, time_s: float) -> bytes:
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError("Preview time must be a finite non-negative number.")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError(f"Video metadata is incomplete or invalid: {path.name}")
        duration = frame_count / fps
        if time_s > duration:
            raise ValueError(f"Preview time exceeds video duration ({duration:.3f}s).")
        frame_index = min(round(time_s * fps), frame_count - 1)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise ValueError(f"Could not seek to frame {frame_index} in {path.name}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not decode frame {frame_index} in {path.name}")
        encoded, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not encoded:
            raise RuntimeError(f"Could not encode preview frame from {path.name}")
        return image.tobytes()
    finally:
        capture.release()


def _text(value: Any, field: str, maximum: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{field} is required.")
    if len(value) > maximum:
        raise ValueError(f"{field} must be at most {maximum} characters.")
    return value


def _time_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def normalize_annotation(
    payload: Any,
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Annotation request must be a JSON object.")
    allowed = {
        "annotation_id",
        "clip_id",
        "verified_gloss",
        "start_s",
        "end_s",
        "phase_events",
        "signer_id",
        "session_id",
        "source_origin",
        "license_or_consent_ref",
        "provenance_status",
        "review_status",
        "reviewer",
        "review_notes",
    }
    if set(payload) - allowed:
        raise ValueError("Annotation request contains unknown fields.")
    if payload.get("clip_id") != record["clip_id"]:
        raise ValueError("Annotation clip does not match the selected source video.")

    start_s = _time_value(payload.get("start_s"), "Sign start")
    end_s = _time_value(payload.get("end_s"), "Sign end")
    duration = float(metadata["duration_s"])
    if start_s < 0 or end_s <= start_s or end_s > duration:
        raise ValueError(f"Sign boundaries must satisfy 0 <= start < end <= {duration:.3f}s.")

    phases = payload.get("phase_events", [])
    if not isinstance(phases, list):
        raise ValueError("Motion phases must be a list.")
    normalized_phases: list[dict[str, Any]] = []
    previous_time = start_s
    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict) or set(phase) != {"name", "time_s"}:
            raise ValueError(f"Motion phase {index} must contain only name and time_s.")
        name = _text(phase["name"], f"Motion phase {index} name", 80, required=True)
        time_s = _time_value(phase["time_s"], f"Motion phase {index} time")
        if time_s < start_s or time_s > end_s or time_s < previous_time:
            raise ValueError("Motion phase timestamps must be ordered inside the sign boundaries.")
        normalized_phases.append({"name": name, "time_s": round(time_s, 3)})
        previous_time = time_s

    annotation_id = _text(payload.get("annotation_id", ""), "Annotation ID", 80)
    if annotation_id and not all(
        character.isalnum() or character in "-_" for character in annotation_id
    ):
        raise ValueError("Annotation ID contains unsupported characters.")
    provenance_status = payload.get("provenance_status", "unknown")
    review_status = payload.get("review_status", "needs_review")
    if not isinstance(provenance_status, str) or provenance_status not in PROVENANCE_STATUSES:
        raise ValueError("Provenance status must be unknown, confirmed, or not_permitted.")
    if not isinstance(review_status, str) or review_status not in REVIEW_STATUSES:
        raise ValueError("Review status must be needs_review, accepted, or rejected.")

    return {
        "annotation_id": annotation_id or uuid.uuid4().hex,
        "source_clip_id": record["clip_id"],
        "relative_path": record["relative_path"],
        "source_sha256": metadata["source_sha256"],
        "phrase_folder": record["phrase_folder"],
        "verified_gloss": _text(payload.get("verified_gloss"), "Verified gloss", 120, True),
        "start_s": f"{start_s:.3f}",
        "end_s": f"{end_s:.3f}",
        "phase_events_json": json.dumps(normalized_phases, ensure_ascii=False),
        "signer_id": _text(payload.get("signer_id", ""), "Signer ID", 120),
        "session_id": _text(payload.get("session_id", ""), "Session ID", 120),
        "source_origin": _text(payload.get("source_origin", ""), "Source origin", 500),
        "license_or_consent_ref": _text(
            payload.get("license_or_consent_ref", ""),
            "License or consent reference",
            500,
        ),
        "provenance_status": provenance_status,
        "review_status": review_status,
        "reviewer": _text(payload.get("reviewer", ""), "Reviewer", 120),
        "review_notes": _text(payload.get("review_notes", ""), "Review notes", 2000),
    }


class VideoReviewStore:
    def __init__(self, video_root: Path, manifest_path: Path):
        self.video_root = video_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.records = video_records(self.video_root)
        self._records_by_id = {record["clip_id"]: record for record in self.records}
        self._lock = threading.Lock()

    def record(self, clip_id: str) -> dict[str, Any]:
        record = self._records_by_id.get(clip_id)
        if record is None:
            raise ValueError("Source video is not in the registered sentence-video collection.")
        return record

    def media_path(self, clip_id: str) -> Path:
        record = self.record(clip_id)
        path = (self.video_root / Path(record["relative_path"])).resolve()
        try:
            path.relative_to(self.video_root)
        except ValueError as exc:
            raise ValueError("Video path escapes the registered collection.") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Source video no longer exists: {record['relative_path']}")
        return path

    def metadata(self, clip_id: str) -> dict[str, Any]:
        return video_metadata(self.media_path(clip_id))

    def annotations(self, clip_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._read_rows()
        if clip_id is not None:
            self.record(clip_id)
            rows = [row for row in rows if row["source_clip_id"] == clip_id]
        return [
            {
                **row,
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "phase_events": json.loads(row["phase_events_json"]),
            }
            for row in rows
        ]

    def save(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Annotation request must be a JSON object.")
        clip_id = payload.get("clip_id")
        if not isinstance(clip_id, str):
            raise ValueError("A registered clip_id is required.")
        record = self.record(clip_id)
        metadata = self.metadata(clip_id)
        row = normalize_annotation(payload, record, metadata)

        with self._lock:
            rows = self._read_rows()
            if payload.get("annotation_id"):
                matches = [
                    index
                    for index, existing in enumerate(rows)
                    if existing["annotation_id"] == row["annotation_id"]
                ]
                if len(matches) != 1:
                    raise ValueError("Annotation to update was not found.")
                if rows[matches[0]]["source_clip_id"] != clip_id:
                    raise ValueError("An annotation cannot be moved to another source video.")
                rows[matches[0]] = row
            else:
                rows.append(row)
            self._write_rows(rows)
        return {
            **row,
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "phase_events": json.loads(row["phase_events_json"]),
        }

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ValueError(f"Annotation manifest has an unexpected schema: {self.manifest_path}")
            rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"Annotation manifest contains a malformed row: {self.manifest_path}")
        return rows

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=self.manifest_path.parent,
                prefix=f".{self.manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, self.manifest_path)
            temporary_path = None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


class VideoReviewServer:
    def __init__(
        self,
        port: int = 8766,
        video_root: Path = DEFAULT_VIDEO_ROOT,
        manifest_path: Path = DEFAULT_MANIFEST,
    ):
        self.store = VideoReviewStore(video_root, manifest_path)
        self._started = False
        store = self.store

        class RequestHandler(BaseHTTPRequestHandler):
            server_version = "MudraVideoReview/1.0"

            def log_message(self, message: str, *args: Any) -> None:
                super().log_message(message, *args)

            def send_json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/api/videos":
                    self.send_json(200, {"total": len(store.records), "videos": store.records})
                    return
                if path.startswith("/api/videos/") and path.endswith("/frame"):
                    clip_id = path.removeprefix("/api/videos/").removesuffix("/frame").rstrip("/")
                    query = parse_qs(urlsplit(self.path).query)
                    try:
                        record = store.record(clip_id)
                        times = query.get("time_s", [])
                        if len(times) != 1:
                            raise ValueError("Provide one time_s preview timestamp.")
                        time_s = float(times[0])
                        if not math.isfinite(time_s):
                            raise ValueError("Preview time must be finite.")
                        image = extract_frame_jpeg(store.media_path(clip_id), time_s)
                    except (FileNotFoundError, ValueError) as exc:
                        self.send_json(422, {"error": str(exc)})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(image)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(image)
                    return
                if path.startswith("/api/videos/"):
                    clip_id = path.removeprefix("/api/videos/")
                    try:
                        record = store.record(clip_id)
                        metadata = store.metadata(clip_id)
                    except (FileNotFoundError, ValueError) as exc:
                        self.send_json(422, {"error": str(exc)})
                        return
                    self.send_json(200, {"video": record, "metadata": metadata})
                    return
                if path == "/api/annotations":
                    query = parse_qs(urlsplit(self.path).query)
                    clip_values = query.get("clip_id")
                    try:
                        annotations = store.annotations(clip_values[0] if clip_values else None)
                    except (ValueError, OSError, json.JSONDecodeError) as exc:
                        self.send_json(500, {"error": f"Could not read annotations: {exc}"})
                        return
                    self.send_json(200, {"annotations": annotations})
                    return
                if path.startswith("/media/"):
                    self.send_media(path.removeprefix("/media/"))
                    return
                asset = ASSETS.get(path)
                if asset is None:
                    self.send_json(404, {"error": "Not found."})
                    return
                filename, content_type = asset
                data = (WEB_ROOT / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)

            def do_HEAD(self) -> None:
                path = urlsplit(self.path).path
                if path.startswith("/media/"):
                    self.send_media(path.removeprefix("/media/"), head_only=True)
                    return
                self.send_error(404)

            def send_media(self, clip_id: str, head_only: bool = False) -> None:
                try:
                    record = store.record(clip_id)
                    media_path = store.media_path(clip_id)
                except (FileNotFoundError, ValueError):
                    self.send_json(404, {"error": "Registered video not found."})
                    return
                size = media_path.stat().st_size
                range_header = self.headers.get("Range")
                start, end, status = 0, size - 1, 200
                if range_header is not None:
                    try:
                        start, end = parse_byte_range(range_header, size)
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    status = 206
                content_length = end - start + 1
                self.send_response(status)
                self.send_header("Content-Type", CONTENT_TYPES[Path(record["filename"]).suffix.lower()])
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(content_length))
                self.send_header("Cache-Control", "no-store")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if head_only:
                    return
                with media_path.open("rb") as source:
                    source.seek(start)
                    remaining = content_length
                    while remaining:
                        block = source.read(min(64 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)

            def do_PUT(self) -> None:
                if urlsplit(self.path).path != "/api/annotations":
                    self.send_json(404, {"error": "Not found."})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= length <= MAX_REQUEST_BYTES:
                        raise ValueError(
                            f"Annotation request must be between 1 and {MAX_REQUEST_BYTES} bytes."
                        )
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    annotation = store.save(payload)
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self.send_json(400, {"error": str(exc)})
                    return
                except OSError as exc:
                    self.send_json(500, {"error": f"Could not save annotation: {exc}"})
                    return
                self.send_json(200, {"annotation": annotation})

        self._server = ThreadingHTTPServer(("127.0.0.1", port), RequestHandler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address[:2]

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def close(self) -> None:
        if self._started:
            self._server.shutdown()
        self._server.server_close()
        if self._started:
            self._thread.join(timeout=2)


def parse_byte_range(header: str, size: int) -> tuple[int, int]:
    if size <= 0 or not header.startswith("bytes=") or "," in header:
        raise ValueError("Unsupported byte range.")
    bounds = header.removeprefix("bytes=").strip().split("-", maxsplit=1)
    if len(bounds) != 2:
        raise ValueError("Malformed byte range.")
    left, right = bounds
    try:
        if left:
            start = int(left)
            end = size - 1 if not right else int(right)
        elif right:
            suffix_length = int(right)
            if suffix_length <= 0:
                raise ValueError("Suffix range must be positive.")
            start = max(0, size - suffix_length)
            end = size - 1
        else:
            raise ValueError("Empty byte range.")
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed byte range.") from exc
    if start < 0 or end < start or start >= size:
        raise ValueError("Byte range is outside the source file.")
    return start, min(end, size - 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review source videos and annotate sign/phase timestamps locally."
    )
    parser.add_argument("--port", type=int, default=8766, help="Local reviewer port (default: 8766)")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help="Folder containing phrase subfolders and source videos",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV file to create/update with reviewed sign segments",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        server = VideoReviewServer(args.port, args.video_root, args.manifest)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    if not server.store.records:
        server.close()
        parser.exit(2, f"ERROR: No supported videos found under {args.video_root}\n")
    server.start()
    host, port = server.address
    url = f"http://{host}:{port}/"
    print(f"Local video reviewer: {url}")
    print(f"Source videos: {len(server.store.records)}")
    print(f"Annotation manifest: {server.store.manifest_path}")
    print("Source videos are read-only; press Ctrl+C to stop the reviewer.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping video reviewer.")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
