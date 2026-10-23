import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import video_review
from video_review import (
    MANIFEST_FIELDS,
    VideoReviewServer,
    VideoReviewStore,
    normalize_annotation,
    parse_byte_range,
    video_records,
)


class VideoReviewTests(unittest.TestCase):
    def test_video_inventory_uses_parent_folder_and_ignores_non_video_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            phrase = root / "are you free today"
            phrase.mkdir()
            (phrase / "free.mp4").write_bytes(b"video")
            (phrase / "notes.txt").write_text("not a video", encoding="utf-8")
            (root / "loose.MP4").write_bytes(b"video")

            records = video_records(root)

        self.assertEqual(len(records), 2)
        sentence_video = next(record for record in records if record["filename"] == "free.mp4")
        self.assertEqual(sentence_video["phrase_folder"], "are you free today")
        self.assertNotIn("verified_gloss", sentence_video)
        self.assertEqual(sentence_video["relative_path"], "are you free today/free.mp4")

    def test_byte_range_parser_handles_closed_open_and_suffix_ranges(self):
        self.assertEqual(parse_byte_range("bytes=2-5", 10), (2, 5))
        self.assertEqual(parse_byte_range("bytes=7-", 10), (7, 9))
        self.assertEqual(parse_byte_range("bytes=-3", 10), (7, 9))
        for header in ("items=0-1", "bytes=10-", "bytes=3-2", "bytes=0-1,4-5"):
            with self.subTest(header=header), self.assertRaises(ValueError):
                parse_byte_range(header, 10)

    def test_annotation_checks_segment_bounds_and_ordered_motion_phases(self):
        record = {
            "clip_id": "a" * 20,
            "relative_path": "phrase/source.mp4",
            "phrase_folder": "phrase",
        }
        metadata = {"duration_s": 4.0, "source_sha256": "b" * 64}
        payload = {
            "clip_id": record["clip_id"],
            "verified_gloss": "HELLO",
            "start_s": 0.5,
            "end_s": 2.5,
            "phase_events": [
                {"name": "stroke", "time_s": 1.0},
                {"name": "hold", "time_s": 2.0},
            ],
            "provenance_status": "confirmed",
            "review_status": "accepted",
        }

        normalized = normalize_annotation(payload, record, metadata)
        self.assertEqual(normalized["phrase_folder"], "phrase")
        self.assertEqual(normalized["verified_gloss"], "HELLO")
        self.assertEqual(
            json.loads(normalized["phase_events_json"]),
            [{"name": "stroke", "time_s": 1.0}, {"name": "hold", "time_s": 2.0}],
        )

        for changed in (
            {**payload, "start_s": -0.1},
            {**payload, "end_s": 4.1},
            {**payload, "end_s": 0.5},
            {
                **payload,
                "phase_events": [
                    {"name": "return", "time_s": 2.1},
                    {"name": "stroke", "time_s": 1.0},
                ],
            },
            {**payload, "provenance_status": "guess"},
        ):
            with self.subTest(payload=changed), self.assertRaises(ValueError):
                normalize_annotation(changed, record, metadata)

    def test_store_creates_and_updates_csv_rows_without_losing_source_folder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "videos"
            folder = root / "hi how are you"
            folder.mkdir(parents=True)
            (folder / "hi how are you (1).mp4").write_bytes(b"not decoded by this store test")
            manifest = Path(temporary_directory) / "manifest.csv"
            store = VideoReviewStore(root, manifest)
            clip_id = store.records[0]["clip_id"]
            payload = {
                "clip_id": clip_id,
                "verified_gloss": "HELLO",
                "start_s": 0.2,
                "end_s": 1.2,
                "phase_events": [{"name": "stroke", "time_s": 0.6}],
            }
            with patch.object(
                video_review,
                "video_metadata",
                return_value={"duration_s": 2.0, "source_sha256": "c" * 64},
            ):
                saved = store.save(payload)
                updated = store.save(
                    {
                        **payload,
                        "annotation_id": saved["annotation_id"],
                        "verified_gloss": "HELLO_GREET",
                    }
                )

            self.assertTrue(manifest.is_file())
            self.assertEqual(updated["annotation_id"], saved["annotation_id"])
            self.assertEqual(updated["verified_gloss"], "HELLO_GREET")
            with manifest.open("r", encoding="utf-8", newline="") as source:
                import csv

                reader = csv.DictReader(source)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames), MANIFEST_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["phrase_folder"], "hi how are you")
            self.assertEqual(rows[0]["relative_path"], "hi how are you/hi how are you (1).mp4")
            self.assertEqual(rows[0]["source_sha256"], "c" * 64)

    def test_local_server_serves_video_ranges_and_persists_annotations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "videos"
            folder = root / "thank you so much"
            folder.mkdir(parents=True)
            video_bytes = b"0123456789abcdef"
            (folder / "thanks.mp4").write_bytes(video_bytes)
            manifest = Path(temporary_directory) / "manifest.csv"
            server = VideoReviewServer(0, root, manifest)
            clip = server.store.records[0]
            metadata = {
                "width": 320,
                "height": 240,
                "fps": 25.0,
                "frames": 50,
                "duration_s": 2.0,
                "source_sha256": "d" * 64,
            }
            try:
                with (
                    patch.object(video_review, "video_metadata", return_value=metadata),
                    patch.object(
                        video_review,
                        "extract_frame_jpeg",
                        return_value=b"jpeg-frame",
                    ) as extract_frame,
                ):
                    server.start()
                    base_url = f"http://127.0.0.1:{server.address[1]}"
                    with urlopen(f"{base_url}/api/videos") as response:
                        inventory = json.loads(response.read())
                    self.assertEqual(inventory["total"], 1)
                    self.assertEqual(inventory["videos"][0]["phrase_folder"], "thank you so much")

                    request = Request(
                        f"{base_url}/media/{clip['clip_id']}",
                        headers={"Range": "bytes=4-8"},
                    )
                    with urlopen(request) as response:
                        self.assertEqual(response.status, 206)
                        self.assertEqual(response.headers["Content-Range"], "bytes 4-8/16")
                        self.assertEqual(response.read(), video_bytes[4:9])

                    with urlopen(
                        f"{base_url}/api/videos/{clip['clip_id']}/frame?time_s=0.480"
                    ) as response:
                        self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                        self.assertEqual(response.read(), b"jpeg-frame")
                    extract_frame.assert_called_once_with(
                        server.store.media_path(clip["clip_id"]),
                        0.48,
                    )

                    payload = {
                        "clip_id": clip["clip_id"],
                        "verified_gloss": "THANK_YOU",
                        "start_s": 0.1,
                        "end_s": 1.4,
                        "phase_events": [],
                    }
                    request = Request(
                        f"{base_url}/api/annotations",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="PUT",
                    )
                    with urlopen(request) as response:
                        result = json.loads(response.read())
                    self.assertEqual(result["annotation"]["verified_gloss"], "THANK_YOU")
                    self.assertTrue(manifest.is_file())

                    bad_request = Request(
                        f"{base_url}/api/annotations",
                        data=json.dumps({**payload, "start_s": -1}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="PUT",
                    )
                    with self.assertRaises(HTTPError) as error:
                        urlopen(bad_request)
                    self.assertEqual(error.exception.code, 400)
                    error.exception.close()
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
