import importlib.util
import json
import sys
import tempfile
import unittest
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import numpy as np

from web_dashboard import (
    DATASET_ROOT,
    DEFAULT_SETTINGS,
    DashboardServer,
    DashboardState,
    dataset_image_records,
)
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "test.py"
SPEC = importlib.util.spec_from_file_location("mudra_phone_webcam_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load sign tracker from {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
SignActivityTracker = MODULE.SignActivityTracker
LatestFrameCapture = MODULE.LatestFrameCapture


class SignActivityTrackerTests(unittest.TestCase):
    def test_latest_frame_capture_returns_newest_frame_without_blocking_readers(self):
        class FakeCapture:
            def __init__(self):
                self.frames = iter(range(100))
                self.released = False

            def read(self):
                if self.released:
                    return False, None
                time.sleep(0.001)
                return True, next(self.frames)

            def release(self):
                self.released = True

        capture = FakeCapture()
        latest = LatestFrameCapture(capture)
        try:
            ok, first_frame = latest.read()
            self.assertTrue(ok)
            self.assertIsInstance(first_frame, int)
            time.sleep(0.01)
            ok, newer_frame = latest.read()
            self.assertTrue(ok)
            self.assertGreater(newer_frame, first_frame)
        finally:
            latest.release()
        self.assertTrue(capture.released)
        self.assertFalse(latest._thread.is_alive())

    def test_latest_frame_capture_surfaces_camera_read_failures(self):
        class FailedCapture:
            def read(self):
                return False, None

            def release(self):
                pass

        latest = LatestFrameCapture(FailedCapture())
        try:
            with self.assertRaisesRegex(RuntimeError, "Camera capture failed"):
                latest.read()
        finally:
            latest.release()

    def test_dashboard_separates_static_and_dynamic_sign_sections(self):
        with patch("web_dashboard.FRONTEND_DIST", Path(__file__).parent / "missing-frontend"):
            dashboard = DashboardServer("127.0.0.1", 0, {})
            dashboard.start()
            try:
                host, port = dashboard.address
                with urlopen(f"http://{host}:{port}/") as response:
                    page = response.read()
                self.assertIn(b"Static Pose Model", page)
                self.assertIn(b"Expanded Intro Model", page)
                self.assertIn(b"Dynamic Pose Model", page)
                self.assertIn(b"STATIC SIGN PREVIEWS", page)
                self.assertIn(b"camera-setup", page)
                self.assertIn(b"trained labels", page)
                self.assertIn(b"static_pose_medoids.npz", page)
            finally:
                dashboard.close()

    def test_dashboard_serves_signline_spa_routes_and_assets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            frontend_root = Path(temporary_directory)
            (frontend_root / "index.html").write_text(
                "<main>SignLine frontend</main>", encoding="utf-8"
            )
            assets = frontend_root / "assets"
            assets.mkdir()
            (assets / "app.js").write_text("globalThis.signline = true", encoding="utf-8")

            with patch("web_dashboard.FRONTEND_DIST", frontend_root):
                dashboard = DashboardServer("127.0.0.1", 0, {})
                dashboard.start()
                try:
                    host, port = dashboard.address
                    base_url = f"http://{host}:{port}"
                    with urlopen(f"{base_url}/model-test") as response:
                        self.assertIn(b"SignLine frontend", response.read())
                    with urlopen(f"{base_url}/assets/app.js") as response:
                        self.assertEqual(response.read(), b"globalThis.signline = true")
                    with self.assertRaises(HTTPError) as error:
                        urlopen(f"{base_url}/api/missing")
                    self.assertEqual(error.exception.code, 404)
                    error.exception.close()
                    with self.assertRaises(HTTPError) as error:
                        urlopen(f"{base_url}/%2e%2e/outside")
                    self.assertEqual(error.exception.code, 404)
                    error.exception.close()
                finally:
                    dashboard.close()

    def test_dashboard_camera_selection_and_start_stop_workflow(self):
        cameras = [
            {
                "index": 0,
                "backend_key": "msmf",
                "backend": "Media Foundation",
                "width": 1280,
                "height": 720,
                "brightness": 30.0,
            }
        ]
        catalog = {
            "static": {
                "trained_words": 20,
                "live_words": 8,
                "words": [{"label": "YOU", "live": True}],
            },
            "dynamic": {"trained_words": 0, "live_words": 0, "words": []},
            "intro": {
                "trained_words": 32,
                "live_words": 10,
                "words": [{"label": "HELLO_HI", "live": False}],
            },
        }
        dashboard = DashboardServer(
            "127.0.0.1",
            0,
            {},
            cameras=cameras,
            model_catalog=catalog,
            camera_setup=True,
        )
        dashboard.start()
        try:
            host, port = dashboard.address
            base_url = f"http://{host}:{port}"
            with urlopen(f"{base_url}/api/status") as response:
                state = json.loads(response.read())
            self.assertTrue(state["camera_setup"])
            self.assertFalse(state["tracking_enabled"])
            self.assertEqual(state["model_catalog"]["static"]["trained_words"], 20)
            self.assertEqual(state["model_catalog"]["static"]["live_words"], 8)
            self.assertEqual(state["model_catalog"]["dynamic"]["trained_words"], 0)
            self.assertEqual(state["model_catalog"]["intro"]["trained_words"], 32)
            self.assertEqual(state["model_catalog"]["intro"]["live_words"], 10)
            self.assertTrue(state["model_catalog"]["static"]["words"][0]["preview_url"])
            self.assertTrue(state["camera_mirrored"])

            request = Request(
                f"{base_url}/api/camera/flip",
                data=json.dumps({"mirrored": False}).encode(),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urlopen(request) as response:
                self.assertFalse(
                    json.loads(response.read())["camera_mirrored"]
                )

            request = Request(
                f"{base_url}/api/camera",
                data=json.dumps({"index": 0, "backend": "msmf"}).encode(),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urlopen(request) as response:
                self.assertEqual(json.loads(response.read())["selected_camera"]["index"], 0)

            request = Request(
                f"{base_url}/api/session",
                data=json.dumps({"enabled": True}).encode(),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urlopen(request) as response:
                self.assertTrue(json.loads(response.read())["tracking_enabled"])

            request = Request(
                f"{base_url}/api/model",
                data=json.dumps({"mode": "intro"}).encode(),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urlopen(request) as response:
                self.assertEqual(json.loads(response.read())["model_mode"], "intro")

            with urlopen(f"{base_url}/api/status") as response:
                state = json.loads(response.read())
            self.assertFalse(state["camera_setup"])
            self.assertTrue(state["tracking_enabled"])
            self.assertEqual(state["model_mode"], "intro")
        finally:
            dashboard.close()

    def test_dashboard_requires_worker_token_when_public_gateway_is_configured(self):
        dashboard = DashboardServer("127.0.0.1", 0, {}, proxy_token="private-token")
        dashboard.start()
        try:
            host, port = dashboard.address
            url = f"http://{host}:{port}/api/status"
            with self.assertRaises(HTTPError) as error:
                urlopen(url)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()

            request = Request(url, headers={"X-Mudra-Proxy-Token": "private-token"})
            with urlopen(request) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b'"camera_setup"', response.read())
        finally:
            dashboard.close()

    def test_dataset_image_index_lists_and_serves_every_image(self):
        expected_images = dataset_image_records(DATASET_ROOT)
        self.assertGreater(len(expected_images), 0)

        dashboard = DashboardServer("127.0.0.1", 0, {})
        dashboard.start()
        try:
            host, port = dashboard.address
            base_url = f"http://{host}:{port}"
            with urlopen(f"{base_url}/images") as response:
                self.assertIn(b"Dataset images", response.read())

            with urlopen(f"{base_url}/api/dataset/images") as response:
                payload = json.loads(response.read())
            self.assertEqual(payload["total"], len(expected_images))
            self.assertEqual(len(payload["images"]), len(expected_images))
            self.assertEqual(
                payload["images"][0]["relative_path"],
                expected_images[0][1].relative_to(DATASET_ROOT.resolve()).as_posix(),
            )

            with urlopen(f"{base_url}{payload['images'][0]['url']}") as response:
                self.assertEqual(
                    response.headers["Content-Type"],
                    expected_images[0][2],
                )
                self.assertGreater(len(response.read()), 0)

            with self.assertRaises(HTTPError) as error:
                urlopen(f"{base_url}/api/dataset/image/{len(expected_images)}")
            self.assertEqual(error.exception.code, 404)
            error.exception.close()
        finally:
            dashboard.close()

    def test_rejects_brief_fidget(self):
        tracker = SignActivityTracker(start_hold_s=0.15, stop_hold_s=0.5)

        self.assertIsNone(tracker.update(0.0, visible=True, raised=True, moving=False))
        self.assertIsNone(tracker.update(0.05, visible=False, raised=False, moving=False))
        self.assertEqual(tracker.state, "idle")

    def test_raised_static_hold_stays_active_and_ends_at_rest_onset(self):
        tracker = SignActivityTracker(start_hold_s=0.15, stop_hold_s=0.5)

        self.assertIsNone(tracker.update(1.0, visible=True, raised=True, moving=False))
        self.assertIsNone(tracker.update(1.1, visible=True, raised=True, moving=False))
        self.assertEqual(tracker.update(1.16, visible=True, raised=True, moving=False), "started")
        self.assertEqual(tracker.sign_started_at, 1.0)
        for timestamp in (1.4, 1.8, 2.0):
            self.assertIsNone(tracker.update(timestamp, visible=True, raised=True, moving=False))
        self.assertEqual(tracker.state, "signing")

        self.assertIsNone(tracker.update(2.1, visible=True, raised=False, moving=False))
        self.assertEqual(tracker.state, "ending")
        self.assertEqual(tracker.update(2.61, visible=False, raised=False, moving=False), "stopped")
        self.assertEqual(tracker.completed_end_at, 2.1)
        self.assertAlmostEqual(tracker.completed_duration_s, 1.1)

    def test_activity_resuming_during_stop_debounce_continues_same_segment(self):
        tracker = SignActivityTracker(start_hold_s=0.1, stop_hold_s=0.5)
        tracker.update(0.0, visible=True, raised=True, moving=False)
        self.assertEqual(tracker.update(0.1, visible=True, raised=True, moving=False), "started")

        tracker.update(0.2, visible=True, raised=False, moving=False)
        self.assertIsNone(tracker.update(0.5, visible=True, raised=True, moving=False))
        self.assertEqual(tracker.state, "signing")
        self.assertIsNone(tracker.update(0.6, visible=True, raised=False, moving=False))
        self.assertEqual(tracker.update(1.11, visible=False, raised=False, moving=False), "stopped")
        self.assertAlmostEqual(tracker.completed_duration_s, 0.6)

    def test_wrist_movement_alone_starts_dynamic_segment(self):
        tracker = SignActivityTracker(start_hold_s=0.15, stop_hold_s=0.5)

        self.assertIsNone(tracker.update(4.0, visible=True, raised=False, moving=True))
        self.assertIsNone(tracker.update(4.1, visible=True, raised=False, moving=True))
        self.assertEqual(tracker.update(4.16, visible=True, raised=False, moving=True), "started")
        self.assertEqual(tracker.sign_started_at, 4.0)
        tracker.update(4.3, visible=True, raised=False, moving=False)
        self.assertEqual(tracker.update(4.81, visible=True, raised=False, moving=False), "stopped")
        self.assertEqual(tracker.completed_end_at, 4.3)
        self.assertAlmostEqual(tracker.completed_duration_s, 0.3)

    def test_segment_refresh_ends_and_restarts_a_static_segment(self):
        tracker = SignActivityTracker(start_hold_s=0.1, stop_hold_s=0.5)
        tracker.update(1.0, visible=True, raised=True, moving=False)
        self.assertEqual(tracker.update(1.1, visible=True, raised=True, moving=False), "started")

        self.assertEqual(
            tracker.restart_segment(
                1.35,
                visible=True,
                raised=False,
                moving=False,
            ),
            "refreshed",
        )
        self.assertEqual(tracker.state, "signing")
        self.assertEqual(tracker.completed_end_at, 1.35)
        self.assertAlmostEqual(tracker.completed_duration_s, 0.35)
        self.assertEqual(tracker.sign_started_at, 1.35)

    def test_static_tracking_refreshes_at_five_seconds_not_before(self):
        tracker = SignActivityTracker(start_hold_s=0.15, stop_hold_s=0.7)
        tracker.update(1.0, visible=True, raised=True, moving=False)
        tracker.update(1.16, visible=True, raised=True, moving=False)

        self.assertFalse(
            tracker.refresh_static_if_due(
                5.99,
                visible=True,
                raised=True,
                moving=False,
            )
        )
        self.assertTrue(
            tracker.refresh_static_if_due(
                6.0,
                visible=True,
                raised=True,
                moving=False,
            )
        )
        self.assertEqual(tracker.state, "signing")
        self.assertEqual(tracker.sign_started_at, 6.0)
        self.assertEqual(tracker.completed_end_at, 6.0)
        self.assertEqual(tracker.elapsed(6.0), 0.0)

    def test_static_refresh_does_not_restart_a_dynamic_segment(self):
        tracker = SignActivityTracker()
        tracker.update(1.0, visible=True, raised=False, moving=True)
        tracker.update(1.16, visible=True, raised=False, moving=True)

        self.assertFalse(
            tracker.refresh_static_if_due(
                7.0,
                visible=True,
                raised=False,
                moving=True,
            )
        )
        self.assertEqual(tracker.sign_started_at, 1.0)

    def test_preview_encoder_caps_stream_size_without_changing_source(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        encoded, (width, height) = MODULE.encode_preview_jpeg(frame, 1280)
        decoded = MODULE.cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            MODULE.cv2.IMREAD_COLOR,
        )

        self.assertEqual((width, height), (1280, 720))
        self.assertEqual(decoded.shape[:2], (720, 1280))
        self.assertEqual(frame.shape[:2], (1080, 1920))

    def test_sign_box_is_drawn_around_all_detected_hands(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hand_landmarks = [
            [SimpleNamespace(x=0.3, y=0.4), SimpleNamespace(x=0.5, y=0.6)],
            [SimpleNamespace(x=0.6, y=0.3), SimpleNamespace(x=0.7, y=0.5)],
        ]

        MODULE.draw_sign_box(frame, hand_landmarks, "SIGN IN ACTION", (80, 240, 160))

        self.assertTrue(np.any(frame[18, 50]))
        self.assertTrue(np.any(frame[50, 82]))
        self.assertTrue(np.any(frame[15:75, 15:85]))

    def test_hand_overlay_draws_index_arrow_and_face_direction(self):
        landmarks = [
            SimpleNamespace(x=0.5, y=0.5, z=0.0)
            for _ in range(21)
        ]
        landmarks[0].x, landmarks[0].y = 0.5, 0.6
        landmarks[5].x, landmarks[5].y = 0.5, 0.5
        landmarks[8].x, landmarks[8].y = 0.5, 0.3
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with (
            patch.object(MODULE.cv2, "arrowedLine") as arrow,
            patch.object(MODULE.cv2, "putText") as put_text,
        ):
            MODULE.draw_hand(frame, landmarks, face_point=(50, 20))

        self.assertEqual(arrow.call_args.args[1:3], ((50, 50), (50, 30)))
        self.assertEqual(put_text.call_args.args[1], "POINTING TOWARD FACE")

    def test_mode_label_waits_for_persistent_change(self):
        tracker = SignActivityTracker(start_hold_s=0.1, stop_hold_s=0.5, mode_hold_s=0.4)
        tracker.update(0.0, visible=True, raised=True, moving=False)
        self.assertEqual(tracker.update(0.1, visible=True, raised=True, moving=False), "started")
        self.assertEqual(tracker.update_mode(0.1, raised=True, moving=False), "static")

        self.assertEqual(tracker.update_mode(0.2, raised=True, moving=True), "static")
        self.assertEqual(tracker.update_mode(0.45, raised=True, moving=True), "static")
        self.assertEqual(tracker.update_mode(0.61, raised=True, moving=True), "dynamic")
        self.assertEqual(tracker.update_mode(0.7, raised=True, moving=False), "dynamic")

    def test_settings_reject_values_outside_supported_range(self):
        settings = DashboardState({"start_hold": 0.2})

        with self.assertRaisesRegex(ValueError, "between"):
            settings.update_settings({"stop_hold": 0.0})
        with self.assertRaisesRegex(ValueError, "Unknown setting"):
            settings.update_settings({"unrecognized": 1.0})

    def test_lower_torso_zone_defaults_to_larger_range(self):
        settings = DashboardState({})
        self.assertEqual(settings.settings()["raised_below_shoulder"], 1.0)
        self.assertEqual(DEFAULT_SETTINGS["raised_below_shoulder"]["max"], 2.0)

    def test_dashboard_serves_page_and_applies_live_setting_updates(self):
        dashboard = DashboardServer("127.0.0.1", 0, {"start_hold": 0.2})
        dashboard.start()
        try:
            host, port = dashboard.address
            base_url = f"http://{host}:{port}"
            dashboard.state.update_runtime({"state": "signing"}, b"JPEG")
            with urlopen(f"{base_url}/") as response:
                self.assertEqual(response.status, 200)
                page = response.read()
                self.assertIn(b"<html", page)

            with urlopen(f"{base_url}/stream.mjpg", timeout=2) as response:
                self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])
                self.assertEqual(response.readline(), b"--frame\r\n")
                while response.readline().strip():
                    pass
                self.assertEqual(response.read(4), b"JPEG")

            request = Request(
                f"{base_url}/api/settings",
                data=json.dumps({"mode_hold": 0.8}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urlopen(request) as response:
                self.assertEqual(response.status, 200)
                payload = json.loads(response.read())
            self.assertEqual(payload["settings"]["mode_hold"], 0.8)
            self.assertEqual(dashboard.state.settings()["mode_hold"], 0.8)
            dashboard.state.update_runtime(
                {
                    "state": "signing",
                    "recognition": {
                        "enabled": True,
                        "state": "recognized",
                        "label": "YOU",
                    },
                },
                b"JPEG",
            )
            self.assertEqual(
                dashboard.state.snapshot()["runtime"]["recognition"]["label"],
                "YOU",
            )

            bad_request = Request(
                f"{base_url}/api/settings",
                data=json.dumps({"mode_hold": 4.0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(bad_request)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        finally:
            dashboard.close()

    def test_learning_api_serves_only_registered_reference_images(self):
        reference = (
            Path(__file__).resolve().parents[1]
            / "training_data"
            / "Frames_Word_Level"
            / "YOU"
            / "YOU (1).jpg"
        )
        self.assertTrue(reference.is_file())
        dashboard = DashboardServer(
            "127.0.0.1",
            0,
            {},
            learning_examples=[("YOU", reference, 2)],
        )
        dashboard.start()
        try:
            host, port = dashboard.address
            base_url = f"http://{host}:{port}"
            with urlopen(f"{base_url}/api/learning/next") as response:
                lesson = json.loads(response.read())
            self.assertEqual(lesson["label"], "YOU")
            self.assertEqual(lesson["expected_hand_count"], 2)
            self.assertEqual(lesson["total"], 1)
            self.assertEqual(dashboard.learning_target(), ("YOU", 2))
            with urlopen(f"{base_url}{lesson['image_url']}") as response:
                self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                self.assertTrue(response.read().startswith(b"\xff\xd8"))

            with self.assertRaises(HTTPError) as error:
                urlopen(f"{base_url}/api/learning/images/99")
            self.assertEqual(error.exception.code, 404)
            error.exception.close()

            clear_target = Request(
                f"{base_url}/api/learning/target",
                data=json.dumps({"sample_id": None}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urlopen(clear_target) as response:
                self.assertIsNone(json.loads(response.read())["target"])
            self.assertIsNone(dashboard.learning_target())

            invalid_target = Request(
                f"{base_url}/api/learning/target",
                data=json.dumps({"sample_id": 99}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(invalid_target)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        finally:
            dashboard.close()

    def test_learning_api_rejects_image_paths_outside_the_dataset(self):
        with self.assertRaisesRegex(ValueError, "inside the selected dataset"):
            DashboardServer(
                "127.0.0.1",
                0,
                {},
                learning_examples=[("YOU", Path(__file__).resolve())],
            )


if __name__ == "__main__":
    unittest.main()
