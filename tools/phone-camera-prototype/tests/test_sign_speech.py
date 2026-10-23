import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sign_speech import Pyttsx3Speaker, SignRecognitionHistory, StablePredictionGate


class SignRecognitionHistoryTests(unittest.TestCase):
    def test_records_each_stable_label_and_repeats_after_new_segment(self):
        history = SignRecognitionHistory()

        self.assertTrue(history.observe(1.0, "YOU"))
        self.assertFalse(history.observe(1.1, "YOU"))
        self.assertTrue(history.observe(1.2, "HELP"))
        self.assertTrue(history.observe(1.3, "YOU"))
        history.reset_segment()
        self.assertTrue(history.observe(2.0, "YOU"))

        self.assertEqual(
            [event["label"] for event in history.events],
            ["YOU", "HELP", "YOU", "YOU"],
        )

    def test_keeps_only_the_configured_number_of_recent_signs(self):
        history = SignRecognitionHistory(limit=2)

        history.observe(1.0, "YOU")
        history.reset_segment()
        history.observe(2.0, "HELP")
        history.reset_segment()
        history.observe(3.0, "FOOD")

        self.assertEqual([event["label"] for event in history.events], ["HELP", "FOOD"])

    def test_history_limit_must_be_positive(self):
        with self.assertRaises(ValueError):
            SignRecognitionHistory(limit=0)


class StablePredictionGateTests(unittest.TestCase):
    def test_emits_after_label_is_held_for_configured_duration(self):
        gate = StablePredictionGate(hold_s=0.5)

        self.assertIsNone(gate.update(1.0, "THANK"))
        self.assertIsNone(gate.update(1.49, "THANK"))
        self.assertEqual(gate.update(1.5, "THANK"), "THANK")

    def test_emits_a_stable_label_only_once_until_it_changes_or_resets(self):
        gate = StablePredictionGate(hold_s=0.5)

        gate.update(2.0, "FOOD")
        self.assertEqual(gate.update(2.5, "FOOD"), "FOOD")
        self.assertIsNone(gate.update(3.0, "FOOD"))
        self.assertIsNone(gate.update(3.1, "THANK"))
        self.assertEqual(gate.update(3.6, "THANK"), "THANK")

    def test_same_label_can_be_spoken_again_after_a_segment_reset(self):
        gate = StablePredictionGate(hold_s=0.5)

        gate.update(2.0, "YOU")
        self.assertEqual(gate.update(2.5, "YOU"), "YOU")
        gate.reset()
        gate.update(3.0, "YOU")
        self.assertEqual(gate.update(3.5, "YOU"), "YOU")

    def test_unknown_prediction_resets_the_hold(self):
        gate = StablePredictionGate(hold_s=0.5)

        gate.update(4.0, "THANK")
        self.assertIsNone(gate.update(4.3, None))
        self.assertIsNone(gate.update(4.4, "THANK"))
        self.assertEqual(gate.update(4.9, "THANK"), "THANK")

    def test_hold_duration_must_be_positive_and_finite(self):
        for hold_s in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(hold_s=hold_s), self.assertRaises(ValueError):
                StablePredictionGate(hold_s)

    def test_speaker_uses_pyttsx3_off_the_calling_thread(self):
        engine = MagicMock()
        pyttsx3 = SimpleNamespace(init=MagicMock(return_value=engine))
        with patch.dict("sys.modules", {"pyttsx3": pyttsx3}):
            speaker = Pyttsx3Speaker()
            speaker.speak("I_ME_MINE_MY")
            speaker.close()

        engine.say.assert_called_once_with("I ME MINE MY")
        engine.runAndWait.assert_called_once_with()

    def test_speaker_processes_multiple_words_on_the_same_engine(self):
        engine = MagicMock()
        pyttsx3 = SimpleNamespace(init=MagicMock(return_value=engine))
        with patch.dict("sys.modules", {"pyttsx3": pyttsx3}):
            speaker = Pyttsx3Speaker()
            speaker.speak("YOU")
            speaker.speak("THANK_YOU")
            speaker.close()

        self.assertEqual(
            [call.args[0] for call in engine.say.call_args_list],
            ["YOU", "THANK YOU"],
        )
        self.assertEqual(engine.runAndWait.call_count, 2)

    def test_speaker_surfaces_engine_initialization_failures(self):
        pyttsx3 = SimpleNamespace(init=MagicMock(side_effect=OSError("no audio device")))
        with patch.dict("sys.modules", {"pyttsx3": pyttsx3}):
            speaker = Pyttsx3Speaker()
            speaker.close()

        with self.assertRaisesRegex(RuntimeError, "no audio device"):
            speaker.raise_if_failed()


if __name__ == "__main__":
    unittest.main()
