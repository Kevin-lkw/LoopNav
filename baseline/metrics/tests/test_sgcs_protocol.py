"""Protocol checks use tiny videos and synthetic masks; no SAM weights needed."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import sgcs_eval
import eval_turn_sgcs


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace(stride=1, tau=0.1, confidence_threshold=0.5, device="cpu")

    def video(self, name, count):
        path = self.root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (32, 32))
        self.assertTrue(writer.isOpened())
        for i in range(count):
            writer.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
        writer.release()
        return path

    def test_tail_alignment_and_zero_weight_frames_stay_in_mean(self):
        a, b = self.video("gt.avi", 8), self.video("prediction.avi", 3)
        scores = [{"sgcs": 1.0, "total_weight": 1.0},
                  {"sgcs": 0.0, "total_weight": 0.0},
                  {"sgcs": 0.5, "total_weight": 1.0}]
        with patch.object(sgcs_eval, "compute_frame_sgcs", side_effect=scores):
            result = sgcs_eval.evaluate_video_pair(None, a, b, self.args, [])
        self.assertEqual([f["video_a_frame_index"] for f in result["frames"]], [5, 6, 7])
        self.assertAlmostEqual(result["mean_sgcs"], 0.5)

    def test_explicit_return_alignment_does_not_shift_to_video_tail(self):
        a, b = self.video("gt.avi", 10), self.video("prediction.avi", 3)
        with patch.object(sgcs_eval, "compute_frame_sgcs", side_effect=lambda **kw: {"sgcs": 1.0}):
            result = sgcs_eval.evaluate_video_pair(None, a, b, self.args, [], 4, 0, "metadata_return_point")
        self.assertEqual([f["video_a_frame_index"] for f in result["frames"]], [4, 5, 6])
        self.assertEqual([f["video_b_frame_index"] for f in result["frames"]], [0, 1, 2])

    def test_return_point_must_be_unique_and_can_be_zero(self):
        path = self.root / "metadata.json"
        path.write_text(json.dumps([{"return_point": True}, {}]))
        self.assertEqual(sgcs_eval.read_explicit_return_point(path), 0)
        for rows in ([{}, {}], [{"return_point": True}, {"return_point": True}]):
            path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                sgcs_eval.read_explicit_return_point(path)

    def test_empty_masks_score_zero_with_audited_prompts(self):
        class EmptyProcessor:
            def __init__(self):
                self.prompts = []
            def set_image(self, image):
                return {}
            def set_text_prompt(self, prompt, state):
                self.prompts.append(prompt)
                return {"masks": []}
        processor = EmptyProcessor()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = sgcs_eval.compute_frame_sgcs(processor, frame, frame, sgcs_eval.DEFAULT_CATEGORIES, 0.1)
        self.assertEqual(result["sgcs"], 0)
        self.assertIn("house", processor.prompts)
        self.assertNotIn("building", processor.prompts)
        self.assertNotIn("path", processor.prompts)

    def test_turn_stops_at_shorter_return_segment(self):
        self.assertEqual(eval_turn_sgcs.sampled_indices_for_counts(40, 100, 40, 30, 5), (10, [0, 5]))


if __name__ == "__main__":
    unittest.main()
