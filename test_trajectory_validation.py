import math
import unittest

from trajectory_validation import (
    validate_horizontal_loop_closure,
    validate_horizontal_speed_transitions,
    validate_native_camera_transitions,
)


def record(yaw_degrees: float, pitch_degrees: float = 0.0) -> dict:
    return {
        "yaw": math.radians(yaw_degrees),
        "pitch": math.radians(pitch_degrees),
    }


class CameraTransitionValidationTest(unittest.TestCase):
    def test_accepts_smooth_10fps_transition(self):
        records = [record(0), record(6), record(12)]
        self.assertEqual(
            validate_native_camera_transitions(records, max_degrees=30),
            (True, None),
        )

    def test_rejects_transition_over_30_degrees(self):
        records = [record(0), record(6), record(31)]
        ok, reason = validate_native_camera_transitions(records, max_degrees=30)
        self.assertFalse(ok)
        self.assertIn("camera_discontinuity", reason)
        self.assertIn("yaw_deg=31.000", reason)

    def test_uses_wrapped_yaw_delta(self):
        records = [record(179), record(-179), record(-178)]
        self.assertEqual(
            validate_native_camera_transitions(records, max_degrees=30),
            (True, None),
        )

    def test_checks_pitch_as_well_as_yaw(self):
        records = [record(0, 0), record(0, 5), record(0, -31)]
        ok, reason = validate_native_camera_transitions(records, max_degrees=30)
        self.assertFalse(ok)
        self.assertIn("pitch_deg=-31.000", reason)

    def test_20fps_check_catches_a_jump_hidden_by_stride_two(self):
        records = [record(0), record(90), record(0)]
        self.assertEqual(
            validate_native_camera_transitions(
                records,
                source_stride=2,
                max_degrees=30,
            ),
            (True, None),
        )
        ok, reason = validate_native_camera_transitions(
            records,
            source_stride=1,
            max_degrees=30,
        )
        self.assertFalse(ok)
        self.assertIn("yaw_deg=90.000", reason)


def position_record(
    x: float,
    z: float = 0.0,
    frame_count: int = 0,
) -> dict:
    return {"x": x, "z": z, "frame_count": frame_count}


class HorizontalSpeedValidationTest(unittest.TestCase):
    def test_accepts_speed_below_15_blocks_per_second(self):
        records = [
            position_record(0.0, frame_count=20),
            position_record(0.749, frame_count=21),
        ]
        self.assertEqual(
            validate_horizontal_speed_transitions(records),
            (True, None),
        )

    def test_rejects_speed_at_15_blocks_per_second(self):
        records = [
            position_record(0.0, frame_count=20),
            position_record(0.75, frame_count=21),
        ]
        ok, reason = validate_horizontal_speed_transitions(records)
        self.assertFalse(ok)
        self.assertIn("horizontal_speed_discontinuity", reason)
        self.assertIn("speed_bps=15.000", reason)

    def test_uses_actual_frame_count_gap(self):
        records = [
            position_record(0.0, frame_count=20),
            position_record(0.75, frame_count=22),
        ]
        self.assertEqual(
            validate_horizontal_speed_transitions(records),
            (True, None),
        )

    def test_rejects_non_positive_frame_gap(self):
        records = [
            position_record(0.0, frame_count=20),
            position_record(0.1, frame_count=20),
        ]
        ok, reason = validate_horizontal_speed_transitions(records)
        self.assertFalse(ok)
        self.assertIn("invalid_frame_gap", reason)


class HorizontalLoopClosureValidationTest(unittest.TestCase):
    def test_accepts_loop_below_two_blocks(self):
        records = [
            position_record(10.0, z=20.0),
            position_record(11.0, z=21.0),
        ]
        self.assertEqual(
            validate_horizontal_loop_closure(records),
            (True, None),
        )

    def test_rejects_loop_at_two_blocks(self):
        records = [
            position_record(10.0, z=20.0),
            position_record(12.0, z=20.0),
        ]
        ok, reason = validate_horizontal_loop_closure(records)
        self.assertFalse(ok)
        self.assertIn("loop_closure_discontinuity", reason)
        self.assertIn("distance_blocks=2.000", reason)

    def test_rejects_invalid_endpoint(self):
        records = [
            position_record(0.0),
            {"x": float("nan"), "z": 0.0},
        ]
        ok, reason = validate_horizontal_loop_closure(records)
        self.assertFalse(ok)
        self.assertIn("invalid_loop_endpoint", reason)


if __name__ == "__main__":
    unittest.main()
