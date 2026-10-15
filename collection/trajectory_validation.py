import math
from collections.abc import Mapping, Sequence
from typing import Any


def wrapped_angle_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2 * math.pi) - math.pi


def validate_native_camera_transitions(
    records: Sequence[Mapping[str, Any]],
    *,
    source_stride: int = 2,
    max_degrees: float = 30.0,
    tolerance: float = 1e-4,
) -> tuple[bool, str | None]:
    if source_stride < 1:
        raise ValueError("source_stride must be >= 1")
    if not math.isfinite(max_degrees) or max_degrees <= 0:
        raise ValueError("max_degrees must be positive and finite")
    if len(records) <= source_stride:
        return False, "not_enough_frames"

    for raw_index in range(source_stride, len(records), source_stride):
        previous = records[raw_index - source_stride]
        current = records[raw_index]
        try:
            delta_yaw = math.degrees(
                wrapped_angle_delta(float(current["yaw"]), float(previous["yaw"]))
            )
            delta_pitch = math.degrees(
                wrapped_angle_delta(
                    float(current["pitch"]),
                    float(previous["pitch"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"invalid_camera_pose raw_index={raw_index}: {exc}"

        magnitude = max(abs(delta_yaw), abs(delta_pitch))
        if magnitude > max_degrees + tolerance:
            return (
                False,
                "camera_discontinuity "
                f"raw_indices={raw_index - source_stride}->{raw_index} "
                f"pitch_deg={delta_pitch:.3f} yaw_deg={delta_yaw:.3f} "
                f"max_deg={magnitude:.3f}",
            )

    return True, None


def validate_horizontal_speed_transitions(
    records: Sequence[Mapping[str, Any]],
    *,
    fps: float = 20.0,
    max_speed_bps: float = 15.0,
) -> tuple[bool, str | None]:
    """Reject non-positive time gaps and horizontal speeds at/above the limit."""

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not math.isfinite(max_speed_bps) or max_speed_bps <= 0:
        raise ValueError("max_speed_bps must be positive and finite")
    if len(records) < 2:
        return False, "not_enough_frames"

    for raw_index in range(1, len(records)):
        previous = records[raw_index - 1]
        current = records[raw_index]
        try:
            delta_frame = float(current["frame_count"]) - float(
                previous["frame_count"]
            )
            if not math.isfinite(delta_frame) or delta_frame <= 0:
                return (
                    False,
                    "invalid_frame_gap "
                    f"raw_indices={raw_index - 1}->{raw_index} "
                    f"delta_frame={delta_frame}",
                )
            dx = float(current["x"]) - float(previous["x"])
            dz = float(current["z"]) - float(previous["z"])
            if not math.isfinite(dx) or not math.isfinite(dz):
                raise ValueError("non-finite position delta")
        except (KeyError, TypeError, ValueError) as exc:
            return (
                False,
                f"invalid_position raw_index={raw_index}: {exc}",
            )

        displacement = math.hypot(dx, dz)
        speed_bps = displacement * fps / delta_frame
        if speed_bps >= max_speed_bps:
            return (
                False,
                "horizontal_speed_discontinuity "
                f"raw_indices={raw_index - 1}->{raw_index} "
                f"frame_counts={previous['frame_count']}->{current['frame_count']} "
                f"dx={dx:.3f} dz={dz:.3f} "
                f"displacement={displacement:.3f} "
                f"speed_bps={speed_bps:.3f} "
                f"max_speed_bps={max_speed_bps:.3f}",
            )

    return True, None


def validate_horizontal_loop_closure(
    records: Sequence[Mapping[str, Any]],
    *,
    max_distance_blocks: float = 2.0,
) -> tuple[bool, str | None]:
    """Reject trajectories whose final XZ position is too far from the start."""

    if not math.isfinite(max_distance_blocks) or max_distance_blocks <= 0:
        raise ValueError("max_distance_blocks must be positive and finite")
    if len(records) < 2:
        return False, "not_enough_frames"

    try:
        start_x = float(records[0]["x"])
        start_z = float(records[0]["z"])
        end_x = float(records[-1]["x"])
        end_z = float(records[-1]["z"])
        if not all(
            math.isfinite(value)
            for value in (start_x, start_z, end_x, end_z)
        ):
            raise ValueError("non-finite loop endpoint")
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid_loop_endpoint: {exc}"

    dx = end_x - start_x
    dz = end_z - start_z
    distance = math.hypot(dx, dz)
    if distance >= max_distance_blocks:
        return (
            False,
            "loop_closure_discontinuity "
            f"dx={dx:.3f} dz={dz:.3f} "
            f"distance_blocks={distance:.3f} "
            f"max_distance_blocks={max_distance_blocks:.3f}",
        )

    return True, None
