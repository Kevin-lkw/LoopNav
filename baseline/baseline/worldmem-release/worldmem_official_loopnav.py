"""Official WorldMem checkpoint inputs for the LoopNav benchmark.

This adapter follows the data protocol in the current ``xizaoqu/WorldMem``
``data_generator.py`` and ``datasets/video/minecraft_video_dataset.py``.  It is
deliberately separate from :mod:`loopnav_eval`, whose conversion functions
reproduce the historical protocol of a locally trained checkpoint.

LoopNav stores camera deltas as ``[yaw, pitch]`` in radians at 20 FPS.  Official
WorldMem uses categorical MineDojo camera actions at 10 FPS, which its dataset
converts to ``{-1, 0, +1}``.  This adapter selects source frames ``0, 2, 4, ...``
and aggregates each selected frame ``i > 0`` from raw actions ``[i-1, i]``:
the four motion buttons represented by the released training converter use
logical OR, and camera deltas are summed before their sign is taken. Yaw maps
to ``cameraX`` (index 16), and pitch maps to ``cameraY`` (index 15). LoopNav
``jump``, ``sneak``, and ``sprint`` are intentionally left zero because the
official checkpoint's dataset converter never activates those columns. The
selected frame at source index zero receives the official initial no-op.

The native model output is therefore 10 FPS.  Raw 20 FPS return points and
frame indices are retained in clip fields and metadata so evaluation can use
the correctly corresponding ground truth instead of tail-aligning by count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from loopnav_eval import (
    LoopNavSample,
    discover_loopnav_samples,
    discover_turn_samples,
    find_return_point,
    read_loopnav_records,
    read_video_rgb,
)


OFFICIAL_ACTION_KEYS = (
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraY",
    "cameraX",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
)

OFFICIAL_ACTION_WIDTH = 25
OFFICIAL_ADAPTER = "xizaoqu-c1cc91b-loopnav20to10-v1"
OFFICIAL_TRAINING_FPS = 10.0
OFFICIAL_MODEL_FPS = 10.0
LOOPNAV_SOURCE_FPS = 20.0
SOURCE_STRIDE = 2
LOOPNAV_FRAME_SIZE = (640, 360)  # width, height
CAMERA_ZERO_EPSILON = 1e-8
CAMERA_ENCODING = "nonzero-sign: yaw->cameraX[16], pitch->cameraY[15]"

SUPPORTED_TASKS = ("TURN", "ABA_FIXED", "ABCA")
DEFAULT_NAV_RANGES = (5, 15, 30, 50)
DEFAULT_TURN_ANGLE_BUCKETS = ("60-90", "90-120", "120-150", "150-180")
DEFAULT_STRICT_TEST_VILLAGE_IDS = ("19", "20")

_BUTTON_TO_INDEX = {
    "forward": 11,
    "back": 12,
    "left": 13,
    "right": 14,
}
DROPPED_LOOPNAV_BUTTONS = ("jump", "sneak", "sprint")


@dataclass
class OfficialLoopNavClip:
    """One LoopNav trajectory adapted for the authors' released checkpoint."""

    sample: LoopNavSample
    video: torch.Tensor
    actions: torch.Tensor
    poses: torch.Tensor
    timestamps: torch.Tensor
    return_point: int
    source_frame_count: int
    evaluation_frame_count: int
    raw_return_point: int
    raw_source_frame_count: int
    fps: float
    metadata: dict[str, Any]

    @property
    def context_frames(self) -> int:
        return self.return_point

    @property
    def prediction_frames(self) -> int:
        return self.evaluation_frame_count - self.return_point

    @property
    def loaded_video_frames(self) -> int:
        return int(self.video.shape[0])

    @property
    def source_first_target_index(self) -> int:
        return SOURCE_STRIDE * self.return_point

    def as_batch(self) -> tuple[torch.Tensor, ...]:
        """Return the tensors expected by WorldMem's validation interface."""

        if self.loaded_video_frames != self.evaluation_frame_count:
            raise ValueError(
                "A context-only clip cannot be passed to validation_step; "
                "use the streaming official-checkpoint rollout path"
            )
        return (
            self.video.unsqueeze(0),
            self.actions.unsqueeze(0),
            self.poses.unsqueeze(0),
            self.timestamps.unsqueeze(0),
            torch.tensor([self.return_point], dtype=torch.long),
        )


def _normalise_task(task: str) -> str:
    normalised = str(task).strip().upper()
    if normalised not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported LoopNav task {task!r}; expected one of {SUPPORTED_TASKS}"
        )
    return normalised


def _root_containing_task(data_root: str | Path, task: str) -> Path:
    """Return the parent directory expected by ``discover_loopnav_samples``."""

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{task} data root is not a directory: {root}")
    if root.name.upper() == task:
        return root.parent
    if (root / task).is_dir():
        return root
    raise FileNotFoundError(
        f"{task} root must be either the task directory itself or its parent: {root}"
    )


def _has_village_id(sample: LoopNavSample, village_ids: set[str]) -> bool:
    return sample.village.rsplit("_", 1)[-1] in village_ids


def discover_official_samples(
    tasks: Sequence[str] = SUPPORTED_TASKS,
    turn_root: str | Path | None = None,
    aba_fixed_root: str | Path | None = None,
    abca_root: str | Path | None = None,
    village_ids: Sequence[str | int] = DEFAULT_STRICT_TEST_VILLAGE_IDS,
    *,
    nav_ranges: Sequence[int] = DEFAULT_NAV_RANGES,
    turn_angle_buckets: Sequence[str] = DEFAULT_TURN_ANGLE_BUCKETS,
) -> list[LoopNavSample]:
    """Discover strict-test TURN, ABA_FIXED, and ABCA samples.

    ``turn_root``, ``aba_fixed_root``, and ``abca_root`` may point either to a
    task directory (for example ``.../test_data/ABCA``) or, for the latter two,
    to its parent.  Only villages whose final underscore-delimited component is
    in ``village_ids`` are returned.  The benchmark's strict test split uses
    village IDs 19 and 20; village 18 is validation and is excluded by default.
    """

    selected_tasks = [_normalise_task(task) for task in tasks]
    if not selected_tasks:
        raise ValueError("At least one LoopNav task must be requested")
    if len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError(f"Duplicate LoopNav tasks are not allowed: {selected_tasks}")

    selected_ids = {str(value) for value in village_ids}
    if not selected_ids:
        raise ValueError("At least one village ID must be selected")

    roots: dict[str, str | Path | None] = {
        "TURN": turn_root,
        "ABA_FIXED": aba_fixed_root,
        "ABCA": abca_root,
    }
    discovered: list[LoopNavSample] = []
    for task in selected_tasks:
        task_root = roots[task]
        if task_root is None:
            raise ValueError(f"{task} was requested but its data root was not provided")

        if task == "TURN":
            samples = discover_turn_samples(
                task_root,
                angle_buckets=tuple(str(value) for value in turn_angle_buckets),
                village_ids=tuple(sorted(selected_ids)),
            )
        else:
            parent_root = _root_containing_task(task_root, task)
            samples = discover_loopnav_samples(
                parent_root,
                path_types=[task],
                nav_ranges=tuple(int(value) for value in nav_ranges),
            )
            samples = [sample for sample in samples if _has_village_id(sample, selected_ids)]
            if not samples:
                raise FileNotFoundError(
                    f"No strict-test {task} samples found for village IDs "
                    f"{sorted(selected_ids)} below {task_root}"
                )
        discovered.extend(samples)

    relative_outputs = [sample.output_relative_path for sample in discovered]
    if len(relative_outputs) != len(set(relative_outputs)):
        raise ValueError("Official LoopNav discovery produced duplicate output paths")
    return discovered


def _validate_return_point(return_point: int, frame_count: int) -> int:
    if not 0 < return_point < frame_count:
        raise ValueError(
            f"Return point {return_point} must leave non-empty context and target "
            f"for {frame_count} frames"
        )
    return return_point


def find_official_return_point(
    records: Sequence[Mapping[str, Any]],
    task: str,
) -> int:
    """Find the first predicted frame under each strict benchmark protocol."""

    task = _normalise_task(task)
    if not records:
        raise ValueError("Cannot find a return point in an empty trajectory")
    explicit = [
        index for index, record in enumerate(records)
        if record.get("return_point") is True
    ]

    if task in {"TURN", "ABA_FIXED"}:
        if len(explicit) != 1:
            raise ValueError(
                f"Strict {task} requires exactly one explicit return_point=true "
                f"marker, found {len(explicit)}"
            )
        return _validate_return_point(explicit[0], len(records))

    if explicit:
        raise ValueError(
            "Strict ABCA uses the reverse-goal return-point heuristic and must "
            "not contain an explicit return_point=true marker"
        )
    return find_return_point(records)


def _pressed(value: Any, *, frame_index: int, key: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    raise ValueError(
        f"Record {frame_index} action {key!r} must be boolean (or numeric 0/1), "
        f"got {value!r}"
    )


def _camera_sign(value: Any, *, frame_index: int, axis: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Record {frame_index} camera {axis} delta is not numeric: {value!r}"
        ) from exc
    if not np.isfinite(numeric):
        raise ValueError(
            f"Record {frame_index} camera {axis} delta is not finite: {numeric}"
        )
    if numeric > CAMERA_ZERO_EPSILON:
        return 1.0
    if numeric < -CAMERA_ZERO_EPSILON:
        return -1.0
    return 0.0


def _camera_delta(value: Any, *, frame_index: int, axis: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Record {frame_index} camera {axis} delta is not numeric: {value!r}"
        ) from exc
    if not np.isfinite(numeric):
        raise ValueError(
            f"Record {frame_index} camera {axis} delta is not finite: {numeric}"
        )
    return numeric


def _native_source_indices(raw_frame_count: int) -> tuple[int, ...]:
    if raw_frame_count <= 0:
        raise ValueError("Official conversion requires at least one raw frame")
    return tuple(range(0, raw_frame_count, SOURCE_STRIDE))


def _validate_source_indices(
    records: Sequence[Mapping[str, Any]],
    source_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    indices = (
        _native_source_indices(len(records))
        if source_indices is None
        else tuple(int(index) for index in source_indices)
    )
    if not indices:
        raise ValueError("Official conversion requires at least one source index")
    if any(index < 0 or index >= len(records) for index in indices):
        raise IndexError(
            f"Source indices {indices[:3]}... are outside {len(records)} raw frames"
        )
    if indices[0] != 0 or any(
        right - left != SOURCE_STRIDE
        for left, right in zip(indices, indices[1:])
    ):
        raise ValueError(
            f"Official source indices must follow 0,{SOURCE_STRIDE},"
            f"{2 * SOURCE_STRIDE},...; got {indices[:5]}"
        )
    return indices


def _read_action(
    record: Mapping[str, Any],
    *,
    frame_index: int,
) -> tuple[Mapping[str, Any], Sequence[Any]]:
    try:
        action = record["action"]
        if not isinstance(action, Mapping):
            raise TypeError("action is not an object")
        camera = action["camera"]
        if (
            not isinstance(camera, Sequence)
            or isinstance(camera, (str, bytes))
            or len(camera) != 2
        ):
            raise TypeError("camera is not a two-element sequence")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Record {frame_index} has a malformed LoopNav action"
        ) from exc
    return action, camera


def convert_actions_official(
    records: Sequence[Mapping[str, Any]],
    source_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Aggregate raw 20 FPS actions into the official native 10 FPS protocol."""

    indices = _validate_source_indices(records, source_indices)
    actions = torch.zeros(
        (len(indices), OFFICIAL_ACTION_WIDTH),
        dtype=torch.float32,
    )
    # Official generated trajectories start with a no-op observation.  Keep
    # model frame zero all-zero even if LoopNav's first record carries input.
    for model_index, source_index in enumerate(indices[1:], start=1):
        aggregate = (
            _read_action(records[source_index - 1], frame_index=source_index - 1),
            _read_action(records[source_index], frame_index=source_index),
        )
        for key, column in _BUTTON_TO_INDEX.items():
            if any(
                _pressed(action[key], frame_index=raw_index, key=key)
                for raw_index, (action, _camera) in zip(
                    (source_index - 1, source_index), aggregate
                )
            ):
                actions[model_index, column] = 1.0

        yaw_sum = sum(
            _camera_delta(
                camera[0], frame_index=raw_index, axis="yaw"
            )
            for raw_index, (_action, camera) in zip(
                (source_index - 1, source_index), aggregate
            )
        )
        pitch_sum = sum(
            _camera_delta(
                camera[1], frame_index=raw_index, axis="pitch"
            )
            for raw_index, (_action, camera) in zip(
                (source_index - 1, source_index), aggregate
            )
        )
        actions[model_index, 16] = _camera_sign(
            yaw_sum, frame_index=source_index, axis="aggregated yaw"
        )
        actions[model_index, 15] = _camera_sign(
            pitch_sum, frame_index=source_index, axis="aggregated pitch"
        )
    return actions


def convert_poses_official(
    records: Sequence[Mapping[str, Any]],
    source_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Select native frames and emit float32 official pose conditioning."""

    indices = _validate_source_indices(records, source_indices)
    poses = np.empty((len(indices), 5), dtype=np.float64)
    for model_index, source_index in enumerate(indices):
        record = records[source_index]
        try:
            values = np.asarray(
                [
                    float(record["x"]),
                    float(record["y"]),
                    float(record["z"]),
                    np.degrees(float(record["pitch"])),
                    np.degrees(float(record["yaw"])),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Record {source_index} has malformed x/y/z/pitch/yaw pose fields"
            ) from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Record {source_index} has non-finite pose fields")
        poses[model_index] = values

    # LoopNav's first pose is valid, so the official training loader's
    # corruption workaround must not be copied here.
    poses[:, :3] -= poses[:1, :3]
    poses[:, 4] = -poses[:, 4]
    poses[:, 3:] %= 360.0
    return torch.from_numpy(poses.astype(np.float32))


def _protocol_metadata(
    sample: LoopNavSample,
    *,
    raw_return_point: int,
    raw_source_frame_count: int,
    source_frame_count: int,
    evaluation_frame_count: int,
) -> dict[str, Any]:
    return_point = (raw_return_point + SOURCE_STRIDE - 1) // SOURCE_STRIDE
    source_first_target_index = SOURCE_STRIDE * return_point
    return {
        "adapter": OFFICIAL_ADAPTER,
        "task": sample.path_type,
        "action_width": OFFICIAL_ACTION_WIDTH,
        "camera_encoding": CAMERA_ENCODING,
        "camera_zero_epsilon": CAMERA_ZERO_EPSILON,
        "action_aggregation": (
            "forward/back/left/right OR; camera deltas sum then sign over "
            "source[i-1:i+1]"
        ),
        "dropped_loopnav_buttons": list(DROPPED_LOOPNAV_BUTTONS),
        "dropped_button_reason": (
            "official released-checkpoint dataset converter leaves action "
            "columns 17/18/19 zero"
        ),
        "pose_encoding": "[relative x,y,z,pitch_deg,-yaw_deg] modulo 360",
        "pose_dtype": "float32",
        "first_frame_height_repaired": False,
        "return_point": return_point,
        "raw_return_point": raw_return_point,
        "raw_source_frame_count": raw_source_frame_count,
        "source_frame_count": source_frame_count,
        "evaluation_frame_count": evaluation_frame_count,
        "native_context_frame_count": return_point,
        "native_target_frame_count": evaluation_frame_count - return_point,
        "source_fps": LOOPNAV_SOURCE_FPS,
        "model_fps": OFFICIAL_MODEL_FPS,
        "official_training_fps": OFFICIAL_TRAINING_FPS,
        "source_stride": SOURCE_STRIDE,
        "source_first_target_index": source_first_target_index,
        "first_target_source_index": source_first_target_index,
        "source_index_rule": "model_index j maps to source_index 2*j",
        "ground_truth_index_rule": (
            "raw_frames[source_first_target_index::source_stride]"
        ),
        "frame_mapping": "model_index j -> source_index 2*j",
        "downsampled": True,
        "fps_domain_shift": False,
    }


def load_official_clip(
    sample: LoopNavSample,
    max_prediction_frames: int | None = None,
    context_only_video: bool = True,
) -> OfficialLoopNavClip:
    """Load one sample with official actions, poses, timestamps, and framing.

    With the default ``context_only_video=True``, only observed native 10 FPS
    RGB frames are retained, while actions, poses, and timestamps cover the
    entire requested native rollout.  ``max_prediction_frames`` is measured in
    native 10 FPS generated frames.
    """

    task = _normalise_task(sample.path_type)
    records = read_loopnav_records(sample.json_path)
    raw_source_frame_count = len(records)
    raw_return_point = find_official_return_point(records, task)
    source_indices = _native_source_indices(raw_source_frame_count)
    source_frame_count = len(source_indices)
    return_point = (raw_return_point + SOURCE_STRIDE - 1) // SOURCE_STRIDE
    source_first_target_index = SOURCE_STRIDE * return_point
    if return_point >= source_frame_count:
        raise ValueError(
            f"20->10 FPS conversion leaves no target frame for "
            f"{sample.relative_stem}: raw frames={raw_source_frame_count}, "
            f"raw return point={raw_return_point}, first target source index="
            f"{source_first_target_index}"
        )

    stop = source_frame_count
    if max_prediction_frames is not None:
        if (
            isinstance(max_prediction_frames, bool)
            or not isinstance(max_prediction_frames, (int, np.integer))
            or max_prediction_frames <= 0
        ):
            raise ValueError("max_prediction_frames must be a positive integer")
        stop = min(stop, return_point + int(max_prediction_frames))

    model_decode_stop = return_point if context_only_video else stop
    # To retain model indices [0, model_decode_stop), decode through the last
    # selected raw source index, inclusive.
    raw_decode_stop = SOURCE_STRIDE * (model_decode_stop - 1) + 1
    video_rgb_raw, source_fps = read_video_rgb(
        sample.video_path,
        expected_frames=raw_source_frame_count,
        expected_size=LOOPNAV_FRAME_SIZE,
        decode_frames=raw_decode_stop,
    )
    if abs(source_fps - LOOPNAV_SOURCE_FPS) > 0.01:
        raise ValueError(
            f"LoopNav source must be 20 FPS before official 10 FPS conversion; "
            f"got {source_fps:g} for {sample.video_path}"
        )

    evaluation_source_indices = source_indices[:stop]
    video_rgb = video_rgb_raw[::SOURCE_STRIDE]
    video = (
        torch.from_numpy(video_rgb.copy())
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
        .contiguous()
    )
    actions = convert_actions_official(
        records, evaluation_source_indices
    ).contiguous()
    poses = convert_poses_official(records, evaluation_source_indices).contiguous()
    timestamps = torch.arange(stop, dtype=torch.long)

    condition_lengths = {len(actions), len(poses), len(timestamps)}
    if condition_lengths != {stop} or len(video) != model_decode_stop:
        raise RuntimeError(
            f"Internal official-adapter alignment error for "
            f"{sample.relative_stem}: video={len(video)}, "
            f"conditions={condition_lengths}, expected_video={model_decode_stop}, "
            f"expected_conditions={stop}"
        )

    return OfficialLoopNavClip(
        sample=sample,
        video=video,
        actions=actions,
        poses=poses,
        timestamps=timestamps,
        return_point=return_point,
        source_frame_count=source_frame_count,
        evaluation_frame_count=stop,
        raw_return_point=raw_return_point,
        raw_source_frame_count=raw_source_frame_count,
        fps=OFFICIAL_MODEL_FPS,
        metadata=_protocol_metadata(
            sample,
            raw_return_point=raw_return_point,
            raw_source_frame_count=raw_source_frame_count,
            source_frame_count=source_frame_count,
            evaluation_frame_count=stop,
        ),
    )


__all__ = [
    "CAMERA_ENCODING",
    "CAMERA_ZERO_EPSILON",
    "DEFAULT_STRICT_TEST_VILLAGE_IDS",
    "DROPPED_LOOPNAV_BUTTONS",
    "LOOPNAV_SOURCE_FPS",
    "OFFICIAL_ACTION_KEYS",
    "OFFICIAL_ACTION_WIDTH",
    "OFFICIAL_MODEL_FPS",
    "OFFICIAL_TRAINING_FPS",
    "SOURCE_STRIDE",
    "OfficialLoopNavClip",
    "convert_actions_official",
    "convert_poses_official",
    "discover_official_samples",
    "find_official_return_point",
    "load_official_clip",
]
