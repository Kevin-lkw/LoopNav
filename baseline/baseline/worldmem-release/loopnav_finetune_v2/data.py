"""Corrected 10 FPS LoopNav training data for WorldMem.

The returned tensor order matches ``WorldMemMinecraft.training_step``:
eight local frames followed by eight causal memory frames.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from loopnav_eval import find_return_point


TASKS = ("TURN", "ABA_FIXED", "ABCA")
ACTION_WIDTH = 25
BUTTON_COLUMNS = {
    "forward": 11,
    "back": 12,
    "left": 13,
    "right": 14,
    "jump": 17,
    "sneak": 18,
    "sprint": 19,
}


def wrap_degrees_360(value: np.ndarray | float) -> np.ndarray | float:
    return np.mod(value, 360.0)


def wrapped_delta_degrees(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return (current - previous + 180.0) % 360.0 - 180.0


def convert_poses_corrected(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Convert raw Mineflayer poses to WorldMem's corrected convention."""

    poses = np.empty((len(records), 5), dtype=np.float32)
    for index, record in enumerate(records):
        values = np.asarray(
            [
                float(record["x"]),
                float(record["y"]),
                float(record["z"]),
                -math.degrees(float(record["pitch"])),
                math.degrees(float(record["yaw"])) + 180.0,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite pose at raw frame {index}")
        poses[index] = values
    poses[:, 3:] = wrap_degrees_360(poses[:, 3:])
    return poses


def _pressed(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key, False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Action {key!r} is not boolean: {value!r}")


def convert_native_10fps(
    records: Sequence[Mapping[str, Any]],
    *,
    source_stride: int = 2,
    camera_clip: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices, corrected poses, and pose-consistent 25D actions."""

    if source_stride != 2:
        raise ValueError("The frozen LoopNav v2 protocol requires source_stride=2")
    source_indices = np.arange(0, len(records), source_stride, dtype=np.int64)
    if len(source_indices) < 2:
        raise ValueError("A trajectory needs at least two native 10 FPS frames")

    raw_poses = convert_poses_corrected(records)
    poses = raw_poses[source_indices].copy()
    actions = np.zeros((len(source_indices), ACTION_WIDTH), dtype=np.float32)

    for native_index, raw_index in enumerate(source_indices[1:], start=1):
        raw_window = records[max(0, raw_index - source_stride + 1) : raw_index + 1]
        for key, column in BUTTON_COLUMNS.items():
            if any(_pressed(record.get("action", {}), key) for record in raw_window):
                actions[native_index, column] = 1.0

        delta_pitch = wrapped_delta_degrees(
            poses[native_index : native_index + 1, 3],
            poses[native_index - 1 : native_index, 3],
        )[0]
        delta_yaw = wrapped_delta_degrees(
            poses[native_index : native_index + 1, 4],
            poses[native_index - 1 : native_index, 4],
        )[0]
        actions[native_index, 15] = np.clip(
            delta_pitch / 15.0, -camera_clip, camera_clip
        )
        actions[native_index, 16] = np.clip(
            -delta_yaw / 15.0, -camera_clip, camera_clip
        )

    return source_indices, poses, actions


def action_pose_mismatch_degrees(poses: np.ndarray, actions: np.ndarray) -> tuple[float, float]:
    """Measure mean absolute pitch/yaw mismatch for native transitions."""

    delta_pitch = wrapped_delta_degrees(poses[1:, 3], poses[:-1, 3])
    delta_yaw = wrapped_delta_degrees(poses[1:, 4], poses[:-1, 4])
    pitch_error = np.abs(delta_pitch - 15.0 * actions[1:, 15])
    yaw_error = np.abs(delta_yaw + 15.0 * actions[1:, 16])
    return float(pitch_error.mean()), float(yaw_error.mean())


def _village_id(path: Path) -> int:
    return int(path.parts[-3].rsplit("_", 1)[-1])


def discover_training_trajectories(
    data_root: str | Path,
    tasks: Sequence[str] = TASKS,
    village_ids: Sequence[int] = tuple(range(1, 17)),
) -> dict[str, list[Path]]:
    root = Path(data_root).expanduser().resolve()
    selected_ids = {int(value) for value in village_ids}
    inventory: dict[str, list[Path]] = {}
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unsupported task {task!r}; expected one of {TASKS}")
        task_root = root / task
        # The public dataset calls the fixed-return ABA task simply ABA.
        if task == "ABA_FIXED" and not task_root.exists():
            task_root = root / "ABA"
        paths = sorted(
            path
            for path in task_root.rglob("*.json")
            if _village_id(path) in selected_ids and path.with_suffix(".avi").is_file()
        )
        if not paths:
            raise FileNotFoundError(
                f"No {task} trajectories for village IDs {sorted(selected_ids)} under {task_root}"
            )
        inventory[task] = paths
    return inventory


@lru_cache(maxsize=128)
def _read_records(path: str) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Trajectory metadata is not a non-empty list: {path}")
    return tuple(payload)


def _return_point(records: Sequence[Mapping[str, Any]], task: str) -> int:
    explicit = [
        index for index, record in enumerate(records)
        if record.get("return_point") is True
    ]
    if task in {"TURN", "ABA_FIXED"}:
        if len(explicit) != 1:
            raise ValueError(
                f"{task} requires exactly one return_point marker, found {explicit}"
            )
        return explicit[0]
    if explicit:
        raise ValueError("ABCA must use its reverse-goal return-point heuristic")
    return find_return_point(records)


def _read_selected_rgb(
    video_path: Path,
    raw_indices: Sequence[int],
    frame_height: int,
    frame_width: int,
) -> np.ndarray:
    unique_indices = sorted({int(index) for index in raw_indices})
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    decoded: dict[int, np.ndarray] = {}
    try:
        for index in unique_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Could not decode raw frame {index} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded[index] = cv2.resize(
                frame,
                (frame_width, frame_height),
                interpolation=cv2.INTER_AREA,
            )
    finally:
        cap.release()
    return np.stack([decoded[int(index)] for index in raw_indices])


def _select_memory_indices(
    poses: np.ndarray,
    *,
    observed_stop: int,
    local_indices: np.ndarray,
    memory_frames: int,
    retrieval_dropout: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if observed_stop <= 0:
        candidates = np.asarray([0], dtype=np.int64)
    else:
        candidates = np.arange(observed_stop, dtype=np.int64)

    if rng.random() < retrieval_dropout:
        replace = len(candidates) < memory_frames
        return np.sort(rng.choice(candidates, size=memory_frames, replace=replace))

    candidate_poses = poses[candidates]
    target_poses = poses[local_indices]
    position_distance = np.linalg.norm(
        candidate_poses[:, None, :3] - target_poses[None, :, :3],
        axis=-1,
    )
    pitch_distance = np.abs(
        wrapped_delta_degrees(
            candidate_poses[:, None, 3],
            target_poses[None, :, 3],
        )
    )
    yaw_distance = np.abs(
        wrapped_delta_degrees(
            candidate_poses[:, None, 4],
            target_poses[None, :, 4],
        )
    )
    score = (
        position_distance / 2.0
        + pitch_distance / 75.0
        + yaw_distance / 105.0
    ).min(axis=1)
    order = candidates[np.argsort(score, kind="stable")]
    if len(order) >= memory_frames:
        selected = order[:memory_frames]
    else:
        selected = np.resize(order, memory_frames)
    return np.sort(selected.astype(np.int64))


def _valid_local_starts(
    poses: np.ndarray,
    *,
    local_frames: int,
    camera_clip: float,
) -> np.ndarray:
    """Return starts whose supervised transitions are representable."""

    max_start = len(poses) - local_frames
    if max_start < 0:
        return np.empty(0, dtype=np.int64)
    pitch_delta = np.abs(wrapped_delta_degrees(poses[1:, 3], poses[:-1, 3]))
    yaw_delta = np.abs(wrapped_delta_degrees(poses[1:, 4], poses[:-1, 4]))
    valid_transition = np.maximum(pitch_delta, yaw_delta) <= (
        15.0 * camera_clip + 1e-4
    )
    starts = []
    for start in range(max_start + 1):
        if valid_transition[start : start + local_frames - 1].all():
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


@dataclass(frozen=True)
class SampleDebug:
    task: str
    metadata_path: Path
    raw_return_point: int
    native_return_point: int
    local_indices: tuple[int, ...]
    memory_indices: tuple[int, ...]
    observed_stop: int


class CorrectedLoopNavTrainingDataset(Dataset):
    """Virtual, exactly task-balanced LoopNav training dataset."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        tasks: Sequence[str] = TASKS,
        village_ids: Sequence[int] = tuple(range(1, 17)),
        local_frames: int = 8,
        memory_frames: int = 8,
        source_stride: int = 2,
        frame_height: int = 360,
        frame_width: int = 640,
        return_focused_probability: float = 0.7,
        memory_retrieval_dropout: float = 0.1,
        virtual_epoch_size: int = 300_000,
        camera_clip: float = 1.0,
        seed: int = 20260725,
    ):
        self.tasks = tuple(tasks)
        self.inventory = discover_training_trajectories(
            data_root, self.tasks, village_ids
        )
        self.local_frames = int(local_frames)
        self.memory_frames = int(memory_frames)
        self.source_stride = int(source_stride)
        self.frame_height = int(frame_height)
        self.frame_width = int(frame_width)
        self.return_focused_probability = float(return_focused_probability)
        self.memory_retrieval_dropout = float(memory_retrieval_dropout)
        self.virtual_epoch_size = int(virtual_epoch_size)
        self.camera_clip = float(camera_clip)
        self.seed = int(seed)

        if self.local_frames != 8 or self.memory_frames != 8:
            raise ValueError("The official checkpoint requires 8 local + 8 memory frames")
        if self.virtual_epoch_size <= 0:
            raise ValueError("virtual_epoch_size must be positive")
        if not 0.0 <= self.return_focused_probability <= 1.0:
            raise ValueError("return_focused_probability must be in [0,1]")
        if not 0.0 <= self.memory_retrieval_dropout <= 1.0:
            raise ValueError("memory_retrieval_dropout must be in [0,1]")

    def __len__(self) -> int:
        return self.virtual_epoch_size

    def inventory_counts(self) -> dict[str, int]:
        return {task: len(paths) for task, paths in self.inventory.items()}

    def _rng(self, index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence([self.seed, int(index)])
        return np.random.default_rng(sequence)

    def build_sample(
        self,
        index: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], SampleDebug]:
        rng = self._rng(index)
        task = self.tasks[index % len(self.tasks)]
        paths = self.inventory[task]
        metadata_path = paths[int(rng.integers(0, len(paths)))]
        records = _read_records(str(metadata_path))
        source_indices, poses, actions = convert_native_10fps(
            records,
            source_stride=self.source_stride,
            camera_clip=self.camera_clip,
        )
        raw_return_point = _return_point(records, task)
        native_return_point = (
            raw_return_point + self.source_stride - 1
        ) // self.source_stride
        max_start = len(source_indices) - self.local_frames
        if max_start < 0:
            raise ValueError(
                f"Trajectory is shorter than {self.local_frames} native frames: {metadata_path}"
            )
        valid_starts = _valid_local_starts(
            poses,
            local_frames=self.local_frames,
            camera_clip=self.camera_clip,
        )
        if not len(valid_starts):
            raise ValueError(
                f"Trajectory has no action-representable local clip: {metadata_path}"
            )

        return_focused = rng.random() < self.return_focused_probability
        if return_focused:
            lower = min(native_return_point, max_start)
            eligible = valid_starts[valid_starts >= lower]
            if not len(eligible):
                eligible = valid_starts
            local_start = int(rng.choice(eligible))
            observed_stop = min(native_return_point, local_start)
        else:
            local_start = int(rng.choice(valid_starts))
            observed_stop = local_start

        local_indices = np.arange(
            local_start, local_start + self.local_frames, dtype=np.int64
        )
        memory_indices = _select_memory_indices(
            poses,
            observed_stop=observed_stop,
            local_indices=local_indices,
            memory_frames=self.memory_frames,
            retrieval_dropout=self.memory_retrieval_dropout,
            rng=rng,
        )
        if np.any(memory_indices >= max(1, observed_stop)):
            raise AssertionError("Causal memory selection returned a future frame")

        combined_indices = np.concatenate([local_indices, memory_indices])
        combined_raw_indices = source_indices[combined_indices]
        video = _read_selected_rgb(
            metadata_path.with_suffix(".avi"),
            combined_raw_indices,
            self.frame_height,
            self.frame_width,
        )

        combined_poses = poses[combined_indices].copy()
        combined_poses[:, :3] -= poses[local_indices[0], :3]
        combined_actions = actions[combined_indices].copy()
        timestamps = combined_indices.astype(np.float32)

        batch = (
            torch.from_numpy(video).permute(0, 3, 1, 2).float().div_(255.0),
            torch.from_numpy(combined_actions),
            torch.from_numpy(combined_poses),
            torch.from_numpy(timestamps),
        )
        debug = SampleDebug(
            task=task,
            metadata_path=metadata_path,
            raw_return_point=raw_return_point,
            native_return_point=native_return_point,
            local_indices=tuple(int(value) for value in local_indices),
            memory_indices=tuple(int(value) for value in memory_indices),
            observed_stop=int(observed_stop),
        )
        return batch, debug

    def __getitem__(self, index: int):
        batch, _debug = self.build_sample(int(index))
        return batch
