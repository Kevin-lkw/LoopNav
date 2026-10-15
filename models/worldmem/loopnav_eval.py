"""LoopNav I/O and the checkpoint-compatible WorldMem input adapter.

This module intentionally does not import ``datasets.video.loopnav_dataset``.
That loader is used by training and is currently under development.  Evaluation
of the local 72.5k checkpoint needs the exact, 25-dimensional protocol used by
the loader at the checkpoint's training commit, including its historical pose
convention.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch


ACTION_KEYS = (
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

CANONICAL_PATH_TYPES = ("ABA", "ABCA")
CANONICAL_RANGES = (5, 15, 30, 50)
CANONICAL_VILLAGES = (
    "desert_village_20",
    "plains_village_20",
    "savanna_village_20",
    "snowy_village_20",
    "taiga_village_20",
    "zombie_village_20",
)

LEGACY_ADAPTER = "legacy-aba5-step72500"
LEGACY_WIDTH = 25
LEGACY_FPS = 20.0
LEGACY_FRAME_SIZE = (640, 360)  # width, height


@dataclass(frozen=True)
class LoopNavSample:
    """One paired LoopNav trajectory in the benchmark directory."""

    data_root: Path
    path_type: str
    nav_range: int | str
    village: str
    stem: str
    video_path: Path
    json_path: Path

    @property
    def relative_stem(self) -> Path:
        return Path(self.path_type) / str(self.nav_range) / self.village / self.stem

    @property
    def output_relative_path(self) -> Path:
        return self.relative_stem.with_suffix(".mp4")

    @property
    def metadata_key(self) -> str:
        # The full-data metadata index uses type/village/range, unlike test_data.
        return f"{self.path_type}/{self.village}/{self.nav_range}/{self.stem}.json"


@dataclass
class LoadedLoopNavClip:
    """A decoded trajectory ready to receive a batch dimension."""

    sample: LoopNavSample
    video: torch.Tensor
    actions: torch.Tensor
    poses: torch.Tensor
    timestamps: torch.Tensor
    return_point: int
    source_frame_count: int
    evaluation_frame_count: int
    fps: float

    @property
    def context_frames(self) -> int:
        return self.return_point

    @property
    def prediction_frames(self) -> int:
        return self.evaluation_frame_count - self.return_point

    @property
    def loaded_video_frames(self) -> int:
        return int(self.video.shape[0])

    def as_batch(self) -> tuple[torch.Tensor, ...]:
        """Return the five tensors expected by WorldMem's validation step."""

        if self.loaded_video_frames != self.evaluation_frame_count:
            raise ValueError(
                "A context-only clip cannot be passed to validation_step; "
                "use the streaming rollout path"
            )
        return (
            self.video.unsqueeze(0),
            self.actions.unsqueeze(0),
            self.poses.unsqueeze(0),
            self.timestamps.unsqueeze(0),
            torch.tensor([self.return_point], dtype=torch.long),
        )


def _ordered(values: Sequence[Any], canonical: Sequence[Any]) -> list[Any]:
    order = {value: index for index, value in enumerate(canonical)}
    return sorted(values, key=lambda value: (order.get(value, len(order)), str(value)))


def discover_loopnav_samples(
    data_root: str | Path,
    *,
    path_types: Sequence[str] | None = None,
    nav_ranges: Sequence[int] | None = None,
    villages: Sequence[str] | None = None,
) -> list[LoopNavSample]:
    """Discover strict JSON/AVI pairs in ``type/range/village/stem`` order."""

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LoopNav data root is not a directory: {root}")

    selected_types = _ordered(
        tuple(path_types) if path_types is not None else tuple(p.name for p in root.iterdir() if p.is_dir()),
        CANONICAL_PATH_TYPES,
    )
    selected_ranges = _ordered(
        tuple(int(value) for value in nav_ranges) if nav_ranges is not None else CANONICAL_RANGES,
        CANONICAL_RANGES,
    )
    selected_villages = set(villages) if villages is not None else None

    samples: list[LoopNavSample] = []
    for path_type in selected_types:
        for nav_range in selected_ranges:
            subtree = root / path_type / str(nav_range)
            if not subtree.exists():
                continue

            json_files = {
                path.relative_to(subtree).with_suffix(""): path
                for path in subtree.glob("*/*.json")
                if selected_villages is None or path.parent.name in selected_villages
            }
            video_files = {
                path.relative_to(subtree).with_suffix(""): path
                for path in subtree.glob("*/*.avi")
                if selected_villages is None or path.parent.name in selected_villages
            }

            missing_video = sorted(json_files.keys() - video_files.keys())
            missing_json = sorted(video_files.keys() - json_files.keys())
            if missing_video or missing_json:
                details = []
                if missing_video:
                    details.append(f"missing AVI for {missing_video[:3]}")
                if missing_json:
                    details.append(f"missing JSON for {missing_json[:3]}")
                raise FileNotFoundError(f"Unpaired LoopNav files below {subtree}: {'; '.join(details)}")

            village_names = _ordered({key.parts[0] for key in json_files}, CANONICAL_VILLAGES)
            for village in village_names:
                for key in sorted((key for key in json_files if key.parts[0] == village), key=str):
                    samples.append(
                        LoopNavSample(
                            data_root=root,
                            path_type=path_type,
                            nav_range=nav_range,
                            village=village,
                            stem=key.name,
                            video_path=video_files[key],
                            json_path=json_files[key],
                        )
                    )

    if not samples:
        filters = f"path_types={path_types}, nav_ranges={nav_ranges}, villages={villages}"
        raise FileNotFoundError(f"No paired LoopNav samples found below {root} ({filters})")

    relative_outputs = [sample.output_relative_path for sample in samples]
    if len(relative_outputs) != len(set(relative_outputs)):
        raise ValueError("LoopNav discovery produced duplicate output paths")
    return samples


def discover_turn_samples(
    data_root: str | Path,
    *,
    angle_buckets: Sequence[str] = ("60-90", "90-120", "120-150", "150-180"),
    villages: Sequence[str] | None = None,
    village_ids: Sequence[str] | None = ("18", "19", "20"),
) -> list[LoopNavSample]:
    """Discover TURN's ``village/angle_bucket/stem`` layout."""

    root = Path(data_root).expanduser().resolve()
    turn_root = root / "TURN" if (root / "TURN").is_dir() else root
    if not turn_root.is_dir():
        raise FileNotFoundError(f"TURN data root is not a directory: {turn_root}")

    selected_villages = set(villages) if villages is not None else None
    selected_ids = set(str(value) for value in village_ids) if village_ids is not None else None
    samples: list[LoopNavSample] = []
    for village_dir in sorted(path for path in turn_root.iterdir() if path.is_dir()):
        village = village_dir.name
        if selected_villages is not None and village not in selected_villages:
            continue
        if selected_ids is not None and village.rsplit("_", 1)[-1] not in selected_ids:
            continue
        for angle_bucket in angle_buckets:
            subtree = village_dir / str(angle_bucket)
            if not subtree.is_dir():
                continue
            json_files = {path.stem: path for path in subtree.glob("*.json")}
            video_files = {path.stem: path for path in subtree.glob("*.avi")}
            missing_video = sorted(json_files.keys() - video_files.keys())
            missing_json = sorted(video_files.keys() - json_files.keys())
            if missing_video or missing_json:
                raise FileNotFoundError(
                    f"Unpaired TURN files below {subtree}: "
                    f"missing AVI={missing_video[:3]}, missing JSON={missing_json[:3]}"
                )
            for stem in sorted(json_files):
                samples.append(
                    LoopNavSample(
                        data_root=turn_root,
                        path_type="TURN",
                        nav_range=str(angle_bucket),
                        village=village,
                        stem=stem,
                        video_path=video_files[stem],
                        json_path=json_files[stem],
                    )
                )
    if not samples:
        raise FileNotFoundError(
            f"No paired TURN samples found below {turn_root} "
            f"(angle_buckets={angle_buckets}, villages={villages}, village_ids={village_ids})"
        )
    return samples


def read_loopnav_records(json_path: str | Path) -> list[Mapping[str, Any]]:
    path = Path(json_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read LoopNav JSON {path}: {exc}") from exc

    if not isinstance(records, list) or not records:
        raise ValueError(f"LoopNav JSON must contain a non-empty list: {path}")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError(f"Every LoopNav record must be an object: {path}")
    return records


def find_return_point(records: Sequence[Mapping[str, Any]]) -> int:
    """Find the first target frame using the canonical LoopNav convention.

    ``return_point`` is a frame count: frames ``[:return_point]`` are observed
    and frames ``[return_point:]`` are predicted.  Canonical ABA/ABCA files do
    not carry an explicit marker, so the fallback exactly reproduces the
    checkpoint-era reverse scan for the final frame with a far-away goal.
    """

    if not records:
        raise ValueError("Cannot find a return point in an empty trajectory")

    explicit = [index for index, record in enumerate(records) if record.get("return_point") is True]
    if len(explicit) > 1:
        raise ValueError(
            "Multiple explicit return points are not supported by the single-stage WorldMem evaluator"
        )
    if explicit:
        return_point = explicit[0]
    else:
        try:
            start_x = float(records[0]["x"])
            start_z = float(records[0]["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("First LoopNav record is missing numeric x/z coordinates") from exc

        return_point = -1
        for index in range(len(records) - 1, -1, -1):
            goal = records[index].get("goal")
            if goal is None:
                continue
            try:
                goal_x = float(goal["x"])
                goal_z = float(goal["z"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Record {index} has a malformed goal") from exc
            if abs(goal_x - start_x) > 2 or abs(goal_z - start_z) > 2:
                return_point = index
                break
        if return_point < 0:
            raise ValueError("No LoopNav return point found: trajectory never has a goal >2 blocks away")

    if not 0 < return_point < len(records):
        raise ValueError(
            f"Return point {return_point} must leave non-empty context and target for {len(records)} frames"
        )
    return return_point


def convert_actions_legacy(records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """Reproduce the 25D action protocol used to train the local checkpoint."""

    actions = torch.zeros((len(records), LEGACY_WIDTH), dtype=torch.float32)
    for index, record in enumerate(records):
        try:
            action = record["action"]
            camera = action["camera"]
            if action["forward"]:
                actions[index, 11] = 1
            if action["jump"]:
                actions[index, 17] = 1
            actions[index, 16] = float(camera[0])
            actions[index, 15] = float(camera[1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Record {index} has a malformed action") from exc
    return actions


def convert_poses_legacy(records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """Reproduce the local checkpoint's historical pose preprocessing.

    The angle conversion and angle order are deliberately bug-compatible.
    Correcting either one changes the protocol and requires another checkpoint.
    """

    # HEAD used NumPy's default float64 here.  Keeping that dtype matters:
    # absolute Minecraft coordinates are large and were differenced only after
    # being stored in this array.
    poses = np.zeros((len(records), 5), dtype=np.float64)
    for index, record in enumerate(records):
        try:
            poses[index, 0] = float(record["x"])
            poses[index, 1] = float(record["y"])
            poses[index, 2] = float(record["z"])
            poses[index, 3] = int(float(record["yaw"]) * 180 / 15) * 15
            poses[index, 4] = int(float(record["pitch"]) * 180 / 15) * 15
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Record {index} has malformed pose fields") from exc

    # The checkpoint-era loader replaced a sometimes-corrupt first height with
    # the second frame before making poses relative.
    if len(poses) >= 2:
        poses[0, 1] = poses[1, 1]
    poses[:, :3] -= poses[:1, :3]
    poses[:, -1] = -poses[:, -1]
    poses[:, 3:] %= 360
    return torch.from_numpy(poses)


def read_video_rgb(
    video_path: str | Path,
    *,
    expected_frames: int | None = None,
    expected_size: tuple[int, int] | None = LEGACY_FRAME_SIZE,
    decode_frames: int | None = None,
) -> tuple[np.ndarray, float]:
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open LoopNav video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if expected_frames is not None and reported_frames > 0 and reported_frames != expected_frames:
        capture.release()
        raise ValueError(
            f"JSON/video length mismatch for {path}: "
            f"JSON={expected_frames}, AVI header={reported_frames}"
        )
    if decode_frames is not None and decode_frames <= 0:
        capture.release()
        raise ValueError("decode_frames must be positive")

    frames = []
    while decode_frames is None or len(frames) < decode_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

    if not frames:
        raise ValueError(f"LoopNav video has no decodable frames: {path}")
    video = np.stack(frames, axis=0)
    required_decoded = decode_frames if decode_frames is not None else expected_frames
    if required_decoded is not None and len(video) != required_decoded:
        raise ValueError(
            f"Could not decode the required frames for {path}: "
            f"required={required_decoded}, decoded={len(video)}"
        )
    if expected_size is not None and (video.shape[2], video.shape[1]) != expected_size:
        raise ValueError(
            f"Unexpected frame size for {path}: got {video.shape[2]}x{video.shape[1]}, "
            f"expected {expected_size[0]}x{expected_size[1]}"
        )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS {fps} for {path}")
    return video, fps


def load_loopnav_clip(
    sample: LoopNavSample,
    *,
    max_prediction_frames: int | None = None,
    expected_return_point: int | None = None,
    expected_length: int | None = None,
    context_only_video: bool = False,
) -> LoadedLoopNavClip:
    """Decode and adapt one canonical LoopNav trajectory."""

    records = read_loopnav_records(sample.json_path)
    source_frame_count = len(records)
    return_point = find_return_point(records)
    if expected_return_point is not None and return_point != expected_return_point:
        raise ValueError(
            f"Return-point mismatch for {sample.metadata_key}: "
            f"derived={return_point}, metadata={expected_return_point}"
        )
    if expected_length is not None and source_frame_count != expected_length:
        raise ValueError(
            f"Length mismatch for {sample.metadata_key}: JSON={source_frame_count}, metadata={expected_length}"
        )

    stop = source_frame_count
    if max_prediction_frames is not None:
        if max_prediction_frames <= 0:
            raise ValueError("max_prediction_frames must be positive")
        stop = min(stop, return_point + max_prediction_frames)

    decode_stop = return_point if context_only_video else stop
    video_rgb, fps = read_video_rgb(
        sample.video_path,
        expected_frames=source_frame_count,
        decode_frames=decode_stop,
    )
    if abs(fps - LEGACY_FPS) > 0.01:
        raise ValueError(
            f"The local checkpoint/SGCS path is locked to 20 FPS; got {fps:g} for {sample.video_path}"
        )

    evaluation_records = records[:stop]
    video = torch.from_numpy(video_rgb.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    actions = convert_actions_legacy(evaluation_records)
    poses = convert_poses_legacy(evaluation_records)
    timestamps = torch.arange(stop, dtype=torch.long)

    condition_lengths = {len(actions), len(poses), len(timestamps)}
    if condition_lengths != {stop} or len(video) != decode_stop:
        raise RuntimeError(
            f"Internal LoopNav alignment error for {sample.relative_stem}: "
            f"video={len(video)}, conditions={condition_lengths}, "
            f"expected_video={decode_stop}, expected_conditions={stop}"
        )
    return LoadedLoopNavClip(
        sample=sample,
        video=video.contiguous(),
        actions=actions.contiguous(),
        poses=poses.contiguous(),
        timestamps=timestamps,
        return_point=return_point,
        source_frame_count=source_frame_count,
        evaluation_frame_count=stop,
        fps=fps,
    )


def load_metadata_index(path: str | Path) -> dict[str, Mapping[str, Any]]:
    metadata_path = Path(path).expanduser().resolve()
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read LoopNav metadata index {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"LoopNav metadata index must be an object: {metadata_path}")
    return metadata


def metadata_expectations(
    sample: LoopNavSample,
    metadata_index: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[int | None, int | None]:
    if metadata_index is None:
        return None, None
    if sample.metadata_key not in metadata_index:
        raise KeyError(f"Sample is absent from metadata index: {sample.metadata_key}")
    entry = metadata_index[sample.metadata_key]
    try:
        return int(entry["return_point"]), int(entry["length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed metadata entry for {sample.metadata_key}: {entry}") from exc


def _prediction_to_uint8(prediction: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().float().cpu().numpy()
    prediction = np.asarray(prediction)
    if prediction.ndim == 5:
        if prediction.shape[1] != 1:
            raise ValueError(f"Video writer only supports batch size 1, got {prediction.shape}")
        prediction = prediction[:, 0]
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError(f"Prediction must have shape T,C,H,W (or T,1,C,H,W), got {prediction.shape}")
    if prediction.shape[0] == 0:
        raise ValueError("Refusing to write an empty prediction")
    prediction = np.clip(prediction, 0.0, 1.0)
    return np.rint(prediction * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)


def inspect_video_file(
    path: str | Path,
    *,
    expected_frames: int | None = None,
    expected_fps: float | None = None,
    expected_size: tuple[int, int] | None = None,
) -> dict[str, float | int]:
    """Read back basic MP4 properties and optionally enforce them."""

    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open generated video: {video_path}")
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()

    if expected_frames is not None and frames != expected_frames:
        raise ValueError(f"Generated video frame count is {frames}, expected {expected_frames}: {video_path}")
    if expected_fps is not None and abs(fps - expected_fps) > 0.01:
        raise ValueError(f"Generated video FPS is {fps:g}, expected {expected_fps:g}: {video_path}")
    if expected_size is not None and (width, height) != expected_size:
        raise ValueError(
            f"Generated video size is {width}x{height}, expected "
            f"{expected_size[0]}x{expected_size[1]}: {video_path}"
        )
    return {"frames": frames, "fps": fps, "width": width, "height": height}


def valid_existing_prediction(
    path: str | Path,
    *,
    expected_frames: int,
    expected_fps: float,
    expected_size: tuple[int, int] = LEGACY_FRAME_SIZE,
) -> bool:
    try:
        inspect_video_file(
            path,
            expected_frames=expected_frames,
            expected_fps=expected_fps,
            expected_size=expected_size,
        )
    except (OSError, ValueError):
        return False
    return True


def save_prediction_video(
    prediction: torch.Tensor | np.ndarray,
    output_path: str | Path,
    *,
    fps: float = LEGACY_FPS,
    overwrite: bool = False,
) -> dict[str, float | int]:
    """Atomically save a pure prediction video and verify it by reopening."""

    frames_rgb = _prediction_to_uint8(prediction)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Prediction already exists (pass --overwrite to replace it): {output}")

    height, width = frames_rgb.shape[1:3]
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create MP4 writer for {temporary}")

    try:
        for frame_rgb in frames_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    try:
        properties = inspect_video_file(
            temporary,
            expected_frames=len(frames_rgb),
            expected_fps=fps,
            expected_size=(width, height),
        )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return properties


def write_json_atomic(payload: Mapping[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
