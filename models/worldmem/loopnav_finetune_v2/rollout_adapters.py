#!/usr/bin/env python3
"""Autoregressive LoopNav rollout for a fine-tuned WorldMem adapter.

This entry point deliberately uses the same corrected continuous 25D actions
and corrected pose convention as ``CorrectedLoopNavTrainingDataset``.  It must
not be replaced with ``eval_loopnav_official.py`` because that zero-shot
adapter quantizes camera input to {-1, 0, 1}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from loopnav_eval import (
    LoopNavSample,
    inspect_video_file,
    read_loopnav_records,
    read_video_rgb,
    write_json_atomic,
)
from loopnav_finetune_v2.data import (
    TASKS,
    convert_native_10fps,
    discover_training_trajectories,
    wrapped_delta_degrees,
)
from loopnav_finetune_v2.model import build_finetune_model
from loopnav_finetune_v2.validate_adapters import _load_adapter
from worldmem_official_loopnav import find_official_return_point
from worldmem_official_runtime import (
    AtomicPredictionWriter,
    amp_dtype,
    rollout_official_streaming,
)


FRAME_SIZE = (640, 360)
MODEL_FPS = 10.0
SOURCE_FPS = 20.0
SOURCE_STRIDE = 2
DEFAULT_RUN_DIR = Path(
    "loopnav_finetune_v2/runs/p1_rank8_updated_20260725"
)
DEFAULT_ADAPTER = (
    DEFAULT_RUN_DIR / "adapters/adapter_step0020000.pt"
)
DEFAULT_OUTPUT = (
    DEFAULT_RUN_DIR / "rollouts/step0020000_fixed20_test19_20"
)


@dataclass(frozen=True)
class RolloutSpec:
    sample: LoopNavSample
    raw_frames: int
    raw_return_point: int
    context_frames: int
    available_prediction_frames: int
    prediction_frames: int
    max_camera_delta_degrees: float


@dataclass
class RolloutClip:
    context_video: torch.Tensor
    ground_truth_rgb: np.ndarray
    context_rgb: np.ndarray
    actions: torch.Tensor
    poses: torch.Tensor
    timestamps: torch.Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("loopnav_finetune_v2/config_p1.yaml"),
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/LoopNav"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument("--village-ids", nargs="+", type=int, default=[19, 20])
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--prediction-frames", type=int, default=20)
    parser.add_argument("--context-preview-frames", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    parser.add_argument("--vae-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.samples_per_task <= 0:
        parser.error("--samples-per-task must be positive")
    if args.prediction_frames <= 0:
        parser.error("--prediction-frames must be positive")
    if args.context_preview_frames < 0:
        parser.error("--context-preview-frames must be non-negative")
    if args.vae_batch_size <= 0:
        parser.error("--vae-batch-size must be positive")
    return args


def _loopnav_sample(path: Path, data_root: Path, task: str) -> LoopNavSample:
    return LoopNavSample(
        data_root=data_root,
        path_type=task,
        nav_range=path.parent.name,
        village=path.parent.parent.name,
        stem=path.stem,
        video_path=path.with_suffix(".avi"),
        json_path=path,
    )


def _camera_limit(
    records: Sequence[dict[str, Any]],
    *,
    native_stop: int,
) -> float:
    _source_indices, poses, _actions = convert_native_10fps(records)
    selected = poses[:native_stop, 3:]
    if len(selected) < 2:
        return 0.0
    pitch = np.abs(wrapped_delta_degrees(selected[1:, 0], selected[:-1, 0]))
    yaw = np.abs(wrapped_delta_degrees(selected[1:, 1], selected[:-1, 1]))
    return float(max(pitch.max(initial=0.0), yaw.max(initial=0.0)))


def _discover_specs(args: argparse.Namespace) -> list[RolloutSpec]:
    selected: list[RolloutSpec] = []
    for task in args.tasks:
        paths = discover_training_trajectories(
            args.data_root,
            tasks=(task,),
            village_ids=args.village_ids,
        )[task]
        candidates: list[RolloutSpec] = []
        for path in paths:
            records = read_loopnav_records(path)
            raw_return = find_official_return_point(records, task)
            context = (raw_return + SOURCE_STRIDE - 1) // SOURCE_STRIDE
            native_frames = (len(records) + SOURCE_STRIDE - 1) // SOURCE_STRIDE
            available = native_frames - context
            if context <= 0 or available <= 0:
                continue
            prediction = min(args.prediction_frames, available)
            max_camera = _camera_limit(
                records,
                native_stop=context + prediction,
            )
            # Fine-tuning excluded local windows that cross >15 degree native
            # transitions, so previews should obey the same support.
            if max_camera > 15.0001:
                continue
            candidates.append(
                RolloutSpec(
                    sample=_loopnav_sample(path, args.data_root, task),
                    raw_frames=len(records),
                    raw_return_point=raw_return,
                    context_frames=context,
                    available_prediction_frames=available,
                    prediction_frames=prediction,
                    max_camera_delta_degrees=max_camera,
                )
            )
        candidates.sort(
            key=lambda spec: (
                spec.available_prediction_frames < args.prediction_frames,
                abs(spec.available_prediction_frames - args.prediction_frames),
                abs(spec.context_frames - 16),
                str(spec.sample.json_path),
            )
        )
        if len(candidates) < args.samples_per_task:
            raise ValueError(
                f"{task} has only {len(candidates)} safe rollout candidates for "
                f"villages {args.village_ids}"
            )
        selected.extend(candidates[: args.samples_per_task])
    return selected


def _load_clip(spec: RolloutSpec, *, camera_clip: float) -> RolloutClip:
    records = read_loopnav_records(spec.sample.json_path)
    source_indices, poses, actions = convert_native_10fps(
        records,
        source_stride=SOURCE_STRIDE,
        camera_clip=camera_clip,
    )
    stop = spec.context_frames + spec.prediction_frames
    source_indices = source_indices[:stop]
    poses = poses[:stop].copy()
    actions = actions[:stop].copy()
    poses[:, :3] -= poses[:1, :3]

    raw_decode_stop = int(source_indices[-1]) + 1
    raw_rgb, fps = read_video_rgb(
        spec.sample.video_path,
        expected_frames=spec.raw_frames,
        expected_size=FRAME_SIZE,
        decode_frames=raw_decode_stop,
    )
    if abs(fps - SOURCE_FPS) > 0.01:
        raise ValueError(
            f"Expected {SOURCE_FPS:g} FPS source, got {fps:g}: "
            f"{spec.sample.video_path}"
        )
    native_rgb = raw_rgb[::SOURCE_STRIDE]
    if len(native_rgb) != stop:
        raise RuntimeError(
            f"Native RGB length mismatch for {spec.sample.relative_stem}: "
            f"{len(native_rgb)} != {stop}"
        )
    context_rgb = native_rgb[: spec.context_frames].copy()
    ground_truth_rgb = native_rgb[spec.context_frames :].copy()
    context_video = (
        torch.from_numpy(context_rgb)
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
        .contiguous()
    )
    return RolloutClip(
        context_video=context_video,
        ground_truth_rgb=ground_truth_rgb,
        context_rgb=context_rgb,
        actions=torch.from_numpy(actions).contiguous(),
        poses=torch.from_numpy(poses).contiguous(),
        timestamps=torch.arange(stop, dtype=torch.float32),
    )


def _read_rgb_video(path: Path, expected_frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open generated video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != expected_frames:
        raise ValueError(
            f"Decoded {len(frames)} frames from {path}, expected {expected_frames}"
        )
    return np.stack(frames)


def _write_rgb_video(path: Path, frames: np.ndarray, *, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError(f"Expected non-empty THWC RGB video, got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.mp4")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {temporary}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    try:
        inspect_video_file(
            temporary,
            expected_frames=len(frames),
            expected_fps=fps,
            expected_size=(width, height),
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _comparison_frames(
    *,
    context_rgb: np.ndarray,
    ground_truth_rgb: np.ndarray,
    prediction_rgb: np.ndarray,
    context_preview_frames: int,
) -> np.ndarray:
    preview = context_rgb[-context_preview_frames:] if context_preview_frames else context_rgb[:0]
    height, width = ground_truth_rgb.shape[1:3]
    output = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    phases = [
        ("CONTEXT", preview, preview),
        ("ROLLOUT", ground_truth_rgb, prediction_rgb),
    ]
    for phase, left_rgb, right_rgb in phases:
        for frame_index, (left, right) in enumerate(zip(left_rgb, right_rgb)):
            canvas = np.zeros((height + 42, width * 2, 3), dtype=np.uint8)
            canvas[42:, :width] = left
            canvas[42:, width:] = right
            left_label = "OBSERVED CONTEXT" if phase == "CONTEXT" else "GROUND TRUTH"
            right_label = "OBSERVED CONTEXT" if phase == "CONTEXT" else "STEP 20K ROLLOUT"
            cv2.putText(
                canvas,
                left_label,
                (16, 29),
                font,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                right_label,
                (width + 16, 29),
                font,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            ordinal = (
                frame_index - len(preview)
                if phase == "CONTEXT"
                else frame_index
            )
            cv2.putText(
                canvas,
                f"{phase} frame {ordinal:+d}",
                (width - 190, 28),
                font,
                0.48,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            output.append(canvas)
    return np.stack(output)


def _quality_metrics(
    ground_truth_rgb: np.ndarray,
    prediction_rgb: np.ndarray,
) -> dict[str, Any]:
    gt = ground_truth_rgb.astype(np.float32)
    pred = prediction_rgb.astype(np.float32)
    mse_per_frame = np.mean((gt - pred) ** 2, axis=(1, 2, 3))
    psnr_per_frame = [
        float("inf") if mse == 0 else float(10.0 * math.log10(255.0**2 / mse))
        for mse in mse_per_frame
    ]

    def sharpness(frames: np.ndarray) -> list[float]:
        return [
            float(
                cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
                    cv2.CV_64F,
                ).var()
            )
            for frame in frames
        ]

    gt_sharpness = sharpness(ground_truth_rgb)
    pred_sharpness = sharpness(prediction_rgb)
    sharpness_ratio = [
        pred_value / max(gt_value, 1e-8)
        for pred_value, gt_value in zip(pred_sharpness, gt_sharpness)
    ]
    return {
        "psnr_per_frame_db": psnr_per_frame,
        "mean_psnr_db": float(np.mean(psnr_per_frame)),
        "gt_laplacian_variance_per_frame": gt_sharpness,
        "prediction_laplacian_variance_per_frame": pred_sharpness,
        "prediction_to_gt_sharpness_ratio_per_frame": sharpness_ratio,
        "mean_prediction_to_gt_sharpness_ratio": float(np.mean(sharpness_ratio)),
    }


def _seed_for_spec(base_seed: int, spec: RolloutSpec) -> int:
    digest = hashlib.sha256(
        spec.sample.relative_stem.as_posix().encode("utf-8")
    ).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**63)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_directory(
    output_root: Path,
    spec: RolloutSpec,
    index: int,
) -> Path:
    safe_stem = spec.sample.stem.replace("/", "_")
    return output_root / spec.sample.path_type / f"sample_{index:02d}_{safe_stem}"


def _write_readme(
    output_root: Path,
    *,
    adapter: Path,
    specs: Sequence[RolloutSpec],
    results: Sequence[dict[str, Any]],
    status: str,
) -> None:
    lines = [
        "# WorldMem LoopNav step-20k rollout preview",
        "",
        f"Status: **{status}**",
        "",
        f"- adapter: `{adapter}`",
        "- split: held-out villages 19–20",
        "- cadence: 10 FPS from raw 20 FPS frames `0,2,4,...`",
        "- action: corrected continuous 25D LoopNav action used by fine-tuning",
        "- pose: corrected `[x,y,z,pitch,yaw]` convention used by fine-tuning",
        "- model context: 8 local tokens plus 8 retrieved memory tokens",
        "- rollout: autoregressive, 20 denoising steps per generated frame",
        "",
        "Each sample folder contains:",
        "",
        "- `comparison.mp4`: target-only GT (left) vs prediction (right); turnaround begins immediately",
        "- `comparison_with_context.mp4`: last observed context followed by the same GT/prediction comparison",
        "- `context_then_prediction.mp4`: last context frames followed by rollout",
        "- `context_then_ground_truth.mp4`: last context frames followed by GT",
        "- `prediction.mp4` and `ground_truth.mp4`: target-only videos",
        "- `metadata.json`: exact source path, protocol, timing, and sharpness/PSNR diagnostics",
        "",
        "## Samples",
        "",
    ]
    result_by_path = {
        item["source_json"]: item
        for item in results
    }
    for task in TASKS:
        task_specs = [spec for spec in specs if spec.sample.path_type == task]
        for task_index, spec in enumerate(task_specs):
            sample_dir = _sample_directory(output_root, spec, task_index)
            result = result_by_path.get(str(spec.sample.json_path))
            state = "complete" if result is not None else "pending"
            lines.extend(
                [
                    f"- `{task}` ({state}): `{sample_dir}`",
                    f"  - source: `{spec.sample.json_path}`",
                    f"  - context/prediction: {spec.context_frames}/{spec.prediction_frames} frames",
                ]
            )
    (output_root / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    args.config = args.config.expanduser().resolve()
    args.adapter = args.adapter.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    specs = _discover_specs(args)
    plan = {
        "adapter": str(args.adapter),
        "output_root": str(args.output_root),
        "tasks": list(args.tasks),
        "village_ids": list(args.village_ids),
        "samples": [
            {
                "task": spec.sample.path_type,
                "source_json": str(spec.sample.json_path),
                "source_video": str(spec.sample.video_path),
                "context_frames": spec.context_frames,
                "available_prediction_frames": spec.available_prediction_frames,
                "prediction_frames": spec.prediction_frames,
                "max_camera_delta_degrees": spec.max_camera_delta_degrees,
            }
            for spec in specs
        ],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if not args.adapter.is_file():
        raise FileNotFoundError(args.adapter)

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_readme(
        args.output_root,
        adapter=args.adapter,
        specs=specs,
        results=[],
        status="running",
    )
    write_json_atomic(
        {
            **plan,
            "status": "starting",
            "created_at": datetime.now().astimezone().isoformat(),
        },
        args.output_root / "run_manifest.json",
    )

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {device}")
    dtype = amp_dtype(args.amp_dtype, device)
    print("[model] building official-initialized adapter model", flush=True)
    model, parameter_counts = build_finetune_model(cfg)
    loaded_step = _load_adapter(
        model,
        args.adapter,
        expected_revision=str(cfg.official.revision),
    )
    print(f"[model] loaded adapter step {loaded_step}", flush=True)
    model = model.to(device=device).eval()

    results: list[dict[str, Any]] = []
    task_counts = {task: 0 for task in TASKS}
    for sequence_index, spec in enumerate(specs, start=1):
        task_index = task_counts[spec.sample.path_type]
        task_counts[spec.sample.path_type] += 1
        sample_dir = _sample_directory(args.output_root, spec, task_index)
        prediction_path = sample_dir / "prediction.mp4"
        if sample_dir.exists() and prediction_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Existing rollout found; pass --overwrite to replace it: {sample_dir}"
            )
        sample_dir.mkdir(parents=True, exist_ok=True)
        clip = _load_clip(spec, camera_clip=float(cfg.data.camera_clip))
        sample_seed = _seed_for_spec(args.seed, spec)
        _set_seed(sample_seed)
        writer = AtomicPredictionWriter(
            prediction_path,
            fps=MODEL_FPS,
            size=FRAME_SIZE,
            overwrite=True,
        )
        started = time.monotonic()
        print(
            f"[sample {sequence_index}/{len(specs)}] "
            f"{spec.sample.relative_stem} context={spec.context_frames} "
            f"prediction={spec.prediction_frames}",
            flush=True,
        )
        try:
            stats = rollout_official_streaming(
                model,
                context_video=clip.context_video,
                actions=clip.actions,
                poses=clip.poses,
                timestamps=clip.timestamps,
                device=device,
                dtype=dtype,
                vae_batch_size=args.vae_batch_size,
                frame_sink=writer.write,
                progress_description=str(spec.sample.relative_stem),
            )
            writer.commit(spec.prediction_frames)
        except Exception:
            writer.abort()
            raise
        elapsed = time.monotonic() - started

        prediction_rgb = _read_rgb_video(
            prediction_path,
            expected_frames=spec.prediction_frames,
        )
        preview_count = min(args.context_preview_frames, len(clip.context_rgb))
        context_preview = clip.context_rgb[-preview_count:] if preview_count else clip.context_rgb[:0]
        _write_rgb_video(
            sample_dir / "ground_truth.mp4",
            clip.ground_truth_rgb,
            fps=MODEL_FPS,
        )
        _write_rgb_video(
            sample_dir / "context_then_prediction.mp4",
            np.concatenate([context_preview, prediction_rgb]),
            fps=MODEL_FPS,
        )
        _write_rgb_video(
            sample_dir / "context_then_ground_truth.mp4",
            np.concatenate([context_preview, clip.ground_truth_rgb]),
            fps=MODEL_FPS,
        )
        comparison_with_context = _comparison_frames(
            context_rgb=clip.context_rgb,
            ground_truth_rgb=clip.ground_truth_rgb,
            prediction_rgb=prediction_rgb,
            context_preview_frames=preview_count,
        )
        _write_rgb_video(
            sample_dir / "comparison_with_context.mp4",
            comparison_with_context,
            fps=MODEL_FPS,
        )
        _write_rgb_video(
            sample_dir / "comparison.mp4",
            _comparison_frames(
                context_rgb=clip.context_rgb,
                ground_truth_rgb=clip.ground_truth_rgb,
                prediction_rgb=prediction_rgb,
                context_preview_frames=0,
            ),
            fps=MODEL_FPS,
        )
        metrics = _quality_metrics(clip.ground_truth_rgb, prediction_rgb)
        result = {
            "status": "complete",
            "task": spec.sample.path_type,
            "source_json": str(spec.sample.json_path),
            "source_video": str(spec.sample.video_path),
            "sample_directory": str(sample_dir),
            "adapter": str(args.adapter),
            "adapter_step": loaded_step,
            "seed": sample_seed,
            "context_frames_10fps": spec.context_frames,
            "prediction_frames_10fps": spec.prediction_frames,
            "available_prediction_frames_10fps": spec.available_prediction_frames,
            "source_first_target_index_20fps": SOURCE_STRIDE * spec.context_frames,
            "max_camera_delta_degrees": spec.max_camera_delta_degrees,
            "protocol": {
                "action": "continuous 25D from corrected pose delta",
                "camera_y": "wrapped_delta(corrected_pitch_deg) / 15",
                "camera_x": "-wrapped_delta(corrected_yaw_deg) / 15",
                "pose": "corrected [x,y,z,-pitch_deg,yaw_deg+180]",
                "source_stride": SOURCE_STRIDE,
                "model_fps": MODEL_FPS,
                "local_tokens": 8,
                "memory_tokens": 8,
                "sampling_timesteps": 20,
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "encode_seconds": stats.encode_seconds,
                "sample_decode_seconds": stats.sample_decode_seconds,
                "seconds_per_prediction_frame": (
                    stats.sample_decode_seconds / spec.prediction_frames
                ),
                "peak_cuda_bytes": stats.peak_cuda_bytes,
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
            },
            "quality": metrics,
            "videos": {
                "comparison": str(sample_dir / "comparison.mp4"),
                "comparison_with_context": str(
                    sample_dir / "comparison_with_context.mp4"
                ),
                "prediction": str(prediction_path),
                "ground_truth": str(sample_dir / "ground_truth.mp4"),
                "context_then_prediction": str(
                    sample_dir / "context_then_prediction.mp4"
                ),
                "context_then_ground_truth": str(
                    sample_dir / "context_then_ground_truth.mp4"
                ),
            },
        }
        write_json_atomic(result, sample_dir / "metadata.json")
        results.append(result)
        _write_readme(
            args.output_root,
            adapter=args.adapter,
            specs=specs,
            results=results,
            status=f"running ({len(results)}/{len(specs)})",
        )
        print(
            f"[saved] {sample_dir / 'comparison.mp4'} "
            f"elapsed={elapsed:.1f}s "
            f"sharpness_ratio={metrics['mean_prediction_to_gt_sharpness_ratio']:.3f}",
            flush=True,
        )

    manifest = {
        **plan,
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "adapter_step": loaded_step,
        "parameter_counts": parameter_counts,
        "amp_dtype": args.amp_dtype,
        "device": str(device),
        "results": results,
    }
    write_json_atomic(manifest, args.output_root / "run_manifest.json")
    _write_readme(
        args.output_root,
        adapter=args.adapter,
        specs=specs,
        results=results,
        status="complete",
    )
    print(f"[done] {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
