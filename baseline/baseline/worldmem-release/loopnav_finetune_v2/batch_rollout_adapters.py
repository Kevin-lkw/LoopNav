#!/usr/bin/env python3
"""Restartable, sharded WorldMem adapter rollout from a frozen sample manifest."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from loopnav_eval import (
    LoopNavSample,
    inspect_video_file,
    write_json_atomic,
)
from loopnav_finetune_v2.model import build_finetune_model
from loopnav_finetune_v2.rollout_adapters import (
    FRAME_SIZE,
    MODEL_FPS,
    RolloutSpec,
    _load_clip,
    _seed_for_spec,
    _set_seed,
)
from loopnav_finetune_v2.validate_adapters import _load_adapter
from worldmem_official_runtime import (
    AtomicPredictionWriter,
    amp_dtype,
    rollout_official_streaming,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("loopnav_finetune_v2/config_p1.yaml"),
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    parser.add_argument("--vae-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < num-shards")
    if args.vae_batch_size <= 0:
        parser.error("--vae-batch-size must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    return args


def _load_selection(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "worldmem_loopnav_comparable_batch_manifest_v1":
        raise ValueError(f"Unexpected selection manifest format: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Selection manifest has no samples: {path}")
    return payload, samples


def _prediction_frames(sample: Mapping[str, Any]) -> int:
    return int(sample["prediction_frames_10fps"])


def _context_frames(sample: Mapping[str, Any]) -> int:
    return int(sample["context_frames_10fps"])


def _raw_frames(sample: Mapping[str, Any]) -> int:
    return int(sample["raw_frames"])


def _raw_return_point(sample: Mapping[str, Any]) -> int:
    return int(sample["raw_return_point"])


def _available_prediction_frames(sample: Mapping[str, Any]) -> int:
    return int(sample["available_prediction_frames_10fps"])


def _drive_fps(sample: Mapping[str, Any]) -> float:
    return MODEL_FPS


def _source_stride(sample: Mapping[str, Any]) -> int:
    return 2


def _max_camera_delta(sample: Mapping[str, Any]) -> float:
    return float(sample["max_camera_delta_degrees"])


def _action_representable(sample: Mapping[str, Any]) -> bool:
    return bool(sample["action_representable_at_15deg"])


def _shard_samples(
    samples: list[dict[str, Any]],
    num_shards: int,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    ordered = sorted(
        samples,
        key=lambda sample: (
            -_prediction_frames(sample),
            -_context_frames(sample),
            int(sample["index"]),
        ),
    )
    for sample in ordered:
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        shards[shard].append(sample)
        loads[shard] += _prediction_frames(sample)
    for shard in shards:
        shard.sort(key=lambda sample: int(sample["index"]))
    return shards, loads


def _rollout_spec(sample: Mapping[str, Any]) -> RolloutSpec:
    source = Path(str(sample["source_json"]))
    loopnav_sample = LoopNavSample(
        data_root=source.parents[3],
        path_type=str(sample["task"]),
        nav_range=str(sample["nav_range"]),
        village=str(sample["village"]),
        stem=str(sample["stem"]),
        video_path=Path(str(sample["source_video"])),
        json_path=source,
    )
    return RolloutSpec(
        sample=loopnav_sample,
        raw_frames=_raw_frames(sample),
        raw_return_point=_raw_return_point(sample),
        context_frames=_context_frames(sample),
        available_prediction_frames=_available_prediction_frames(sample),
        prediction_frames=_prediction_frames(sample),
        max_camera_delta_degrees=_max_camera_delta(sample),
    )


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(".worldmem_ft.json")


def _valid_existing(
    *,
    output_path: Path,
    sample: Mapping[str, Any],
    adapter: Path,
) -> bool:
    try:
        inspect_video_file(
            output_path,
            expected_frames=_prediction_frames(sample),
            expected_fps=_drive_fps(sample),
            expected_size=FRAME_SIZE,
        )
        metadata = json.loads(_sidecar_path(output_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("status") == "complete"
        and metadata.get("source_json") == sample["source_json"]
        and metadata.get("adapter") == str(adapter)
        and metadata.get("prediction_frames_10fps") == _prediction_frames(sample)
    )


def _shard_manifest(
    *,
    args: argparse.Namespace,
    selection_manifest: Path,
    samples: list[dict[str, Any]],
    loads: list[int],
    status: str,
    completed: int,
    skipped: int,
    failed: int,
) -> dict[str, Any]:
    return {
        "format": "worldmem_loopnav_adapter_batch_shard_v1",
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(),
        "selection_manifest": str(selection_manifest),
        "adapter": str(args.adapter),
        "config": str(args.config),
        "output_root": str(args.output_root),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "all_shard_prediction_frame_loads": loads,
        "selected_samples": len(samples),
        "drive_fps": _drive_fps(samples[0]) if samples else None,
        "selected_prediction_frames": sum(
            _prediction_frames(sample) for sample in samples
        ),
        "completed_this_run": completed,
        "skipped_existing": skipped,
        "failed_this_run": failed,
    }


def main() -> None:
    args = _parse_args()
    args.selection_manifest = args.selection_manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.adapter = args.adapter.expanduser().resolve()
    _selection, all_samples = _load_selection(args.selection_manifest)
    shards, loads = _shard_samples(all_samples, args.num_shards)
    samples = shards[args.shard_index]
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(
        json.dumps(
            {
                "shard": f"{args.shard_index}/{args.num_shards}",
                "samples": len(samples),
                "drive_fps": _drive_fps(samples[0]),
                "prediction_frames": sum(
                    _prediction_frames(sample) for sample in samples
                ),
                "all_shard_loads": loads,
                "tasks": {
                    task: sum(sample["task"] == task for sample in samples)
                    for task in sorted({str(sample["task"]) for sample in samples})
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_path = (
        args.output_root
        / "_runs"
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    )
    completed = 0
    skipped = 0
    failed = 0
    pending = []
    for sample in samples:
        output_path = args.output_root / str(sample["output_relative_path"])
        if (
            not args.overwrite
            and _valid_existing(
                output_path=output_path,
                sample=sample,
                adapter=args.adapter,
            )
        ):
            skipped += 1
        else:
            pending.append(sample)
    write_json_atomic(
        _shard_manifest(
            args=args,
            selection_manifest=args.selection_manifest,
            samples=samples,
            loads=loads,
            status="starting" if pending else "complete",
            completed=completed,
            skipped=skipped,
            failed=failed,
        ),
        run_path,
    )
    if not pending:
        print(f"[done] all {skipped} selected outputs already valid", flush=True)
        return

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for {device}")
    dtype = amp_dtype(args.amp_dtype, device)
    print("[model] building official-initialized adapter model", flush=True)
    model, parameter_counts = build_finetune_model(cfg)
    loaded_step = _load_adapter(
        model,
        args.adapter,
        expected_revision=str(cfg.official.revision),
    )
    model = model.to(device=device).eval()
    print(
        f"[model] adapter step={loaded_step} device={device} "
        f"pending={len(pending)}",
        flush=True,
    )

    for sequence_index, sample in enumerate(pending, start=1):
        spec = _rollout_spec(sample)
        output_path = args.output_root / str(sample["output_relative_path"])
        sidecar = _sidecar_path(output_path)
        writer = None
        started = time.monotonic()
        try:
            clip = _load_clip(
                spec,
                camera_clip=float(cfg.data.camera_clip),
            )
            sample_seed = _seed_for_spec(args.seed, spec)
            _set_seed(sample_seed)
            writer = AtomicPredictionWriter(
                output_path,
                fps=_drive_fps(sample),
                size=FRAME_SIZE,
                overwrite=True,
            )
            print(
                f"[sample {sequence_index}/{len(pending)}] "
                f"{spec.sample.relative_stem} "
                f"context={spec.context_frames} pred={spec.prediction_frames}",
                flush=True,
            )
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
            writer = None
            elapsed = time.monotonic() - started
            result = {
                "format": "worldmem_loopnav_adapter_batch_prediction_v1",
                "status": "complete",
                "task": sample["task"],
                **(
                    {"cycle": sample["cycle"]}
                    if "cycle" in sample
                    else {}
                ),
                **(
                    {"segment": sample["segment"]}
                    if "segment" in sample
                    else {}
                ),
                "village_id": sample["village_id"],
                "nav_range": sample["nav_range"],
                "stem": sample["stem"],
                "source_json": sample["source_json"],
                "source_video": sample["source_video"],
                "output_video": str(output_path),
                "selection_reference": sample["selection_reference"],
                "adapter": str(args.adapter),
                "adapter_step": loaded_step,
                "seed": sample_seed,
                "context_frames_10fps": spec.context_frames,
                "prediction_frames_10fps": spec.prediction_frames,
                "source_first_target_index_20fps": 2 * spec.context_frames,
                "max_camera_delta_degrees": spec.max_camera_delta_degrees,
                "action_representable_at_15deg": _action_representable(sample),
                "protocol": {
                    "action": "continuous 25D from corrected pose delta",
                    "pose": "corrected [x,y,z,-pitch_deg,yaw_deg+180]",
                    "source_stride": _source_stride(sample),
                    "drive_fps": _drive_fps(sample),
                    "target_horizon": "manifest prediction cap",
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
            }
            write_json_atomic(result, sidecar)
            completed += 1
            print(
                f"[saved] {output_path} elapsed={elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            if writer is not None:
                writer.abort()
            failed += 1
            error = f"{type(exc).__name__}: {exc}"
            write_json_atomic(
                {
                    "format": "worldmem_loopnav_adapter_batch_prediction_v1",
                    "status": "failed",
                    "task": sample["task"],
                    "source_json": sample["source_json"],
                    "output_video": str(output_path),
                    "adapter": str(args.adapter),
                    "error": error,
                    "traceback": traceback.format_exc(),
                },
                sidecar,
            )
            print(f"[failed] {sample['source_json']}: {error}", flush=True)
            if args.fail_fast:
                raise
        finally:
            if "clip" in locals():
                del clip
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_json_atomic(
                {
                    **_shard_manifest(
                        args=args,
                        selection_manifest=args.selection_manifest,
                        samples=samples,
                        loads=loads,
                        status="running",
                        completed=completed,
                        skipped=skipped,
                        failed=failed,
                    ),
                    "adapter_step": loaded_step,
                    "parameter_counts": parameter_counts,
                },
                run_path,
            )

    status = "complete" if failed == 0 else "complete_with_failures"
    write_json_atomic(
        {
            **_shard_manifest(
                args=args,
                selection_manifest=args.selection_manifest,
                samples=samples,
                loads=loads,
                status=status,
                completed=completed,
                skipped=skipped,
                failed=failed,
            ),
            "adapter_step": loaded_step,
            "parameter_counts": parameter_counts,
        },
        run_path,
    )
    print(
        f"[done] completed={completed} skipped={skipped} failed={failed} "
        f"manifest={run_path}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
