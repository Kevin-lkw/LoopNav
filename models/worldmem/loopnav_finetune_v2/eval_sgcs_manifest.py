#!/usr/bin/env python3
"""Evaluate native-10-Hz WorldMem predictions with repository SGCS.

Each generated frame is paired with its exact raw LoopNav target as
``GT[2 * context + 2 * t]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


METRICS_ROOT = Path(os.environ.get(
    "LOOPNAV_METRICS_DIR", Path(__file__).resolve().parents[3] / "evaluation"
)).expanduser().resolve()
if str(METRICS_ROOT) not in sys.path:
    sys.path.insert(0, str(METRICS_ROOT))

import sgcs_eval  # noqa: E402


FORMAT = "worldmem_loopnav_sgcs_shard_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=sgcs_eval.DEFAULT_CATEGORIES,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.tau <= 0:
        parser.error("--tau must be positive")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be in [0, 1]")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < num-shards")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    return args


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_selection(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "worldmem_loopnav_comparable_batch_manifest_v1":
        raise ValueError(f"Unexpected selection manifest format: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Selection manifest has no samples: {path}")
    return payload, samples


def prediction_frames(sample: dict[str, Any]) -> int:
    return int(sample["prediction_frames_10fps"])


def context_frames(sample: dict[str, Any]) -> int:
    return int(sample["context_frames_10fps"])


def raw_frames(sample: dict[str, Any]) -> int:
    return int(sample["raw_frames"])


def source_stride(sample: dict[str, Any]) -> int:
    return 2


def prediction_fps(sample: dict[str, Any]) -> int:
    return 10


def alignment_description(sample: dict[str, Any]) -> str:
    return "GT[2*context+2*t] <-> prediction[t]"


def sampled_frame_count(sample: dict[str, Any], stride: int) -> int:
    return len(range(0, prediction_frames(sample), stride))


def shard_samples(
    samples: list[dict[str, Any]],
    num_shards: int,
    stride: int,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    ordered = sorted(
        samples,
        key=lambda sample: (
            -sampled_frame_count(sample, stride),
            -prediction_frames(sample),
            int(sample["index"]),
        ),
    )
    for sample in ordered:
        shard = min(
            range(num_shards),
            key=lambda index: (loads[index], len(shards[index]), index),
        )
        shards[shard].append(sample)
        loads[shard] += sampled_frame_count(sample, stride)
    for shard in shards:
        shard.sort(key=lambda sample: int(sample["index"]))
    return shards, loads


def generated_path(prediction_root: Path, sample: dict[str, Any]) -> Path:
    return prediction_root / str(sample["output_relative_path"])


def validate_pair_geometry(
    prediction_root: Path,
    sample: dict[str, Any],
) -> dict[str, int]:
    source = Path(str(sample["source_video"]))
    generated = generated_path(prediction_root, sample)
    if not source.is_file():
        raise FileNotFoundError(f"Missing source video: {source}")
    if not generated.is_file():
        raise FileNotFoundError(f"Missing generated video: {generated}")

    cap_source = sgcs_eval.open_video(source)
    cap_generated = sgcs_eval.open_video(generated)
    try:
        source_count = sgcs_eval.get_frame_count(cap_source)
        generated_count = sgcs_eval.get_frame_count(cap_generated)
    finally:
        cap_source.release()
        cap_generated.release()

    expected_source = raw_frames(sample)
    expected_generated = prediction_frames(sample)
    if source_count != expected_source:
        raise ValueError(
            f"Source frame count mismatch {source_count} != {expected_source}: {source}"
        )
    if generated_count != expected_generated:
        raise ValueError(
            "Generated frame count mismatch "
            f"{generated_count} != {expected_generated}: {generated}"
        )
    cadence_stride = source_stride(sample)
    first_target = cadence_stride * context_frames(sample)
    last_target = first_target + cadence_stride * (generated_count - 1)
    if first_target < 0 or last_target >= source_count:
        raise ValueError(
            f"Aligned GT indices [{first_target}, {last_target}] outside "
            f"{source_count} frames: {source}"
        )
    return {
        "source_frame_count": source_count,
        "generated_frame_count": generated_count,
        "source_first_target_index_20fps": first_target,
        "source_last_target_index_20fps": last_target,
    }


def evaluate_pair(
    processor: Any,
    args: argparse.Namespace,
    sample: dict[str, Any],
    categories: list[str],
) -> dict[str, Any]:
    geometry = validate_pair_geometry(args.prediction_root, sample)
    source = Path(str(sample["source_video"]))
    generated = generated_path(args.prediction_root, sample)
    sampled_indices = list(
        range(0, geometry["generated_frame_count"], args.stride)
    )
    cap_source = sgcs_eval.open_video(source)
    cap_generated = sgcs_eval.open_video(generated)
    frame_results = []
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda")
        else nullcontext()
    )
    try:
        with torch.inference_mode(), autocast_context:
            for generated_index in sampled_indices:
                source_index = (
                    geometry["source_first_target_index_20fps"]
                    + source_stride(sample) * generated_index
                )
                raw_source = sgcs_eval.read_frame_at(cap_source, source_index)
                raw_generated = sgcs_eval.read_frame_at(
                    cap_generated, generated_index
                )
                source_frame, generated_frame, shape_info = (
                    sgcs_eval.align_frame_shapes(raw_source, raw_generated)
                )
                frame_result = sgcs_eval.compute_frame_sgcs(
                    processor=processor,
                    frame_a=source_frame,
                    frame_b=generated_frame,
                    categories=categories,
                    tau=args.tau,
                    shape_info=shape_info,
                )
                frame_result["generated_frame_index"] = int(generated_index)
                frame_result["frame_index_10fps"] = int(generated_index)
                frame_result["video_a_frame_index"] = int(source_index)
                frame_result["video_b_frame_index"] = int(generated_index)
                frame_results.append(frame_result)
    finally:
        cap_source.release()
        cap_generated.release()

    scores = [float(frame["sgcs"]) for frame in frame_results]
    result = {
        "sample_index": int(sample["index"]),
        "task": str(sample["task"]),
        "village_id": int(sample["village_id"]),
        "village": str(sample["village"]),
        "biome": str(sample["village"]).rsplit("_village_", 1)[0],
        "nav_range": str(sample["nav_range"]),
        "stem": str(sample["stem"]),
        "video_a": str(source),
        "video_b": str(generated),
        "video_a_role": "gt_target_at_native_10fps",
        "video_b_role": "worldmem_generated_target_10fps",
        **geometry,
        "alignment": alignment_description(sample),
        "source_stride": source_stride(sample),
        "prediction_fps": prediction_fps(sample),
        "stride": int(args.stride),
        "tau": float(args.tau),
        "confidence_threshold": float(args.confidence_threshold),
        "categories": categories,
        "num_sampled_frames": len(frame_results),
        "mean_sgcs": float(np.mean(scores)) if scores else 0.0,
        "frames": frame_results,
    }
    for optional_key in ("cycle", "segment", "source_stem", "derived_stem"):
        if optional_key in sample:
            result[optional_key] = sample[optional_key]
    return result


def shard_path(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    return (
        output_dir
        / "shards"
        / f"worldmem_shard_{shard_index:02d}_of_{num_shards:02d}.json"
    )


def config_for_args(
    args: argparse.Namespace,
    categories: list[str],
    samples: list[dict[str, Any]],
    selected_samples: int,
    selected_sampled_frames: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("SGCS shard has no selected samples")
    return {
        "sgcs_pipeline_version": sgcs_eval.SGCS_PIPELINE_VERSION,
        "selection_manifest": str(args.selection_manifest),
        "prediction_root": str(args.prediction_root),
        "stride": int(args.stride),
        "tau": float(args.tau),
        "confidence_threshold": float(args.confidence_threshold),
        "categories": categories,
        "category_prompts": sgcs_eval.resolved_category_prompts(categories),
        "category_merges": sgcs_eval.resolved_category_merges(categories),
        "source_stride": 2,
        "prediction_fps": 10,
        "alignment": "GT[2*context+2*t] <-> prediction[t]",
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "max_samples": int(args.max_samples),
        "selected_samples": selected_samples,
        "selected_sampled_frames": selected_sampled_frames,
    }


def shard_payload(
    *,
    config: dict[str, Any],
    status: str,
    videos: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = [float(video["mean_sgcs"]) for video in videos]
    return {
        "format": FORMAT,
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(),
        "config": config,
        "num_videos": len(videos),
        "num_failures": len(failures),
        "num_sampled_frames": sum(
            int(video["num_sampled_frames"]) for video in videos
        ),
        "mean_sgcs": float(np.mean(scores)) if scores else 0.0,
        "std_sgcs": float(np.std(scores)) if scores else 0.0,
        "videos": videos,
        "failures": failures,
    }


def load_existing(
    path: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if overwrite or not path.is_file():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"Unexpected existing shard format: {path}")
    if payload.get("config") != config:
        raise ValueError(
            f"Existing shard config differs: {path}. Use --overwrite to replace it."
        )
    return list(payload.get("videos", [])), list(payload.get("failures", []))


def evaluate_shard(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    categories: list[str],
    loads: list[int],
) -> None:
    selected_sampled_frames = sum(
        sampled_frame_count(sample, args.stride) for sample in samples
    )
    config = config_for_args(
        args,
        categories,
        samples,
        selected_samples=len(samples),
        selected_sampled_frames=selected_sampled_frames,
    )
    output_path = shard_path(args.output_dir, args.shard_index, args.num_shards)
    videos, failures = load_existing(output_path, config, args.overwrite)
    complete_indices = {int(video["sample_index"]) for video in videos}
    failures_by_index = {
        int(failure["sample_index"]): failure for failure in failures
    }
    pending = [
        sample
        for sample in samples
        if int(sample["index"]) not in complete_indices
    ]
    print(
        json.dumps(
            {
                "shard": f"{args.shard_index}/{args.num_shards}",
                "samples": len(samples),
                "sampled_frames": selected_sampled_frames,
                "all_shard_sampled_frame_loads": loads,
                "already_complete": len(complete_indices),
                "pending": len(pending),
            },
            indent=2,
        ),
        flush=True,
    )
    if not pending:
        status = "complete" if not failures_by_index else "complete_with_failures"
        write_json_atomic(
            shard_payload(
                config=config,
                status=status,
                videos=videos,
                failures=list(failures_by_index.values()),
            ),
            output_path,
        )
        return

    processor = sgcs_eval.build_processor(args)
    new_since_save = 0
    for sequence_index, sample in enumerate(pending, start=1):
        sample_index = int(sample["index"])
        try:
            print(
                f"[sample {sequence_index}/{len(pending)}] "
                f"index={sample_index} {sample['task']}/{sample['nav_range']}/"
                f"{sample['village']}/{sample['stem']}",
                flush=True,
            )
            result = evaluate_pair(processor, args, sample, categories)
            videos.append(result)
            videos.sort(key=lambda video: int(video["sample_index"]))
            failures_by_index.pop(sample_index, None)
            print(
                f"[saved] index={sample_index} "
                f"sgcs={result['mean_sgcs']:.6f} "
                f"frames={result['num_sampled_frames']}",
                flush=True,
            )
        except Exception as exc:
            failures_by_index[sample_index] = {
                "sample_index": sample_index,
                "task": sample["task"],
                "source_video": sample["source_video"],
                "generated_video": str(generated_path(args.prediction_root, sample)),
                "error": f"{type(exc).__name__}: {exc}",
            }
            for optional_key in ("cycle", "segment"):
                if optional_key in sample:
                    failures_by_index[sample_index][optional_key] = sample[
                        optional_key
                    ]
            print(
                f"[failed] index={sample_index} "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        new_since_save += 1
        if new_since_save >= args.save_every:
            write_json_atomic(
                shard_payload(
                    config=config,
                    status="running",
                    videos=videos,
                    failures=list(failures_by_index.values()),
                ),
                output_path,
            )
            new_since_save = 0

    failures = list(failures_by_index.values())
    status = (
        "complete"
        if len(videos) == len(samples) and not failures
        else "complete_with_failures"
    )
    write_json_atomic(
        shard_payload(
            config=config,
            status=status,
            videos=videos,
            failures=failures,
        ),
        output_path,
    )
    print(
        f"[done] status={status} videos={len(videos)} failures={len(failures)} "
        f"output={output_path}",
        flush=True,
    )
    if status != "complete":
        raise SystemExit(1)


def group_summary(
    scope: str,
    key: str,
    videos: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    selected = list(videos)
    scores = [float(video["mean_sgcs"]) for video in selected]
    return {
        "scope": scope,
        "key": key,
        "num_videos": len(selected),
        "num_sampled_frames": sum(
            int(video["num_sampled_frames"]) for video in selected
        ),
        "mean_sgcs": float(np.mean(scores)) if scores else 0.0,
        "std_sgcs": float(np.std(scores)) if scores else 0.0,
    }


def merge_shards(
    args: argparse.Namespace,
    selection: dict[str, Any],
    samples: list[dict[str, Any]],
    categories: list[str],
) -> None:
    if not samples:
        raise ValueError("Selection has no samples")
    videos: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        path = shard_path(args.output_dir, shard_index, args.num_shards)
        if not path.is_file():
            raise FileNotFoundError(f"Missing SGCS shard: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise ValueError(f"Unexpected SGCS shard format: {path}")
        if payload.get("status") != "complete":
            raise RuntimeError(
                f"SGCS shard is not complete: {path} status={payload.get('status')}"
            )
        if payload.get("failures"):
            raise RuntimeError(f"SGCS shard contains failures: {path}")
        videos.extend(payload.get("videos", []))

    videos.sort(key=lambda video: int(video["sample_index"]))
    indices = [int(video["sample_index"]) for video in videos]
    expected_indices = sorted(int(sample["index"]) for sample in samples)
    if indices != expected_indices:
        raise RuntimeError(
            f"Merged sample indices do not match selection: "
            f"{len(indices)} != {len(expected_indices)}"
        )

    groups = [group_summary("overall", "ALL", videos)]
    tasks = sorted({str(video["task"]) for video in videos})
    for task in tasks:
        task_videos = [video for video in videos if video["task"] == task]
        groups.append(group_summary("task", task, task_videos))
        for village_id in sorted(
            {int(video["village_id"]) for video in task_videos}
        ):
            selected = [
                video
                for video in task_videos
                if int(video["village_id"]) == village_id
            ]
            groups.append(
                group_summary("task_village", f"{task}/v{village_id}", selected)
            )
        for nav_range in sorted({str(video["nav_range"]) for video in task_videos}):
            selected = [
                video
                for video in task_videos
                if str(video["nav_range"]) == nav_range
            ]
            groups.append(
                group_summary("task_range", f"{task}/{nav_range}", selected)
            )
        cycles = sorted(
            {
                str(video["cycle"])
                for video in task_videos
                if "cycle" in video
            }
        )
        for cycle in cycles:
            cycle_videos = [
                video
                for video in task_videos
                if str(video.get("cycle")) == cycle
            ]
            segment = str(cycle_videos[0].get("segment", cycle))
            groups.append(
                group_summary(
                    "task_cycle",
                    f"{task}/{cycle}/{segment}",
                    cycle_videos,
                )
            )
            for nav_range in sorted(
                {str(video["nav_range"]) for video in cycle_videos},
                key=lambda value: int(value) if value.isdigit() else value,
            ):
                selected = [
                    video
                    for video in cycle_videos
                    if str(video["nav_range"]) == nav_range
                ]
                groups.append(
                    group_summary(
                        "task_cycle_range",
                        f"{task}/{cycle}/{segment}/{nav_range}",
                        selected,
                    )
                )

    fine_groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for video in videos:
        fine_groups[
            (
                str(video["task"]),
                int(video["village_id"]),
                str(video["nav_range"]),
                str(video["biome"]),
            )
        ].append(video)
    for (task, village_id, nav_range, biome), selected in sorted(
        fine_groups.items()
    ):
        groups.append(
            group_summary(
                "task_village_range_biome",
                f"{task}/v{village_id}/{nav_range}/{biome}",
                selected,
            )
        )

    compact_videos = []
    for video in videos:
        compact = {
            key: video[key]
            for key in (
                "sample_index",
                "task",
                "village_id",
                "village",
                "biome",
                "nav_range",
                "stem",
                "video_a",
                "video_b",
                "num_sampled_frames",
                "mean_sgcs",
            )
        }
        for optional_key in ("cycle", "segment", "source_stem", "derived_stem"):
            if optional_key in video:
                compact[optional_key] = video[optional_key]
        compact_videos.append(compact)
    summary = {
        "format": "worldmem_loopnav_sgcs_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_manifest": str(args.selection_manifest),
        "prediction_root": str(args.prediction_root),
        "selection_samples": int(selection["total_samples"]),
        "num_videos": len(videos),
        "num_failures": 0,
        "protocol": {
            "sgcs_pipeline_version": sgcs_eval.SGCS_PIPELINE_VERSION,
            "metric": "repository SAM3 SGCS",
            "metric_stride_generated_frames": int(args.stride),
            "tau": float(args.tau),
            "confidence_threshold": float(args.confidence_threshold),
            "categories": categories,
            "category_prompts": sgcs_eval.resolved_category_prompts(categories),
            "category_merges": sgcs_eval.resolved_category_merges(categories),
            "alignment": "GT[2*context+2*t] <-> prediction[t]",
            "source_stride": 2,
            "source_fps": 20,
            "prediction_fps": 10,
        },
        "groups": groups,
        "videos": compact_videos,
    }
    write_json_atomic(summary, args.output_dir / "summary.json")

    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    with (args.output_dir / "per_video.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        compact_fieldnames = list(compact_videos[0])
        for video in compact_videos[1:]:
            for key in video:
                if key not in compact_fieldnames:
                    compact_fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=compact_fieldnames)
        writer.writeheader()
        writer.writerows(compact_videos)
    print(
        f"[merged] videos={len(videos)} "
        f"mean_sgcs={groups[0]['mean_sgcs']:.6f} "
        f"output={args.output_dir / 'summary.json'}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.selection_manifest = args.selection_manifest.expanduser().resolve()
    args.prediction_root = args.prediction_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    categories = list(dict.fromkeys(str(value) for value in args.categories))
    selection, all_samples = load_selection(args.selection_manifest)

    if args.merge_only:
        merge_shards(args, selection, all_samples, categories)
        return

    shards, loads = shard_samples(all_samples, args.num_shards, args.stride)
    samples = shards[args.shard_index]
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(
        f"[audit] all_samples={len(all_samples)} "
        f"all_sampled_frames={sum(loads)} shard_loads={loads}",
        flush=True,
    )
    for sample in samples:
        validate_pair_geometry(args.prediction_root, sample)
    print(
        f"[audit] shard={args.shard_index} pairs={len(samples)} "
        f"sampled_frames={sum(sampled_frame_count(s, args.stride) for s in samples)} "
        "geometry=valid",
        flush=True,
    )
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_shard(args, samples, categories, loads)


if __name__ == "__main__":
    main()
