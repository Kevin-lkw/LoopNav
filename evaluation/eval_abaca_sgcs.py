#!/usr/bin/env python3
"""Compute SAM3-based SGCS for the generated stages of ABACA rollouts.

The ABACA_FIXED manifest contains two independently generated return stages:

    cycle1: B -> A
    cycle2: C -> A

For both stages, the derived input video contains the context followed by the
ground-truth return.  The generated baseline video begins at ``return_point``.
The only comparison performed here is therefore:

    derived_gt[return_point + t]  <->  generated[t]

In particular, the A -> C bridge before cycle2's return point is context only
and is never included in SGCS.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import sgcs_eval


DEFAULT_MANIFEST = Path("benchmark/prepared/ABACA_FIXED_manifest.json")
DEFAULT_BASELINE_ROOTS = {
    name: Path("predictions") / name
    for name in ("oasis", "nwm", "mineworld", "diamond")
}
DEFAULT_CYCLES = ["cycle1", "cycle2"]
DEFAULT_LENGTHS = ["5", "15", "30"]
DEFAULT_VILLAGE_IDS = ["19", "20"]
CYCLE_LABELS = {
    "cycle1": "B-A",
    "cycle2": "C-A",
}


def parse_multi_value(tokens: list[str]) -> list[str]:
    values: list[str] = []
    for token in tokens:
        values.extend(part.strip() for part in str(token).split(",") if part.strip())
    return values


def parse_baseline_root_overrides(items: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                "--baseline-root must use NAME=PATH, e.g. --baseline-root nwm=/tmp"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid baseline root override: {item}")
        overrides[name] = Path(path).expanduser()
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SGCS for the generated B-A and C-A stages in ABACA_FIXED."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(DEFAULT_BASELINE_ROOTS),
        help="Baseline names to evaluate.",
    )
    parser.add_argument(
        "--baseline-root",
        action="append",
        default=None,
        help="Override or add a baseline root as NAME=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--cycles",
        nargs="+",
        default=DEFAULT_CYCLES,
        help="ABACA generated cycles to evaluate: cycle1 (B-A), cycle2 (C-A).",
    )
    parser.add_argument("--lengths", nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--village-ids", nargs="+", default=DEFAULT_VILLAGE_IDS)
    parser.add_argument(
        "--max-videos-per-setting",
        type=int,
        default=5,
        help=(
            "Keep the lexicographically first N source videos for each "
            "(cycle, length, biome-village) setting."
        ),
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=sgcs_eval.DEFAULT_CATEGORIES,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "eval_SGCS_ABACA_FIXED_TEST19_20_5",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frame-progress", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing shards without loading SAM3.",
    )
    return parser.parse_args()


def normalize_manifest_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def sample_id(sample: dict[str, Any]) -> str:
    return (
        f"{sample['cycle']}:{sample['length']}:{sample['village']}:"
        f"{sample['derived_stem']}"
    )


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError(f"Manifest must be a list: {args.manifest}")

    cycle_set = set(args.cycles)
    length_set = set(args.lengths)
    village_id_set = set(args.village_ids)
    selected: list[dict[str, Any]] = []
    for raw_item in manifest:
        item = dict(raw_item)
        cycle = str(item.get("cycle", ""))
        length = str(item.get("length", ""))
        village = str(item.get("village", ""))
        village_id = village.rsplit("_", 1)[-1]
        if (
            cycle not in cycle_set
            or length not in length_set
            or village_id not in village_id_set
        ):
            continue
        if cycle not in CYCLE_LABELS:
            raise ValueError(f"Unknown ABACA cycle: {cycle}")
        item["cycle"] = cycle
        item["length"] = length
        item["village"] = village
        item["derived_video"] = str(
            normalize_manifest_path(str(item["derived_video"]), args.manifest)
        )
        item["derived_metadata"] = str(
            normalize_manifest_path(str(item["derived_metadata"]), args.manifest)
        )
        item["sample_id"] = sample_id(item)
        selected.append(item)

    selected.sort(
        key=lambda item: (
            item["cycle"],
            int(item["length"]),
            item["village"],
            item["source_stem"],
            item["derived_stem"],
        )
    )
    if args.max_videos_per_setting is not None:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in selected:
            grouped[(item["cycle"], item["length"], item["village"])].append(item)
        selected = []
        for key in sorted(grouped, key=lambda value: (value[0], int(value[1]), value[2])):
            selected.extend(grouped[key][: args.max_videos_per_setting])
    return selected


def generated_video_for_sample(
    baseline_root: Path, sample: dict[str, Any]
) -> Path:
    return (
        baseline_root
        / "ABACA_FIXED"
        / str(sample["length"])
        / str(sample["village"])
        / f"{sample['derived_stem']}.mp4"
    )


def verify_manifest_return_point(sample: dict[str, Any]) -> tuple[int, int]:
    metadata_path = Path(sample["derived_metadata"])
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Derived metadata must be a list: {metadata_path}")
    points = [
        index
        for index, record in enumerate(records)
        if isinstance(record, dict) and record.get("return_point") is True
    ]
    if len(points) != 1:
        raise ValueError(
            f"Expected one return_point in {metadata_path}, found {points}"
        )
    metadata_return_point = int(points[0])
    manifest_return_point = int(sample["return_point"])
    if metadata_return_point != manifest_return_point:
        raise ValueError(
            f"Manifest/metadata return_point mismatch for {metadata_path}: "
            f"{manifest_return_point} != {metadata_return_point}"
        )
    manifest_metadata_frames = int(sample["metadata_frames"])
    if len(records) != manifest_metadata_frames:
        raise ValueError(
            f"Manifest/metadata length mismatch for {metadata_path}: "
            f"{manifest_metadata_frames} != {len(records)}"
        )
    return metadata_return_point, len(records)


def sampled_indices_for_counts(
    gt_frame_count: int,
    generated_frame_count: int,
    metadata_length: int,
    return_point: int,
    stride: int,
) -> tuple[int, list[int]]:
    gt_available = metadata_length - return_point
    if gt_frame_count > 0:
        gt_available = min(gt_available, gt_frame_count - return_point)
    if gt_available <= 0:
        raise ValueError(
            f"No GT generated-stage frames after return_point={return_point}; "
            f"gt_frame_count={gt_frame_count}, metadata_length={metadata_length}"
        )
    compare_count = gt_available
    if generated_frame_count > 0:
        compare_count = min(compare_count, generated_frame_count)
    if compare_count <= 0:
        raise ValueError(
            f"No comparable generated-stage frames: gt_available={gt_available}, "
            f"generated_frame_count={generated_frame_count}"
        )
    return int(compare_count), list(range(0, int(compare_count), stride))


def evaluate_abaca_pair(
    processor: Any,
    sample: dict[str, Any],
    generated_video: Path,
    args: argparse.Namespace,
    categories: list[str],
) -> dict[str, Any]:
    gt_video = Path(sample["derived_video"])
    return_point, metadata_length = verify_manifest_return_point(sample)
    cap_gt = sgcs_eval.open_video(gt_video)
    cap_generated = sgcs_eval.open_video(generated_video)
    gt_frame_count = sgcs_eval.get_frame_count(cap_gt)
    generated_frame_count = sgcs_eval.get_frame_count(cap_generated)
    compare_count, sampled_indices = sampled_indices_for_counts(
        gt_frame_count=gt_frame_count,
        generated_frame_count=generated_frame_count,
        metadata_length=metadata_length,
        return_point=return_point,
        stride=args.stride,
    )

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device == "cuda"
        else nullcontext()
    )
    frame_results: list[dict[str, Any]] = []
    try:
        with torch.inference_mode(), autocast_context:
            for frame_idx in tqdm(
                sampled_indices,
                desc=(
                    f"{sample['cycle']}/{sample['length']}/"
                    f"{sample['village']}/{sample['derived_stem']}"
                ),
                leave=False,
                disable=not args.frame_progress,
            ):
                gt_frame_index = return_point + frame_idx
                generated_frame_index = frame_idx
                raw_gt_frame = sgcs_eval.read_frame_at(cap_gt, gt_frame_index)
                raw_generated_frame = sgcs_eval.read_frame_at(
                    cap_generated, generated_frame_index
                )
                gt_frame, generated_frame, shape_info = sgcs_eval.align_frame_shapes(
                    raw_gt_frame, raw_generated_frame
                )
                frame_result = sgcs_eval.compute_frame_sgcs(
                    processor=processor,
                    frame_a=gt_frame,
                    frame_b=generated_frame,
                    categories=categories,
                    tau=args.tau,
                    shape_info=shape_info,
                )
                frame_result["frame_index"] = int(frame_idx)
                frame_result["video_a_frame_index"] = int(gt_frame_index)
                frame_result["video_b_frame_index"] = int(generated_frame_index)
                frame_results.append(frame_result)
    finally:
        cap_gt.release()
        cap_generated.release()

    scores = [item["sgcs"] for item in frame_results]
    return {
        "sgcs_pipeline_version": sgcs_eval.SGCS_PIPELINE_VERSION,
        "sample_id": sample["sample_id"],
        "cycle": sample["cycle"],
        "segment": CYCLE_LABELS[sample["cycle"]],
        "length": sample["length"],
        "village": sample["village"],
        "source_stem": sample["source_stem"],
        "derived_stem": sample["derived_stem"],
        "video_a": str(gt_video),
        "video_b": str(generated_video),
        "video_a_role": "derived_gt_generated_stage",
        "video_b_role": "generated_stage",
        "video_a_frame_count": int(gt_frame_count),
        "video_b_frame_count": int(generated_frame_count),
        "metadata_length": int(metadata_length),
        "return_point": int(return_point),
        "compare_frame_count": int(compare_count),
        "stride": int(args.stride),
        "tau": float(args.tau),
        "confidence_threshold": float(args.confidence_threshold),
        "categories": categories,
        "category_prompts": sgcs_eval.resolved_category_prompts(categories),
        "category_merges": sgcs_eval.resolved_category_merges(categories),
        "alignment": "manifest_return_point",
        "video_a_frame_offset": int(return_point),
        "video_b_frame_offset": 0,
        "context_before_return_point_scored": False,
        "num_sampled_frames": len(frame_results),
        "mean_sgcs": float(np.mean(scores)) if scores else 0.0,
        "frames": frame_results,
    }


def config_for_args(
    args: argparse.Namespace,
    baseline_name: str,
    baseline_root: Path,
    categories: list[str],
) -> dict[str, Any]:
    return {
        "sgcs_pipeline_version": sgcs_eval.SGCS_PIPELINE_VERSION,
        "baseline": baseline_name,
        "baseline_root": str(baseline_root),
        "manifest": str(args.manifest),
        "cycles": list(args.cycles),
        "cycle_labels": {cycle: CYCLE_LABELS[cycle] for cycle in args.cycles},
        "lengths": list(args.lengths),
        "village_ids": list(args.village_ids),
        "max_videos_per_setting": args.max_videos_per_setting,
        "stride": args.stride,
        "tau": args.tau,
        "confidence_threshold": args.confidence_threshold,
        "categories": categories,
        "category_prompts": sgcs_eval.resolved_category_prompts(categories),
        "category_merges": sgcs_eval.resolved_category_merges(categories),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "scored_region": "generated_stage_only",
        "alignment": "derived_gt[return_point+t] vs generated[t]",
    }


def shard_output_path(
    output_dir: Path,
    baseline_name: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    return (
        output_dir
        / "shards"
        / f"{baseline_name}_ABACA_shard{shard_index:02d}_of_{num_shards:02d}.json"
    )


def load_existing_shard(
    output_path: Path,
    expected_config: dict[str, Any],
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if overwrite or not output_path.exists():
        return [], []
    data = json.loads(output_path.read_text(encoding="utf-8"))
    if data.get("config") != expected_config:
        raise ValueError(
            f"Existing shard config differs from this run: {output_path}. "
            "Use --overwrite to replace it."
        )
    return list(data.get("videos", [])), list(data.get("failures", []))


def write_shard(
    output_path: Path,
    config: dict[str, Any],
    videos: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    scores = [item["mean_sgcs"] for item in videos]
    output = {
        "config": config,
        "num_videos": len(videos),
        "num_sampled_frames": sum(
            int(item.get("num_sampled_frames", 0)) for item in videos
        ),
        "mean_sgcs": float(np.mean(scores)) if scores else 0.0,
        "std_sgcs": float(np.std(scores)) if scores else 0.0,
        "videos": videos,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(output_path)


def evaluate_baseline_shard(
    processor: Any,
    args: argparse.Namespace,
    categories: list[str],
    baseline_name: str,
    baseline_root: Path,
    samples: list[dict[str, Any]],
) -> None:
    config = config_for_args(args, baseline_name, baseline_root, categories)
    output_path = shard_output_path(
        args.output_dir, baseline_name, args.shard_index, args.num_shards
    )
    videos, failures = load_existing_shard(output_path, config, args.overwrite)
    processed = {str(item["sample_id"]) for item in videos}
    processed.update(str(item.get("sample_id", "")) for item in failures)
    newly_evaluated = 0

    print(
        f"[{baseline_name}] shard {args.shard_index}/{args.num_shards}: "
        f"{len(samples)} assigned, {len(processed)} already recorded"
    )
    for sample_index, sample in enumerate(samples, 1):
        if sample["sample_id"] in processed:
            continue
        gt_video = Path(sample["derived_video"])
        generated_video = generated_video_for_sample(baseline_root, sample)
        try:
            if not gt_video.is_file():
                raise FileNotFoundError(f"Missing derived GT video: {gt_video}")
            if not Path(sample["derived_metadata"]).is_file():
                raise FileNotFoundError(
                    f"Missing derived metadata: {sample['derived_metadata']}"
                )
            if not generated_video.is_file():
                raise FileNotFoundError(f"Missing generated video: {generated_video}")
            print(
                f"[{baseline_name}] {sample_index}/{len(samples)} "
                f"{sample['cycle']}({CYCLE_LABELS[sample['cycle']]}) "
                f"length={sample['length']} {sample['village']} "
                f"{sample['derived_stem']} return_point={sample['return_point']}"
            )
            result = evaluate_abaca_pair(
                processor=processor,
                sample=sample,
                generated_video=generated_video,
                args=args,
                categories=categories,
            )
            result["baseline"] = baseline_name
            result["gt_metadata"] = str(sample["derived_metadata"])
            videos.append(result)
        except Exception as exc:
            print(
                f"[{baseline_name}] failure {sample['sample_id']}: {exc}",
                file=sys.stderr,
            )
            failures.append(
                {
                    "baseline": baseline_name,
                    "sample_id": sample["sample_id"],
                    "cycle": sample["cycle"],
                    "segment": CYCLE_LABELS[sample["cycle"]],
                    "length": sample["length"],
                    "village": sample["village"],
                    "gt_video": str(gt_video),
                    "gt_metadata": str(sample["derived_metadata"]),
                    "generated_video": str(generated_video),
                    "error": str(exc),
                }
            )
        processed.add(sample["sample_id"])
        newly_evaluated += 1
        if newly_evaluated % args.save_every == 0:
            write_shard(output_path, config, videos, failures)
            print(f"[{baseline_name}] checkpointed {output_path}")

    write_shard(output_path, config, videos, failures)
    print(
        f"[{baseline_name}] wrote {output_path} "
        f"videos={len(videos)} failures={len(failures)}"
    )


def aggregate_videos(videos: list[dict[str, Any]]) -> tuple[float, float, int]:
    scores = [float(item["mean_sgcs"]) for item in videos]
    sampled_frames = sum(int(item.get("num_sampled_frames", 0)) for item in videos)
    if not scores:
        return 0.0, 0.0, sampled_frames
    return float(np.mean(scores)), float(np.std(scores)), sampled_frames


def deduplicate_by_sample(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("sample_id", ""))
        if not key:
            key = f"{row.get('video_a', '')}:{row.get('video_b', '')}"
        deduplicated[key] = row
    return list(deduplicated.values())


def write_group_output(
    args: argparse.Namespace,
    baseline_name: str,
    baseline_root: Path,
    cycle: str,
    length: str,
    videos: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    categories: list[str],
) -> tuple[dict[str, Any], Path]:
    mean_sgcs, std_sgcs, sampled_frames = aggregate_videos(videos)
    output = {
        "sgcs_pipeline_version": sgcs_eval.SGCS_PIPELINE_VERSION,
        "baseline": baseline_name,
        "baseline_root": str(baseline_root),
        "manifest": str(args.manifest),
        "path_type": "ABACA_FIXED",
        "cycle": cycle,
        "segment": CYCLE_LABELS[cycle],
        "length": length,
        "scored_region": "generated_stage_only",
        "context_before_return_point_scored": False,
        "stride": args.stride,
        "tau": args.tau,
        "confidence_threshold": args.confidence_threshold,
        "categories": categories,
        "category_prompts": sgcs_eval.resolved_category_prompts(categories),
        "category_merges": sgcs_eval.resolved_category_merges(categories),
        "village_ids": args.village_ids,
        "max_videos_per_setting": args.max_videos_per_setting,
        "num_shards": args.num_shards,
        "num_videos": len(videos),
        "num_sampled_frames": sampled_frames,
        "mean_sgcs": mean_sgcs,
        "std_sgcs": std_sgcs,
        "videos": sorted(
            videos,
            key=lambda item: (
                item.get("village", ""),
                item.get("source_stem", ""),
            ),
        ),
        "failures": failures,
    }
    output_path = (
        args.output_dir
        / f"{baseline_name}_ABACA_{cycle}_{CYCLE_LABELS[cycle]}_{length}.json"
    )
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output, output_path


def merge_shards(
    args: argparse.Namespace,
    baseline_roots: dict[str, Path],
    categories: list[str],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for baseline_name in args.baselines:
        all_videos: list[dict[str, Any]] = []
        all_failures: list[dict[str, Any]] = []
        missing_shards: list[str] = []
        for shard_index in range(args.num_shards):
            path = shard_output_path(
                args.output_dir, baseline_name, shard_index, args.num_shards
            )
            if not path.is_file():
                missing_shards.append(str(path))
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            all_videos.extend(data.get("videos", []))
            all_failures.extend(data.get("failures", []))
        all_videos = deduplicate_by_sample(all_videos)
        all_failures = deduplicate_by_sample(all_failures)
        if missing_shards:
            print(
                f"[{baseline_name}] warning: missing {len(missing_shards)} shard(s)",
                file=sys.stderr,
            )

        for cycle in args.cycles:
            cycle_videos = [
                item for item in all_videos if item.get("cycle") == cycle
            ]
            cycle_failures = [
                item for item in all_failures if item.get("cycle") == cycle
            ]
            for length in args.lengths:
                videos = [
                    item
                    for item in cycle_videos
                    if str(item.get("length")) == str(length)
                ]
                failures = [
                    item
                    for item in cycle_failures
                    if str(item.get("length")) == str(length)
                ]
                output, output_path = write_group_output(
                    args=args,
                    baseline_name=baseline_name,
                    baseline_root=baseline_roots[baseline_name],
                    cycle=cycle,
                    length=length,
                    videos=videos,
                    failures=failures,
                    categories=categories,
                )
                summary_rows.append(
                    {
                        "baseline": baseline_name,
                        "cycle": cycle,
                        "segment": CYCLE_LABELS[cycle],
                        "length": length,
                        "num_videos": output["num_videos"],
                        "num_sampled_frames": output["num_sampled_frames"],
                        "num_failures": len(failures),
                        "mean_sgcs": output["mean_sgcs"],
                        "std_sgcs": output["std_sgcs"],
                        "output": str(output_path),
                    }
                )
                print(
                    f"wrote {output_path} mean={output['mean_sgcs']:.6f} "
                    f"videos={len(videos)} sampled_frames="
                    f"{output['num_sampled_frames']} failures={len(failures)}"
                )

            cycle_mean, cycle_std, cycle_sampled = aggregate_videos(cycle_videos)
            summary_rows.append(
                {
                    "baseline": baseline_name,
                    "cycle": cycle,
                    "segment": CYCLE_LABELS[cycle],
                    "length": "ALL",
                    "num_videos": len(cycle_videos),
                    "num_sampled_frames": cycle_sampled,
                    "num_failures": len(cycle_failures),
                    "mean_sgcs": cycle_mean,
                    "std_sgcs": cycle_std,
                    "output": "",
                }
            )

        overall_mean, overall_std, overall_sampled = aggregate_videos(all_videos)
        summary_rows.append(
            {
                "baseline": baseline_name,
                "cycle": "ALL",
                "segment": "B-A+C-A",
                "length": "ALL",
                "num_videos": len(all_videos),
                "num_sampled_frames": overall_sampled,
                "num_failures": len(all_failures),
                "mean_sgcs": overall_mean,
                "std_sgcs": overall_std,
                "output": "",
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = args.output_dir / "summary.csv"
    columns = [
        "baseline",
        "cycle",
        "segment",
        "length",
        "num_videos",
        "num_sampled_frames",
        "num_failures",
        "mean_sgcs",
        "std_sgcs",
        "output",
    ]
    csv_lines = [",".join(columns)]
    for row in summary_rows:
        csv_lines.append(
            ",".join(
                [
                    str(row["baseline"]),
                    str(row["cycle"]),
                    str(row["segment"]),
                    str(row["length"]),
                    str(row["num_videos"]),
                    str(row["num_sampled_frames"]),
                    str(row["num_failures"]),
                    f"{row['mean_sgcs']:.10f}",
                    f"{row['std_sgcs']:.10f}",
                    str(row["output"]),
                ]
            )
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")


def validate_args(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if (
        args.max_videos_per_setting is not None
        and args.max_videos_per_setting <= 0
    ):
        raise ValueError("--max-videos-per-setting must be positive")


def print_sample_summary(samples: list[dict[str, Any]], stride: int) -> None:
    setting_counts: dict[tuple[str, str], int] = defaultdict(int)
    for sample in samples:
        setting_counts[(sample["cycle"], sample["length"])] += 1
    expected_sampled = sum(
        (int(sample["prediction_frames_inclusive"]) + stride - 1) // stride
        for sample in samples
    )
    print(
        f"selected {len(samples)} manifest samples; "
        f"expected GT sampled frames at stride {stride}: {expected_sampled}"
    )
    for key in sorted(setting_counts, key=lambda value: (value[0], int(value[1]))):
        print(
            f"  {key[0]}({CYCLE_LABELS[key[0]]}) "
            f"length={key[1]} videos={setting_counts[key]}"
        )


def main() -> None:
    args = parse_args()
    args.cycles = parse_multi_value(args.cycles)
    args.lengths = parse_multi_value(args.lengths)
    args.village_ids = parse_multi_value(args.village_ids)
    args.baselines = parse_multi_value(args.baselines)
    validate_args(args)
    unknown_cycles = [cycle for cycle in args.cycles if cycle not in CYCLE_LABELS]
    if unknown_cycles:
        raise ValueError(
            f"Unknown cycles: {unknown_cycles}; valid cycles: {sorted(CYCLE_LABELS)}"
        )

    categories = list(dict.fromkeys(args.categories))
    baseline_roots = dict(DEFAULT_BASELINE_ROOTS)
    baseline_roots.update(parse_baseline_root_overrides(args.baseline_root))
    unknown_baselines = [
        name for name in args.baselines if name not in baseline_roots
    ]
    if unknown_baselines:
        raise ValueError(
            f"Unknown baselines: {unknown_baselines}; "
            f"known baselines: {sorted(baseline_roots)}"
        )

    if args.merge_only:
        merge_shards(args, baseline_roots, categories)
        return

    samples = load_samples(args)
    if args.limit is not None:
        samples = samples[: args.limit]
    print_sample_summary(samples, args.stride)
    shard_samples = samples[args.shard_index :: args.num_shards]
    print(
        f"shard {args.shard_index}/{args.num_shards} gets "
        f"{len(shard_samples)} samples"
    )

    if args.dry_run:
        missing_gt = [
            Path(sample["derived_video"])
            for sample in shard_samples
            if not Path(sample["derived_video"]).is_file()
        ]
        missing_metadata = [
            Path(sample["derived_metadata"])
            for sample in shard_samples
            if not Path(sample["derived_metadata"]).is_file()
        ]
        print(
            f"derived_gt_missing={len(missing_gt)} "
            f"derived_metadata_missing={len(missing_metadata)}"
        )
        for baseline_name in args.baselines:
            missing = [
                generated_video_for_sample(baseline_roots[baseline_name], sample)
                for sample in shard_samples
                if not generated_video_for_sample(
                    baseline_roots[baseline_name], sample
                ).is_file()
            ]
            print(
                f"[{baseline_name}] root={baseline_roots[baseline_name]} "
                f"found={len(shard_samples) - len(missing)} "
                f"missing_generated={len(missing)}"
            )
            for path in missing[:10]:
                print(f"  missing: {path}")
        return

    processor = sgcs_eval.build_processor(args)
    for baseline_name in args.baselines:
        evaluate_baseline_shard(
            processor=processor,
            args=args,
            categories=categories,
            baseline_name=baseline_name,
            baseline_root=baseline_roots[baseline_name],
            samples=shard_samples,
        )


if __name__ == "__main__":
    main()
