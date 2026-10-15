#!/usr/bin/env python3
"""Compute SAM3-based SGCS for TURN rollouts.

TURN ground-truth videos contain the outbound turn and the return segment.  The
generated baseline videos start at the recorded return point, so the comparison
is:

    GT[return_point + t]  <->  generated[t]

for sampled frame indices t.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import sgcs_eval


DEFAULT_GT_ROOT = Path("benchmark/prepared/turn_gt")
DEFAULT_BASELINE_ROOTS = {
    name: Path("predictions") / name / "TURN"
    for name in ("oasis", "nwm", "mineworld", "diamond")
}
DEFAULT_ANGLE_BUCKETS = ["60-90", "90-120", "120-150", "150-180"]
DEFAULT_VILLAGE_IDS = ["19", "20"]


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
                "--baseline-root must use NAME=PATH, e.g. --baseline-root nwm=/tmp/TURN"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid baseline root override: {item}")
        overrides[name] = Path(path).expanduser()
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SGCS for TURN baseline outputs using return_point alignment."
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
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
        "--angle-buckets",
        nargs="+",
        default=DEFAULT_ANGLE_BUCKETS,
        help="TURN angle buckets to evaluate.",
    )
    parser.add_argument(
        "--village-ids",
        nargs="+",
        default=DEFAULT_VILLAGE_IDS,
        help="Village id suffixes to evaluate, e.g. 18 19 20 or 18,19,20.",
    )
    parser.add_argument("--stride", type=int, default=5, help="Frame sampling stride.")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Normalized centroid distance threshold for a valid mask match.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="SAM3 mask confidence threshold.",
    )
    parser.add_argument(
        "--gt-center-crop-aspect-ratio",
        type=float,
        default=None,
        help=(
            "Optionally center-crop each GT frame to this width/height aspect "
            "ratio before aligning it with the generated frame."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=sgcs_eval.DEFAULT_CATEGORIES,
        help="Text prompts/categories to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local SAM3 checkpoint path. If omitted, SAM3 downloads from HF.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device used by SAM3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_SGCS_TURN"),
        help="Directory for shard outputs, merged JSON, and summary files.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug limit applied before sharding.",
    )
    parser.add_argument(
        "--max-videos-per-setting",
        type=int,
        default=None,
        help=(
            "Keep at most the lexicographically first N videos in each "
            "(village, angle-bucket) setting."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Checkpoint shard JSON after this many newly evaluated videos.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--frame-progress",
        action="store_true",
        help="Show per-frame tqdm bars inside each video.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing shard JSON files; do not load SAM3 or evaluate.",
    )
    return parser.parse_args()


def find_return_point(json_path: Path) -> tuple[int, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"TURN metadata is not a list: {json_path}")
    points = [idx for idx, row in enumerate(data) if row.get("return_point") is True]
    if len(points) != 1:
        raise ValueError(
            f"Expected exactly one return_point in {json_path}, found {len(points)}"
        )
    return int(points[0]), len(data)


def discover_turn_samples(
    gt_root: Path,
    angle_buckets: list[str],
    village_ids: list[str],
    max_videos_per_setting: int | None = None,
) -> list[dict[str, Any]]:
    village_id_set = set(village_ids)
    bucket_set = set(angle_buckets)
    samples: list[dict[str, Any]] = []
    for village_dir in sorted(path for path in gt_root.iterdir() if path.is_dir()):
        village = village_dir.name
        if "_" not in village:
            continue
        village_id = village.rsplit("_", 1)[-1]
        if village_id not in village_id_set:
            continue
        for bucket in angle_buckets:
            if bucket not in bucket_set:
                continue
            bucket_dir = village_dir / bucket
            if not bucket_dir.is_dir():
                continue
            gt_videos = sorted(bucket_dir.glob("*.avi"))
            if max_videos_per_setting is not None:
                gt_videos = gt_videos[:max_videos_per_setting]
            for gt_video in gt_videos:
                json_path = gt_video.with_suffix(".json")
                if not json_path.is_file():
                    samples.append(
                        {
                            "angle_bucket": bucket,
                            "village": village,
                            "gt_video": gt_video,
                            "json_path": json_path,
                            "missing_json": True,
                        }
                    )
                    continue
                samples.append(
                    {
                        "angle_bucket": bucket,
                        "village": village,
                        "gt_video": gt_video,
                        "json_path": json_path,
                        "missing_json": False,
                    }
                )
    return samples


def generated_video_for_sample(baseline_root: Path, sample: dict[str, Any]) -> Path:
    gt_video = Path(sample["gt_video"])
    return (
        baseline_root
        / str(sample["angle_bucket"])
        / str(sample["village"])
        / f"{gt_video.stem}.mp4"
    )


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
        "gt_root": str(args.gt_root),
        "angle_buckets": list(args.angle_buckets),
        "village_ids": list(args.village_ids),
        "stride": args.stride,
        "tau": args.tau,
        "confidence_threshold": args.confidence_threshold,
        "categories": categories,
        "category_prompts": sgcs_eval.resolved_category_prompts(categories),
        "category_merges": sgcs_eval.resolved_category_merges(categories),
        "max_videos_per_setting": args.max_videos_per_setting,
        "gt_center_crop_aspect_ratio": args.gt_center_crop_aspect_ratio,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }


def load_existing_shard(
    output_path: Path,
    expected_config: dict[str, Any],
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if overwrite or not output_path.exists():
        return [], []
    data = json.loads(output_path.read_text(encoding="utf-8"))
    old_config = data.get("config")
    if old_config != expected_config:
        raise ValueError(
            f"Existing shard config differs from this run: {output_path}. "
            "Use --overwrite to replace it."
        )
    return list(data.get("videos", [])), list(data.get("failures", []))


def sampled_indices_for_counts(
    gt_frame_count: int,
    generated_frame_count: int,
    metadata_length: int,
    return_point: int,
    stride: int,
) -> tuple[int, list[int]]:
    gt_remaining = (
        gt_frame_count - return_point if gt_frame_count > 0 else metadata_length - return_point
    )
    if gt_remaining <= 0:
        raise ValueError(
            f"No GT frames remain after return_point={return_point}; "
            f"gt_frame_count={gt_frame_count}, metadata_length={metadata_length}"
        )
    if generated_frame_count > 0:
        compare_count = min(gt_remaining, generated_frame_count)
    else:
        compare_count = gt_remaining
    if compare_count <= 0:
        raise ValueError(
            f"No comparable frames; gt_remaining={gt_remaining}, "
            f"generated_frame_count={generated_frame_count}"
        )
    return int(compare_count), list(range(0, int(compare_count), stride))


def center_crop_to_aspect_ratio(
    frame: np.ndarray, aspect_ratio: float
) -> np.ndarray:
    height, width = frame.shape[:2]
    current_ratio = width / height
    if np.isclose(current_ratio, aspect_ratio):
        return frame
    if current_ratio > aspect_ratio:
        target_width = max(1, min(width, int(round(height * aspect_ratio))))
        x0 = (width - target_width) // 2
        return frame[:, x0 : x0 + target_width]
    target_height = max(1, min(height, int(round(width / aspect_ratio))))
    y0 = (height - target_height) // 2
    return frame[y0 : y0 + target_height, :]


def evaluate_turn_pair(
    processor: Any,
    gt_video: Path,
    generated_video: Path,
    return_point: int,
    metadata_length: int,
    args: argparse.Namespace,
    categories: list[str],
) -> dict[str, Any]:
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
    frame_results = []
    try:
        with torch.inference_mode(), autocast_context:
            for frame_idx in tqdm(
                sampled_indices,
                desc=f"{gt_video.parent.parent.name}/{gt_video.parent.name}/{gt_video.stem}",
                leave=False,
                disable=not args.frame_progress,
            ):
                gt_frame_index = return_point + frame_idx
                generated_frame_index = frame_idx
                raw_gt_frame = sgcs_eval.read_frame_at(cap_gt, gt_frame_index)
                raw_generated_frame = sgcs_eval.read_frame_at(
                    cap_generated, generated_frame_index
                )
                gt_pre_crop_shape = [int(value) for value in raw_gt_frame.shape[:2]]
                if args.gt_center_crop_aspect_ratio is not None:
                    raw_gt_frame = center_crop_to_aspect_ratio(
                        raw_gt_frame, args.gt_center_crop_aspect_ratio
                    )
                gt_frame, generated_frame, shape_info = sgcs_eval.align_frame_shapes(
                    raw_gt_frame, raw_generated_frame
                )
                shape_info["video_a_pre_crop_shape"] = gt_pre_crop_shape
                shape_info["gt_center_crop_aspect_ratio"] = (
                    args.gt_center_crop_aspect_ratio
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
        "video_a": str(gt_video),
        "video_b": str(generated_video),
        "video_a_role": "gt_return_segment",
        "video_b_role": "generated_from_return_point",
        "video_a_frame_count": gt_frame_count,
        "video_b_frame_count": generated_frame_count,
        "metadata_length": int(metadata_length),
        "return_point": int(return_point),
        "compare_frame_count": int(compare_count),
        "stride": int(args.stride),
        "tau": float(args.tau),
        "confidence_threshold": float(args.confidence_threshold),
        "categories": categories,
        "category_prompts": sgcs_eval.resolved_category_prompts(categories),
        "category_merges": sgcs_eval.resolved_category_merges(categories),
        "video_a_frame_offset": int(return_point),
        "video_b_frame_offset": 0,
        "num_sampled_frames": len(frame_results),
        "mean_sgcs": float(np.mean(scores)) if scores else 0.0,
        "frames": frame_results,
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
        / f"{baseline_name}_TURN_shard{shard_index:02d}_of_{num_shards:02d}.json"
    )


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
        "mean_sgcs": float(np.mean(scores)) if scores else 0.0,
        "std_sgcs": float(np.std(scores)) if scores else 0.0,
        "videos": videos,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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
    processed = {item["video_a"] for item in videos}
    processed.update(item.get("gt_video", "") for item in failures)
    newly_evaluated = 0

    print(
        f"[{baseline_name}] shard {args.shard_index}/{args.num_shards}: "
        f"{len(samples)} assigned, {len(processed)} already recorded"
    )
    for sample_idx, sample in enumerate(samples, 1):
        gt_video = Path(sample["gt_video"])
        if str(gt_video) in processed:
            continue
        generated_video = generated_video_for_sample(baseline_root, sample)
        try:
            if sample.get("missing_json"):
                raise FileNotFoundError(f"Missing TURN metadata: {sample['json_path']}")
            if not generated_video.is_file():
                raise FileNotFoundError(f"Missing generated video: {generated_video}")
            return_point, metadata_length = find_return_point(Path(sample["json_path"]))
            print(
                f"[{baseline_name}] {sample_idx}/{len(samples)} "
                f"{sample['angle_bucket']} {sample['village']} {gt_video.stem} "
                f"return_point={return_point}"
            )
            result = evaluate_turn_pair(
                processor=processor,
                gt_video=gt_video,
                generated_video=generated_video,
                return_point=return_point,
                metadata_length=metadata_length,
                args=args,
                categories=categories,
            )
            result["baseline"] = baseline_name
            result["angle_bucket"] = sample["angle_bucket"]
            result["village"] = sample["village"]
            result["gt_metadata"] = str(sample["json_path"])
            videos.append(result)
        except Exception as exc:
            print(
                f"[{baseline_name}] failure {sample['angle_bucket']} "
                f"{sample['village']} {gt_video.name}: {exc}",
                file=sys.stderr,
            )
            failures.append(
                {
                    "baseline": baseline_name,
                    "angle_bucket": sample["angle_bucket"],
                    "village": sample["village"],
                    "gt_video": str(gt_video),
                    "gt_metadata": str(sample["json_path"]),
                    "generated_video": str(generated_video),
                    "error": str(exc),
                }
            )
        processed.add(str(gt_video))
        newly_evaluated += 1
        if newly_evaluated % args.save_every == 0:
            write_shard(output_path, config, videos, failures)
            print(f"[{baseline_name}] checkpointed {output_path}")

    write_shard(output_path, config, videos, failures)
    print(
        f"[{baseline_name}] wrote {output_path} "
        f"videos={len(videos)} failures={len(failures)}"
    )


def aggregate_videos(videos: list[dict[str, Any]]) -> tuple[float, float]:
    scores = [item["mean_sgcs"] for item in videos]
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


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
        missing_shards = []
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
        if missing_shards:
            print(
                f"[{baseline_name}] warning: missing {len(missing_shards)} shard(s)",
                file=sys.stderr,
            )

        for angle_bucket in args.angle_buckets:
            videos = [
                item for item in all_videos if item.get("angle_bucket") == angle_bucket
            ]
            failures = [
                item
                for item in all_failures
                if item.get("angle_bucket") == angle_bucket
            ]
            mean_sgcs, std_sgcs = aggregate_videos(videos)
            output = {
                "sgcs_pipeline_version": sgcs_eval.SGCS_PIPELINE_VERSION,
                "baseline": baseline_name,
                "baseline_root": str(baseline_roots[baseline_name]),
                "gt_root": str(args.gt_root),
                "path_type": "TURN",
                "angle_bucket": angle_bucket,
                "stride": args.stride,
                "tau": args.tau,
                "confidence_threshold": args.confidence_threshold,
                "categories": categories,
                "category_prompts": sgcs_eval.resolved_category_prompts(categories),
                "category_merges": sgcs_eval.resolved_category_merges(categories),
                "village_ids": args.village_ids,
                "num_shards": args.num_shards,
                "num_videos": len(videos),
                "mean_sgcs": mean_sgcs,
                "std_sgcs": std_sgcs,
                "videos": sorted(
                    videos,
                    key=lambda item: (
                        item.get("village", ""),
                        Path(item.get("video_a", "")).stem,
                    ),
                ),
                "failures": failures,
            }
            output_path = args.output_dir / f"{baseline_name}_TURN_{angle_bucket}.json"
            output_path.write_text(
                json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            summary_rows.append(
                {
                    "baseline": baseline_name,
                    "angle_bucket": angle_bucket,
                    "num_videos": len(videos),
                    "num_failures": len(failures),
                    "mean_sgcs": mean_sgcs,
                    "std_sgcs": std_sgcs,
                    "output": str(output_path),
                }
            )
            print(
                f"wrote {output_path} mean={mean_sgcs:.6f} "
                f"std={std_sgcs:.6f} videos={len(videos)} failures={len(failures)}"
            )

        mean_sgcs, std_sgcs = aggregate_videos(all_videos)
        summary_rows.append(
            {
                "baseline": baseline_name,
                "angle_bucket": "ALL",
                "num_videos": len(all_videos),
                "num_failures": len(all_failures),
                "mean_sgcs": mean_sgcs,
                "std_sgcs": std_sgcs,
                "output": "",
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = args.output_dir / "summary.csv"
    csv_lines = ["baseline,angle_bucket,num_videos,num_failures,mean_sgcs,std_sgcs,output"]
    for row in summary_rows:
        csv_lines.append(
            ",".join(
                [
                    str(row["baseline"]),
                    str(row["angle_bucket"]),
                    str(row["num_videos"]),
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


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")
    if (
        args.gt_center_crop_aspect_ratio is not None
        and args.gt_center_crop_aspect_ratio <= 0
    ):
        raise ValueError("--gt-center-crop-aspect-ratio must be positive")
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

    args.angle_buckets = parse_multi_value(args.angle_buckets)
    args.village_ids = parse_multi_value(args.village_ids)
    args.baselines = parse_multi_value(args.baselines)
    categories = list(dict.fromkeys(args.categories))
    baseline_roots = dict(DEFAULT_BASELINE_ROOTS)
    baseline_roots.update(parse_baseline_root_overrides(args.baseline_root))
    unknown = [name for name in args.baselines if name not in baseline_roots]
    if unknown:
        raise ValueError(
            f"Unknown baseline(s): {unknown}. Known baselines: {sorted(baseline_roots)}"
        )

    if args.merge_only:
        merge_shards(args, baseline_roots, categories)
        return

    samples = discover_turn_samples(
        gt_root=args.gt_root,
        angle_buckets=args.angle_buckets,
        village_ids=args.village_ids,
        max_videos_per_setting=args.max_videos_per_setting,
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    shard_samples = samples[args.shard_index :: args.num_shards]
    print(
        f"discovered {len(samples)} TURN samples; "
        f"shard {args.shard_index}/{args.num_shards} gets {len(shard_samples)}"
    )

    if args.dry_run:
        for baseline_name in args.baselines:
            missing = [
                generated_video_for_sample(baseline_roots[baseline_name], sample)
                for sample in shard_samples
                if not generated_video_for_sample(
                    baseline_roots[baseline_name], sample
                ).is_file()
            ]
            print(
                f"[{baseline_name}] baseline_root={baseline_roots[baseline_name]} "
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
