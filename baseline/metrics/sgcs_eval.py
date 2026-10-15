#!/usr/bin/env python3
"""Evaluate scene graph consistency between two equal-length videos with SAM 3.

For every sampled frame pair, this script segments each text category in both
frames, matches masks by normalized centroid distance, and reports the
area-weighted Scene Graph Consistency Score (SGCS).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


DEFAULT_CATEGORIES = [
    "building",
    "tree",
    "flower",
    "grass",
    "water",
    "sky",
    "sand",
    "dirt",
]

# Audit-calibrated evaluator configuration.  `path` is intentionally not a
# separate category: the `dirt` prompt is evaluated against the merged
# dirt/path ground semantics established by the object-level audit.
SGCS_PIPELINE_VERSION = "sam3_audit_v2_house_merged_ground_20260801"
DEFAULT_CATEGORY_PROMPTS = {
    "building": "house",
}
DEFAULT_CATEGORY_MERGES = {"dirt": ["dirt", "path"]}


def resolved_category_prompts(categories: list[str]) -> dict[str, str]:
    """Return the exact SAM text prompt used for every semantic category."""
    return {
        category: DEFAULT_CATEGORY_PROMPTS.get(category, category)
        for category in categories
    }


def resolved_category_merges(categories: list[str]) -> dict[str, list[str]]:
    """Return audited semantic-label merges active in this evaluation."""
    return {
        category: list(DEFAULT_CATEGORY_MERGES[category])
        for category in categories
        if category in DEFAULT_CATEGORY_MERGES
    }

DEFAULT_GT_ROOT = Path("baseline/benchmark/prepared/gt")

DEFAULT_BASELINES = {
    name: Path("predictions") / name
    for name in ("oasis", "mineworld", "nwm", "diamond")
}
DEFAULT_PATH_TYPES = ["ABA_FIXED", "ABCA"]
DEFAULT_LENGTHS = ["5", "15", "30", "50"]
DEFAULT_VILLAGES = [
    f"{biome}_village_{village_id}"
    for village_id in (19, 20)
    for biome in ("desert", "plains", "savanna", "snowy", "taiga", "zombie")
]
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def ensure_local_sam3_importable() -> None:
    repo_root = Path(__file__).resolve().parent
    local_sam3 = repo_root / "sam3"
    if local_sam3.exists():
        sys.path.insert(0, str(local_sam3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SAM3-based SGCS for one video pair or for a GT/baseline "
            "directory tree."
        )
    )
    parser.add_argument("--video_a", type=Path, default=None)
    parser.add_argument("--video_b", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        choices=sorted(DEFAULT_BASELINES),
        default=None,
        help="Named baseline root for dataset evaluation.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Custom baseline root. Overrides --baseline path when both are set.",
    )
    parser.add_argument("--baseline-name", default=None, help="Name used in output JSON.")
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--path-types", nargs="+", default=DEFAULT_PATH_TYPES)
    parser.add_argument("--lengths", nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--villages", nargs="+", default=DEFAULT_VILLAGES)
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Sorted GT trajectory index to evaluate in each village.",
    )
    parser.add_argument(
        "--eval-all-trajectories",
        action="store_true",
        help="Deprecated alias for --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every trajectory with a matching generated-video stem.",
    )
    parser.add_argument("--stride", type=int, default=5, help="Frame sampling stride")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Normalized centroid distance threshold for a valid match",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="SAM3 mask confidence threshold",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help=(
            "Semantic categories to evaluate. The default prompt mapping uses "
            "'house' for building; path is merged into dirt and is not run as "
            "a separate prompt."
        ),
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
        help="Device used by SAM3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single-pair output path. Dataset mode writes one JSON per path/length.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for dataset evaluation JSON files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume dataset evaluation from a per-video partial checkpoint, "
            "or skip a completed path/length JSON when its pipeline and "
            "evaluation configuration match this run and it has no failures."
        ),
    )
    parser.add_argument(
        "--vis-output",
        type=Path,
        default=None,
        help=(
            "Optional path for an N-row x 3-column visualization image: "
            "video A frame, video B frame, SGCS score."
        ),
    )
    return parser.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def list_videos(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def read_explicit_return_point(metadata_path: Path) -> int:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing trajectory metadata: {metadata_path}")
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Trajectory metadata must be a list: {metadata_path}")
    points = [
        index
        for index, record in enumerate(records)
        if isinstance(record, dict) and record.get("return_point") is True
    ]
    if len(points) != 1:
        raise ValueError(
            f"Expected exactly one return_point in {metadata_path}, found {points}"
        )
    return int(points[0])


def pick_video_pairs(
    gt_dir: Path,
    baseline_dir: Path,
    trajectory_index: int,
    eval_all_trajectories: bool,
) -> list[tuple[Path, Path]]:
    gt_videos = list_videos(gt_dir)
    baseline_videos = list_videos(baseline_dir)
    baseline_by_stem = {path.stem: path for path in baseline_videos}
    if eval_all_trajectories:
        return [
            (gt_video, baseline_by_stem[gt_video.stem])
            for gt_video in gt_videos
            if gt_video.stem in baseline_by_stem
        ]

    if not gt_videos:
        raise FileNotFoundError(f"No GT videos found in {gt_dir}")
    if trajectory_index < 0 or trajectory_index >= len(gt_videos):
        raise IndexError(
            f"--trajectory-index {trajectory_index} out of range for {gt_dir}; "
            f"found {len(gt_videos)} GT videos"
        )

    gt_video = gt_videos[trajectory_index]
    if gt_video.stem in baseline_by_stem:
        return [(gt_video, baseline_by_stem[gt_video.stem])]
    if trajectory_index < len(baseline_videos):
        return [(gt_video, baseline_videos[trajectory_index])]
    raise FileNotFoundError(
        f"No baseline video matching {gt_video.stem} and no index "
        f"{trajectory_index} fallback in {baseline_dir}"
    )


def get_frame_count(cap: cv2.VideoCapture) -> int:
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return count if count > 0 else -1


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def frame_to_pil(frame_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_rgb)


def resize_rgb(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)


def align_frame_shapes(
    frame_a: np.ndarray, frame_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shape_a = frame_a.shape[:2]
    shape_b = frame_b.shape[:2]
    target_height = min(shape_a[0], shape_b[0])
    target_width = min(shape_a[1], shape_b[1])
    target_shape = (target_height, target_width)

    def resize_if_needed(frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] == target_shape:
            return frame
        return resize_rgb(frame, target_width, target_height)

    aligned_a = resize_if_needed(frame_a)
    aligned_b = resize_if_needed(frame_b)
    return aligned_a, aligned_b, {
        "video_a_original_shape": [int(shape_a[0]), int(shape_a[1])],
        "video_b_original_shape": [int(shape_b[0]), int(shape_b[1])],
        "aligned_shape": [int(target_height), int(target_width)],
        "resized": bool(shape_a != target_shape or shape_b != target_shape),
    }


def tensor_to_masks(masks: Any) -> list[np.ndarray]:
    if masks is None:
        return []
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.size == 0:
        return []
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    return [np.asarray(mask).astype(bool) for mask in masks]


def segment_with_prompt(processor: Any, state: dict[str, Any], prompt: str) -> list[np.ndarray]:
    output = processor.set_text_prompt(state=state, prompt=prompt)
    return tensor_to_masks(output.get("masks"))


def mask_stats(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    areas = []
    for mask in masks:
        area = int(mask.sum())
        if area <= 0:
            continue
        ys, xs = np.nonzero(mask)
        centers.append([float(xs.mean()), float(ys.mean())])
        areas.append(float(area))
    if not centers:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64), np.asarray(areas, dtype=np.float64)


def fallback_max_bipartite_matching(valid: np.ndarray) -> int:
    if valid.size == 0:
        return 0
    n_left, n_right = valid.shape
    match_right = [-1] * n_right

    def dfs(left: int, seen: list[bool]) -> bool:
        for right in np.flatnonzero(valid[left]):
            if seen[right]:
                continue
            seen[right] = True
            if match_right[right] == -1 or dfs(match_right[right], seen):
                match_right[right] = left
                return True
        return False

    matched = 0
    for left in range(n_left):
        if dfs(left, [False] * n_right):
            matched += 1
    return matched


def hungarian_match_count(
    centers_a: np.ndarray, centers_b: np.ndarray, max_distance: float
) -> int:
    if len(centers_a) == 0 or len(centers_b) == 0:
        return 0

    distances = np.linalg.norm(centers_a[:, None, :] - centers_b[None, :, :], axis=2)
    valid = distances < max_distance
    if not valid.any():
        return 0

    try:
        from scipy.optimize import linear_sum_assignment

        invalid_cost = max_distance + distances.max(initial=0.0) + 1.0
        cost = np.where(valid, distances, invalid_cost)
        row_ind, col_ind = linear_sum_assignment(cost)
        return int(valid[row_ind, col_ind].sum())
    except Exception:
        return fallback_max_bipartite_matching(valid)


def score_category(
    masks_a: list[np.ndarray],
    masks_b: list[np.ndarray],
    height: int,
    width: int,
    tau: float,
) -> dict[str, Any]:
    centers_a, areas_a = mask_stats(masks_a)
    centers_b, areas_b = mask_stats(masks_b)
    m = int(len(areas_a))
    n = int(len(areas_b))
    p = hungarian_match_count(
        centers_a=centers_a,
        centers_b=centers_b,
        max_distance=tau * math.hypot(height, width),
    )
    score = 1.0 if m == 0 and n == 0 else (2.0 * p / float(m + n))
    area_a = float(areas_a.sum())
    area_b = float(areas_b.sum())
    weight = (area_a + area_b) / 2.0
    return {
        "masks_a": m,
        "masks_b": n,
        "matches": int(p),
        "score": float(score),
        "area_a": area_a,
        "area_b": area_b,
        "weight": weight,
    }


def compute_frame_sgcs(
    processor: Any,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    categories: list[str],
    tau: float,
    shape_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if shape_info is None:
        frame_a, frame_b, shape_info = align_frame_shapes(frame_a, frame_b)
    height, width = frame_a.shape[:2]
    state_a = processor.set_image(frame_to_pil(frame_a))
    state_b = processor.set_image(frame_to_pil(frame_b))

    category_results = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for category in categories:
        prompt = DEFAULT_CATEGORY_PROMPTS.get(category, category)
        masks_a = segment_with_prompt(processor, state_a, prompt)
        masks_b = segment_with_prompt(processor, state_b, prompt)
        result = score_category(masks_a, masks_b, height, width, tau)
        category_results[category] = result
        weighted_sum += result["weight"] * result["score"]
        total_weight += result["weight"]

    sgcs = weighted_sum / total_weight if total_weight > 0.0 else 0.0
    return {
        "sgcs": float(sgcs),
        "total_weight": float(total_weight),
        **shape_info,
        "categories": category_results,
    }


def render_sgcs_grid(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    max_cell_width = 640
    first_frame = rows[0]["frame_a"]
    src_height, src_width = first_frame.shape[:2]
    cell_width = min(src_width, max_cell_width)
    cell_height = max(1, round(src_height * cell_width / src_width))
    score_width = max(220, min(360, cell_width // 2))

    canvas_width = cell_width * 2 + score_width
    canvas_height = cell_height * len(rows)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, cell_height // 10))
        small_font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, cell_height // 18))
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for row_idx, row in enumerate(rows):
        y0 = row_idx * cell_height
        frame_a = resize_rgb(row["frame_a"], cell_width, cell_height)
        frame_b = resize_rgb(row["frame_b"], cell_width, cell_height)
        canvas.paste(frame_to_pil(frame_a), (0, y0))
        canvas.paste(frame_to_pil(frame_b), (cell_width, y0))

        score_x0 = cell_width * 2
        draw.rectangle(
            [score_x0, y0, canvas_width, y0 + cell_height],
            fill=(255, 255, 255),
            outline=(210, 210, 210),
        )
        sgcs_text = f"SGCS\n{row['sgcs']:.4f}"
        frame_text = (
            f"A:{row['video_a_frame_index']}  "
            f"B:{row['video_b_frame_index']}"
        )
        bbox = draw.multiline_textbbox((0, 0), sgcs_text, font=font, spacing=6)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = score_x0 + (score_width - text_w) // 2
        text_y = y0 + max(8, (cell_height - text_h) // 2)
        draw.multiline_text(
            (text_x, text_y),
            sgcs_text,
            fill=(20, 20, 20),
            font=font,
            spacing=6,
            align="center",
        )
        draw.text(
            (score_x0 + 12, y0 + cell_height - 24),
            frame_text,
            fill=(90, 90, 90),
            font=small_font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_processor(args: argparse.Namespace) -> Any:
    ensure_local_sam3_importable()
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    build_kwargs = {
        "device": args.device,
        "eval_mode": True,
        "compile": False,
    }
    if args.checkpoint is not None:
        build_kwargs["checkpoint_path"] = str(args.checkpoint)
        build_kwargs["load_from_HF"] = False
    model = build_sam3_image_model(**build_kwargs)
    return Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )


def evaluate_video_pair(
    processor: Any,
    video_a: Path,
    video_b: Path,
    args: argparse.Namespace,
    categories: list[str],
    frame_offset_a: int | None = None,
    frame_offset_b: int | None = None,
    alignment: str = "tail",
) -> dict[str, Any]:
    cap_a = open_video(video_a)
    cap_b = open_video(video_b)
    count_a = get_frame_count(cap_a)
    count_b = get_frame_count(cap_b)
    known_counts = [c for c in [count_a, count_b] if c > 0]
    if not known_counts:
        raise RuntimeError("Could not determine frame counts from either video")

    if (frame_offset_a is None) != (frame_offset_b is None):
        raise ValueError("Both frame offsets must be provided together")
    if frame_offset_a is None:
        offset_a = 0
        offset_b = 0
        if count_a > 0 and count_b > 0:
            if count_a > count_b:
                offset_a = count_a - count_b
            elif count_b > count_a:
                offset_b = count_b - count_a
    else:
        offset_a = int(frame_offset_a)
        offset_b = int(frame_offset_b)

    if offset_a < 0 or offset_b < 0:
        raise ValueError(f"Frame offsets must be non-negative: {offset_a}, {offset_b}")
    available_counts = []
    if count_a > 0:
        available_counts.append(count_a - offset_a)
    if count_b > 0:
        available_counts.append(count_b - offset_b)
    if not available_counts or min(available_counts) <= 0:
        raise ValueError(
            "No overlapping frames after alignment: "
            f"counts=({count_a}, {count_b}) offsets=({offset_a}, {offset_b})"
        )
    frame_count = min(available_counts)
    sampled_indices = list(range(0, frame_count, args.stride))

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device == "cuda"
        else nullcontext()
    )
    results = []
    vis_rows = []
    try:
        with torch.inference_mode(), autocast_context:
            for frame_idx in tqdm(sampled_indices, desc="Evaluating frames"):
                source_frame_a = offset_a + frame_idx
                source_frame_b = offset_b + frame_idx
                raw_frame_a = read_frame_at(cap_a, source_frame_a)
                raw_frame_b = read_frame_at(cap_b, source_frame_b)
                frame_a, frame_b, shape_info = align_frame_shapes(
                    raw_frame_a, raw_frame_b
                )
                frame_result = compute_frame_sgcs(
                    processor=processor,
                    frame_a=frame_a,
                    frame_b=frame_b,
                    categories=categories,
                    tau=args.tau,
                    shape_info=shape_info,
                )
                frame_result["frame_index"] = int(frame_idx)
                frame_result["video_a_frame_index"] = int(source_frame_a)
                frame_result["video_b_frame_index"] = int(source_frame_b)
                results.append(frame_result)
                if getattr(args, "vis_output", None) is not None:
                    vis_rows.append(
                        {
                            "frame_a": frame_a,
                            "frame_b": frame_b,
                            "sgcs": frame_result["sgcs"],
                            "video_a_frame_index": source_frame_a,
                            "video_b_frame_index": source_frame_b,
                        }
                    )
    finally:
        cap_a.release()
        cap_b.release()

    mean_sgcs = float(np.mean([item["sgcs"] for item in results])) if results else 0.0
    output = {
        "sgcs_pipeline_version": SGCS_PIPELINE_VERSION,
        "video_a": str(video_a),
        "video_b": str(video_b),
        "video_a_frame_count": count_a,
        "video_b_frame_count": count_b,
        "stride": args.stride,
        "tau": args.tau,
        "confidence_threshold": args.confidence_threshold,
        "categories": categories,
        "category_prompts": resolved_category_prompts(categories),
        "category_merges": resolved_category_merges(categories),
        "alignment": alignment,
        "video_a_frame_offset": offset_a,
        "video_b_frame_offset": offset_b,
        "overlap_frame_count": frame_count,
        "num_sampled_frames": len(results),
        "mean_sgcs": mean_sgcs,
        "frames": results,
    }
    if getattr(args, "vis_output", None) is not None:
        render_sgcs_grid(vis_rows, args.vis_output)
    return output


def write_single_pair_result(
    processor: Any, args: argparse.Namespace, categories: list[str]
) -> None:
    if args.video_a is None or args.video_b is None:
        raise ValueError("Single-pair mode requires both --video_a and --video_b")

    output = evaluate_video_pair(
        processor=processor,
        video_a=args.video_a,
        video_b=args.video_b,
        args=args,
        categories=categories,
    )
    output_path = args.output or Path("sgcs_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"mean_sgcs: {output['mean_sgcs']:.6f}")
    print(f"num_sampled_frames: {output['num_sampled_frames']}")
    print(f"wrote: {output_path}")
    if args.vis_output is not None:
        print(f"wrote visualization: {args.vis_output}")


def resolve_baseline(args: argparse.Namespace) -> tuple[str, Path]:
    if args.baseline_root is not None:
        name = args.baseline_name or args.baseline or args.baseline_root.name
        return name, args.baseline_root
    if args.baseline is None:
        raise ValueError("Dataset mode requires --baseline or --baseline-root")
    return args.baseline_name or args.baseline, DEFAULT_BASELINES[args.baseline]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without exposing a truncated checkpoint to a resumed worker."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_dataset_results(
    processor: Any, args: argparse.Namespace, categories: list[str]
) -> None:
    baseline_name, baseline_root = resolve_baseline(args)
    eval_all = args.all or args.eval_all_trajectories
    is_legacy_single_trajectory_nwm = (
        baseline_root.resolve() == DEFAULT_BASELINES["nwm"].resolve()
    )
    if (
        eval_all
        and is_legacy_single_trajectory_nwm
        and "ABA_FIXED" not in args.path_types
    ):
        raise ValueError("NWM does not support --all because it has one trajectory only")

    output_dir = args.output_dir or Path("./eval_SGCS_all" if eval_all else "./eval_SGCS")
    output_dir.mkdir(parents=True, exist_ok=True)

    for path_type in args.path_types:
        for length in args.lengths:
            output_path = output_dir / f"{baseline_name}_{path_type}_{length}.json"
            partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
            expected = {
                "sgcs_pipeline_version": SGCS_PIPELINE_VERSION,
                "baseline": baseline_name,
                "baseline_root": str(baseline_root),
                "gt_root": str(args.gt_root),
                "path_type": path_type,
                "length": str(length),
                "stride": args.stride,
                "tau": args.tau,
                "confidence_threshold": args.confidence_threshold,
                "categories": categories,
                "category_prompts": resolved_category_prompts(categories),
                "category_merges": resolved_category_merges(categories),
                "trajectory_index": args.trajectory_index,
                "all": eval_all,
                "eval_all_trajectories": eval_all,
            }
            if args.resume and output_path.is_file():
                try:
                    existing = json.loads(output_path.read_text(encoding="utf-8"))
                    if (
                        all(existing.get(key) == value for key, value in expected.items())
                        and int(existing.get("num_videos", 0)) > 0
                        and not existing.get("failures")
                    ):
                        print(f"resume: skipping complete output {output_path}")
                        continue
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            videos = []
            failures = []
            if args.resume and partial_path.is_file():
                try:
                    partial = json.loads(partial_path.read_text(encoding="utf-8"))
                    if all(
                        partial.get(key) == value for key, value in expected.items()
                    ):
                        videos = list(partial.get("videos", []))
                        print(
                            f"resume: loaded {len(videos)} completed videos from "
                            f"{partial_path}"
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            completed_pairs = {
                (str(item.get("video_a")), str(item.get("video_b")))
                for item in videos
            }
            for village in args.villages:
                gt_dir = args.gt_root / path_type / str(length) / village
                baseline_dir = baseline_root / path_type / str(length) / village
                try:
                    pairs = pick_video_pairs(
                        gt_dir=gt_dir,
                        baseline_dir=baseline_dir,
                        trajectory_index=args.trajectory_index,
                        eval_all_trajectories=eval_all,
                    )
                    for pair_index, (gt_video, baseline_video) in enumerate(pairs):
                        pair_key = (str(gt_video), str(baseline_video))
                        if pair_key in completed_pairs:
                            print(
                                "resume: skipping completed video pair "
                                f"{gt_video.name} / {baseline_video.name}"
                            )
                            continue
                        trajectory_index = (
                            pair_index if eval_all else args.trajectory_index
                        )
                        frame_offset_a = None
                        frame_offset_b = None
                        alignment = "tail"
                        if path_type == "ABA_FIXED":
                            frame_offset_a = read_explicit_return_point(
                                gt_video.with_suffix(".json")
                            )
                            frame_offset_b = 0
                            alignment = "metadata_return_point"
                        print(
                            "evaluating "
                            f"{baseline_name} {path_type} {length} {village} "
                            f"traj={trajectory_index}: {gt_video.name} "
                            f"alignment={alignment}"
                        )
                        result = evaluate_video_pair(
                            processor=processor,
                            video_a=gt_video,
                            video_b=baseline_video,
                            args=args,
                            categories=categories,
                            frame_offset_a=frame_offset_a,
                            frame_offset_b=frame_offset_b,
                            alignment=alignment,
                        )
                        result["village"] = village
                        result["path_type"] = path_type
                        result["length"] = str(length)
                        result["trajectory_index"] = trajectory_index
                        videos.append(result)
                        completed_pairs.add(pair_key)
                        checkpoint_scores = [item["mean_sgcs"] for item in videos]
                        checkpoint = {
                            **expected,
                            "num_videos": len(videos),
                            "mean_sgcs": float(np.mean(checkpoint_scores)),
                            "std_sgcs": float(np.std(checkpoint_scores)),
                            "videos": videos,
                            "failures": [],
                            "checkpoint_complete": False,
                        }
                        write_json_atomic(partial_path, checkpoint)
                except Exception as exc:
                    failures.append(
                        {
                            "village": village,
                            "gt_dir": str(gt_dir),
                            "baseline_dir": str(baseline_dir),
                            "error": str(exc),
                        }
                    )

            video_scores = [item["mean_sgcs"] for item in videos]
            output = {
                "sgcs_pipeline_version": SGCS_PIPELINE_VERSION,
                "baseline": baseline_name,
                "baseline_root": str(baseline_root),
                "gt_root": str(args.gt_root),
                "path_type": path_type,
                "length": str(length),
                "stride": args.stride,
                "tau": args.tau,
                "confidence_threshold": args.confidence_threshold,
                "categories": categories,
                "category_prompts": resolved_category_prompts(categories),
                "category_merges": resolved_category_merges(categories),
                "trajectory_index": args.trajectory_index,
                "all": eval_all,
                "eval_all_trajectories": eval_all,
                "num_videos": len(videos),
                "mean_sgcs": float(np.mean(video_scores)) if video_scores else 0.0,
                "std_sgcs": float(np.std(video_scores)) if video_scores else 0.0,
                "videos": videos,
                "failures": failures,
            }
            write_json_atomic(output_path, output)
            partial_path.unlink(missing_ok=True)
            print(
                f"wrote: {output_path} "
                f"mean={output['mean_sgcs']:.6f} std={output['std_sgcs']:.6f} "
                f"videos={len(videos)} failures={len(failures)}"
            )


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")
    if args.trajectory_index < 0:
        raise ValueError("--trajectory-index must be non-negative")
    if args.vis_output is not None and (args.video_a is None or args.video_b is None):
        raise ValueError("--vis-output is only supported in single-pair mode")

    categories = list(dict.fromkeys(args.categories))
    processor = build_processor(args)
    if args.video_a is not None or args.video_b is not None:
        write_single_pair_result(processor, args, categories)
    else:
        write_dataset_results(processor, args, categories)


if __name__ == "__main__":
    main()
