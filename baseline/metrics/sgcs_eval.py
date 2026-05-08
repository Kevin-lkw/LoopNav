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
    "path",
    "water",
    "sky",
    "sand",
    "dirt",
    "snow",
]


def ensure_local_sam3_importable() -> None:
    repo_root = Path(__file__).resolve().parent
    local_sam3 = repo_root / "sam3"
    if local_sam3.exists():
        sys.path.insert(0, str(local_sam3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SAM3-based Scene Graph Consistency Score for two videos."
        )
    )
    parser.add_argument("--video_a", type=Path, required=True)
    parser.add_argument("--video_b", type=Path, required=True)
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
        help="Text prompts/categories to evaluate",
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
        help="Output JSON path. Defaults to sgcs_results.json.",
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
        masks_a = segment_with_prompt(processor, state_a, category)
        masks_b = segment_with_prompt(processor, state_b, category)
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
) -> dict[str, Any]:
    cap_a = open_video(video_a)
    cap_b = open_video(video_b)
    count_a = get_frame_count(cap_a)
    count_b = get_frame_count(cap_b)
    known_counts = [c for c in [count_a, count_b] if c > 0]
    if not known_counts:
        raise RuntimeError("Could not determine frame counts from either video")
    frame_count = min(known_counts)
    sampled_indices = list(range(0, frame_count, args.stride))
    offset_a = 0
    offset_b = 0
    if count_a > 0 and count_b > 0:
        if count_a > count_b:
            offset_a = count_a - count_b
        elif count_b > count_a:
            offset_b = count_b - count_a

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
        "video_a": str(video_a),
        "video_b": str(video_b),
        "video_a_frame_count": count_a,
        "video_b_frame_count": count_b,
        "stride": args.stride,
        "tau": args.tau,
        "confidence_threshold": args.confidence_threshold,
        "categories": categories,
        "video_a_frame_offset": offset_a,
        "video_b_frame_offset": offset_b,
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


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")

    categories = list(dict.fromkeys(args.categories))
    processor = build_processor(args)
    write_single_pair_result(processor, args, categories)


if __name__ == "__main__":
    main()
