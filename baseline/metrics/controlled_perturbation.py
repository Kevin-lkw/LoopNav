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
import os
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
        description="Compute SAM3 SGCS ablations on one video with synthetic perturbations."
    )
    parser.add_argument("--video_a", type=Path)
    parser.add_argument("--video_b", type=Path)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help=(
            "Optional batch mode root. Each immediate subdirectory is treated "
            "as one village, and the first sorted .avi in it is evaluated."
        ),
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
        help="Path to write per-frame JSON results",
    )
    parser.add_argument(
        "--vis-output",
        type=Path,
        default=None,
        help=(
            "Optional path for an N-row visualization image: GT, then each "
            "perturbation frame and its SGCS score."
        ),
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=["all"],
        help=(
            "Perturbation types to run. Use 'all' for every type. Choices: "
            "small_translation, small_rotation, crop_resize, color_change, "
            "delete_object, swap_objects."
        ),
    )
    parser.add_argument(
        "--ssim",
        action="store_true",
        help=(
            "Compute SSIM for each perturbation instead of SGCS. This skips "
            "SGCS segmentation; SAM is still used only for object-aware "
            "delete/swap perturbation generation."
        ),
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help=(
            "Compute sim-LPIPS = 1 - LPIPS for each perturbation instead of "
            "SGCS. This skips SGCS segmentation; SAM is still used only for "
            "object-aware delete/swap perturbation generation."
        ),
    )
    parser.add_argument(
        "--translation-frac",
        type=float,
        default=0.04,
        help="Small translation magnitude as a fraction of frame width/height.",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=5.0,
        help="Small rotation angle in degrees.",
    )
    parser.add_argument(
        "--crop-frac",
        type=float,
        default=0.9,
        help="Centered crop fraction before resizing back to original size.",
    )
    parser.add_argument(
        "--color-brightness",
        type=float,
        default=18.0,
        help="Brightness delta for color-change perturbation.",
    )
    parser.add_argument(
        "--color-contrast",
        type=float,
        default=1.12,
        help="Contrast multiplier for color-change perturbation.",
    )
    parser.add_argument(
        "--object-mask-min-area-frac",
        type=float,
        default=0.002,
        help="Minimum SAM object mask area fraction for delete/swap perturbations.",
    )
    parser.add_argument(
        "--object-mask-max-area-frac",
        type=float,
        default=0.4,
        help="Maximum SAM object mask area fraction for delete/swap perturbations.",
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


def rgb_to_gray(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    frame_a, frame_b, _ = align_frame_shapes(frame_a, frame_b)
    gray_a = rgb_to_gray(frame_a)
    gray_b = rgb_to_gray(frame_b)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel_size = (11, 11)
    sigma = 1.5

    mu_a = cv2.GaussianBlur(gray_a, kernel_size, sigma)
    mu_b = cv2.GaussianBlur(gray_b, kernel_size, sigma)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a * gray_a, kernel_size, sigma) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b * gray_b, kernel_size, sigma) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, kernel_size, sigma) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    score_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(score_map))


def frame_to_lpips_tensor(frame_rgb: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0) * 2.0 - 1.0
    return tensor.to(device)


def build_lpips_model(device: str) -> Any:
    try:
        import lpips
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent
        local_lpips_parent = repo_root / "common_metrics_on_video_quality"
        if local_lpips_parent.exists():
            sys.path.insert(0, str(local_lpips_parent))
        try:
            import lpips
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LPIPS dependencies are unavailable. Install lpips or ensure "
                "common_metrics_on_video_quality/lpips and torchvision are importable."
            ) from exc

    try:
        model = lpips.LPIPS(net="alex", spatial=False)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LPIPS model initialization failed because a dependency is missing. "
            "Install torchvision and the LPIPS dependencies."
        ) from exc
    model.eval()
    return model.to(device)


def compute_lpips_score(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    lpips_model: Any,
    device: str,
) -> float:
    frame_a, frame_b, _ = align_frame_shapes(frame_a, frame_b)
    tensor_a = frame_to_lpips_tensor(frame_a, device)
    tensor_b = frame_to_lpips_tensor(frame_b, device)
    with torch.inference_mode():
        return float(lpips_model(tensor_a, tensor_b).mean().detach().cpu().item())


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


def perturb_small_translation(frame_rgb: np.ndarray, frac: float) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    dx = max(1, round(width * frac))
    dy = max(1, round(height * frac))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def perturb_small_rotation(frame_rgb: np.ndarray, degrees: float) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(
        frame_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def perturb_crop_resize(frame_rgb: np.ndarray, crop_frac: float) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    crop_frac = float(np.clip(crop_frac, 0.1, 1.0))
    crop_h = max(1, round(height * crop_frac))
    crop_w = max(1, round(width * crop_frac))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    crop = frame_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return resize_rgb(crop, width, height)


def perturb_color_change(
    frame_rgb: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    adjusted = frame_rgb.astype(np.float32) * contrast + brightness
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def deduplicate_object_masks(objects: list[dict[str, Any]], iou_threshold: float = 0.85) -> list[dict[str, Any]]:
    kept = []
    for obj in sorted(objects, key=lambda item: item["area"], reverse=True):
        if all(mask_iou(obj["mask"], kept_obj["mask"]) < iou_threshold for kept_obj in kept):
            kept.append(obj)
    return kept


def collect_sam_object_masks(
    processor: Any,
    state: dict[str, Any],
    categories: list[str],
    height: int,
    width: int,
    min_area_frac: float,
    max_area_frac: float,
) -> list[dict[str, Any]]:
    min_area = max(1.0, height * width * min_area_frac)
    max_area = height * width * max_area_frac
    objects = []
    for category in categories:
        for mask in segment_with_prompt(processor, state, category):
            area = int(mask.sum())
            if area < min_area or area > max_area:
                continue
            bbox = mask_bbox(mask)
            if bbox is None:
                continue
            objects.append(
                {
                    "category": category,
                    "mask": mask,
                    "area": area,
                    "bbox": bbox,
                }
            )
    return deduplicate_object_masks(objects)


def inpaint_rgb(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(frame_bgr, mask_uint8, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def perturb_delete_object(
    frame_rgb: np.ndarray, objects: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    if not objects:
        return frame_rgb.copy(), {"status": "skipped_no_object"}
    obj = objects[0]
    return inpaint_rgb(frame_rgb, obj["mask"]), {
        "status": "ok",
        "deleted_category": obj["category"],
        "deleted_area": int(obj["area"]),
        "deleted_bbox": list(obj["bbox"]),
    }


def paste_resized_object(
    canvas: np.ndarray,
    source_patch: np.ndarray,
    source_mask: np.ndarray,
    target_bbox: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = target_bbox
    target_w = max(1, x1 - x0)
    target_h = max(1, y1 - y0)
    resized_patch = resize_rgb(source_patch, target_w, target_h)
    resized_mask = cv2.resize(
        source_mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    canvas[y0:y1, x0:x1][resized_mask] = resized_patch[resized_mask]


def perturb_swap_objects(
    frame_rgb: np.ndarray, objects: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(objects) < 2:
        return frame_rgb.copy(), {"status": "skipped_less_than_two_objects"}
    obj_a, obj_b = objects[0], objects[1]
    union_mask = np.logical_or(obj_a["mask"], obj_b["mask"])
    canvas = inpaint_rgb(frame_rgb, union_mask)

    bbox_a = obj_a["bbox"]
    bbox_b = obj_b["bbox"]
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    patch_a = frame_rgb[ay0:ay1, ax0:ax1]
    patch_b = frame_rgb[by0:by1, bx0:bx1]
    mask_a = obj_a["mask"][ay0:ay1, ax0:ax1]
    mask_b = obj_b["mask"][by0:by1, bx0:bx1]

    paste_resized_object(canvas, patch_a, mask_a, bbox_b)
    paste_resized_object(canvas, patch_b, mask_b, bbox_a)
    return canvas, {
        "status": "ok",
        "object_a_category": obj_a["category"],
        "object_a_area": int(obj_a["area"]),
        "object_a_bbox": list(bbox_a),
        "object_b_category": obj_b["category"],
        "object_b_area": int(obj_b["area"]),
        "object_b_bbox": list(bbox_b),
    }


def build_perturbations(args: argparse.Namespace) -> list[dict[str, Any]]:
    perturbations = [
        {
            "name": "small_translation",
            "label": "Small translation",
            "object_aware": False,
            "apply": lambda frame: perturb_small_translation(
                frame, args.translation_frac
            ),
            "params": {"translation_frac": args.translation_frac},
        },
        {
            "name": "small_rotation",
            "label": "Small rotation",
            "object_aware": False,
            "apply": lambda frame: perturb_small_rotation(frame, args.rotation_deg),
            "params": {"rotation_deg": args.rotation_deg},
        },
        {
            "name": "crop_resize",
            "label": "Crop + resize",
            "object_aware": False,
            "apply": lambda frame: perturb_crop_resize(frame, args.crop_frac),
            "params": {"crop_frac": args.crop_frac},
        },
        {
            "name": "color_change",
            "label": "Color change",
            "object_aware": False,
            "apply": lambda frame: perturb_color_change(
                frame, args.color_brightness, args.color_contrast
            ),
            "params": {
                "color_brightness": args.color_brightness,
                "color_contrast": args.color_contrast,
            },
        },
        {
            "name": "delete_object",
            "label": "Delete object",
            "object_aware": True,
            "apply": perturb_delete_object,
            "params": {
                "object_mask_min_area_frac": args.object_mask_min_area_frac,
                "object_mask_max_area_frac": args.object_mask_max_area_frac,
            },
        },
        {
            "name": "swap_objects",
            "label": "Swap objects",
            "object_aware": True,
            "apply": perturb_swap_objects,
            "params": {
                "object_mask_min_area_frac": args.object_mask_min_area_frac,
                "object_mask_max_area_frac": args.object_mask_max_area_frac,
            },
        },
    ]
    requested = list(dict.fromkeys(args.perturbations))
    if requested == ["all"] or "all" in requested:
        return perturbations

    by_name = {item["name"]: item for item in perturbations}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(
            "Unknown perturbation(s): "
            + ", ".join(unknown)
            + ". Valid choices are: "
            + ", ".join(by_name)
            + ", all"
        )
    return [by_name[name] for name in requested]


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
    return compute_sgcs_from_states(
        processor=processor,
        state_a=state_a,
        state_b=state_b,
        height=height,
        width=width,
        categories=categories,
        tau=tau,
        shape_info=shape_info,
    )


def compute_sgcs_from_states(
    processor: Any,
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    height: int,
    width: int,
    categories: list[str],
    tau: float,
    shape_info: dict[str, Any],
) -> dict[str, Any]:
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


def draw_centered_multiline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (20, 20, 20),
    spacing: int = 6,
) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    x = x0 + max(0, (x1 - x0 - text_w) // 2)
    y = y0 + max(0, (y1 - y0 - text_h) // 2)
    draw.multiline_text(
        (x, y),
        text,
        fill=fill,
        font=font,
        spacing=spacing,
        align="center",
    )


def draw_cell_label(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((0, 0), label, font=font)
    padding_x = 6
    padding_y = 3
    draw.rectangle(
        [
            x0 + 6,
            y0 + 6,
            x0 + 6 + (bbox[2] - bbox[0]) + padding_x * 2,
            y0 + 6 + (bbox[3] - bbox[1]) + padding_y * 2,
        ],
        fill=(0, 0, 0),
    )
    draw.text(
        (x0 + 6 + padding_x, y0 + 6 + padding_y),
        label,
        fill=(255, 255, 255),
        font=font,
    )


def render_ablation_grid(
    rows: list[dict[str, Any]],
    perturbations: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        return

    max_cell_width = 260
    first_frame = rows[0]["gt_frame"]
    src_height, src_width = first_frame.shape[:2]
    image_width = min(src_width, max_cell_width)
    image_height = max(1, round(src_height * image_width / src_width))
    score_width = 150
    n_cols = 1 + len(perturbations) * 2

    canvas_width = image_width * (1 + len(perturbations)) + score_width * len(
        perturbations
    )
    canvas_height = image_height * len(rows)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        label_font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, image_height // 16))
        score_font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, image_height // 10))
        small_font = ImageFont.truetype("DejaVuSans.ttf", size=max(11, image_height // 20))
    except OSError:
        label_font = ImageFont.load_default()
        score_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for row_idx, row in enumerate(rows):
        y0 = row_idx * image_height
        x = 0

        gt = resize_rgb(row["gt_frame"], image_width, image_height)
        canvas.paste(frame_to_pil(gt), (x, y0))
        draw_cell_label(draw, x, y0, "GT", label_font)
        draw.text(
            (x + 8, y0 + image_height - 22),
            f"frame {row['frame_index']}",
            fill=(255, 255, 255),
            font=small_font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        x += image_width

        for perturbation in perturbations:
            name = perturbation["name"]
            label = perturbation["label"]
            perturbed = resize_rgb(
                row["perturbations"][name]["frame"], image_width, image_height
            )
            canvas.paste(frame_to_pil(perturbed), (x, y0))
            draw_cell_label(draw, x, y0, label, label_font)
            x += image_width

            score_box = (x, y0, x + score_width, y0 + image_height)
            draw.rectangle(
                score_box,
                fill=(255, 255, 255),
                outline=(210, 210, 210),
            )
            if row["perturbations"][name].get("skipped", False):
                metadata = row["perturbations"][name].get("perturbation_metadata", {})
                status = metadata.get("status", "skipped")
                score_text = f"SKIPPED\n{status}"
            elif "ssim" in row["perturbations"][name]:
                score_text = f"SSIM\n{row['perturbations'][name]['ssim']:.4f}"
            elif "sim_lpips" in row["perturbations"][name]:
                score_text = f"sim-LPIPS\n{row['perturbations'][name]['sim_lpips']:.4f}"
            else:
                score_text = f"SGCS\n{row['perturbations'][name]['sgcs']:.4f}"
            draw_centered_multiline(
                draw,
                score_box,
                score_text,
                score_font,
            )
            x += score_width

        if row_idx > 0:
            draw.line(
                [(0, y0), (canvas_width, y0)],
                fill=(220, 220, 220),
                width=1,
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


def selected_metric(args: argparse.Namespace) -> str:
    if args.ssim and args.lpips:
        raise ValueError("--ssim and --lpips are mutually exclusive")
    if args.ssim:
        return "ssim"
    if args.lpips:
        return "sim_lpips"
    return "sgcs"


def summarize_video_results(
    results: list[dict[str, Any]],
    perturbations: list[dict[str, Any]],
    metric_name: str,
) -> tuple[dict[str, float | None], dict[str, int], dict[str, int]]:
    mean_score_by_perturbation = {}
    valid_frames_by_perturbation = {}
    skipped_frames_by_perturbation = {}
    for perturbation in perturbations:
        name = perturbation["name"]
        scores = [
            item["perturbations"][name][metric_name]
            for item in results
            if metric_name in item["perturbations"][name]
        ]
        valid_frames_by_perturbation[name] = len(scores)
        skipped_frames_by_perturbation[name] = len(results) - len(scores)
        mean_score_by_perturbation[name] = (
            float(np.mean(scores)) if scores else None
        )
    return (
        mean_score_by_perturbation,
        valid_frames_by_perturbation,
        skipped_frames_by_perturbation,
    )


def evaluate_video(
    args: argparse.Namespace,
    processor: Any | None,
    lpips_model: Any | None,
    categories: list[str],
    perturbations: list[dict[str, Any]],
    video_path: Path,
    vis_output: Path | None = None,
) -> dict[str, Any]:
    cap_a = open_video(video_path)
    count_a = get_frame_count(cap_a)
    if count_a <= 0:
        raise RuntimeError(f"Could not determine frame count from video: {video_path}")
    frame_count = count_a
    sampled_indices = list(range(0, frame_count, args.stride))
    results = []
    vis_rows = []
    needs_object_masks = any(item.get("object_aware", False) for item in perturbations)
    metric_name = selected_metric(args)
    desc = f"Evaluating {video_path.parent.name}/{video_path.name}"

    for frame_idx in tqdm(sampled_indices, desc=desc):
        gt_frame = read_frame_at(cap_a, frame_idx)
        gt_state = None
        object_masks = []
        if metric_name == "sgcs" or needs_object_masks:
            if processor is None:
                raise RuntimeError("SAM processor is required for this evaluation")
            gt_state = processor.set_image(frame_to_pil(gt_frame))
        if needs_object_masks:
            gt_height, gt_width = gt_frame.shape[:2]
            object_masks = collect_sam_object_masks(
                processor=processor,
                state=gt_state,
                categories=categories,
                height=gt_height,
                width=gt_width,
                min_area_frac=args.object_mask_min_area_frac,
                max_area_frac=args.object_mask_max_area_frac,
            )

        frame_result = {
            "frame_index": int(frame_idx),
            "video_a_frame_index": int(frame_idx),
            "video_b_frame_index": int(frame_idx),
            "num_object_candidates": len(object_masks),
            "perturbations": {},
        }
        vis_row = {
            "frame_index": int(frame_idx),
            "gt_frame": gt_frame,
            "perturbations": {},
        }

        for perturbation in perturbations:
            perturbation_metadata = {}
            if perturbation.get("object_aware", False):
                perturbed_frame, perturbation_metadata = perturbation["apply"](
                    gt_frame, object_masks
                )
            else:
                perturbed_frame = perturbation["apply"](gt_frame)
            name = perturbation["name"]
            if perturbation_metadata.get("status") not in (None, "ok"):
                skipped_result = {
                    "skipped": True,
                    "perturbation_metadata": perturbation_metadata,
                }
                frame_result["perturbations"][name] = skipped_result
                if vis_output is not None:
                    vis_row["perturbations"][name] = {
                        "frame": perturbed_frame,
                        **skipped_result,
                    }
                continue

            aligned_gt, aligned_perturbed, shape_info = align_frame_shapes(
                gt_frame, perturbed_frame
            )
            if metric_name == "ssim":
                score_result = {
                    "ssim": compute_ssim(aligned_gt, aligned_perturbed),
                    **shape_info,
                }
                if perturbation_metadata:
                    score_result["perturbation_metadata"] = perturbation_metadata
                frame_result["perturbations"][name] = score_result
                if vis_output is not None:
                    vis_row["gt_frame"] = aligned_gt
                    vis_row["perturbations"][name] = {
                        "frame": aligned_perturbed,
                        "ssim": score_result["ssim"],
                    }
                continue
            if metric_name == "sim_lpips":
                if lpips_model is None:
                    raise RuntimeError("LPIPS model is required for LPIPS evaluation")
                lpips_distance = compute_lpips_score(
                    aligned_gt, aligned_perturbed, lpips_model, args.device
                )
                score_result = {
                    "sim_lpips": 1.0 - lpips_distance,
                    "lpips_distance": lpips_distance,
                    **shape_info,
                }
                if perturbation_metadata:
                    score_result["perturbation_metadata"] = perturbation_metadata
                frame_result["perturbations"][name] = score_result
                if vis_output is not None:
                    vis_row["gt_frame"] = aligned_gt
                    vis_row["perturbations"][name] = {
                        "frame": aligned_perturbed,
                        "sim_lpips": score_result["sim_lpips"],
                    }
                continue

            if aligned_gt.shape[:2] == gt_frame.shape[:2]:
                state_a = gt_state
            else:
                state_a = processor.set_image(frame_to_pil(aligned_gt))
            aligned_height, aligned_width = aligned_gt.shape[:2]
            state_b = processor.set_image(frame_to_pil(aligned_perturbed))
            score_result = compute_sgcs_from_states(
                processor=processor,
                state_a=state_a,
                state_b=state_b,
                height=aligned_height,
                width=aligned_width,
                categories=categories,
                tau=args.tau,
                shape_info=shape_info,
            )
            if perturbation_metadata:
                score_result["perturbation_metadata"] = perturbation_metadata
            frame_result["perturbations"][name] = score_result
            if vis_output is not None:
                vis_row["gt_frame"] = aligned_gt
                vis_row["perturbations"][name] = {
                    "frame": aligned_perturbed,
                    "sgcs": score_result["sgcs"],
                }

        results.append(frame_result)
        if vis_output is not None:
            vis_rows.append(vis_row)

    cap_a.release()

    (
        mean_score_by_perturbation,
        valid_frames_by_perturbation,
        skipped_frames_by_perturbation,
    ) = summarize_video_results(results, perturbations, metric_name)
    mean_key = f"mean_{metric_name}_by_perturbation"
    output = {
        "video_a": str(video_path),
        "video_b": str(video_path),
        "metric": metric_name,
        "stride": args.stride,
        "tau": args.tau,
        "confidence_threshold": args.confidence_threshold,
        "categories": categories,
        "selected_perturbations": [item["name"] for item in perturbations],
        "perturbations": {
            item["name"]: {
                "label": item["label"],
                "params": item["params"],
                f"mean_{metric_name}": mean_score_by_perturbation[item["name"]],
                "num_valid_frames": valid_frames_by_perturbation[item["name"]],
                "num_skipped_frames": skipped_frames_by_perturbation[item["name"]],
            }
            for item in perturbations
        },
        "num_sampled_frames": len(results),
        mean_key: mean_score_by_perturbation,
        "valid_frames_by_perturbation": valid_frames_by_perturbation,
        "skipped_frames_by_perturbation": skipped_frames_by_perturbation,
        "frames": results,
    }
    if vis_output is not None:
        render_ablation_grid(vis_rows, perturbations, vis_output)
    return output


def first_avi_per_village(video_root: Path) -> list[tuple[str, Path]]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")
    videos = []
    for village_dir in sorted(path for path in video_root.iterdir() if path.is_dir()):
        avi_files = sorted(village_dir.glob("*.avi"))
        if not avi_files:
            continue
        videos.append((village_dir.name, avi_files[0]))
    if not videos:
        raise RuntimeError(f"No village .avi files found under: {video_root}")
    return videos


def suffix_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def summarize_batch(
    video_outputs: list[dict[str, Any]], perturbations: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = {}
    metric_name = video_outputs[0].get("metric", "sgcs") if video_outputs else "sgcs"
    mean_key = f"mean_{metric_name}_by_perturbation"
    for perturbation in perturbations:
        name = perturbation["name"]
        values = [
            item[mean_key][name]
            for item in video_outputs
            if item[mean_key][name] is not None
        ]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "num_valid_villages": len(values),
            "num_skipped_villages": len(video_outputs) - len(values),
            f"per_village_mean_{metric_name}": {
                item["village"]: item[mean_key][name]
                for item in video_outputs
            },
        }
    return summary


def print_video_summary(output: dict[str, Any]) -> None:
    metric_name = output.get("metric", "sgcs")
    mean_key = f"mean_{metric_name}_by_perturbation"
    for name, mean_score in output[mean_key].items():
        valid_count = output["valid_frames_by_perturbation"][name]
        skipped_count = output["skipped_frames_by_perturbation"][name]
        if mean_score is None:
            print(
                f"{name}_mean_{metric_name}: skipped_all_frames "
                f"(valid={valid_count}, skipped={skipped_count})"
            )
        else:
            print(
                f"{name}_mean_{metric_name}: {mean_score:.6f} "
                f"(valid={valid_count}, skipped={skipped_count})"
            )
    print(f"num_sampled_frames: {output['num_sampled_frames']}")


def main() -> None:
    args = parse_args()
    args.video_b = args.video_a
    metric_name = selected_metric(args)
    if args.output is None:
        args.output = Path(f"{metric_name}_results.json")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")

    categories = list(dict.fromkeys(args.categories))
    perturbations = build_perturbations(args)
    needs_object_masks = any(item.get("object_aware", False) for item in perturbations)
    processor = (
        build_processor(args)
        if (metric_name == "sgcs" or needs_object_masks)
        else None
    )
    lpips_model = build_lpips_model(args.device) if metric_name == "sim_lpips" else None

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device == "cuda" 
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        if args.video_root is None:
            output = evaluate_video(
                args=args,
                processor=processor,
                lpips_model=lpips_model,
                categories=categories,
                perturbations=perturbations,
                video_path=args.video_a,
                vis_output=args.vis_output,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print_video_summary(output)
            print(f"wrote: {args.output}")
            if args.vis_output is not None:
                print(f"wrote visualization: {args.vis_output}")
            return

        village_videos = first_avi_per_village(args.video_root)
        video_outputs = []
        for village, video_path in village_videos:
            vis_output = (
                suffix_path(args.vis_output, village)
                if args.vis_output is not None
                else None
            )
            village_output = evaluate_video(
                args=args,
                processor=processor,
                lpips_model=lpips_model,
                categories=categories,
                perturbations=perturbations,
                video_path=video_path,
                vis_output=vis_output,
            )
            village_output["village"] = village
            if vis_output is not None:
                village_output["vis_output"] = str(vis_output)
            video_outputs.append(village_output)
            print(f"\n[{village}] {video_path}")
            print_video_summary(village_output)

    batch_summary = summarize_batch(video_outputs, perturbations)
    output = {
        "video_root": str(args.video_root),
        "metric": metric_name,
        "stride": args.stride,
        "tau": args.tau,
        "confidence_threshold": args.confidence_threshold,
        "categories": categories,
        "selected_perturbations": [item["name"] for item in perturbations],
        "num_villages": len(video_outputs),
        "batch_summary": batch_summary,
        "videos": video_outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nBatch summary across villages:")
    for name, stats in batch_summary.items():
        if stats["mean"] is None:
            print(
                f"{name}: skipped_all_villages "
                f"(valid_villages={stats['num_valid_villages']})"
            )
        else:
            print(
                f"{name}: mean_{metric_name}={stats['mean']:.6f}, "
                f"std_{metric_name}={stats['std']:.6f} "
                f"(valid_villages={stats['num_valid_villages']})"
            )
    print(f"wrote: {args.output}")
    if args.vis_output is not None:
        print(
            "wrote visualizations with suffixes like: "
            f"{suffix_path(args.vis_output, village_videos[0][0])}"
        )


if __name__ == "__main__":
    main()
