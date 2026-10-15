#!/usr/bin/env python3
"""Fixed, lightweight held-out diffusion-loss validation for P1 adapters."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from loopnav_finetune_v2.data import CorrectedLoopNavTrainingDataset
from loopnav_finetune_v2.model import build_finetune_model


TASKS = ("TURN", "ABA_FIXED", "ABCA")
STEP_PATTERN = re.compile(r"adapter_step(\d+)\.pt$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("loopnav_finetune_v2/config_p1.yaml"),
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="Evaluate only these adapter steps instead of every discovered adapter.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _discover_adapters(directory: Path) -> list[tuple[int, Path]]:
    adapters = []
    for path in directory.glob("adapter_step*.pt"):
        match = STEP_PATTERN.fullmatch(path.name)
        if match:
            adapters.append((int(match.group(1)), path.resolve()))
    return sorted(adapters)


def _load_adapter(
    model: torch.nn.Module,
    path: Path,
    *,
    expected_revision: str,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "worldmem_loopnav_adapter_v2":
        raise ValueError(f"Unexpected adapter format in {path}")
    step = int(payload.get("global_step", -1))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Adapter metadata is missing in {path}")
    if metadata.get("official_revision") != expected_revision:
        raise ValueError(
            f"Adapter revision mismatch in {path}: "
            f"{metadata.get('official_revision')} != {expected_revision}"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Adapter state_dict is missing in {path}")

    expected = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    actual = set(state)
    if actual != expected:
        raise ValueError(
            f"Adapter key mismatch in {path}: "
            f"missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )
    model_state = model.state_dict()
    for name, tensor in state.items():
        if name not in model_state:
            raise ValueError(f"Adapter key {name} is absent from model")
        if tuple(tensor.shape) != tuple(model_state[name].shape):
            raise ValueError(
                f"Adapter shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {tuple(model_state[name].shape)}"
            )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected adapter keys in {path}: {incompatible.unexpected_keys}"
        )
    return step


def _batch_to_device(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.unsqueeze(0).to(device, non_blocking=True) for tensor in batch)


def _autocast(device: torch.device, name: str):
    if device.type != "cuda" or name == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if name == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def main() -> None:
    args = _parse_args()
    if args.samples_per_task <= 0:
        raise ValueError("--samples-per-task must be positive")
    cfg = OmegaConf.load(args.config.resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {device}")

    adapters = _discover_adapters(args.adapter_dir.resolve())
    if args.steps:
        requested_steps = set(args.steps)
        adapters = [
            (step, path)
            for step, path in adapters
            if step in requested_steps
        ]
        found_steps = {step for step, _path in adapters}
        missing_steps = requested_steps - found_steps
        if missing_steps:
            raise FileNotFoundError(
                f"Requested adapter steps are missing under {args.adapter_dir}: "
                f"{sorted(missing_steps)}"
            )
    if not adapters and not args.include_base:
        raise FileNotFoundError(f"No adapters found under {args.adapter_dir}")

    validation_samples = args.samples_per_task * len(TASKS)
    dataset = CorrectedLoopNavTrainingDataset(
        data_root=cfg.data.root,
        tasks=TASKS,
        village_ids=(17, 18),
        local_frames=cfg.data.local_frames,
        memory_frames=cfg.data.memory_frames,
        source_stride=cfg.data.source_stride,
        frame_height=cfg.data.frame_height,
        frame_width=cfg.data.frame_width,
        return_focused_probability=cfg.data.return_focused_probability,
        memory_retrieval_dropout=0.0,
        virtual_epoch_size=validation_samples,
        camera_clip=cfg.data.camera_clip,
        seed=cfg.run.seed + 170018,
    )
    expected_inventory = {"TURN": 240, "ABA_FIXED": 960, "ABCA": 960}
    if dataset.inventory_counts() != expected_inventory:
        raise RuntimeError(
            f"Held-out inventory mismatch: {dataset.inventory_counts()} "
            f"!= {expected_inventory}"
        )

    sample_manifest = []
    for index in range(validation_samples):
        _batch, debug = dataset.build_sample(index)
        sample_manifest.append(
            {
                "index": index,
                "task": debug.task,
                "metadata_path": str(debug.metadata_path),
                "local_indices": list(debug.local_indices),
                "memory_indices": list(debug.memory_indices),
                "observed_stop": debug.observed_stop,
            }
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "inventory": dataset.inventory_counts(),
                    "samples_per_task": args.samples_per_task,
                    "samples": sample_manifest,
                    "adapters": [
                        {"step": step, "path": str(path)}
                        for step, path in adapters
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        return

    print("[validation] constructing official-initialized adapter model", flush=True)
    model, parameter_counts = build_finetune_model(cfg)
    model = model.to(device=device).eval()
    checkpoints: list[tuple[int, Path | None]] = []
    if args.include_base:
        checkpoints.append((0, None))
    checkpoints.extend(adapters)

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for expected_step, adapter_path in checkpoints:
        if adapter_path is not None:
            loaded_step = _load_adapter(
                model,
                adapter_path,
                expected_revision=str(cfg.official.revision),
            )
            if loaded_step != expected_step:
                raise ValueError(
                    f"Adapter filename/payload step mismatch: "
                    f"{expected_step} != {loaded_step}"
                )
        model.eval()
        by_task: dict[str, list[float]] = defaultdict(list)
        all_losses = []
        print(
            f"[validation] checkpoint step={expected_step} "
            f"path={adapter_path or 'official_base'}",
            flush=True,
        )
        for index in range(validation_samples):
            batch, debug = dataset.build_sample(index)
            batch_device = _batch_to_device(batch, device)
            sample_seed = int(cfg.run.seed) + 9_000_000 + index
            _seed_everything(sample_seed)
            with torch.inference_mode(), _autocast(device, args.amp_dtype):
                output = model.training_step(batch_device, batch_idx=1)
                loss = float(output["loss"].detach().float().cpu())
            if not math.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at step={expected_step}, "
                    f"sample={index}: {loss}"
                )
            by_task[debug.task].append(loss)
            all_losses.append(loss)
            rows.append(
                {
                    "checkpoint_step": expected_step,
                    "adapter_path": str(adapter_path or "official_base"),
                    "sample_index": index,
                    "task": debug.task,
                    "metadata_path": str(debug.metadata_path),
                    "seed": sample_seed,
                    "loss": loss,
                }
            )
            del batch, batch_device, output

        summaries[str(expected_step)] = {
            "adapter_path": str(adapter_path or "official_base"),
            "overall_mean_loss": fmean(all_losses),
            "task_mean_loss": {
                task: fmean(by_task[task])
                for task in TASKS
            },
            "samples_per_task": args.samples_per_task,
        }
        print(json.dumps(summaries[str(expected_step)], indent=2), flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "format": "worldmem_loopnav_adapter_fixed_validation_v1",
        "data_root": str(Path(cfg.data.root).resolve()),
        "validation_village_ids": [17, 18],
        "inventory": dataset.inventory_counts(),
        "sample_seed": int(cfg.run.seed) + 170018,
        "noise_seed_rule": f"{int(cfg.run.seed)} + 9000000 + sample_index",
        "samples": sample_manifest,
        "parameter_counts": parameter_counts,
        "amp_dtype": args.amp_dtype,
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[validation] saved {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
