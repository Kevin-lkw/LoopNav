#!/usr/bin/env python3
"""Launch the isolated WorldMem LoopNav P1 fine-tune."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from loopnav_finetune_v2.data import CorrectedLoopNavTrainingDataset
from loopnav_finetune_v2.model import AdapterCheckpoint, build_finetune_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        help=(
            "Lightning checkpoint to resume from. This restores the global step, "
            "optimizer state, and warmup state; it is not an adapter-only load."
        ),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--accumulate-grad-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config.resolve())
    if args.max_steps is not None:
        cfg.run.max_steps = int(args.max_steps)
    if args.num_workers is not None:
        cfg.trainer.num_workers = int(args.num_workers)
    if args.accumulate_grad_batches is not None:
        if args.accumulate_grad_batches <= 0:
            raise ValueError("--accumulate-grad-batches must be positive")
        cfg.trainer.accumulate_grad_batches = int(args.accumulate_grad_batches)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_path.resolve() if args.ckpt_path is not None else None
    if ckpt_path is not None and not ckpt_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {ckpt_path}")
    pl.seed_everything(int(cfg.run.seed), workers=True)
    torch.set_float32_matmul_precision("high")

    dataset = CorrectedLoopNavTrainingDataset(
        data_root=cfg.data.root,
        tasks=cfg.data.tasks,
        village_ids=cfg.data.train_village_ids,
        local_frames=cfg.data.local_frames,
        memory_frames=cfg.data.memory_frames,
        source_stride=cfg.data.source_stride,
        frame_height=cfg.data.frame_height,
        frame_width=cfg.data.frame_width,
        return_focused_probability=cfg.data.return_focused_probability,
        memory_retrieval_dropout=cfg.data.memory_retrieval_dropout,
        virtual_epoch_size=cfg.data.virtual_epoch_size,
        camera_clip=cfg.data.camera_clip,
        seed=cfg.run.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.trainer.per_device_batch_size),
        shuffle=True,
        num_workers=int(cfg.trainer.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.trainer.num_workers) > 0,
        drop_last=True,
    )

    model, parameter_counts = build_finetune_model(cfg)
    devices = torch.cuda.device_count()
    if devices <= 0:
        raise RuntimeError("Fine-tuning requires CUDA")

    manifest = {
        "format": "worldmem_loopnav_finetune_v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "config_path": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "resume_checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
        "effective_global_batch": (
            devices
            * int(cfg.trainer.per_device_batch_size)
            * int(cfg.trainer.accumulate_grad_batches)
        ),
        "inventory": dataset.inventory_counts(),
        "parameter_counts": parameter_counts,
        "official_repo_id": str(cfg.official.repo_id),
        "official_revision": str(cfg.official.revision),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        AdapterCheckpoint(
            output_dir / "adapters",
            every_n_train_steps=int(cfg.run.checkpoint_every),
            metadata={
                "official_repo_id": str(cfg.official.repo_id),
                "official_revision": str(cfg.official.revision),
                "parameter_counts": parameter_counts,
            },
        ),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="step{step}",
            every_n_train_steps=int(cfg.run.full_checkpoint_every),
            save_top_k=-1,
            save_last=True,
            enable_version_counter=False,
        ),
    ]
    strategy = (
        DDPStrategy(find_unused_parameters=False)
        if devices > 1
        else "auto"
    )
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        logger=CSVLogger(save_dir=output_dir / "logs", name="csv"),
        callbacks=callbacks,
        precision=str(cfg.trainer.precision),
        max_steps=int(cfg.run.max_steps),
        max_epochs=-1,
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        log_every_n_steps=int(cfg.run.log_every),
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        deterministic=False,
    )
    trainer.fit(
        model,
        train_dataloaders=loader,
        ckpt_path=str(ckpt_path) if ckpt_path is not None else None,
    )


if __name__ == "__main__":
    main()
