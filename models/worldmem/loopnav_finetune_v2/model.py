"""Official-checkpoint WorldMem with isolated LoopNav adapters."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from omegaconf import DictConfig

import algorithms.worldmem.df_video as df_video
from worldmem_official_runtime import (
    _load_component_strict,
    load_official_config,
    validate_official_release,
)


class LoRALinear(torch.nn.Module):
    """Frozen linear layer plus a zero-initialized low-rank update."""

    def __init__(
        self,
        base: torch.nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = float(alpha) / float(rank)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.lora_B(self.lora_A(self.dropout(value))) * self.scale


class LoopNavAdapterModel(df_video.WorldMemMinecraft):
    def __init__(self, cfg: DictConfig, optimizer_cfg: DictConfig):
        self.ft_optimizer_cfg = optimizer_cfg
        super().__init__(cfg)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    def configure_optimizers(self):
        action_lora: list[torch.nn.Parameter] = []
        pose: list[torch.nn.Parameter] = []
        for name, parameter in self.diffusion_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "pose_embedder" in name or "pose_cond_mlp" in name:
                pose.append(parameter)
            else:
                action_lora.append(parameter)
        if not action_lora or not pose:
            raise RuntimeError(
                f"Unexpected trainable groups: action_lora={len(action_lora)}, pose={len(pose)}"
            )
        groups = [
            {
                "params": action_lora,
                "lr": float(self.ft_optimizer_cfg.lora_and_action_lr),
                "target_lr": float(self.ft_optimizer_cfg.lora_and_action_lr),
                "name": "lora_and_action",
            },
            {
                "params": pose,
                "lr": float(self.ft_optimizer_cfg.pose_lr),
                "target_lr": float(self.ft_optimizer_cfg.pose_lr),
                "name": "pose",
            },
        ]
        return torch.optim.AdamW(
            groups,
            weight_decay=float(self.ft_optimizer_cfg.weight_decay),
            betas=tuple(float(value) for value in self.ft_optimizer_cfg.betas),
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        warmup_steps = int(self.ft_optimizer_cfg.warmup_steps)
        if self.trainer.global_step < warmup_steps:
            scale = min(1.0, float(self.trainer.global_step + 1) / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = scale * float(group["target_lr"])

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable
        }


def _enable_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def install_adapters(model: LoopNavAdapterModel, adapter_cfg: DictConfig) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    diffusion = model.diffusion_model.model
    if bool(adapter_cfg.tune_external_cond):
        _enable_module(diffusion.external_cond)
    if bool(adapter_cfg.tune_pose_modules):
        _enable_module(diffusion.pose_embedder)
        for block in diffusion.blocks:
            _enable_module(block.pose_cond_mlp)

    requested = set(str(value) for value in adapter_cfg.branches)
    valid = {"spatial", "temporal", "memory"}
    if not requested <= valid:
        raise ValueError(f"Unknown LoRA branches: {sorted(requested - valid)}")

    installed = 0
    for block in diffusion.blocks:
        branches: Iterable[tuple[str, torch.nn.Module]] = (
            ("spatial", block.s_attn),
            ("temporal", block.t_attn),
            ("memory", block.r_attn),
        )
        for branch, attention in branches:
            if branch not in requested:
                continue
            if not isinstance(attention.to_qkv, torch.nn.Linear):
                raise TypeError(f"{branch} to_qkv is already wrapped: {type(attention.to_qkv)}")
            attention.to_qkv = LoRALinear(
                attention.to_qkv,
                rank=int(adapter_cfg.rank),
                alpha=float(adapter_cfg.alpha),
                dropout=float(adapter_cfg.dropout),
            )
            installed += 1

    counts = {
        "lora_modules": installed,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    counts["frozen_parameters"] = counts["total_parameters"] - counts["trainable_parameters"]
    return counts


def build_finetune_model(config: DictConfig) -> tuple[LoopNavAdapterModel, dict[str, int]]:
    release_paths, _manifest = validate_official_release(
        config.official.release_dir,
        verify_sha256=False,
    )
    official_cfg = load_official_config(config.official.config)
    official_cfg.lr = float(config.optimizer.lora_and_action_lr)
    official_cfg.warmup_steps = int(config.optimizer.warmup_steps)
    official_cfg.log_video = False

    df_video.LearnedPerceptualImagePatchSimilarity = lambda: torch.nn.Identity()
    model = LoopNavAdapterModel(official_cfg, config.optimizer)
    _load_component_strict(model.diffusion_model, release_paths.diffusion, "diffusion")
    _load_component_strict(model.vae, release_paths.vae, "VAE")
    counts = install_adapters(model, config.adapter)
    model.vae.eval()
    return model, counts


class AdapterCheckpoint(Callback):
    def __init__(
        self,
        output_dir: str | Path,
        every_n_train_steps: int,
        metadata: dict[str, Any],
    ):
        self.output_dir = Path(output_dir)
        self.every_n_train_steps = int(every_n_train_steps)
        self.metadata = metadata
        self._last_saved_step = -1

    def _save(self, trainer: Trainer, module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        step = int(trainer.global_step)
        if step == self._last_saved_step:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "worldmem_loopnav_adapter_v2",
            "global_step": step,
            "metadata": self.metadata,
            "state_dict": module.trainable_state_dict(),
        }
        target = self.output_dir / f"adapter_step{step:07d}.pt"
        with tempfile.NamedTemporaryFile(
            dir=self.output_dir,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        latest = self.output_dir / "latest.json"
        latest.write_text(
            json.dumps(
                {
                    "global_step": step,
                    "adapter": str(target.resolve()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._last_saved_step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step > 0 and step % self.every_n_train_steps == 0:
            self._save(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._save(trainer, pl_module)
