"""Runtime helpers for the authors' released WorldMem checkpoint.

The official release stores the diffusion model and VAE in separate Lightning
checkpoint files.  This module loads those components strictly and implements
the authors' validation rollout without encoding or decoding target GT frames.
Keeping the latent history on CPU makes long LoopNav trajectories practical on
24 GB GPUs while preserving WorldMem's 8-token local window and 8 retrieved
memory frames.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


OFFICIAL_GITHUB_COMMIT = "c1cc91bdfb2dda52fe309280e5ed046b1faba448"
OFFICIAL_HF_REPO = "zeqixiao/worldmem_checkpoints"
OFFICIAL_HF_REVISION = "a3ae4d1c3a89260f800ddeab01ca56b7e63b10e2"
OFFICIAL_DIFFUSION_FILE = "diffusion_only.ckpt"
OFFICIAL_VAE_FILE = "vae_only.ckpt"
OFFICIAL_CONFIG_SHA256 = (
    "13e9750ee2c1ff89b65902e85c3bf8defc0e43667893bfe8b361cff1426ec2fa"
)
OFFICIAL_RELEASE_FILES = {
    OFFICIAL_DIFFUSION_FILE: {
        "size": 3_780_882_509,
        "sha256": "b63155939ae584e9819254c90754d3b39c5b3f08297525fa0eae9ae5f8f37f70",
    },
    OFFICIAL_VAE_FILE: {
        "size": 917_053_138,
        "sha256": "82c285609570d0a9a99815f73c1f0f44ba4bbf52460d3f7319bae03a675c947b",
    },
}


@dataclass(frozen=True)
class ReleasePaths:
    diffusion: Path
    vae: Path
    manifest: Path


@dataclass(frozen=True)
class RolloutStats:
    context_frames: int
    prediction_frames: int
    encode_seconds: float
    sample_decode_seconds: float
    peak_cuda_bytes: int


def _sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_release(
    release_dir: str | Path,
    *,
    verify_sha256: bool = False,
) -> tuple[ReleasePaths, Mapping[str, object]]:
    """Validate the pinned official release manifest and component files."""

    root = Path(release_dir).expanduser().resolve()
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Official release manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read official release manifest {manifest_path}: {exc}") from exc

    if manifest.get("repo_id") != OFFICIAL_HF_REPO:
        raise ValueError(
            f"Release manifest repo_id={manifest.get('repo_id')!r}; expected {OFFICIAL_HF_REPO!r}"
        )
    if manifest.get("revision") != OFFICIAL_HF_REVISION:
        raise ValueError(
            f"Release manifest revision={manifest.get('revision')!r}; "
            f"expected {OFFICIAL_HF_REVISION!r}"
        )

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"Release manifest has no files mapping: {manifest_path}")

    resolved: dict[str, Path] = {}
    for filename, official in OFFICIAL_RELEASE_FILES.items():
        entry = files.get(filename)
        if not isinstance(entry, Mapping):
            raise ValueError(f"Release manifest has no entry for {filename}")
        if (
            int(entry.get("size", -1)) != official["size"]
            or str(entry.get("sha256", "")).lower() != official["sha256"]
        ):
            raise ValueError(
                f"Release manifest identity mismatch for {filename}; expected "
                f"size={official['size']} sha256={official['sha256']}"
            )
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Official release file is missing: {path}")
        expected_size = int(official["size"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Official release size mismatch for {path}: {actual_size} != {expected_size}"
            )
        if verify_sha256:
            actual_sha = _sha256(path)
            expected_sha = str(official["sha256"])
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Official release SHA256 mismatch for {path}: {actual_sha} != {expected_sha}"
                )
        resolved[filename] = path

    return (
        ReleasePaths(
            diffusion=resolved[OFFICIAL_DIFFUSION_FILE],
            vae=resolved[OFFICIAL_VAE_FILE],
            manifest=manifest_path,
        ),
        manifest,
    )


def load_official_config(
    config_path: str | Path,
    *,
    sampling_timesteps: int | None = None,
) -> DictConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Official WorldMem evaluation config is missing: {path}")
    actual_sha = _sha256(path)
    if actual_sha != OFFICIAL_CONFIG_SHA256:
        raise ValueError(
            f"Official WorldMem config SHA256 mismatch for {path}: "
            f"{actual_sha} != {OFFICIAL_CONFIG_SHA256}"
        )
    cfg = OmegaConf.load(path)
    if sampling_timesteps is not None:
        if sampling_timesteps <= 0:
            raise ValueError("sampling_timesteps must be positive")
        cfg.diffusion.sampling_timesteps = int(sampling_timesteps)

    expected = {
        "action_cond_dim": 25,
        "pose_cond_dim": 5,
        "frame_stack": 1,
        "n_tokens": 8,
        "memory_condition_length": 8,
        "chunk_size": 1,
    }
    mismatches = {
        key: (cfg.get(key), value)
        for key, value in expected.items()
        if int(cfg.get(key)) != value
    }
    if mismatches:
        raise ValueError(f"Official WorldMem config invariant mismatch: {mismatches}")
    if not bool(cfg.use_plucker) or not bool(cfg.relative_embedding):
        raise ValueError("Official WorldMem release requires relative Plucker pose conditioning")
    if not bool(cfg.add_timestamp_embedding):
        raise ValueError("Official WorldMem release requires timestamp embeddings")
    if float(cfg.focal_length) != 0.35 or str(cfg.ref_mode) != "sequential":
        raise ValueError(
            "Official WorldMem release requires focal_length=0.35 and ref_mode=sequential"
        )
    return cfg


def _load_component_strict(module: torch.nn.Module, checkpoint_path: Path, label: str) -> None:
    print(f"[official:model] loading {label}: {checkpoint_path}", flush=True)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError(f"Official {label} checkpoint has no state_dict: {checkpoint_path}")
    state_dict = checkpoint["state_dict"]
    try:
        incompatible = module.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        incompatible = module.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Official {label} checkpoint mismatch; missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state_dict, checkpoint
    gc.collect()


def load_official_model(
    release_paths: ReleasePaths,
    cfg: DictConfig,
    device: torch.device,
):
    """Construct WorldMem and strictly load only the two official components."""

    if "outputs" in release_paths.diffusion.parts or "outputs" in release_paths.vae.parts:
        raise ValueError("Refusing an outputs/... checkpoint for official-release evaluation")

    torch.set_float32_matmul_precision("high")

    # LPIPS is unrelated to generation.  Avoid constructing or downloading its
    # backbone; component checkpoints do not contain LPIPS weights.
    import algorithms.worldmem.df_video as df_video

    df_video.LearnedPerceptualImagePatchSimilarity = lambda: torch.nn.Identity()
    print("[official:model] constructing WorldMemMinecraft", flush=True)
    model = df_video.WorldMemMinecraft(cfg)
    _load_component_strict(model.diffusion_model, release_paths.diffusion, "diffusion")
    _load_component_strict(model.vae, release_paths.vae, "VAE")

    action_weight = model.diffusion_model.model.external_cond.weight
    if tuple(action_weight.shape) != (1024, 25):
        raise RuntimeError(
            f"Official checkpoint action layer has shape {tuple(action_weight.shape)}, expected (1024, 25)"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"[official:model] moving FP32 release weights to {device}", flush=True)
    model = model.to(device=device).eval()
    return model


def autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def amp_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        device = torch.device(spec)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {spec}")
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("Official WorldMem inference requires CUDA")
    free_memory = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, _ = torch.cuda.mem_get_info()
        free_memory.append((free_bytes, index))
    return torch.device(f"cuda:{max(free_memory)[1]}")


def _encode_context(
    model,
    context_video: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    if context_video.ndim != 4 or context_video.shape[1] != 3:
        raise ValueError(f"context_video must have shape T,C,H,W, got {tuple(context_video.shape)}")
    if batch_size <= 0:
        raise ValueError("VAE encode batch size must be positive")

    latents = []
    for start in range(0, len(context_video), batch_size):
        frames = context_video[start : start + batch_size].unsqueeze(1).to(device)
        with autocast_context(device, dtype):
            encoded = model.encode(frames)
        latents.append(encoded.detach().cpu())
        del frames, encoded
    return torch.cat(latents, dim=0)


def _decode_latents(
    model,
    latents: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with autocast_context(device, dtype):
        decoded = model.decode(latents.to(device))
    return decoded[:, 0].detach().float().cpu()


def rollout_official_streaming(
    model,
    *,
    context_video: torch.Tensor,
    actions: torch.Tensor,
    poses: torch.Tensor,
    timestamps: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    vae_batch_size: int,
    frame_sink: Callable[[torch.Tensor], None],
    progress_description: str | None = None,
) -> RolloutStats:
    """Generate all post-context frames and stream decoded RGB batches.

    The sampling loop intentionally follows the authors' current
    ``WorldMemMinecraft.validation_step`` ordering: CPU noise creation, local
    8-token sliding window, official FOV memory selection, 20 denoising steps,
    and removal of the 8 temporary memory tokens after every generated frame.
    """

    if actions.ndim != 2 or actions.shape[1] != 25:
        raise ValueError(f"actions must have shape T,25, got {tuple(actions.shape)}")
    if poses.ndim != 2 or poses.shape[1] != 5:
        raise ValueError(f"poses must have shape T,5, got {tuple(poses.shape)}")
    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must have shape T, got {tuple(timestamps.shape)}")
    total_frames = len(actions)
    context_frames = len(context_video)
    if len(poses) != total_frames or len(timestamps) != total_frames:
        raise ValueError(
            f"Condition length mismatch: actions={len(actions)}, poses={len(poses)}, "
            f"timestamps={len(timestamps)}"
        )
    if not 0 < context_frames < total_frames:
        raise ValueError(
            f"Context must be non-empty and leave prediction frames: {context_frames}/{total_frames}"
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    encode_started = time.monotonic()
    with torch.inference_mode():
        xs_pred = _encode_context(
            model,
            context_video,
            device=device,
            dtype=dtype,
            batch_size=vae_batch_size,
        )
    encode_seconds = time.monotonic() - encode_started

    # Match the official _preprocess_batch action offset exactly.
    conditions = torch.cat([torch.zeros_like(actions[:1]), actions[1:]], dim=0)
    conditions = conditions[:, None].contiguous().to(device)
    pose_conditions = poses[:, None].contiguous().to(device)
    frame_idx = timestamps[:, None].contiguous().to(device)

    from algorithms.worldmem.df_video import euler_to_camera_to_world_matrix

    # Lightning's official ``16-mixed`` test precision wraps the complete
    # validation_step, including pose-matrix and Plucker construction.
    with autocast_context(device, dtype):
        c2w_mat = euler_to_camera_to_world_matrix(pose_conditions)
    curr_frame = context_frames
    prediction_frames = total_frames - context_frames
    memory_length = int(model.memory_condition_length)
    pending_decode: list[torch.Tensor] = []
    sample_started = time.monotonic()

    progress = tqdm(
        total=prediction_frames,
        desc=progress_description or "WorldMem official",
        leave=False,
        dynamic_ncols=True,
    )
    try:
        with torch.inference_mode(), autocast_context(device, dtype):
            while curr_frame < total_frames:
                horizon = min(total_frames - curr_frame, int(model.chunk_size))
                if horizon <= 0 or horizon > int(model.n_tokens):
                    raise RuntimeError(
                        f"Invalid official rollout horizon {horizon} for n_tokens={model.n_tokens}"
                    )

                scheduling_matrix = model._generate_scheduling_matrix(horizon)
                chunk = torch.randn((horizon, 1, *xs_pred.shape[2:]))
                chunk = torch.clamp(chunk, -model.clip_noise, model.clip_noise)
                xs_pred = torch.cat([xs_pred, chunk.to(xs_pred.device)], dim=0)
                start_frame = max(0, curr_frame + horizon - int(model.n_tokens))

                random_idx = model._generate_condition_indices(
                    curr_frame,
                    memory_length,
                    xs_pred,
                    pose_conditions,
                    frame_idx,
                    horizon,
                )
                memory_latents = xs_pred[
                    random_idx[:, range(xs_pred.shape[1])],
                    range(xs_pred.shape[1]),
                ].clone()
                xs_pred = torch.cat([xs_pred, memory_latents], dim=0)

                input_condition, input_pose_condition, frame_idx_list = model._prepare_conditions(
                    start_frame,
                    curr_frame,
                    horizon,
                    conditions,
                    pose_conditions,
                    c2w_mat,
                    frame_idx,
                    random_idx,
                    image_width=int(context_video.shape[-1]),
                    image_height=int(context_video.shape[-2]),
                )

                for step in range(scheduling_matrix.shape[0] - 1):
                    from_noise, to_noise = model._prepare_noise_levels(
                        scheduling_matrix,
                        step,
                        curr_frame,
                        1,
                        memory_length,
                    )
                    sampled = model.diffusion_model.sample_step(
                        xs_pred[start_frame:].to(input_condition.device),
                        input_condition,
                        input_pose_condition,
                        from_noise[start_frame:],
                        to_noise[start_frame:],
                        current_frame=curr_frame,
                        mode="validation",
                        reference_length=memory_length,
                        frame_idx=frame_idx_list,
                    )
                    xs_pred[start_frame:] = sampled.detach().cpu()
                    del sampled, from_noise, to_noise

                xs_pred = xs_pred[:-memory_length]
                pending_decode.append(xs_pred[curr_frame : curr_frame + horizon].clone())
                curr_frame += horizon
                progress.update(horizon)

                if len(pending_decode) >= vae_batch_size or curr_frame == total_frames:
                    decoded = _decode_latents(
                        model,
                        torch.cat(pending_decode, dim=0),
                        device=device,
                        dtype=dtype,
                    )
                    frame_sink(decoded)
                    pending_decode.clear()
                    del decoded

                del (
                    chunk,
                    random_idx,
                    memory_latents,
                    input_condition,
                    input_pose_condition,
                    frame_idx_list,
                )
    finally:
        progress.close()

    elapsed = time.monotonic() - sample_started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return RolloutStats(
        context_frames=context_frames,
        prediction_frames=prediction_frames,
        encode_seconds=encode_seconds,
        sample_decode_seconds=elapsed,
        peak_cuda_bytes=int(peak),
    )


class AtomicPredictionWriter:
    """Incrementally write an MP4 and expose it only after full validation."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        fps: float,
        size: tuple[int, int],
        overwrite: bool,
    ) -> None:
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.width, self.height = size
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(self.output_path)
        self.temporary_path = self.output_path.with_name(
            f".{self.output_path.name}.{uuid.uuid4().hex}.tmp.mp4"
        )
        self.writer = cv2.VideoWriter(
            str(self.temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create prediction video: {self.temporary_path}")
        self.frames_written = 0
        self._closed = False

    def write(self, frames: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed prediction writer")
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Decoded frames must have shape T,C,H,W, got {tuple(frames.shape)}")
        if (int(frames.shape[-1]), int(frames.shape[-2])) != (self.width, self.height):
            raise ValueError(
                f"Decoded frame size is {frames.shape[-1]}x{frames.shape[-2]}, "
                f"expected {self.width}x{self.height}"
            )
        rgb = (
            frames.clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .numpy()
        )
        for frame in rgb:
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.frames_written += 1

    def commit(self, expected_frames: int) -> None:
        if self._closed:
            raise RuntimeError("Prediction writer is already closed")
        self.writer.release()
        self._closed = True
        capture = cv2.VideoCapture(str(self.temporary_path))
        if not capture.isOpened():
            self.temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not reopen prediction video: {self.temporary_path}")
        actual_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        actual_size = (
            int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        capture.release()
        if (
            self.frames_written != expected_frames
            or actual_frames != expected_frames
            or abs(actual_fps - self.fps) > 0.01
            or actual_size != (self.width, self.height)
        ):
            self.temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Generated-video validation failed: "
                f"written={self.frames_written}, decoded={actual_frames}, expected={expected_frames}, "
                f"fps={actual_fps}, size={actual_size}"
            )
        os.replace(self.temporary_path, self.output_path)

    def abort(self) -> None:
        if not self._closed:
            self.writer.release()
            self._closed = True
        self.temporary_path.unlink(missing_ok=True)
