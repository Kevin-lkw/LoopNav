"""Evaluate Oasis-500M on the TURN split.

The first frame marked ``return_point`` is the first prediction target.  Ground
truth frames at and after that point are never passed to the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import load_model
from torch import autocast
from torchvision.io import read_video, write_video
from tqdm import tqdm

from dit import DiT_models
from utils import process_json_actions, sigmoid_beta_schedule
from vae import VAE_models


DEFAULT_DATA_ROOT = Path("data/LoopNav/TURN")
DEFAULT_VILLAGE_IDS = (18, 19, 20)


def parse_village_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def find_return_point(records: list[dict]) -> int:
    points = [i for i, record in enumerate(records) if record.get("return_point") is True]
    if len(points) != 1:
        raise ValueError(f"expected exactly one return_point marker, found {points}")
    return points[0]


def discover_samples(data_root: Path, village_ids: tuple[int, ...]) -> list[tuple[Path, Path]]:
    allowed_suffixes = tuple(f"_{village_id}" for village_id in village_ids)
    samples = []
    for video_path in sorted(data_root.glob("*/*/*.avi")):
        if not video_path.parent.parent.name.endswith(allowed_suffixes):
            continue
        action_path = video_path.with_suffix(".json")
        if not action_path.is_file():
            raise FileNotFoundError(f"missing action file for {video_path}: {action_path}")
        samples.append((video_path, action_path))
    return samples


def output_path_for(video_path: Path, data_root: Path, output_root: Path) -> Path:
    village, angle_bucket, filename = video_path.relative_to(data_root).parts
    return output_root / "TURN" / angle_bucket / village / Path(filename).with_suffix(".mp4")


def left_pad_context(tensor: torch.Tensor, target_length: int, value: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] > target_length:
        return tensor[-target_length:]
    pad_length = target_length - tensor.shape[0]
    if pad_length == 0:
        return tensor
    return torch.cat((value.expand(pad_length, *tensor.shape[1:]), tensor), dim=0)


class OasisTurnEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Oasis TURN evaluation requires CUDA")
        self.device = torch.device(args.device)
        self.context_frames = args.context_frames
        self.ddim_steps = args.ddim_steps

        self.model = DiT_models["DiT-S/2"]()
        load_model(self.model, str(args.oasis_ckpt))
        self.model = self.model.to(self.device).eval()

        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        load_model(self.vae, str(args.vae_ckpt))
        self.vae = self.vae.to(self.device).eval()

        self.max_noise_level = 1000
        self.noise_abs_max = 20
        self.stabilization_level = 15
        self.scaling_factor = 0.07843137255
        betas = sigmoid_beta_schedule(self.max_noise_level).float().to(self.device)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.alphas_cumprod = rearrange(alphas_cumprod, "t -> t 1 1 1")
        self.noise_range = torch.linspace(
            -1, self.max_noise_level - 1, self.ddim_steps + 1, device=self.device
        )

    @torch.no_grad()
    def generate(self, video_path: Path, action_path: Path, seed: int) -> tuple[torch.Tensor, float, int]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        records = json.loads(action_path.read_text())
        return_point = find_return_point(records)
        if return_point <= 0:
            raise ValueError(f"return point must have preceding context: {action_path}")

        video, _, info = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
        if video.shape[0] != len(records):
            raise ValueError(
                f"frame/action length mismatch for {video_path}: {video.shape[0]} vs {len(records)}"
            )
        fps = float(info.get("video_fps", 20.0))
        actions = process_json_actions(records)

        context = video[max(0, return_point - self.context_frames) : return_point].float().div(255)
        context = left_pad_context(context, self.context_frames, context[:1])
        context_actions = actions[max(0, return_point - self.context_frames) : return_point]
        zero_action = torch.zeros_like(actions[:1])
        context_actions = left_pad_context(context_actions, self.context_frames, zero_action)
        rollout_actions = torch.cat((context_actions, actions[return_point:]), dim=0).unsqueeze(0)

        x = context.unsqueeze(0).to(self.device)
        rollout_actions = rollout_actions.to(self.device)
        batch_size, _, _, height, width = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with autocast("cuda", dtype=torch.float16):
            x = self.vae.encode(x * 2 - 1).mean * self.scaling_factor
        x = rearrange(
            x,
            "(b t) (h w) c -> b t c h w",
            b=batch_size,
            t=self.context_frames,
            h=height // self.vae.patch_size,
            w=width // self.vae.patch_size,
        )

        total_frames = rollout_actions.shape[1]
        for frame_index in tqdm(
            range(self.context_frames, total_frames),
            desc=video_path.stem,
            leave=False,
        ):
            chunk = torch.randn((batch_size, 1, *x.shape[-3:]), device=self.device)
            x = torch.cat((x, chunk.clamp(-self.noise_abs_max, self.noise_abs_max)), dim=1)
            start_frame = max(0, frame_index + 1 - self.model.max_frames)

            for noise_index in reversed(range(1, self.ddim_steps + 1)):
                t_context = torch.full(
                    (batch_size, frame_index),
                    self.stabilization_level - 1,
                    dtype=torch.long,
                    device=self.device,
                )
                t = torch.full(
                    (batch_size, 1),
                    self.noise_range[noise_index],
                    dtype=torch.long,
                    device=self.device,
                )
                t_next = torch.full(
                    (batch_size, 1),
                    self.noise_range[noise_index - 1],
                    dtype=torch.long,
                    device=self.device,
                )
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat((t_context, t), dim=1)[:, start_frame:]
                t_next = torch.cat((t_context, t_next), dim=1)[:, start_frame:]
                x_current = x[:, start_frame:].clone()

                with autocast("cuda", dtype=torch.float16):
                    velocity = self.model(
                        x_current,
                        t,
                        rollout_actions[:, start_frame : frame_index + 1],
                    )
                alpha = self.alphas_cumprod[t]
                x_start = alpha.sqrt() * x_current - (1 - alpha).sqrt() * velocity
                x_noise = ((1 / alpha).sqrt() * x_current - x_start) / (1 / alpha - 1).sqrt()
                alpha_next = self.alphas_cumprod[t_next]
                alpha_next[:, :-1] = 1
                if noise_index == 1:
                    alpha_next[:, -1:] = 1
                prediction = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = prediction[:, -1:]

        generated = x[:, self.context_frames :]
        generated = rearrange(generated, "b t c h w -> (b t) (h w) c")
        decoded = []
        for start in range(0, generated.shape[0], 16):
            decoded.append(self.vae.decode(generated[start : start + 16] / self.scaling_factor))
        generated = torch.cat(decoded).add(1).div(2).clamp(0, 1)
        generated = rearrange(generated, "(b t) c h w -> b t h w c", b=batch_size)
        return generated[0].mul(255).byte().cpu(), fps, return_point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Oasis-500M on TURN")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("output_turn"))
    parser.add_argument("--oasis-ckpt", type=Path, default=Path("oasis500m.safetensors"))
    parser.add_argument("--vae-ckpt", type=Path, default=Path("vit-l-20.safetensors"))
    parser.add_argument("--village-ids", type=parse_village_ids, default=DEFAULT_VILLAGE_IDS)
    parser.add_argument("--context-frames", type=int, default=32)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = discover_samples(args.data_root, args.village_ids)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    samples = samples[args.shard_index :: args.num_shards]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError(f"no TURN samples found under {args.data_root}")

    pending = []
    for video_path, action_path in samples:
        output_path = output_path_for(video_path, args.data_root, args.output_root)
        if args.overwrite or not output_path.exists():
            pending.append((video_path, action_path, output_path))
    print(f"selected={len(samples)} pending={len(pending)} output_root={args.output_root}")
    if args.dry_run:
        for video_path, _, output_path in pending:
            print(f"{video_path} -> {output_path}")
        return

    evaluator = OasisTurnEvaluator(args)
    for video_path, action_path, output_path in tqdm(pending, desc="TURN/Oasis"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated, fps, return_point = evaluator.generate(video_path, action_path, args.seed)
        write_video(str(output_path), generated, fps=fps)
        print(f"saved {output_path} (return_point={return_point}, frames={len(generated)})")


if __name__ == "__main__":
    main()
