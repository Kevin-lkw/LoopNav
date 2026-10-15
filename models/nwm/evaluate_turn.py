"""Autoregressive TURN evaluation for the trained NWM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.io import read_video, write_video
from tqdm import tqdm

import misc
from diffusion import create_diffusion
from misc import angle_difference, normalize_data, to_local_coords, transform
from models import CDiT_models


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
    suffixes = tuple(f"_{village_id}" for village_id in village_ids)
    samples = []
    for video_path in sorted(data_root.glob("*/*/*.avi")):
        if not video_path.parent.parent.name.endswith(suffixes):
            continue
        action_path = video_path.with_suffix(".json")
        if not action_path.is_file():
            raise FileNotFoundError(f"missing action file for {video_path}")
        samples.append((video_path, action_path))
    return samples


def output_path_for(video_path: Path, data_root: Path, output_root: Path) -> Path:
    village, angle_bucket, filename = video_path.relative_to(data_root).parts
    return output_root / "TURN" / angle_bucket / village / Path(filename).with_suffix(".mp4")


def load_config(config_path: Path, eval_config_path: Path) -> dict:
    with eval_config_path.open() as handle:
        config = yaml.safe_load(handle)
    with config_path.open() as handle:
        config.update(yaml.safe_load(handle))
    return config


def make_action_deltas(records: list[dict], config: dict) -> torch.Tensor:
    positions = np.asarray([[record["x"], record["z"]] for record in records], dtype=np.float32)
    yaw = np.asarray([record["yaw"] for record in records], dtype=np.float32)
    local_positions = to_local_coords(positions, positions[0], yaw[0])
    local_yaw = angle_difference(yaw[0], yaw)
    states = np.concatenate((local_positions, local_yaw[:, None]), axis=-1)

    action_stats = {
        key: np.expand_dims(np.asarray(value, dtype=np.float32), axis=0)
        for key, value in config["action_stats"].items()
    }
    states[:, :2] = normalize_data(states[:, :2], action_stats)
    deltas = np.zeros_like(states)
    deltas[1:] = states[1:] - states[:-1]
    return torch.from_numpy(deltas).float()


def load_frames(video_path: Path) -> tuple[torch.Tensor, float]:
    video, _, info = read_video(str(video_path), pts_unit="sec", output_format="THWC")
    frames = torch.stack([transform(Image.fromarray(frame.numpy())) for frame in video])
    return frames, int(round(float(info.get("video_fps", 20.0))))


def left_pad_frames(frames: torch.Tensor, target_length: int) -> torch.Tensor:
    if frames.shape[0] >= target_length:
        return frames[-target_length:]
    return torch.cat((frames[:1].expand(target_length - frames.shape[0], -1, -1, -1), frames))


class NWMTurnEvaluator:
    def __init__(self, args: argparse.Namespace, config: dict) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NWM TURN evaluation requires CUDA")
        self.device = torch.device(args.device)
        self.config = config
        self.context_frames = config["context_size"]
        latent_size = config["image_size"] // 8

        model = CDiT_models[config["model"]](
            context_size=self.context_frames,
            input_size=latent_size,
            in_channels=4,
        )
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["ema"], strict=True)
        model = model.to(self.device).eval()
        self.model = torch.compile(model) if args.compile else model
        self.diffusion = create_diffusion(str(args.diffusion_steps))
        self.vae = AutoencoderKL.from_pretrained(args.vae).to(self.device).eval()
        self.latent_size = latent_size

    @torch.no_grad()
    def predict(self, context: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        context = context.unsqueeze(0).to(self.device)
        delta = delta.unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            batch_size, time = context.shape[:2]
            latents = self.vae.encode(context.flatten(0, 1)).latent_dist.sample().mul_(0.18215)
            latents = latents.unflatten(0, (batch_size, time))
            noise = torch.randn(
                batch_size,
                4,
                self.latent_size,
                self.latent_size,
                device=self.device,
            )
            model_kwargs = {
                "y": delta,
                "x_cond": latents,
                "rel_t": torch.full((batch_size,), 1.0 / 128.0, device=self.device),
            }
            sample = self.diffusion.p_sample_loop(
                self.model.forward,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=self.device,
            )
            return self.vae.decode(sample / 0.18215).sample.clamp(-1, 1)[0]

    @torch.no_grad()
    def generate(self, video_path: Path, action_path: Path, seed: int) -> tuple[torch.Tensor, float, int]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        records = json.loads(action_path.read_text())
        return_point = find_return_point(records)
        frames, fps = load_frames(video_path)
        if len(frames) != len(records):
            raise ValueError(f"frame/action mismatch: {video_path} ({len(frames)} vs {len(records)})")
        deltas = make_action_deltas(records, self.config)
        context = left_pad_frames(frames[:return_point], self.context_frames)

        generated = []
        for target_index in tqdm(
            range(return_point, len(records)),
            desc=video_path.stem,
            leave=False,
        ):
            prediction = self.predict(context, deltas[target_index])
            generated.append(prediction)
            context = torch.cat((context[1:], prediction.unsqueeze(0).cpu()), dim=0)

        generated = torch.stack(generated)
        generated = misc.unnormalize(generated).clamp(0, 1)
        generated = generated.permute(0, 2, 3, 1).mul(255).byte()
        # Keep parity with the original NWM Minecraft inference path: decoded
        # frames come out in BGR order, while torchvision.write_video expects RGB.
        generated = generated[..., [2, 1, 0]]
        return generated, fps, return_point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate NWM on TURN")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("output_turn_32_0200000"))
    parser.add_argument("--config", type=Path, default=Path("config/nwm_cdit_b.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("config/eval_config.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/nwm_cdit_b_32/checkpoints/0200000.pth.tar"),
    )
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--diffusion-steps", type=int, default=250)
    parser.add_argument("--village-ids", type=parse_village_ids, default=DEFAULT_VILLAGE_IDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.eval_config)
    samples = discover_samples(args.data_root, args.village_ids)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    samples = samples[args.shard_index :: args.num_shards]
    if args.limit is not None:
        samples = samples[: args.limit]
    pending = []
    for video_path, action_path in samples:
        output_path = output_path_for(video_path, args.data_root, args.output_root)
        if args.overwrite or not output_path.exists():
            pending.append((video_path, action_path, output_path))
    print(
        f"selected={len(samples)} pending={len(pending)} model={config['model']} "
        f"context={config['context_size']} checkpoint={args.checkpoint}"
    )
    if args.dry_run:
        for video_path, _, output_path in pending:
            print(f"{video_path} -> {output_path}")
        return

    evaluator = NWMTurnEvaluator(args, config)
    for video_path, action_path, output_path in tqdm(pending, desc="TURN/NWM"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated, fps, return_point = evaluator.generate(video_path, action_path, args.seed)
        write_video(str(output_path), generated, fps=fps)
        print(f"saved {output_path} (return_point={return_point}, frames={len(generated)})")


if __name__ == "__main__":
    main()
