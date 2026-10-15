"""Evaluate the 32-frame DIAMOND checkpoint on the TURN split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from agent import Agent
from data.MCdata import CameraQuantizer, QuantizationScheme
from envs import WorldModelEnv


DEFAULT_DATA_ROOT = Path("data/LoopNav/TURN")
DEFAULT_RUN_DIR = Path("outputs/2025-09-25/17-13-54")
DEFAULT_VILLAGE_IDS = (18, 19, 20)


CAMERA_QUANTIZER = CameraQuantizer(
    camera_maxval=0.1,
    camera_binsize=0.02,
    quantization_scheme=QuantizationScheme.MU_LAW,
    mu=5,
)


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


def output_path_for(video_path: Path, data_root: Path, output_root: Path, epoch: str) -> Path:
    village, angle_bucket, filename = video_path.relative_to(data_root).parts
    return output_root / epoch / "TURN" / angle_bucket / village / Path(filename).with_suffix(".mp4")


def encode_action(record: dict) -> torch.Tensor:
    action = record["action"]
    values = [int(bool(action.get("forward", False))), int(bool(action.get("jump", False)))]
    yaw = int(CAMERA_QUANTIZER.discretize(action.get("camera", [0.0, 0.0])[0]))
    pitch = int(CAMERA_QUANTIZER.discretize(action.get("camera", [0.0, 0.0])[1]))
    values.extend(F.one_hot(torch.tensor(yaw), num_classes=11).tolist())
    values.extend(F.one_hot(torch.tensor(pitch), num_classes=11).tolist())
    return torch.tensor(values, dtype=torch.long)


def read_video(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float().div(127.5).sub(1))
    capture.release()
    if not frames:
        raise ValueError(f"could not read frames from {video_path}")
    return torch.stack(frames).numpy()


def write_video_rgb(output_path: Path, frames: torch.Tensor, fps: int) -> None:
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames.shape[2], frames.shape[1]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output_path}")
    for frame in frames.numpy():
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def left_pad_context(
    frames: np.ndarray,
    actions: torch.Tensor,
    return_point: int,
    context_frames: int,
) -> tuple[np.ndarray, torch.Tensor]:
    start = max(0, return_point - context_frames)
    frame_context = frames[start:return_point]
    action_context = actions[start:return_point]
    if len(frame_context) == 0:
        raise ValueError("return point has no preceding context")
    pad_length = context_frames - len(frame_context)
    if pad_length:
        frame_context = np.concatenate(
            (np.repeat(frame_context[:1], pad_length, axis=0), frame_context),
            axis=0,
        )
        no_op = encode_action({"action": {"camera": [0.0, 0.0]}})
        action_context = torch.cat((no_op.unsqueeze(0).expand(pad_length, -1), action_context))
    return frame_context, action_context


class DiamondTurnEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("DIAMOND TURN evaluation requires CUDA")
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        config_path = args.run_dir / "config" / "trainer.yaml"
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(args.device)

        agent_config = instantiate(self.config.agent, num_actions=self.config.env.num_actions)
        self.agent = Agent(agent_config).to(self.device).eval()
        checkpoint = args.run_dir / "checkpoints" / "agent_versions" / f"agent_epoch_{args.epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.agent.load(checkpoint)

        sequence_length = self.config.agent.denoiser.inner_model.num_steps_conditioning
        if self.agent.upsampler is not None:
            sequence_length = max(
                sequence_length,
                self.config.agent.upsampler.inner_model.num_steps_conditioning,
            )
        world_model_config = instantiate(self.config.world_model_env, num_batches_to_preload=1)
        self.env = WorldModelEnv(
            self.agent.denoiser,
            self.agent.upsampler,
            self.agent.rew_end_model,
            None,
            1,
            sequence_length,
            world_model_config,
            return_denoising_trajectory=False,
        )
        self.context_frames = sequence_length

    @torch.no_grad()
    def generate(
        self,
        video_path: Path,
        action_path: Path,
        seed: int,
    ) -> tuple[torch.Tensor, int]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        records = json.loads(action_path.read_text())
        return_point = find_return_point(records)
        frames = read_video(video_path)
        actions = torch.stack([encode_action(record) for record in records])
        if len(frames) != len(actions):
            raise ValueError(f"frame/action mismatch: {video_path} ({len(frames)} vs {len(actions)})")

        context_frames, context_actions = left_pad_context(
            frames,
            actions,
            return_point,
            self.context_frames,
        )
        self.env.reset(context_frames, context_actions, self.context_frames)

        generated = []
        for target_index in tqdm(
            range(return_point, len(records)),
            desc=video_path.stem,
            leave=False,
        ):
            observation, _, _, _, _ = self.env.step(actions[target_index - 1])
            generated.append(observation[0].cpu())
        generated = torch.stack(generated).permute(0, 2, 3, 1)
        return generated.add(1).mul(127.5).clamp(0, 255).byte(), return_point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DIAMOND-32 on TURN")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("output_turn_32"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epoch", default="00200")
    parser.add_argument("--village-ids", type=parse_village_ids, default=DEFAULT_VILLAGE_IDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=30)
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
    pending = []
    for video_path, action_path in samples:
        output_path = output_path_for(
            video_path,
            args.data_root,
            args.output_root,
            args.epoch,
        )
        if args.overwrite or not output_path.exists():
            pending.append((video_path, action_path, output_path))
    print(
        f"selected={len(samples)} pending={len(pending)} run_dir={args.run_dir} "
        f"epoch={args.epoch}"
    )
    if args.dry_run:
        for video_path, _, output_path in pending:
            print(f"{video_path} -> {output_path}")
        return

    evaluator = DiamondTurnEvaluator(args)
    print(f"DIAMOND context={evaluator.context_frames}")
    for video_path, action_path, output_path in tqdm(pending, desc="TURN/DIAMOND"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated, return_point = evaluator.generate(video_path, action_path, args.seed)
        write_video_rgb(output_path, generated, args.fps)
        print(f"saved {output_path} (return_point={return_point}, frames={len(generated)})")


if __name__ == "__main__":
    main()
