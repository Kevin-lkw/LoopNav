"""Evaluate MineWorld-1200M-32f on the TURN split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from mcdataset import MCDataset, NOOP_ACTION
from utils import load_model


DEFAULT_DATA_ROOT = Path("data/LoopNav/TURN")
DEFAULT_CHECKPOINT = Path("checkpoints/1200M_32f.ckpt")
DEFAULT_VILLAGE_IDS = (18, 19, 20)
TOKEN_PER_IMAGE = 336
ACTION_TOKENS = 11


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


def read_video_rgb(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (384, 224), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise ValueError(f"could not read frames from {video_path}")
    return np.stack(frames)


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


class MineWorldTurnEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MineWorld TURN evaluation requires CUDA")
        config = OmegaConf.load(args.config)
        self.model = load_model(config, args.checkpoint, gpu=True, eval_mode=True)
        self.device = self.model.transformer.device
        self.context_frames = args.context_frames
        self.top_p = args.top_p
        self.action_encoder = MCDataset()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        self.noop_tokens = self.encode_action({})

    def encode_action(self, record_action: dict) -> list[int]:
        action = NOOP_ACTION.copy()
        action["camera"] = np.array([0.0, 0.0])
        for key in (
            "forward",
            "back",
            "left",
            "right",
            "jump",
            "sprint",
            "sneak",
            "attack",
            "use",
            "drop",
            "pickItem",
        ):
            action[key] = int(bool(record_action.get(key, False)))
        camera = record_action.get("camera", [0.0, 0.0])
        action["camera"] = -np.asarray(camera[::-1], dtype=np.float32) / math.pi * 180.0
        return self.action_encoder.get_action_index_from_actiondict(action, action_vocab_offset=8192)

    @torch.no_grad()
    def tokenize_context(self, frames: np.ndarray, return_point: int) -> tuple[torch.Tensor, list[int | None]]:
        source_indices: list[int | None] = list(range(max(0, return_point - self.context_frames), return_point))
        if not source_indices:
            raise ValueError("return point has no preceding context")
        source_indices = [None] * (self.context_frames - len(source_indices)) + source_indices
        first_real_index = next(index for index in source_indices if index is not None)
        context = np.stack(
            [frames[first_real_index if index is None else index] for index in source_indices]
        )
        tensor = torch.from_numpy(context).permute(0, 3, 1, 2).float().div(255).to(self.device)
        tensor = self.normalize(tensor)
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = self.model.tokenizer.tokenize_images(tensor)
        return rearrange(tokens, "t h w -> t (h w)"), source_indices

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
        frames = read_video_rgb(video_path)
        if len(frames) != len(records):
            raise ValueError(f"frame/action mismatch: {video_path} ({len(frames)} vs {len(records)})")
        action_tokens = [self.encode_action(record["action"]) for record in records]
        image_tokens, source_indices = self.tokenize_context(frames, return_point)

        prompt_parts = []
        for prompt_index in range(self.context_frames):
            prompt_parts.append(image_tokens[prompt_index])
            if prompt_index == self.context_frames - 1:
                continue
            source_index = source_indices[prompt_index]
            next_source_index = source_indices[prompt_index + 1]
            transition = (
                action_tokens[source_index]
                if source_index is not None and next_source_index == source_index + 1
                else self.noop_tokens
            )
            prompt_parts.append(torch.tensor(transition, device=self.device))
        prompt = torch.cat(prompt_parts).unsqueeze(0)
        expected_prompt_length = self.context_frames * TOKEN_PER_IMAGE + (self.context_frames - 1) * ACTION_TOKENS
        if prompt.shape[1] != expected_prompt_length:
            raise AssertionError(f"unexpected prompt length {prompt.shape[1]} != {expected_prompt_length}")

        generated_frames = []
        for target_index in tqdm(
            range(return_point, len(records)),
            desc=video_path.stem,
            leave=False,
        ):
            transition = torch.tensor(
                action_tokens[target_index - 1],
                device=self.device,
            ).view(1, 1, ACTION_TOKENS)
            with torch.autocast("cuda", dtype=torch.float16):
                output = self.model.transformer.naive_generate(
                    input_ids=prompt,
                    max_new_tokens=TOKEN_PER_IMAGE,
                    action_all=transition,
                    top_p=self.top_p,
                    compile_prefill=False,
                )
            frame = self.model.tokenizer.token2image(output[0])
            generated_frames.append(torch.from_numpy(np.asarray(frame).copy()))

            prompt = torch.cat((prompt, transition[:, 0], output), dim=1)
            prompt = prompt[:, -expected_prompt_length:]

        return torch.stack(generated_frames).byte(), return_point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MineWorld-1200M-32f on TURN")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("output_turn_32"))
    parser.add_argument("--config", type=Path, default=Path("configs/1200M_32f.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--village-ids", type=parse_village_ids, default=DEFAULT_VILLAGE_IDS)
    parser.add_argument("--context-frames", type=int, default=31)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.context_frames != 31:
        raise ValueError("1200M-32f uses 31 observed/generated frames plus one target frame")
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
        f"selected={len(samples)} pending={len(pending)} checkpoint={args.checkpoint} "
        f"context={args.context_frames}+1"
    )
    if args.dry_run:
        for video_path, _, output_path in pending:
            print(f"{video_path} -> {output_path}")
        return

    evaluator = MineWorldTurnEvaluator(args)
    for video_path, action_path, output_path in tqdm(pending, desc="TURN/MineWorld"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated, return_point = evaluator.generate(video_path, action_path, args.seed)
        write_video_rgb(output_path, generated, args.fps)
        print(f"saved {output_path} (return_point={return_point}, frames={len(generated)})")


if __name__ == "__main__":
    main()
