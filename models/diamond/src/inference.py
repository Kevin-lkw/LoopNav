import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from huggingface_hub import snapshot_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from agent import Agent
from envs import WorldModelEnv

from data.MCdata import CameraQuantizer, QuantizationScheme, MinecraftDataset
import torch.nn.functional as F
from torchvision.io import write_video
from tqdm import tqdm
import os

quantizer = CameraQuantizer(
            camera_maxval=0.1,
            camera_binsize=0.02,
            quantization_scheme=QuantizationScheme.MU_LAW,
            mu=5
        )

OmegaConf.register_new_resolver("eval", eval)
def action_to_tensor(action:dict) -> torch.Tensor:
    action = action["action"]
    ls = []
    ls.append(1 if action["forward"] else 0)
    ls.append(1 if action["jump"] else 0)
    yaw = action["camera"][0]
    yaw = quantizer.discretize(yaw)
    # convert to one hot vector
    yaw_one_hot = F.one_hot(torch.tensor(yaw), num_classes=11)
    ls.extend(yaw_one_hot)
    pitch = action["camera"][1]
    pitch = quantizer.discretize(pitch)
    pitch_one_hot = F.one_hot(torch.tensor(pitch), num_classes=11)
    ls.extend(pitch_one_hot)
    return torch.tensor(ls)

def get_offset_actions(path):
    actions = json.load(open(path))
    length = len(actions)
    return_points = [
        index for index, action in enumerate(actions)
        if action.get("return_point") is True
    ]
    if return_points:
        if len(return_points) != 1:
            raise ValueError(
                f"Expected one return_point in {path}, found {len(return_points)}"
            )
        print(f"using explicit return_point at {return_points[0]}, total length {length}")
        return return_points[0]
    start_pos ={"x":actions[0]["x"],"z":actions[0]["z"]}
    print(f"start_pos: {start_pos}")

    for i in reversed(range(length)):
        # print(i)
        if actions[i]["goal"] is not None:
            gx = actions[i]["goal"]["x"]
            gz = actions[i]["goal"]["z"]
            # print(f"gx: {gx}, gz: {gz}")
            if abs(gx - start_pos["x"]) > 2 or abs(gz - start_pos["z"]) > 2:
                print(f"find returning point at {i}, total length {length}")
                return i
    return None

def load_video_and_actions(video_path: Path, action_path: Path) -> tuple[np.ndarray, list]:
    """Load video frames and actions from files"""
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 180))
        frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0 * 2 - 1
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    # Load actions
    with open(action_path) as f:
        actions = json.load(f)

    offset = get_offset_actions(action_path)
    tensor_actions = []
    for i in range(len(actions)):
        tensor_action = action_to_tensor(actions[i])
        tensor_actions.append(tensor_action)
    tensor_actions = torch.stack(tensor_actions)
    ### some method to test answer leak
    # total_frames = frames.shape[0]
    # keep_start = max(offset, 0)
    # keep_end = min(offset, total_frames)

    # for i in range(total_frames):
    #     if i < keep_start or i >= keep_end:
    #         # Replace with uniform noise in [-1, 1]
    #         frames[i] = np.random.rand(*frames[i].shape).astype(np.float32) * 2 - 1

    return frames, tensor_actions, offset

def prepare_inference(cfg: DictConfig, checkpoint_dir: Path, ckpt: str) -> tuple[WorldModelEnv, Agent]:
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for inference")
    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob(f"checkpoints/agent_versions/agent_epoch_{ckpt}.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}/checkpoints/")
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
    print(f"Using checkpoint: {latest_checkpoint}")

    # Setup models
    num_actions = cfg.env.num_actions
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(latest_checkpoint)

    # Setup world model environment
    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)


    # Create environment
    wm_env = WorldModelEnv(
        agent.denoiser,
        agent.upsampler,
        agent.rew_end_model,
        None, # spawn_dir
        1,  # num_envs
        sl,
        wm_env_cfg,
        return_denoising_trajectory=True
    )


    return wm_env, agent

def prepare_observation(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert frames to model input format"""
    # Normalize to [-1, 1]
    frames = frames.astype(np.float32) / 127.5 - 1.0
    # Convert to torch tensor
    frames = torch.from_numpy(frames).to(device)
    # Add batch dimension if needed
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    return frames


@torch.no_grad()
def run_inference(env: WorldModelEnv, frames: np.ndarray, actions: list, offset: int, \
                    location: str, nvtype: str, nvrange: int, ckpt: str, name: str, output: str) -> None:
    """Run inference with frames and actions"""
    # Prepare initial observation
    obs = prepare_observation(frames[0], env.device)

    print("Starting inference...")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Number of actions: {len(actions)}")

    try:
        # Reset environment with num_envs=1
        obs, _ = env.reset(frames, actions, offset)
        print(f"Environment reset successful, observation shape: {obs.shape}")
        obs_list = []
        for i in tqdm(range(offset-1, len(actions)), desc="Running inference"):

            # Take a step in the environment
            next_obs, reward, done, truncated, info = env.step(actions[i])
            obs_list.append(next_obs[0])

            # print(f"\nStep {i+1}:")
            # print(f"Action taken: {actions[i]}")
            # print(f"Reward: {reward.item()}")
            # print(f"Done: {bool(done.item())}")
            # print(f"Truncated: {bool(truncated.item())}")
            # print(f"Observation shape: {next_obs.shape}")

            if done.item() or truncated.item():
                print("\nEpisode finished")
                break
        output_dir = Path(output) / f"{ckpt}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # output obs_list as a video
        output_video_path = output_dir / f"{nvtype}" / f"{nvrange}" / f"{location}"
        output_video_path.mkdir(parents=True, exist_ok=True)
        output_video_path = output_video_path / f"{name}.mp4"

        obs_list = torch.stack(obs_list).cpu()
        # (T,C,H,W) -> (T,H,W,C)
        obs_list = obs_list.permute(0, 2, 3, 1)
        # convert [-1,1] to [0,255]
        obs_list = (obs_list + 1.0) * 127.5
        obs_list = obs_list.numpy().astype(np.uint8)
        height, width = obs_list.shape[1:3]
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_video_path}")
        for frame in obs_list:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        # Clean up
        if Path("temp_spawn").exists():
            shutil.rmtree("temp_spawn")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="../outputs/2025-09-25/17-13-54",
                      help="Directory containing the checkpoint")
    parser.add_argument("--ckpt", type=str, default="00200",
                      help="Directory containing the checkpoint")
    parser.add_argument("--output_dir", type=str, default="output_inference",)
    parser.add_argument("--data_root", type=str, default="../../test_data")
    parser.add_argument("--nvtype", type=str, default="ABA")
    parser.add_argument("--village_ids", nargs="+", default=["18", "19", "20"])
    parser.add_argument("--lengths", nargs="+", type=int, default=[5, 15, 30, 50])
    args = parser.parse_args()
    # Initialize Hydra
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")
    checkpoint_dir = Path(args.checkpoint_dir)
    # Prepare environment and agent
    env, agent = prepare_inference(cfg, checkpoint_dir, args.ckpt)

    for nvtype in [args.nvtype]:
        for nvrange in args.lengths:
            dataset_dir = Path(args.data_root) / nvtype / str(nvrange)
            locations = sorted(
                p.name for p in dataset_dir.iterdir()
                if p.is_dir() and p.name.rsplit('_', 1)[-1] in args.village_ids
            )
            for location in locations:
                # Convert paths
                path = str(dataset_dir / location)
                video_names = sorted(f for f in os.listdir(path) if f.endswith('.avi'))
                for video_name in video_names:
                    action_name = f"{Path(video_name).stem}.json"
                    output_file = Path(args.output_dir) / args.ckpt / nvtype / str(nvrange) / location / f"{Path(video_name).stem}.mp4"
                    if output_file.exists():
                        print(f"Skipping existing {output_file}")
                        continue
                    video_path = Path(path) / video_name
                    action_path = Path(path) / action_name
                    # Load data
                    frames, actions, offset = load_video_and_actions(video_path, action_path)
                    print(f"Loaded {len(frames)} frames and {len(actions)} actions")
                    # Run inference
                    run_inference(env, frames, actions, offset, location, nvtype, nvrange, \
                                   args.ckpt, video_name.split(".")[0], args.output_dir)
if __name__ == "__main__":
    main()
