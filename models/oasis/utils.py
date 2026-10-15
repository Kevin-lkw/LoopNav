"""
Adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py
Action format derived from VPT https://github.com/openai/Video-Pre-Training
"""

import math
import torch
from torch import nn
from torchvision.io import read_image, read_video
from torchvision.transforms.functional import resize
from einops import rearrange
from typing import Mapping, Sequence
import json

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

def process_json_actions(actions: Sequence[Mapping[str, int]]) -> torch.Tensor:
    actions_one_hot = torch.zeros(len(actions), len(ACTION_KEYS))
    for i, current_actions in enumerate(actions):
        current_actions = current_actions["action"]
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][1]
                elif action_key == "cameraY":
                    value = current_actions["camera"][0]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
                # max_val = 20
                # bin_size = 0.5
                # num_buckets = int(max_val / bin_size)

                # expect value to be in [-1, 1]
                # origin value need to be in [0,80] (pre-quantized to )
                # I don't know how the camera action is quantized to [0,80]
                """
                My camera action is [-0.1,0.1] rad
                I used it directly.
                """
                # value = (value - num_buckets) / num_buckets
                # ABA_FIXED can contain large camera deltas at angle wrapping
                # boundaries. Oasis was trained with camera controls in [-1, 1],
                # so saturate out-of-range values to the model's action space.
                value = max(-1.0, min(1.0, float(value)))
            else:
                if action_key not in current_actions:
                    value = 0
                else:
                    value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            # if action_key == "forward":
            #     actions_one_hot[i,j] = 1
            # else:
            #     actions_one_hot[i,j] = 0
            actions_one_hot[i, j] = value

    return actions_one_hot

def one_hot_actions(actions: Sequence[Mapping[str, int]]) -> torch.Tensor:
    actions_one_hot = torch.zeros(len(actions), len(ACTION_KEYS))
    for i, current_actions in enumerate(actions):
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][0]
                elif action_key == "cameraY":
                    value = current_actions["camera"][1]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                value = (value - num_buckets) / num_buckets
                assert -1 - 1e-3 <= value <= 1 + 1e-3, f"Camera action value must be in [-1, 1], got {value}"
            else:
                value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            actions_one_hot[i, j] = value

    return actions_one_hot


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
VIDEO_EXTENSIONS = {"mp4", "avi"}


def load_prompt(path, video_offset=None, n_prompt_frames=1):
    if path.lower().split(".")[-1] in IMAGE_EXTENSIONS:
        print("prompt is image; ignoring video_offset and n_prompt_frames")
        prompt = read_image(path)
        # add frame dimension
        prompt = rearrange(prompt, "c h w -> 1 c h w")
    elif path.lower().split(".")[-1] in VIDEO_EXTENSIONS:
        prompt = read_video(path, pts_unit="sec")[0]
        if video_offset is not None:
            prompt = prompt[video_offset:]
        prompt = prompt[:n_prompt_frames]
    else:
        raise ValueError(f"unrecognized prompt file extension; expected one in {IMAGE_EXTENSIONS} or {VIDEO_EXTENSIONS}")
    assert prompt.shape[0] == n_prompt_frames, f"input prompt {path} had less than n_prompt_frames={n_prompt_frames} frames"
    # prompt = resize(prompt, (360, 640))
    # import ipdb; ipdb.set_trace()
    # add batch dimension
    # prompt = rearrange(prompt, "t c h w -> 1 t c h w")
    prompt = rearrange(prompt, "t h w c -> 1 t c h w")
    prompt = prompt.float() / 255.0
    print("prompt.shape", prompt.shape)
    return prompt


def load_actions(path, action_offset=None):
    if path.endswith(".actions.pt"):
        actions = one_hot_actions(torch.load(path))
    elif path.endswith(".one_hot_actions.pt"):
        actions = torch.load(path, weights_only=True)
    elif path.endswith(".json"):
        actions = json.load(open(path))
        actions = process_json_actions(actions)
    else:
        raise ValueError("unrecognized action file extension; expected '*.actions.pt' or '*.one_hot_actions.pt'")
    if action_offset is not None:
        actions = actions[action_offset:]
    # actions = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0)
    # CANNOT UNDERSTAND WHY THIS IS NEEDED!


    # add batch dimension
    actions = rearrange(actions, "t d -> 1 t d")
    return actions
def get_offset_actions(path):
    assert path.endswith(".json")
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
        if actions[i]["goal"] is not None:
            gx = actions[i]["goal"]["x"]
            gz = actions[i]["goal"]["z"]
            if abs(gx - start_pos["x"]) > 2 or abs(gz - start_pos["z"]) > 2:
                print(f"find returning point at {i}, total length {length}")
                return i
    return None

if __name__ == "__main__":
    # offset = load_actions("data/LoopNav/ABA/snowy_village_1/15/05-04_13-57-20.json")
    action = load_actions("../05-04_13-57-20.json")
    # actions = load_actions("sample_data/sample_actions_0.one_hot_actions.pt")
    # import ipdb; ipdb.set_trace()
