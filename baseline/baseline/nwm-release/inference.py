# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#from distributed import init_distributed
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

import misc
import distributed as dist
from models import CDiT_models
from datasets import EvalDataset
from PIL import Image
import json
import cv2
from misc import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords, transform


def save_image(output_file, img, unnormalize_img):
    img = img.detach().cpu()
    if unnormalize_img:
        img = misc.unnormalize(img)
        
    img = img * 255
    img = img.byte()
    img = img.permute(1, 2, 0).numpy()
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img, mode='RGB')

    image.save(output_file)
    
    

@torch.no_grad()
def model_forward_wrapper(all_models, curr_obs, curr_delta, num_timesteps, latent_size, device, num_cond, num_goals=1, rel_t=None, progress=False):
    model, diffusion, vae = all_models
    x = curr_obs.to(device)
    y = curr_delta.to(device)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            rel_t = (torch.ones(B)* (1. / 128.)).to(device)
            rel_t *= num_timesteps

        x = x.flatten(0,1)
        x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
        x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
        z = torch.randn(B*num_goals, 4, latent_size, latent_size, device=device)
        y = y.flatten(0, 1)
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)      
        samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device
        )
        samples = vae.decode(samples / 0.18215).sample

        return torch.clip(samples, -1., 1.)

def get_offset_actions(path):
    assert path.endswith(".json")
    actions = json.load(open(path))
    length = len(actions)
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
@torch.no_grad
def main(args, file, output_path, model_lst, config, device):
    # _, _, device, _ = init_distributed()
    print(args)

    os.makedirs(output_path, exist_ok=True)    
    input_avi_path = file
    input_action_path = file.replace('avi','json')
    actions = json.load(open(input_action_path))

    ### process actions ###
    positions = np.array([[step["x"], step["z"]] for step in actions], dtype=np.float32)
    yaw = np.array([step["yaw"] for step in actions], dtype=np.float32)
    waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
    waypoints_yaw = angle_difference(yaw[0], yaw)
    actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
    actions = actions[1:]
    ACTION_STATS = {}
    for key in config["action_stats"]:
        ACTION_STATS[key] = np.expand_dims(config["action_stats"][key], axis=0)
    actions[:, :2] = normalize_data(actions[:, :2], ACTION_STATS)
    ex_actions = np.concatenate((np.zeros((1, actions.shape[1])), actions), axis=0)
    deltas = ex_actions[1:] - ex_actions[:-1]
    deltas = torch.from_numpy(deltas).to(device)
    offset = get_offset_actions(input_action_path)
    total = len(actions)
    frames = []
    cap = cv2.VideoCapture(str(input_avi_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = transform(Image.fromarray(frame))
        frames.append(frame)
    cap.release()

    frames = torch.stack(frames)
    num_cond = config["context_size"]
    curr_obs = frames[offset-num_cond:offset].unsqueeze(0).to(device)
    generated_frames = []
    for i in tqdm(range(offset,total)):
        curr_delta = deltas[i-1:i].to(device)
        # create a tensor 0 of shape curr_delta
        # curr_delta = torch.zeros_like(curr_delta)

        x_pred_pixels = model_forward_wrapper(model_lst, curr_obs, curr_delta, 1, args.latent_size,
                                               num_cond=num_cond, num_goals=1, device=device)

        curr_obs = torch.cat((curr_obs, x_pred_pixels.unsqueeze(1)), dim=1) # append current prediction
        curr_obs = curr_obs[:, 1:] # remove first observation
        generated_frames.append(x_pred_pixels.squeeze(0))

    # output as video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 编码
    writer = cv2.VideoWriter(output_path + f"/{file.split('/')[-1].split('.')[0]}.mp4", fourcc, 10, (224, 224))

    for img in generated_frames:
        img = img.detach().cpu()
        img = misc.unnormalize(img)
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)

    writer.release()
    # for i, frame in enumerate(generated_frames):
    #     save_image(output_path + f"/frame_{i}.png", frame, unnormalize_img=True)
    # import ipdb; ipdb.set_trace()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default="output", help="output directory")
    parser.add_argument("--exp", type=str, default="config/nwm_cdit_b.yaml", help="experiment name")
    parser.add_argument("--ckp", type=str, default='0200000')
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--datasets", type=str, default="mc", help="dataset name")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--eval_type", type=str, default="rollout", help="type of evaluation has to be either 'time' or 'rollout'")
    # Rollout Evaluation Args
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    
    # main(args)
    exp_eval = args.exp
    device = torch.device("cuda")
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(exp_eval, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    latent_size = config['image_size'] // 8
    args.latent_size = config['image_size'] // 8

    num_cond = config['context_size']
    print("loading")
    model_lst = (None, None, None)
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4)
    ckp = torch.load(f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{args.ckp}.pth.tar', map_location='cpu', weights_only=False)
    print(model.load_state_dict(ckp["ema"], strict=True))
    model.eval()
    model.to(device)
    model = torch.compile(model)
    diffusion = create_diffusion(str(250))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    model_lst = (model, diffusion, vae)
    print("loading finish")

    for nvtype in ["ABA","ABCA"]:
        for length in [5,15,30,50]:
            for loc in ["snowy_village_20", "desert_village_20", "plains_village_20", "taiga_village_20", "savanna_village_20", "zombie_village_20"]:
                print(f"Processing {nvtype} {length} {loc}")
                path = "../test_data/" + nvtype + "/" + str(length) + "/" + loc
                output_path = f"./output_32_{args.ckp}/" + nvtype + "/" + str(length) + "/" + loc 
                file_list = os.listdir(path)
                file_list.sort()
                file = os.path.join(path, file_list[0])
                main(args, file, output_path, model_lst, config, device)