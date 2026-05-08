import os
import cv2
import torch
import time
import numpy as np
import math
from tqdm import tqdm
from rich import print
from PIL import Image
from pathlib import Path
from torch import autocast
from einops import rearrange
from mcdataset import MCDataset
from omegaconf import OmegaConf
from torchvision import transforms
from argparse import ArgumentParser
from utils import load_model, tensor_to_uint8
import json
from mcdataset import NOOP_ACTION
torch.backends.cuda.matmul.allow_tf32 = False

ACCELERATE_ALGO = [
    'naive','image_diagd'
]

TARGET_SIZE=(224,384)
TOKEN_PER_IMAGE = 347 # IMAGE = PIX+ACTION
TOKEN_PER_PIX = 336

safe_globals = {"array": np.array}


def token2video(code_list, tokenizer, save_path, fps, device = 'cuda'):
    """
    change log:  we don't perform path processing inside functions to enable extensibility
    save_path: str, path to save the video, expect to endwith .mp4
    
    """
    if len(code_list) % TOKEN_PER_PIX != 0:
        print(f"code_list length {len(code_list)} is not multiple of {TOKEN_PER_PIX}")
        return
    num_images = len(code_list) // TOKEN_PER_PIX
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (384, 224))
    for i in range(num_images):
        code = code_list[i*TOKEN_PER_PIX:(i+1)*TOKEN_PER_PIX]
        code = torch.tensor([int(x) for x in code], dtype=torch.long).to(device)
        img = tokenizer.token2image(code) # pixel
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../test_data")
    parser.add_argument('--model_ckpt', type=str, default="./mineworld/checkpoints")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--demo_num', type=int, default=1)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--accelerate-algo', type=str, default='naive', help=f"Accelerate Algorithm Option: {ACCELERATE_ALGO}")
    parser.add_argument('--fps', type=int, default=6)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--top_k', type=int, help='Use top-k sampling')
    group.add_argument('--top_p', type=float, help='Use top-p (nucleus) sampling')
    parser.add_argument('--nvtype', type=str, default=None)
    parser.add_argument('--index', type=int, default=None)
    args = parser.parse_args()
    return args

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
    
def lvm_generate(args, model, input_path, output_path, file):
    """
    """
    ### 1. set video input/output path
    input_mp4_path = file
    input_action_path = file.replace('avi','json')
    os.makedirs(output_path, exist_ok=True)
    output_mp4_path = str(output_path + "/" + f"{file.split('/')[-1].split('.')[0]}.mp4")

    # output_action_path = output_mp4_path.replace('.avi', '.json')
    # backup action  
    # os.system(f"cp {input_action_path} {output_action_path}")
    if os.path.exists(output_mp4_path):
        print(f"output path {output_mp4_path} exist")
        return {}
    
    device = model.transformer.device
    ### 2. load action into list 
    action_list = []
    mcdataset = MCDataset()
    actions = json.load(open(input_action_path))
    for action in actions:
        action = action['action']
        action_dict = NOOP_ACTION
        action_dict["forward"] = action['forward']
        action_dict["jump"]  = action['jump']
        action_dict["right"] = action['right']
        action_dict['camera'] = np.array(action['camera'][::-1]) / math.pi * 180
        action_dict['camera'] = -action_dict['camera']
        # import ipdb; ipdb.set_trace()
        act_index = mcdataset.get_action_index_from_actiondict(action_dict, action_vocab_offset=8192)
        action_list.append(act_index)
    
    ### 3. load video frames 
    cap = cv2.VideoCapture(input_mp4_path)
    offset = get_offset_actions(input_action_path)
    start_frame = offset - args.demo_num
    end_frame = offset
    frames = []
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error in reading frame {frame_idx}")
            continue
        cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
        frame = cv2.resize(frame, (384, 224), interpolation=cv2.INTER_AREA)
        frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
        frame = torch.from_numpy(frame)
        frames.append(frame)
    frames = torch.stack(frames, dim=0).to(device)
    frames = frames.permute(0, 3, 1, 2)
    frames = frames.float() / 255.0
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frames = normalize(frames)
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        img_index = model.tokenizer.tokenize_images(frames)
        #frames num,3,224,384   num is args.demo_num
        #img_index shape num ,14, 24
    # import ipdb; ipdb.set_trace() 
    img_index = rearrange(img_index, '(b t) h w -> b t (h w)', b=1)
    # img_index shape 1, num ,336
    ### ###
    # img_index shape 1, num ,336
    # action_list [N,11]
    # image_action_input [1, (num-1) * 347 + 336]
    image_action_input = []
    for i in range(args.demo_num):
        image_action_input.append(img_index[0][i])
        action_index = start_frame + i
        if i < args.demo_num - 1:
            image_action_input.append(torch.tensor(action_list[action_index]).to(device))

    # import ipdb; ipdb.set_trace()
    image_action_input = torch.cat(image_action_input, dim=0).unsqueeze(0)
    # shape [1,(num-1) * 347 + 336]
    ### ###
    
    
    # image_input = rearrange(img_index, 'b t c -> b (t c)')
    # shape [1,(num-1) * 347 + 336]
    
    # add prompt
    all_generated_tokens = []
    total = len(action_list)
    # 发现args.frames + args.demo_num > 16 的时候就会挂掉
    step = args.frames
    for i in range(end_frame, total, step):
        # currently predicting [i, i + step)
        cur_step = total - i if i + step > total else step
        action_all = action_list[i - 1: i + cur_step - 1] # as action i is performed after image i(in my dataset)
        action_all = torch.tensor(action_all).unsqueeze(1).to(device)
        
        start_t = time.time()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            if args.accelerate_algo == 'naive':
                outputs = model.transformer.naive_generate( input_ids=image_action_input, max_new_tokens=TOKEN_PER_PIX*cur_step, action_all=action_all, top_k=args.top_k, top_p=args.top_p)
            elif args.accelerate_algo == 'image_diagd':
                outputs = model.transformer.img_diagd_generate(input_ids=image_action_input, max_new_tokens=TOKEN_PER_PIX*cur_step, action_all=action_all,windowsize = args.window_size, top_k=args.top_k, top_p=args.top_p)
            else:
                raise ValueError(f"Unknown accelerate algorithm {args.accelerate_algo}")
        end_t = time.time()
        print(f"use {end_t - start_t} seconds to generate {cur_step} frames")
        all_generated_tokens.extend(outputs.tolist()[0])
        for j in range(cur_step):
            image_action_input = torch.cat([image_action_input, torch.tensor(action_all[j]).to(device)], dim=1)
            image_action_input = torch.cat([image_action_input, outputs[:,j*TOKEN_PER_PIX:(j+1)*TOKEN_PER_PIX]], dim=1)
        image_action_input = image_action_input[:,-((args.demo_num-1)*TOKEN_PER_IMAGE + TOKEN_PER_PIX):]
    new_length = len(all_generated_tokens)
    time_costed = end_t - start_t 
    token_per_sec = new_length / time_costed
    frame_per_sec = token_per_sec / TOKEN_PER_PIX
    print(f"{new_length} token generated; cost {time_costed:.3f} second; {token_per_sec:.3f} token/sec {frame_per_sec:.3f} fps")
    token2video(all_generated_tokens, model.tokenizer, output_mp4_path, args.fps, device)  
    # return for evaluation 
    return_item = {
        "time_costed": time_costed,
        "token_num": new_length,
    }
    return return_item
if __name__ == '__main__':
    args = get_args()
    config = OmegaConf.load(args.config)
    output_path = Path(args.output_dir)
    precision_scope = autocast
    os.makedirs(output_path, exist_ok=True)

    model = load_model(config, args.model_ckpt, gpu=True, eval_mode=True)
    print(f"[bold magenta][MINEWORLD][INFERENCE][/bold magenta] Load Model From {args.model_ckpt}")
    # get accelearte algoritm
    args.accelerate_algo = args.accelerate_algo.lower()
    if args.accelerate_algo not in ACCELERATE_ALGO:
        print(f"[bold red][Warning][/bold red] {args.accelerate_algo} is not in {ACCELERATE_ALGO}, use naive")
        args.accelerate_algo = 'naive'
    num_item = 0
    nvtype_list = ["ABA","ABCA"]
    if args.nvtype is not None:
        nvtype_list = [args.nvtype]
    for nvtype in nvtype_list:
        for length in [5,15,30,50]:
            for loc in ["snowy_village_20", "desert_village_20", "plains_village_20", "taiga_village_20", "savanna_village_20", "zombie_village_20"]:
                print(f"Processing {nvtype} {length} {loc}")
                path = "../test_data/" + nvtype + "/" + str(length) + "/" + loc
                output_path = f"./output_32/" + nvtype + "/" + str(length) + "/" + loc 
                file_list = os.listdir(path)
                file_list.sort()
                if args.index is not None:
                    file_list = [file_list[args.index],file_list[args.index+1]]
                
                for i in range(0,len(file_list),2):
                    file = os.path.join(path, file_list[i])
                    return_item = lvm_generate(args, model, input_path=path, output_path=output_path, file=file)