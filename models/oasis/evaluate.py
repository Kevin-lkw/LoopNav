"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import torch
import av
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video
from utils import load_prompt, load_actions, sigmoid_beta_schedule, get_offset_actions
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
import os

def initialize_models(args):
    global model, vae, device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    oasis_ckpt = args.oasis_ckpt
    vae_ckpt = args.vae_ckpt
    # load DiT checkpoint
    model = DiT_models["DiT-S/2"]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(oasis_ckpt)}...")
    if oasis_ckpt.endswith(".pt"):
        ckpt = torch.load(oasis_ckpt, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
    elif oasis_ckpt.endswith(".safetensors"):
        load_model(model, oasis_ckpt)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    if vae_ckpt.endswith(".pt"):
        vae_ckpt = torch.load(vae_ckpt, weights_only=True)
        vae.load_state_dict(vae_ckpt)
    elif vae_ckpt.endswith(".safetensors"):
        load_model(vae, vae_ckpt)
    vae = vae.to(device).eval()

def main(args,loc,length,name):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # sampling params
    n_prompt_frames = args.n_prompt_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15
    
    # video offset
    # video offset is the first frame that agent begin to return to the start position
    # [A------B------A]
    #     [     offset]
    #     [ x ]                
    #     [    actions]
    #      total_frames
    video_offset = get_offset_actions(args.actions_path)

    assert video_offset - n_prompt_frames >= 0
    video_offset = video_offset - n_prompt_frames
    # get prompt image/video
    x = load_prompt(
        args.prompt_path,
        video_offset=video_offset,
        n_prompt_frames=n_prompt_frames,
    )
    # x is [video_offset, video_offset+n_prompt_frames]
    # x.shape = [B=1,T=n_prompt_frames,C=3,H=360,W=640]

    # get input action stream
    actions = load_actions(args.actions_path, action_offset=video_offset)
    
    # actions is [video_offset:]
    # actions shape = [B=1,T=total_frames,D=2]
    total_frames = actions.shape[1]
    # import ipdb; ipdb.set_trace()

    # sampling inputs
    x = x.to(device)
    actions = actions.to(device)
    # import ipdb; ipdb.set_trace()
    # vae encoding
    B = x.shape[0]
    H, W = x.shape[-2:]
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t c h w -> (b t) c h w")
    # x shape = [t c h w]
    with torch.no_grad():
        with autocast("cuda", dtype=torch.half):
            x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
    # vae.patch_size = 20
    # 640 / 20 = 32 . 360 / 20 = 18 C=16?
    # x shape = [B=1,T=n_prompt_frames,C=16,H=18,W=32]
    x = x[:, :n_prompt_frames]

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    if args.num_frames is not None:
        total_frames = min(total_frames, args.num_frames)
    # sampling loop
    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)
        # import ipdb; ipdb.set_trace()
        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            # set up noise values
            t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # sliding window
            x_curr = x.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            # get model predictions
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    v = model(x_curr, t, actions[:, start_frame : i + 1])

            x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

            # get frame prediction
            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]


    # Decode and encode generated frames incrementally.  Keeping all decoded
    # 360x640 float frames on the GPU requires more than 8 GiB for long ABACA
    # trajectories and previously caused an OOM after generation had finished.
    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    output_path = os.path.join(args.output_path, nvtype, str(length), loc)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{name}.mp4")
    temporary_output_file = output_file + ".tmp.mp4"
    if os.path.exists(temporary_output_file):
        os.remove(temporary_output_file)

    generated_frame_count = total_frames - n_prompt_frames
    print(
        "streaming decoded output:",
        (generated_frame_count, H, W, 3),
    )
    try:
        with av.open(temporary_output_file, mode="w") as container:
            stream = container.add_stream("libx264", rate=args.fps)
            stream.width = W
            stream.height = H
            stream.pix_fmt = "yuv420p"
            stream.options = {}

            with torch.no_grad():
                chunk_size = 16
                for start in tqdm(
                    range(n_prompt_frames, x.shape[0], chunk_size),
                    desc="Decoding and encoding frames",
                ):
                    end = min(start + chunk_size, x.shape[0])
                    decoded = (
                        vae.decode(x[start:end] / scaling_factor) + 1
                    ) / 2
                    decoded = (
                        decoded.clamp_(0, 1)
                        .mul_(255)
                        .byte()
                        .permute(0, 2, 3, 1)
                        .cpu()
                        .numpy()
                    )
                    for image in decoded:
                        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                        frame.pict_type = "NONE"
                        for packet in stream.encode(frame):
                            container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
        os.replace(temporary_output_file, output_file)
    except Exception:
        if os.path.exists(temporary_output_file):
            os.remove(temporary_output_file)
        raise
    print(f"generation saved to {output_file}.")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--prompt-path",
        type=str,
        help="Path to image or video to condition generation on.",
        # default="sample_data/sample_image_0.png",
        default="data/LoopNav/ABA/snowy_village_1/15/05-04_13-57-20.avi"
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
        default="data/LoopNav/ABA/snowy_village_1/15/05-04_13-57-20.json",
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        help="If the prompt is a video, how many frames to condition on.",
        default=32,
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=None,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="output_more",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=10,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=10)
    parse.add_argument("--data-root", type=str, default="../test_data")
    parse.add_argument("--nvtype", type=str, default="ABA")
    parse.add_argument("--village-ids", nargs="+", default=["18", "19", "20"])
    parse.add_argument("--lengths", nargs="+", type=int, default=[5, 15, 30, 50])
    parse.add_argument("--oasis-ckpt", default="oasis500m.safetensors")
    parse.add_argument("--vae-ckpt", default="vit-l-20.safetensors")
    parse.add_argument("--device", default="cuda")
    args = parse.parse_args()
    initialize_models(args)
    # args.prompt_path = "data/LoopNav/ABA/plains_village_1/15/05-04_14-11-01.avi"
    # args.actions_path = "data/LoopNav/ABA/plains_village_1/15/05-04_14-11-01.json"
    for nvtype in [args.nvtype]:
        for length in args.lengths:
            dataset_dir = os.path.join(args.data_root, nvtype, str(length))
            locations = sorted(
                name for name in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, name))
                and name.rsplit('_', 1)[-1] in args.village_ids
            )
            for loc in locations:
                path = os.path.join(dataset_dir, loc)
                video_names = sorted(name for name in os.listdir(path) if name.endswith('.avi'))
                for video_name in video_names:
                    name = os.path.splitext(video_name)[0]
                    output_file = os.path.join(args.output_path, nvtype, str(length), loc, f"{name}.mp4")
                    if os.path.exists(output_file):
                        print(f"skipping existing {output_file}")
                        continue
                    args.prompt_path = os.path.join(path, video_name)
                    args.actions_path = os.path.join(path, f"{name}.json")

                    print("inference args:")
                    pprint(vars(args))
                    main(args,loc,length,name)
