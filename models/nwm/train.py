# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

from isolated_nwm_infer import model_forward_wrapper
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import matplotlib.pyplot as plt 
import yaml
import wandb


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL

from distributed import init_distributed
from models import CDiT_models
from diffusion import create_diffusion
from datasets import TrainingDataset
from misc import transform

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    _, rank, device, _ = init_distributed()
    # rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    
    # Setup an experiment folder:
    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'nwm'),
                config=config
            )
            logger.info("Weights & Biases logging enabled")
    else:
        logger = create_logger(None)

    # Create model:
    tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    latent_size = config['image_size'] // 8

    assert config['image_size'] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    num_cond = config['context_size']
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4).to(device)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = float(config.get('lr', 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)
    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # load existing checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print('Searching for model from ', checkpoint_dir)
    start_epoch = 0
    train_steps = 0
    if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
        if os.path.isfile(latest_path) and config.get('from_checkpoint', 0):
            raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")
        latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
        print("Loading model from ", latest_path)
        latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False) 

        if "model" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['model'].items()}
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
            res = ema.load_state_dict(model_ckp, strict=True)
            print("Loading EMA model weights", res)
        else:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

        if "opt" in latest_checkpoint:
            opt_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['opt'].items()}
            opt.load_state_dict(opt_ckp)
            print("Loading optimizer params")
        
        if "epoch" in latest_checkpoint:
            start_epoch = latest_checkpoint['epoch'] + 1
        
        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]
        
        if "scaler" in latest_checkpoint:
            scaler.load_state_dict(latest_checkpoint["scaler"])
        
    # ~40% speedup but might leads to worse performance depending on pytorch version
    if args.torch_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[device])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = []
    test_dataset = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    goals_per_obs = int(data_config["goals_per_obs"])
                    if data_split_type == 'test':
                        goals_per_obs = 4 # standardize testing
                    
                    if "distance" in data_config:
                        min_dist_cat=data_config["distance"]["min_dist_cat"]
                        max_dist_cat=data_config["distance"]["max_dist_cat"]
                    else:
                        min_dist_cat=config["distance"]["min_dist_cat"]
                        max_dist_cat=config["distance"]["max_dist_cat"]

                    if "len_traj_pred" in data_config:
                        len_traj_pred=data_config["len_traj_pred"]
                    else:
                        len_traj_pred=config["len_traj_pred"]

                    dataset = TrainingDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_split_type,
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        min_dist_cat=min_dist_cat,
                        max_dist_cat=max_dist_cat,
                        len_traj_pred=len_traj_pred,
                        context_size=config["context_size"],
                        normalize=config["normalize"],
                        goals_per_obs=goals_per_obs,
                        transform=transform,
                        predefined_index=None,
                        traj_stride=1,
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        test_dataset.append(dataset)
                    print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    # combine all the datasets from different robots
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y, rel_t in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    B, T = x.shape[:2]
                    x = x.flatten(0,1)
                    x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215)
                    x = x.unflatten(0, (B, T))
                
                num_goals = T - num_cond
                x_start = x[:, num_cond:].flatten(0, 1)
                x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
                y = y.flatten(0, 1)
                rel_t = rel_t.flatten(0, 1)
                
                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)
                model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
                loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            if not bfloat_enable:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                if config.get('grad_clip_val', 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                scaler.step(opt)
                scaler.update()
            
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.detach().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = dist.get_world_size()*x_cond.shape[0]*steps_per_sec
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")
                
                # Log to wandb if enabled
                if rank == 0 and args.use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/steps_per_sec": steps_per_sec,
                        "train/samples_per_sec": samples_per_sec,
                        "train/epoch": epoch,
                        "train/step": train_steps
                    })
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    if bfloat_enable:
                        checkpoint.update({"scaler": scaler.state_dict()})
                    checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
                    torch.save(checkpoint, checkpoint_path)
                    if train_steps % (10*args.ckpt_every) == 0 and train_steps > 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                        torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if train_steps % args.eval_every == 0 and train_steps > 0:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, rank, config["batch_size"], config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond)
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}")
                
                # Log evaluation metrics to wandb if enabled
                if rank == 0 and args.use_wandb:
                    wandb.log({
                        "eval/perceptual_loss": sim_score,
                        "eval/eval_time": eval_time,
                        "eval/step": train_steps
                    })
                    
                    # Log sample images
                    if os.path.exists(save_dir):
                        for i in range(min(10, len(os.listdir(save_dir)))):
                            if f"{i}.png" in os.listdir(save_dir):
                                wandb.log({
                                    f"eval/samples_{i}": wandb.Image(os.path.join(save_dir, f"{i}.png")),
                                    "eval/step": train_steps
                                })

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


@torch.no_grad
def evaluate(model, vae, diffusion, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond):
    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )
    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True)
    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    for x, y, rel_t in loader:
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            samples = model_forward_wrapper((model, diffusion, vae), x, y, num_timesteps=None, latent_size=latent_size, device=device, num_cond=num_cond, num_goals=num_goals, rel_t=rel_t)
            x_start_pixels = x[:, num_cond:].flatten(0, 1)
            x_cond_pixels = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
            samples = samples * 0.5 + 0.5
            x_start_pixels = x_start_pixels * 0.5 + 0.5
            x_cond_pixels = x_cond_pixels * 0.5 + 0.5
            res = eval_model(x_start_pixels, samples)
            score += res.sum()
            n_samples += len(res)
        break
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(samples.shape[0], 10)):
            _, ax = plt.subplots(1,3,dpi=256)
            ax[0].imshow((x_cond_pixels[i, -1].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[1].imshow((x_start_pixels[i].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[2].imshow((samples[i].permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
            plt.savefig(f'{save_dir}/{i}.png')
            plt.close()

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score/n_samples
    return sim_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="nwm", help="Weights & Biases project name")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
