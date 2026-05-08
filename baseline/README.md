# World Model Baselines and Scene Graph Consistency Score

This repository contains adapted evaluation scripts for Minecraft world model
baselines and metric scripts for computing Scene Graph Consistency Score (SGCS)
on our dataset.

The currently supported baselines are:

- Open OASIS
- DIAMOND
- MineWorld
- NWM

The metric code provides:

- `metrics/sgcs_eval.py`: SAM3-based Scene Graph Consistency Score evaluation.
- `metrics/controlled_perturbation.py`: controlled perturbations on GT videos/labels,
  with SGCS, SSIM, or sim-LPIPS evaluation.

## Repository Layout

```text
.
├── baseline/
│   ├── open-oasis-release/
│   │   └── evaluate.py
│   ├── mineworld-release/
│   │   └── inference.py
│   ├── nwm-release/
│   │   ├── inference.py
│   │   └── processmc.py
│   └── diamond-release/
│       └── src/
│           ├── data/MCdata.py
│           └── envs/world_model_env.py
├── metrics/
│   ├── sgcs_eval.py
│   ├── controlled_perturbation.py
│   └── calcmetric.py
└── README.md
```

## Dataset Format

The scripts assume the dataset is organized by path type, rollout length, and
village:

```text
baseline/test_data/
├── ABA/
│   ├── 5/
│   │   └── desert_village_20/
│   │       ├── trajectory_0.avi
│   │       └── trajectory_0.json
│   └── ...
└── ABCA/
    └── ...
```

Default SGCS evaluation covers:

- path types: `ABA`, `ABCA`
- lengths: `5`, `15`, `30`, `50`
- villages: `desert_village_20`, `plains_village_20`, `savanna_village_20`,
  `snowy_village_20`, `taiga_village_20`, `zombie_village_20`

Generated baseline videos should follow the same tree structure:

```text
baseline/<baseline-output-root>/
└── ABA/
    └── 5/
        └── desert_village_20/
            └── trajectory_0.mp4
```

## Environment Setup

Install the common Python dependencies used by the metric scripts:

```bash
pip install opencv-python pillow numpy tqdm torch torchvision
```

For `controlled_perturbation.py --lpips`, also install:

```bash
pip install lpips
```

### Install SAM3

Both `metrics/sgcs_eval.py` and `metrics/controlled_perturbation.py` require
Meta SAM3:

```bash
git clone https://github.com/facebookresearch/sam3.git metrics/sam3
cd metrics/sam3
pip install -e .
```

Alternatively, install SAM3 somewhere else and make sure `sam3` is importable in
the active Python environment. If `--checkpoint` is not provided, SAM3 will use
its default checkpoint loading behavior.

## Baseline Evaluation

Each baseline has its own upstream environment and checkpoint requirements.
Before running a baseline, install the corresponding original repository and
dependencies, then use the adapted files in `baseline/` to run on this dataset.

### Open OASIS

Adapted entry:

```text
baseline/open-oasis-release/evaluate.py
```

Example:

```bash
cd baseline/open-oasis-release
python evaluate.py \
  --output-path output \
  --n-prompt-frames 32 \
  --fps 10 \
  --ddim-steps 10
```

The script iterates over `ABA`/`ABCA`, lengths `5,15,30,50`, and the default six
villages. Update the dataset path inside the script if your local data root is
different.

### MineWorld

Adapted entry:

```text
baseline/mineworld-release/inference.py
```

Example:

```bash
cd baseline/mineworld-release
python inference.py \
    --data_root "/path/to/validation/dataset" \
    --model_ckpt "path/to/ckpt" \
    --config "path/to/config" \
    --demo_num 1 \
    --frames 31 \
    --accelerate-algo 'naive' \
    --top_p 0.8 \
    --output_dir "path/to/output"
```

Optional arguments:

- `--nvtype ABA` or `--nvtype ABCA` to evaluate path type.
- `--index <idx>` to evaluate a specific trajectory pair.

### NWM

Adapted entries:

```text
baseline/nwm-release/processmc.py
baseline/nwm-release/inference.py
```

First preprocess videos into the frame layout expected by NWM:

```bash
cd baseline/nwm-release
python processmc.py
```
Train the model and obtain the corresponding ckpt, then

```bash
torchrun train.py --config config/nwm_cdit_b.yaml --ckpt-every 10000 --eval-every 10000 --bfloat16 1 --epochs 600 --torch-compile 0 --use-wandb 
```

Then run inference:

```bash
python inference.py \
  --output_dir output \
  --exp config/nwm_cdit_b.yaml \
  --ckp ckpt_path \
  --num_sec_eval 5 \
  --input_fps 4 \
  --datasets mc \
  --eval_type rollout
```

Check `processmc.py` and `inference.py` for local paths, checkpoint locations,
and GPU device settings before running.

### DIAMOND

Adapted files:

```text
baseline/diamond-release/src/data/MCdata.py
baseline/diamond-release/src/envs/world_model_env.py
```

Use these files inside the DIAMOND codebase to load our Minecraft dataset and
run DIAMOND evaluation with its original training/evaluation scripts. Place the
generated videos under a tree matching the dataset layout so SGCS can compare
them against GT videos.

## Scene Graph Consistency Score
`metrics/sgcs_eval.py` computes SGCS using SAM3 for object segmentation and matching.
### Single Pair

```bash
python metrics/sgcs_eval.py \
  --video_a path/to/video_a.mp4 \
  --video_b path/to/video_b.mp4 \
```

Important options:

- `--stride`: frame sampling interval. Default: `5`.
- `--tau`: normalized centroid distance threshold for object matching. Default:
  `0.1`.
- `--confidence-threshold`: SAM3 mask threshold. Default: `0.5`.
- `--categories`: text prompts used for segmentation. Defaults to Minecraft
  scene categories such as `building`, `tree`, `grass`, `path`, `sky`.
- `--checkpoint`: optional local SAM3 checkpoint path.
- `--device`: `cuda` or `cpu`.

The output JSON contains per-video scores, per-frame scores, failed cases, and
the aggregate `mean_sgcs`/`std_sgcs`.

## Controlled Perturbation

`metrics/controlled_perturbation.py` generates synthetic perturbations from GT
videos and evaluates how each metric responds.

Supported perturbations:

- `small_translation`
- `small_rotation`
- `crop_resize`
- `color_change`
- `delete_object`
- `swap_objects`

Run SGCS perturbation evaluation on one video:

```bash
python metrics/controlled_perturbation.py \
  --video_a baseline/test_data/ABA/5/desert_village_20/trajectory_0.avi \
  --perturbations all \
  --stride 5 \
  --output perturb_sgcs.json \
  --vis-output perturb_sgcs.jpg
```

Run batch mode over one directory containing village subdirectories:

```bash
python metrics/controlled_perturbation.py \
  --video-root baseline/test_data/ABA/5 \
  --perturbations all \
  --output perturb_batch_sgcs.json
```

Compare against SSIM:

```bash
python metrics/controlled_perturbation.py \
  --video_a baseline/test_data/ABA/5/desert_village_20/trajectory_0.avi \
  --perturbations all \
  --ssim \
  --output perturb_ssim.json
```

Compare against sim-LPIPS:

```bash
python metrics/controlled_perturbation.py \
  --video_a baseline/test_data/ABA/5/desert_village_20/trajectory_0.avi \
  --perturbations all \
  --lpips \
  --output perturb_lpips.json
```

Object-aware perturbations (`delete_object`, `swap_objects`) still require SAM3
even when evaluating SSIM or sim-LPIPS.
