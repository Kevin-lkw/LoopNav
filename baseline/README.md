# Model setup and evaluation

The five `baseline/*-release/` directories are overlays for upstream model repositories. They contain LoopNav-specific files and their dependencies, not standalone model packages. Each upstream retains its own license, environment, and checkpoint requirements.

## Install an overlay

Use a separate clone/environment per model. The source revisions used locally are pinned in [release_sources.json](release_sources.json):

| Overlay | Upstream | Revision |
| --- | --- | --- |
| `open-oasis-release` | [etched-ai/open-oasis](https://github.com/etched-ai/open-oasis) | `f59deef2c019c212bd0c5a3a5b986a51f3701847` |
| `mineworld-release` | [microsoft/mineworld](https://github.com/microsoft/mineworld) | `b1b5a385aeef20aed0cc1e0d5f25f74776b17fdc` |
| `nwm-release` | [facebookresearch/nwm](https://github.com/facebookresearch/nwm) | `16bec8770d73ff9e82531bad36a8229cd1a02554` |
| `diamond-release` | [eloialonso/diamond](https://github.com/eloialonso/diamond) | `851cefb497733d27f1b85c804104638765860fca` |
| `worldmem-release` | [Kevin-lkw/worldmem](https://github.com/Kevin-lkw/worldmem) | `ac7f3cb1b5c6065f91547dd5ef8c149045db27ac` |

For example, starting at the LoopNav repository root:

```bash
export LOOPNAV_REPO="$PWD"
git clone https://github.com/etched-ai/open-oasis.git ../open-oasis-loopnav
git -C ../open-oasis-loopnav checkout f59deef2c019c212bd0c5a3a5b986a51f3701847
cp -a baseline/baseline/open-oasis-release/. ../open-oasis-loopnav/
```

Repeat with the matching repository, revision, and overlay for each model. Install upstream dependencies and obtain upstream or LoopNav-trained checkpoints before inference. No fine-tuned weights are included in this repository.

Prepare the fixed evaluation inputs with the [benchmark guide](benchmark/README.md#prepare-the-evaluation-subset), then set absolute paths so commands work from upstream directories:

```bash
export LOOPNAV_INPUTS="$LOOPNAV_REPO/baseline/benchmark/prepared"
export LOOPNAV_PREDICTIONS="$LOOPNAV_REPO/predictions"
```

The following commands run from each model's upstream repository root. Navigation examples use `ABA_FIXED`; repeat with `ABCA`. Both cover ranges 5/15/30/50 and villages 19/20. The prepared tree contains only the five selected trajectories per setting.

## Open OASIS

Download `oasis500m.safetensors` and `vit-l-20.safetensors` using the upstream instructions. The overlay includes its action loader and VAE changes, including chunked decoding for long trajectories.

```bash
python evaluate.py \
  --data-root "$LOOPNAV_INPUTS/gt" --nvtype ABA_FIXED \
  --village-ids 19 20 --n-prompt-frames 32 --ddim-steps 10 \
  --oasis-ckpt oasis500m.safetensors --vae-ckpt vit-l-20.safetensors \
  --output-path "$LOOPNAV_PREDICTIONS/oasis"

python evaluate_turn.py \
  --data-root "$LOOPNAV_INPUTS/turn_gt" --village-ids 19,20 \
  --context-frames 32 --ddim-steps 10 \
  --output-root "$LOOPNAV_PREDICTIONS/oasis"
```

## MineWorld

Use the 1.2B, 32-frame checkpoint and corresponding configuration. Set `--model_ckpt` to the checkpoint **file**. The navigation entry requires `--frames` and `--config`.

```bash
python inference.py \
  --config configs/1200M_32f.yaml --model_ckpt /path/to/mineworld.ckpt \
  --frames 32 --top_p 0.8 \
  --data_root "$LOOPNAV_INPUTS/gt" --nvtype ABA_FIXED --village_ids 19 20 \
  --output_dir "$LOOPNAV_PREDICTIONS/mineworld"

python evaluate_turn.py \
  --config configs/1200M_32f.yaml --checkpoint /path/to/mineworld.ckpt \
  --data-root "$LOOPNAV_INPUTS/turn_gt" --village-ids 19,20 \
  --output-root "$LOOPNAV_PREDICTIONS/mineworld"
```

## NWM

After installing the upstream environment, run `python -m pip install -r requirements-loopnav.txt` for the metric dependencies imported by the training utilities.

The adapted configuration uses 32 context frames. Navigation inference locates the checkpoint using `results_dir`, `run_name`, and `--ckp`: by default `logs/nwm_cdit_b_32/checkpoints/0200000.pth.tar`. TURN accepts a checkpoint path directly.

```bash
python inference.py --exp config/nwm_cdit_b.yaml --ckp 0200000 \
  --data_root "$LOOPNAV_INPUTS/gt" --nvtype ABA_FIXED --village_ids 19 20 \
  --output_dir "$LOOPNAV_PREDICTIONS/nwm"

python evaluate_turn.py --checkpoint /path/to/0200000.pth.tar \
  --data-root "$LOOPNAV_INPUTS/turn_gt" --village-ids 19,20 \
  --output-root "$LOOPNAV_PREDICTIONS/nwm"
```

For training, `processmc.py --data-root /path/to/LoopNav --output-root data/LoopNav_img --tasks ABA ABCA` extracts JPEG frames and `traj_data.pkl` annotations expected by the upstream loader. Set `datasets.mc.data_folder` in `config/nwm_cdit_b.yaml` to the chosen processed task. The loader uses villages 1–16 for training and 17–18 for validation; keep villages 19–20 for benchmark evaluation. Use the adapted `train.py` with the upstream distributed-training setup.

## DIAMOND

The overlay includes training/data changes, the world-model environment, and both inference entrypoints. Use a trained run directory containing `checkpoints/agent_versions/agent_epoch_00200.pt`.

```bash
cd src
python inference.py --checkpoint_dir /path/to/diamond-run --ckpt 00200 \
  --data_root "$LOOPNAV_INPUTS/gt" --nvtype ABA_FIXED --village_ids 19 20 \
  --output_dir "$LOOPNAV_PREDICTIONS/diamond"
cd ..

python src/evaluate_turn.py --run-dir /path/to/diamond-run --epoch 00200 \
  --data-root "$LOOPNAV_INPUTS/turn_gt" --village-ids 19,20 \
  --output-root "$LOOPNAV_PREDICTIONS/diamond"
```

DIAMOND saves both navigation and TURN outputs below `<output>/00200/`. Point SGCS at that epoch directory, or its `TURN` child for TURN evaluation. Configure data paths in `config/env/csgo.yaml` before using the upstream training entrypoint.

## WorldMem

See the dedicated [WorldMem guide](baseline/worldmem-release/README.md) for pinned official checkpoints, LoRA training, the 20,000-step adapter configuration, native 10 FPS inference, and manifest-based SGCS evaluation.

## SGCS

Use the [benchmark guide](benchmark/README.md#evaluate-generated-videos) for exact sample selection, alignment, and model-specific frame cadence. Use a separate SAM 3 environment from the model-generation environments.

## Controlled perturbations

From the LoopNav repository root, in the SAM 3 environment:

```bash
python baseline/metrics/controlled_perturbation.py \
  --video_a /path/to/ground_truth.avi --perturbations all --stride 5 \
  --output results/perturbation_sgcs.json \
  --vis-output results/perturbation_sgcs.jpg
```

Add `--ssim` or `--lpips` to compare pixel/perceptual metrics; install `lpips` for the latter. Object deletion and swapping still require SAM 3 for object masks. Run `--help` for perturbation strengths and options.

`metrics/calcmetric.py` is the legacy SSIM/LPIPS/FVD utility. It requires the external `common_metrics_on_video_quality` package and local path configuration; it is separate from the released SGCS benchmark.
