# WorldMem on LoopNav

This overlay adds an official-checkpoint loader, LoopNav LoRA adaptation, causal memory retrieval, batch generation, and native-10-FPS SGCS evaluation. It is intended for the [pinned WorldMem fork](https://github.com/Kevin-lkw/worldmem/tree/ac7f3cb1b5c6065f91547dd5ef8c149045db27ac), with upstream model code and dependencies installed.

## Setup

Starting from the LoopNav repository root:

```bash
export LOOPNAV_REPO="$PWD"
git clone https://github.com/Kevin-lkw/worldmem.git ../worldmem-loopnav
git -C ../worldmem-loopnav checkout ac7f3cb1b5c6065f91547dd5ef8c149045db27ac
cp -a models/worldmem/. ../worldmem-loopnav/
cd ../worldmem-loopnav
```

Install the upstream environment. Obtain the official diffusion and VAE checkpoints from the pinned Hugging Face revision:

```bash
hf download zeqixiao/worldmem_checkpoints \
  diffusion_only.ckpt vae_only.ckpt \
  --revision a3ae4d1c3a89260f800ddeab01ca56b7e63b10e2 \
  --local-dir checkpoints/official_worldmem
cp release_manifest.json checkpoints/official_worldmem/release_manifest.json
```

The runtime validates the repository/revision, expected filenames and sizes, and optionally file hashes. The official evaluation configuration is also hash-checked; keep `configurations/worldmem_official_eval.yaml` unchanged. Set data paths and training options in `loopnav_finetune_v2/config_p1.yaml` instead.

## Training

The adapter trains spatial, temporal, and memory LoRA branches with rank 8, action conditioning, and pose modules. Base diffusion weights and VAE remain frozen. The default configuration runs for 20,000 optimizer steps, with eight local frames and eight causally retrieved memory frames.

Set `data.root` in `config_p1.yaml` to your extracted dataset. Both `ABA` and `ABA_FIXED` directory names are supported for fixed-return training data. Training uses villages 1–16; villages 17–20 are held out. The recorded run used eight GPUs, batch size 2 per GPU, and no gradient accumulation (effective batch size 16). Adjust accumulation when changing the number of GPUs if you want to preserve that batch size.

```bash
python -m loopnav_finetune_v2.train \
  --config loopnav_finetune_v2/config_p1.yaml \
  --output-dir runs/loopnav-p1 --max-steps 20000
```

The trainer uses all visible CUDA devices and supports `--ckpt-path` for resuming a full Lightning checkpoint. Adapter checkpoints and full training checkpoints are saved separately. Use the adapter checkpoint with the rollout scripts below. Fine-tuned weights are not bundled with this code release.

## Generate the frozen subset

Prepare inputs using LoopNav's [benchmark script](../../benchmark/README.md#prepare-the-evaluation-subset). From the WorldMem root:

```bash
export LOOPNAV_INPUTS="$LOOPNAV_REPO/benchmark/prepared"
export LOOPNAV_PREDICTIONS="$LOOPNAV_REPO/predictions/worldmem"

python -m loopnav_finetune_v2.batch_rollout_adapters \
  --selection-manifest "$LOOPNAV_INPUTS/selection_manifest.json" \
  --config loopnav_finetune_v2/config_p1.yaml \
  --adapter /path/to/step0020000_adapter.pt \
  --output-root "$LOOPNAV_PREDICTIONS" \
  --device cuda:0 --amp-dtype fp16
```

Use the actual adapter filename produced by training. The manifest specifies context length and the complete return horizon for each sample. `--dry-run` checks batch planning without loading model weights. `--num-shards` and `--shard-index` partition work; existing matching outputs are reused unless `--overwrite` is set.

The native cadence is 10 FPS, obtained by taking every second source frame at 20 FPS. Pose ordering is `[x, y, z, pitch, yaw]`; pitch changes sign from the Minecraft convention and yaw is shifted by 180 degrees and wrapped. Camera actions follow pose differences (`cameraY = delta_pitch / 15`, `cameraX = -delta_yaw / 15`). Memory retrieval uses observed history only.

## SGCS

Run this command in the SAM 3 environment. This evaluator uses LoopNav's canonical audited metric without requiring the WorldMem training environment:

```bash
export LOOPNAV_METRICS_DIR="$LOOPNAV_REPO/evaluation"
python -m loopnav_finetune_v2.eval_sgcs_manifest \
  --selection-manifest "$LOOPNAV_INPUTS/selection_manifest.json" \
  --prediction-root "$LOOPNAV_PREDICTIONS" \
  --output-dir results/sgcs --device cuda \
  --stride 5 --tau 0.1 --confidence-threshold 0.5
```

Generated frame `t` aligns to raw source frame `2 * context_frames_10fps + 2 * t`. The evaluator checks exact source and prediction frame counts. Do not interpolate or duplicate generated frames, or use the four-baseline tail/return-point evaluator on native WorldMem outputs.

The environment variable points to the metric code when this overlay is copied outside LoopNav. Within the model directory, the evaluator can locate it automatically. Shards, resumption, `--dry-run`, and `--merge-only` are supported.

## Validation

In the WorldMem environment, from its repository root:

```bash
python -B -m loopnav_finetune_v2.tests.run_tests
```

These checks cover continuous camera action/pose agreement, button aggregation, and LoRA initialization/gradients, without loading checkpoints. To also check the full training inventory and real-data memory sampling:

```bash
LOOPNAV_DATA_ROOT=/path/to/LoopNav \
  python -B -m loopnav_finetune_v2.tests.run_tests --with-data
```

The real-data checks expect all 23,040 training trajectories. They are optional for users who downloaded only the benchmark subset.

## Additional utilities

`rollout_adapters.py` and `validate_adapters.py` support individual rollouts and adapter checks. `loopnav_eval.py`, `worldmem_official_runtime.py`, and `worldmem_official_loopnav.py` provide the shared data and inference helpers. Use the released 720-sample manifest and preparation script for the main benchmark.
