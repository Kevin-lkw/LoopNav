# SGCS evaluation

This directory contains the audited SGCS pipeline `sam3_audit_v2_house_merged_ground_20260801`. Run commands from the LoopNav repository root in a separate SAM 3 environment.

## Environment

Follow the [SAM 3 installation and checkpoint instructions](https://github.com/facebookresearch/sam3#installation), including its PyTorch/CUDA requirements. Then install the metric dependencies:

```bash
python -m pip install opencv-python pillow numpy tqdm scipy
git clone https://github.com/facebookresearch/sam3.git evaluation/sam3
python -m pip install -e evaluation/sam3
```

Supply `--checkpoint /path/to/sam3.pt` to use local weights; otherwise SAM 3 uses its default checkpoint loading.

## Choose the entrypoint

| Entry | Use |
| --- | --- |
| [sgcs_eval.py](sgcs_eval.py) | Two clips, or batched ABA_FIXED/ABCA predictions |
| [eval_turn_sgcs.py](eval_turn_sgcs.py) | TURN predictions aligned to annotated return points |
| [eval_abaca_sgcs.py](eval_abaca_sgcs.py) | Additional ABACA cycle evaluation with a supplied manifest |
| [controlled_perturbation.py](controlled_perturbation.py) | SGCS sensitivity to controlled visual changes |
| [WorldMem manifest evaluator](../models/worldmem/README.md#sgcs) | Native-10-FPS WorldMem predictions |
| [tests/](tests/README.md) | Protocol checks without SAM weights |
| [legacy/](legacy/README.md) | Earlier SSIM/LPIPS/FVD utility |

ABACA is an additional workflow, separate from the main 720-sample benchmark; provide its cycle manifest and prediction roots explicitly. Run each entry with `--help` for options.

## Compare two clips

```bash
python evaluation/sgcs_eval.py \
  --video_a /path/to/ground_truth.avi --video_b /path/to/generated.mp4 \
  --device cuda --stride 5 --output results/sgcs.json
```

Pair mode aligns video tails. For the fixed benchmark, use [task-specific commands](../benchmark/README.md#evaluate-generated-videos): ABA_FIXED and TURN align to `return_point`, ABCA aligns tails, and WorldMem maps its native 10 FPS outputs to the raw 20 FPS source. Container FPS alone does not define frame correspondence.

## Metric protocol

| Setting | Value |
| --- | --- |
| Categories | building, tree, flower, grass, water, sky, sand, dirt |
| SAM prompts | `house` for building; category names for the others |
| Ground semantics | Dirt represents dirt/path; no separate path prompt |
| Confidence threshold | 0.5 |
| Centroid match threshold | 0.1 of the image diagonal |
| Temporal sampling | Every fifth aligned frame |
| Aggregation | Mean frame score per video, then mean across videos |

Zero-area frames receive zero and remain in the mean. Benchmark standard deviations use the population convention (`ddof=0`). Use [benchmark/verify_results.py](../benchmark/verify_results.py) to validate the archived summaries without SAM or GPU inference.

## Controlled perturbations

```bash
python evaluation/controlled_perturbation.py \
  --video_a /path/to/ground_truth.avi --perturbations all --stride 5 \
  --output results/perturbation_sgcs.json --vis-output results/perturbation_sgcs.jpg
```

Supported changes include translation, rotation, crop/resize, color changes, object deletion, and object swaps. Add `--ssim` or `--lpips` for alternative metrics; install `lpips` for the latter. Object deletion/swapping still requires SAM 3 masks. See `--help` for perturbation strengths.
