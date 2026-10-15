# Minecraft benchmark

This directory freezes the sample selection and compact per-video SGCS results used by the [project website](https://kevin-lkw.github.io/LoopNav/). The metric version is `sam3_audit_v2_house_merged_ground_20260801`.

## Selection and results

| Property | Value |
| --- | --- |
| Models | OASIS, MineWorld, NWM, DIAMOND, WorldMem |
| Tasks | TURN, ABA_FIXED (public dataset name: ABA), ABCA |
| Navigation ranges | 5, 15, 30, 50 blocks |
| TURN angle buckets | 60–90, 90–120, 120–150, 150–180 degrees |
| Locations | Desert, plains, savanna, snowy, taiga, zombie villages; IDs 19 and 20 |
| Samples per location/task/range | 5 |
| Samples per task | 240 |
| Videos per model/task/range | 60 |
| Total | 720 unique trajectories, 3,600 model/video scores |

[manifests/minecraft.json](manifests/minecraft.json) is the authoritative selection. All five models share the same identities. ABA/ABCA incorporate the September 2026 quality replacements and return-point repairs. TURN scores come from the audited TURN run and the WorldMem native-10-FPS extension. Selection paths are relative to the extracted dataset root.

[results/summary.csv](results/summary.csv) contains 60 task/range/model rows and 15 pooled `ALL` rows. Each result JSON contains video identities, scores, sampled-frame counts, and the SHA-256 of its original local result file. The compact export omits frame masks and category-level arrays. Source videos and model outputs are not bundled.

The reported mean is the unweighted mean of per-video means. Standard deviations use the population convention (`ddof=0`). `ALL` pools the 240 video means for a task/model; it does not average four standard deviations.

```bash
# From the LoopNav repository root; standard library only.
python baseline/benchmark/verify_results.py
```

## Prepare the evaluation subset

Download the [dataset](https://huggingface.co/datasets/kevinLian/LoopNav) location archives for all six biomes, village IDs 19 and 20, and tasks ABA/ABCA/TURN. Extract each archive into `<data-root>/<task>/<location>/`; see the [root README](../../README.md#download) for an example. Use the released Hugging Face split manifests.

```bash
python baseline/benchmark/prepare_inputs.py \
  --data-root data/LoopNav \
  --output-root baseline/benchmark/prepared \
  --dry-run

# Run the same command without --dry-run to prepare the inputs.
python baseline/benchmark/prepare_inputs.py \
  --data-root data/LoopNav \
  --output-root baseline/benchmark/prepared
```

Preparation verifies all source files and metadata lengths before writing. For TURN/ABA_FIXED, it checks an existing return-point marker against the frozen selection, or restores an absent marker in the prepared metadata copy. Conflicting markers fail. ABCA metadata is preserved. Original annotations and videos remain untouched.

Videos are symlinks, so keep the source dataset available. The script also accepts a local dataset whose fixed-return directory is named `ABA_FIXED`. Use a fresh output directory when changing selection or data roots; identical reruns are accepted.

```text
prepared/
├── selection_manifest.json             # Absolute runtime paths for WorldMem
├── gt/<task>/<range>/<village>/        # Navigation adapters and SGCS
│   ├── <stem>.avi                     # Link to the original video
│   └── <stem>.json                    # Prepared metadata
└── turn_gt/<village>/<angle>/          # TURN adapter input layout
    ├── <stem>.avi
    └── <stem>.json
```

`--tasks ABA_FIXED ABCA` or `--tasks TURN` prepares a subset. Defaults include all three tasks. The script checks metadata lengths and file presence; it does not decode all videos. WorldMem's evaluator additionally validates video frame counts against the manifest.

## Evaluate generated videos

Install the [SGCS environment](../../README.md#set-up-sam-3). Generate predictions using the [model overlays](../README.md), keeping each selected trajectory's filename. Commands below run from the LoopNav root.

### ABA_FIXED and ABCA: OASIS, MineWorld, NWM, DIAMOND

```bash
python baseline/metrics/sgcs_eval.py \
  --gt-root baseline/benchmark/prepared/gt \
  --baseline-root /absolute/path/to/predictions/oasis \
  --baseline-name oasis \
  --path-types ABA_FIXED ABCA --lengths 5 15 30 50 \
  --all --stride 5 --tau 0.1 --confidence-threshold 0.5 \
  --device cuda --output-dir results/oasis --resume
```

Replace the model name and prediction root for each model. Default villages are the 12 benchmark locations. Prediction layout is `<prediction-root>/<task>/<range>/<village>/<stem>.mp4`. DIAMOND adds an epoch directory: point `--baseline-root` at that directory.

ABA_FIXED aligns `GT[return_point + t]` with `prediction[t]`. ABCA aligns tails, matching the original adapter return segment. Both compare frame indices, without resampling based on video container FPS.

### TURN: OASIS, MineWorld, NWM, DIAMOND

```bash
python baseline/metrics/eval_turn_sgcs.py \
  --gt-root baseline/benchmark/prepared/turn_gt \
  --baselines oasis \
  --baseline-root oasis=/absolute/path/to/predictions/oasis/TURN \
  --village-ids 19 20 --stride 5 --tau 0.1 \
  --confidence-threshold 0.5 --device cuda \
  --output-dir results/oasis-turn
```

TURN also aligns `GT[return_point + t]` with `prediction[t]`, stopping at the shorter available segment. The evaluator supports `--dry-run`, resumable shard files, `--num-shards`, `--shard-index`, and `--merge-only`. Baseline roots for this command must include the final `TURN` directory.

### WorldMem: all tasks

Use WorldMem's [manifest evaluator](../baseline/worldmem-release/README.md#sgcs). Its predictions remain at native 10 FPS: generated frame `t` corresponds to raw frame `2 * context_frames_10fps + 2 * t`. Do not evaluate these outputs with the four-baseline index alignment above, duplicate frames, or interpolate them to 20 FPS.

## Metric settings

The eight categories are building, tree, flower, grass, water, sky, sand, and dirt. Building uses SAM's `house` prompt; dirt represents the audited dirt/path ground semantics with a single `dirt` prompt. The default confidence threshold is 0.5 and centroid threshold is 0.1 of the image diagonal. Sample every fifth aligned frame.

Zero-area frames receive zero and remain in the average. Keep this convention when comparing to the archived scores. Model checkpoints, sampling seeds, and generation settings also affect fresh runs; the code release contains no fine-tuned weights and does not claim a new full GPU reproduction.
