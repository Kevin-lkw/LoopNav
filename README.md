# LoopNav

### Benchmarking Spatial Consistency in World Models

[Kewei Lian](https://kevin-lkw.github.io/) · [Shaofei Cai](https://phython96.github.io/) · [Yitao Liang](https://zero-lab-pku.github.io/personwise/liangyitao/) · [Anji Liu](https://liuanji.github.io/)

**[Project Website](https://kevin-lkw.github.io/LoopNav/)** · **[Paper](https://arxiv.org/abs/2505.22976)** · **[Dataset](https://huggingface.co/datasets/kevinLian/LoopNav)** · **[Citation](#citation)**

LoopNav is a dataset and benchmark for evaluating whether world models preserve spatial consistency when an agent revisits a previously observed location. It combines loop-based navigation in Minecraft with action annotations and a **Scene Graph Consistency Score (SGCS)** for comparing scene structure across observations.

[![LoopNav overview: loop trajectories and observations in a Minecraft village](https://kevin-lkw.github.io/LoopNav/assets/loop-overview.webp)](https://kevin-lkw.github.io/LoopNav/)

## Overview

The dataset contains **250 hours of navigation video** across diverse Minecraft environments. Three trajectory types test complementary aspects of spatial memory:

| Task | Trajectory | What it tests |
| --- | --- | --- |
| **TURN** | Rotate away from a view and return to it | Consistency under changes in viewing direction |
| **ABA** | Travel from A to B and return to A | Reconstruction after retracing a route |
| **ABCA** | Travel from A through B and C, then return to A | Consistency when revisiting a location via a different route |

ABA and ABCA use navigation ranges of **5, 15, 30, and 50 blocks**. SGCS compares segmented scene elements to assess spatial consistency beyond pixel-level similarity. See the [project website](https://kevin-lkw.github.io/LoopNav/) for trajectory videos, qualitative comparisons, and benchmark results.

This repository provides the trajectory collection framework, adaptations for five world model baselines, and SGCS evaluation tools.

## Getting Started

```bash
git clone https://github.com/Kevin-lkw/LoopNav.git
cd LoopNav
```

All commands below run from this repository root unless stated otherwise. Install the dependencies for the workflow you need; data collection and model evaluation use separate environments.

| Goal | Start here |
| --- | --- |
| Use the released trajectories | [Dataset](#dataset) |
| Evaluate generated videos | [SGCS Evaluation](#sgcs-evaluation) |
| Run a world model baseline | [World Model Baselines](#world-model-baselines) |
| Collect new trajectories | [Data Collection](#data-collection) |
| Find a script | [Repository Structure](#repository-structure) |

## Dataset

The released data is hosted on [Hugging Face](https://huggingface.co/datasets/kevinLian/LoopNav). It includes `train.csv`, `val.csv`, and `test.csv` manifests, with videos and JSON annotations packaged into location archives under `TURN/`, `ABA/`, and `ABCA/`.

| Split | Trajectories |
| --- | ---: |
| Train | 23,040 |
| Validation | 2,880 |
| Test | 2,880 |

Each split covers all three trajectory types. Use the manifests distributed with the dataset to select trajectories.

### Download

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and download the split manifests:

```bash
python -m pip install -U huggingface_hub
hf download kevinLian/LoopNav train.csv val.csv test.csv \
  --repo-type dataset --local-dir data/LoopNav
```

Download and extract the locations you need. For example, this retrieves ABA trajectories for `desert_village_1`:

```bash
hf download kevinLian/LoopNav ABA/desert_village_1.tar \
  --repo-type dataset --local-dir data/LoopNav

mkdir -p data/LoopNav/ABA/desert_village_1
tar -xf data/LoopNav/ABA/desert_village_1.tar \
  -C data/LoopNav/ABA/desert_village_1
```

### Format

Each manifest row contains `path_type`, `radius`, `biome`, `video_path`, and `json_path`. Resolve the video and annotation paths relative to your extracted dataset root. Keep `radius` as a string: ABA/ABCA use numeric ranges, while TURN uses angular intervals such as `60-90`.

```text
data/LoopNav/
├── train.csv
├── val.csv
├── test.csv
├── ABA/
│   └── desert_village_1/
│       └── 5/
│           ├── <trajectory>.avi
│           └── <trajectory>.json
├── ABCA/
│   └── ...
└── TURN/
    └── ...
```

The collection code records **640 × 360 video at 20 FPS** alongside per-frame actions, position, camera orientation, and navigation metadata. It skips the initial 20 received frames, so the recorded `frame_count` field is offset from the saved video index; use the JSON list position when aligning annotations with saved frames.

The public dataset calls fixed-return trajectories `ABA`; code and benchmark result files use `ABA_FIXED`. The [benchmark preparation script](baseline/benchmark/README.md#prepare-the-evaluation-subset) maps the released data into the directory layout expected by the adapters.

## SGCS Evaluation

The released [SGCS evaluator](baseline/metrics/sgcs_eval.py) is the audited pipeline used for the reported results: `sam3_audit_v2_house_merged_ground_20260801`. It segments scene categories with SAM 3, matches objects by normalized centroid distance, and combines category scores with area weighting.

| Setting | Value |
| --- | --- |
| Categories | building, tree, flower, grass, water, sky, sand, dirt |
| SAM text prompts | `house` for building; category names for the others |
| Ground semantics | Dirt represents dirt/path; no separate path prompt |
| Confidence / centroid threshold | `0.5` / `0.1` |
| Temporal sampling | Every fifth aligned frame |
| Aggregation | Mean of frame scores per video, then mean across videos |

Frames with no detected category area receive zero and remain in the video average. Reported standard deviations are population standard deviations across videos.

### Set up SAM 3

Install the dependencies and checkpoints following the [SAM 3 instructions](https://github.com/facebookresearch/sam3#installation) in a dedicated environment, then install the metric dependencies:

```bash
python -m pip install opencv-python pillow numpy tqdm scipy
git clone https://github.com/facebookresearch/sam3.git baseline/metrics/sam3
python -m pip install -e baseline/metrics/sam3
```

Pass `--checkpoint /path/to/sam3.pt` to use local weights; otherwise SAM 3 uses its default checkpoint loading. Follow upstream access requirements for the weights.

### Compare two clips

```bash
python baseline/metrics/sgcs_eval.py \
  --video_a /path/to/ground_truth.avi \
  --video_b /path/to/generated.mp4 \
  --device cuda --stride 5 \
  --output results/sgcs.json
```

Pair mode aligns video tails. For the full benchmark, use the task-specific commands in the [benchmark guide](baseline/benchmark/README.md): ABA_FIXED and TURN use the recorded return point, ABCA uses tail alignment, and WorldMem uses native 10 FPS alignment against the 20 FPS source. Video container FPS alone does not define correspondence.

### Reproduce and inspect the benchmark

The release includes a fixed [720-sample selection](baseline/benchmark/manifests/minecraft.json), [per-video scores](baseline/benchmark/results), and a [CSV summary](baseline/benchmark/results/summary.csv) for all five models. Each task/range/model cell contains 60 videos from 12 locations. These are compact archived scores, not generated videos or newly rerun experiments.

```bash
# Validate selection consistency and recompute every released summary statistic.
python baseline/benchmark/verify_results.py
```

For SGCS sensitivity experiments, [controlled_perturbation.py](baseline/metrics/controlled_perturbation.py) supports translation, rotation, crop/resize, color changes, object deletion, and object swaps. See [usage](baseline/README.md#controlled-perturbations), including optional SSIM and sim-LPIPS comparisons.

## World Model Baselines

These directories contain overlays for the original model repositories. Install each upstream environment and checkpoints, then copy in the adapted files. Pinned upstream revisions and source hashes are recorded in [release_sources.json](baseline/release_sources.json).

| Model | Included LoopNav adaptation |
| --- | --- |
| [Open OASIS](baseline/baseline/open-oasis-release) | ABA/ABCA and TURN generation, action conversion, VAE helpers |
| [MineWorld](baseline/baseline/mineworld-release) | Generation, TURN evaluation, dataset/model helpers, configurations |
| [NWM](baseline/baseline/nwm-release) | Preprocessing, training, generation, TURN evaluation, configurations |
| [DIAMOND](baseline/baseline/diamond-release) | Dataset loader, training configuration, generation, TURN evaluation |
| [WorldMem](baseline/baseline/worldmem-release/README.md) | Official checkpoint loader, LoopNav adapters, training, native 10 FPS generation and SGCS |

See the [model setup guide](baseline/README.md) and [WorldMem guide](baseline/baseline/worldmem-release/README.md). Model weights, training outputs, and generated videos are obtained separately; this code release does not include fine-tuned checkpoints.

## Data Collection

The collection framework combines Mineflayer agent control, A* navigation, and Prismarine Viewer first-person rendering. Current drivers support TURN, ABA_FIXED, ABCA, and the additional ABACA_FIXED trajectory. The main benchmark covers the first three tasks.

### 1. Install dependencies

Use Linux with Python, Node.js/npm, and Java compatible with your Minecraft server. Headless rendering also needs the native Canvas/WebGL dependencies.

```bash
npm install
python -m pip install opencv-python numpy
```

`package.json` includes the required direct dependencies and the LoopNav forks of Mineflayer, Pathfinder, and Prismarine Viewer. See [Changes.md](Changes.md) for the navigation/rendering changes.

### 2. Configure the world

Start a Minecraft Java server, for example `java -Xmx2G -Xms2G -jar server.jar nogui`, then configure:

| Setting | Location |
| --- | --- |
| Target coordinates and names | `target.js` and `target.py` |
| Minecraft server ports | `--mc_ports`; default `8105` |
| Bot permissions | Grant each bot operator permission, e.g. `op Bot` |
| Recorded seed | `extra_info.seed` in `receiver.py`; set it to your world's seed |

`senderhl.js` accepts a local `targets.js` override and falls back to the included `target.js`. Receiver stream ports are separate from Minecraft ports and start at `8089` by default.

### 3. Record trajectories

Run from the repository root:

```bash
python run_navigation.py --nvtype ABA_FIXED \
  --output_path ./output --workers 1 --bot_names Bot --mc_ports 8105

python run_navigation.py --nvtype ABCA \
  --output_path ./output --workers 1 --bot_names Bot --mc_ports 8105

# The historical filename is run_trun.py.
python run_trun.py --output_path ./output \
  --workers 1 --bot_names Bot --mc_ports 8105
```

Navigation defaults to 20 accepted recordings per configuration. Use `--location` and `--nvrange` to restrict collection, or pass multiple bot names/server ports with `--workers` for concurrent recording. Run `--help` for all options.

The navigation driver validates camera discontinuities, horizontal speed, and loop closure before accepting a trajectory. Default limits are 30 degrees per native frame, 15 blocks/s, and 2 blocks of closure error. Videos and annotations are written under `output/<task>/<location>/<range>/<timestamp>.avi` and `.json`.

### 4. Audit recordings

```bash
python audit_dataset_quality.py --data_root ./output \
  --tasks TURN ABA_FIXED ABCA --output results/quality_audit.json
```

The audit produces reports without deleting recordings. Additional utilities handle camera discontinuities, orphaned files, and excess trajectories; inspect their `--help` before applying cleanup. `run.py` and `process.py` are retained as legacy utilities; `process.py` deletes files in place and has no dry run.

## Repository Structure

```text
LoopNav/
├── run_navigation.py          # Navigation collection and validation
├── run_trun.py                # TURN collection
├── senderhl.js / sender_trun.js
├── receiver.py                # Video, action, and return-point recording
├── smooth_camera.js           # Camera motion interpolation
├── trajectory_validation.py   # Trajectory quality checks
├── audit_dataset_quality.py
├── target.py / target.js
└── baseline/
    ├── README.md              # Model setup and commands
    ├── release_sources.json   # Upstream revisions and source hashes
    ├── baseline/              # Five model overlays
    ├── benchmark/             # Selection, preparation, archived results
    └── metrics/               # SGCS and controlled perturbations
```

## Validation

Lightweight checks for collection and benchmark preparation need no model weights:

```bash
python -B -m unittest -q test_trajectory_validation
node --test smooth_camera.test.js
python -B -m unittest discover -s baseline/benchmark/tests -q
python baseline/benchmark/verify_results.py
```

In the SGCS environment, run `python -B -m unittest discover -s baseline/metrics/tests -q` to check prompts, return-point alignment, and zero-weight frame handling. WorldMem has separate [adapter checks](baseline/baseline/worldmem-release/README.md#validation). These checks do not replace full GPU generation and evaluation.

## Citation

If you use LoopNav in your research, please cite the [paper](https://arxiv.org/abs/2505.22976):

```bibtex
@misc{lian2026loopnavbenchmarkingspatialconsistency,
  title={LoopNav: Benchmarking Spatial Consistency in World Models},
  author={Kewei Lian and Shaofei Cai and Yitao Liang and Anji Liu},
  year={2026},
  eprint={2505.22976},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.22976},
}
```

## Acknowledgments

Our collection framework builds on [Mineflayer](https://github.com/PrismarineJS/mineflayer), [Mineflayer Pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder), and [Prismarine Viewer](https://github.com/PrismarineJS/prismarine-viewer). The modified versions are available in our [Mineflayer](https://github.com/Kevin-lkw/mineflayer), [Pathfinder](https://github.com/Kevin-lkw/mineflayer-pathfinder), and [Viewer](https://github.com/Kevin-lkw/prismarine-viewer) forks. We also thank the authors of the world model baselines and [SAM 3](https://github.com/facebookresearch/sam3) for releasing their code and models.

For questions or reproducibility issues, please [open an issue](https://github.com/Kevin-lkw/LoopNav/issues).
