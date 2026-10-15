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

This repository provides the trajectory collection framework, adaptations for four world model baselines, and SGCS evaluation tools.

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

The baseline adapters use a different directory order, typically `<task>/<range>/<location>/`. Arrange the evaluation subset to match each adapter's paths; see [World Model Baselines](#world-model-baselines).

## SGCS Evaluation

[SGCS](baseline/metrics/sgcs_eval.py) uses SAM 3 to segment scene categories, matches objects by normalized centroid distance, and aggregates their consistency with area weighting.

**Release scope:** this branch currently includes the original two-video evaluator. The paper's reported results use the audited `sam3_audit_v2_house_merged_ground_20260801` pipeline, which changes the category vocabulary and adds task-specific alignment and sample selection. That complete benchmark pipeline is not yet included here; the example below describes the released evaluator.

### Environment

Set up Python and CUDA-enabled PyTorch using the [SAM 3 installation instructions](https://github.com/facebookresearch/sam3#installation). The current upstream requirements include Python 3.12+, PyTorch 2.7+, and CUDA 12.6+ for GPU inference. Then install the metric dependencies and SAM 3:

```bash
python -m pip install opencv-python pillow numpy tqdm
git clone https://github.com/facebookresearch/sam3.git baseline/metrics/sam3
python -m pip install -e baseline/metrics/sam3
```

Follow SAM 3's checkpoint access instructions before downloading weights. You can supply a local checkpoint with `--checkpoint /path/to/sam3.pt`; otherwise the evaluator uses SAM 3's default Hugging Face loading behavior.

### Compare two videos

Replace the example paths with a ground-truth video and its corresponding generated video:

```bash
python baseline/metrics/sgcs_eval.py \
  --video_a /path/to/ground_truth.avi \
  --video_b /path/to/generated.mp4 \
  --device cuda \
  --stride 5 \
  --output results/sgcs.json \
  --vis-output results/sgcs.jpg
```

The JSON contains `mean_sgcs`, per-frame scores, sampled frame indices, and alignment offsets. The optional image shows the paired frames and their scores.

| Option | Default | Description |
| --- | --- | --- |
| `--stride` | `5` | Sample every Nth frame |
| `--tau` | `0.1` | Centroid matching threshold, normalized by the image diagonal |
| `--confidence-threshold` | `0.5` | SAM 3 mask confidence threshold |
| `--categories` | Ten Minecraft scene categories | Text prompts, e.g. `building tree grass path sky` |
| `--checkpoint` | SAM 3 default loading | Path to local model weights |
| `--device` | CUDA when available, otherwise CPU | Inference device |

**Frame alignment:** the evaluator aligns the ends of unequal-length videos by skipping leading frames from the longer video. It also resizes frames to a common resolution. Prepare temporally corresponding clips at the same sampling rate before evaluation; this script does not read trajectory JSON or select the return phase automatically.

### Controlled perturbations

Use [controlled_perturbation.py](baseline/metrics/controlled_perturbation.py) to measure how SGCS responds to translations, rotations, crop-and-resize, color changes, object deletion, and object swaps:

```bash
python baseline/metrics/controlled_perturbation.py \
  --video_a /path/to/ground_truth.avi \
  --perturbations all \
  --stride 5 \
  --output results/perturbation_sgcs.json \
  --vis-output results/perturbation_sgcs.jpg
```

The script also supports `--ssim` and `--lpips` comparisons; install `lpips` for the latter. Object deletion and swapping require SAM 3 even when evaluating SSIM or sim-LPIPS. Additional examples are in the [evaluation guide](baseline/README.md#controlled-perturbation).

## World Model Baselines

The repository includes adapted files for the following models. Install each model's original codebase, dependencies, and checkpoints, then integrate the corresponding adaptations.

| Baseline | Included adaptations |
| --- | --- |
| Open OASIS | [Evaluation script](baseline/baseline/open-oasis-release/evaluate.py) |
| MineWorld | [Inference script](baseline/baseline/mineworld-release/inference.py) |
| NWM | [Preprocessing](baseline/baseline/nwm-release/processmc.py), [training](baseline/baseline/nwm-release/train.py), and [inference](baseline/baseline/nwm-release/inference.py) |
| DIAMOND | [Dataset loader](baseline/baseline/diamond-release/src/data/MCdata.py) and [world model environment](baseline/baseline/diamond-release/src/envs/world_model_env.py) |

The paper also evaluates WorldMem; its LoopNav adaptation is not included in this branch.

The included batch loops focus on ABA/ABCA and ranges `5, 15, 30, 50`. A typical evaluation layout is:

```text
<evaluation-data-root>/                 <generated-video-root>/
└── ABA/                                └── ABA/
    └── 5/                                  └── 5/
        └── desert_village_20/                   └── desert_village_20/
            ├── <trajectory>.avi                    └── <trajectory>.mp4
            └── <trajectory>.json
```

Before running, check the adapter's dataset paths, output paths, checkpoint locations, and device settings. Some batch paths are hard-coded, including `../test_data_more/` in Open OASIS and `../test_data/` in MineWorld. Run each adapter from its intended upstream project directory.

See [baseline/README.md](baseline/README.md) for model-specific commands. Paths in that guide are relative to the outer `baseline/` directory. The additional [calcmetric.py](baseline/metrics/calcmetric.py) script computes SSIM, LPIPS, and FVD; it requires the external `common_metrics_on_video_quality` package and local path configuration.

## Data Collection

The collection framework uses Mineflayer for agent control, A* path planning for navigation, and Prismarine Viewer for first-person rendering. The current `run.py` driver collects **ABA and ABCA** trajectories; TURN is included in the released dataset.

### 1. Install dependencies

Use Linux with Python, Node.js/npm, and a Java runtime compatible with your Minecraft server. The receiver uses Linux socket utilities, and headless rendering requires the native dependencies used by Canvas/WebGL.

```bash
npm install
npm install --no-save minimist vec3
python -m pip install opencv-python numpy
```

`package.json` installs the LoopNav forks of Mineflayer, Mineflayer Pathfinder, and Prismarine Viewer. Their navigation and recording changes are described in [Changes.md](Changes.md).

The current `senderhl.js` imports `./targets.js`, while the coordinate file in this checkout is named `target.js`. Before collecting, change that import to `./target.js`, or create this local compatibility link:

```bash
ln -s target.js targets.js
```

### 2. Configure the Minecraft server

Start your Minecraft Java server, for example from the directory containing its JAR:

```bash
java -Xmx2G -Xms2G -jar server.jar nogui
```

Configure the following before starting collection:

| Setting | Where to configure it |
| --- | --- |
| Server host and port | `mineflayer.createBot(...)` in `senderhl.js`; defaults to `localhost:8105` |
| Teleport permissions | Grant the bot operator permissions with `op Bot` in the server console |
| Target coordinates | `target.js`; update coordinates to match your Minecraft world |
| Target lists | `target.py`; keep names consistent with `target.js` |
| Recorded world seed | `extra_info.seed` in `receiver.py`; currently set to `42`, not read from the server |

The Minecraft server port is separate from the local video-stream port. `run.py` chooses a free receiver port in the range `8089–9089`.

### 3. Record trajectories

From the LoopNav repository root:

```bash
python run.py --name Bot --target village --output_path ./output
```

| Argument | Default | Description |
| --- | --- | --- |
| `--name` | `Bot` | Minecraft bot username |
| `--target` | `village` | Target group: `village`, `biome`, or `structure` |
| `--output_path` | `./output` | Root directory for recorded trajectories |

The driver iterates over the selected locations, ABA/ABCA, and ranges `5, 15, 30, 50`. It targets 20 recordings per configuration and checks existing JSON files before collecting more. Videos and annotations are saved as matching `.avi`/`.json` pairs under:

```text
output/<task>/<location>/<range>/<timestamp>.avi
output/<task>/<location>/<range>/<timestamp>.json
```

### 4. Inspect and clean recordings

[process.py](process.py) removes invalid trajectory pairs and orphaned videos, trims each configuration to at most 25 recordings by removing the longest excess trajectories, and prints counts.

**Run cleanup on a copy of your recordings:** it deletes files in place and has no dry-run mode. In the current release, the script always uses `village_targets` even though `--target` is required; edit its `targets` selection before processing biome or structure recordings.

```bash
python process.py --output_path ./output-copy --target village
```

Its 25-recording cleanup target differs from the collector's default of 20. Adjust `target_number` in `process.py` and `total_eps_number` in `run.py` if you need a consistent quota.

## Repository Structure

```text
LoopNav/
├── run.py                     # Collection orchestration and retries
├── senderhl.js                # Agent navigation and headless rendering
├── receiver.py                # Video and annotation recording
├── target.py                  # Location groups for the collector
├── target.js                  # Minecraft target coordinates
├── process.py                 # Recording cleanup and counts
├── bird.js                    # Auxiliary bird's-eye viewer
├── package.json               # JavaScript dependencies and LoopNav forks
├── Changes.md                 # Changes to upstream navigation/rendering code
└── baseline/
    ├── README.md              # Model and metric usage guide
    ├── baseline/              # Adaptations for OASIS, MineWorld, NWM, DIAMOND
    └── metrics/
        ├── sgcs_eval.py
        ├── controlled_perturbation.py
        └── calcmetric.py
```

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
