# Dataset

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

The public dataset calls fixed-return trajectories `ABA`; code and benchmark result files use `ABA_FIXED`. The [benchmark preparation script](../benchmark/README.md#prepare-the-evaluation-subset) maps the released data into the directory layout expected by the adapters.
