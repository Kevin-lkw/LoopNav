# World model adapters

Each directory is an overlay for a pinned upstream repository. Install the upstream code and dependencies, then copy the matching overlay into that checkout. Run model commands from the upstream root; DIAMOND's navigation entry runs from its `src/` directory. Model weights and generated videos are not bundled.

| Model | Setup, generation, and training |
| --- | --- |
| Open OASIS | [oasis/](oasis/README.md) |
| MineWorld | [mineworld/](mineworld/README.md) |
| NWM | [nwm/](nwm/README.md) |
| DIAMOND | [diamond/](diamond/README.md) |
| WorldMem | [worldmem/](worldmem/README.md) |

Upstream revisions, licenses, and source hashes are recorded in [release_sources.json](../docs/release_sources.json) and the individual model folders. Keep separate environments for the models and SAM 3 evaluation.

## Workflow

1. Download the [dataset](../docs/dataset.md).
2. Prepare the fixed 720-sample subset using [benchmark/](../benchmark/README.md#prepare-the-evaluation-subset).
3. Follow a model's README to install its overlay and generate predictions.
4. Use the matching [evaluation protocol](../evaluation/README.md), preserving filenames and task/range/location layout.

WorldMem uses native 10 FPS and a dedicated manifest evaluator. The other four models use the navigation/TURN frame-index alignment described in the benchmark guide.
