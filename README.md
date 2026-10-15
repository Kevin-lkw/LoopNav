# LoopNav

### Benchmarking Spatial Consistency in World Models

[Kewei Lian](https://kevin-lkw.github.io/) · [Shaofei Cai](https://phython96.github.io/) · [Yitao Liang](https://zero-lab-pku.github.io/personwise/liangyitao/) · [Anji Liu](https://liuanji.github.io/)

**[Project Website](https://kevin-lkw.github.io/LoopNav/)** · **[Paper](https://arxiv.org/abs/2505.22976)** · **[Dataset](https://huggingface.co/datasets/kevinLian/LoopNav)** · **[Citation](#citation)**

LoopNav is a dataset and benchmark for evaluating whether world models preserve spatial consistency when an agent revisits a previously observed location. It combines loop-based navigation in Minecraft with action annotations and a **Scene Graph Consistency Score (SGCS)** for comparing scene structure across observations.

[![LoopNav overview: loop trajectories and observations in a Minecraft village](https://kevin-lkw.github.io/LoopNav/assets/loop-overview.webp)](https://kevin-lkw.github.io/LoopNav/)

## Start here

| Directory | Contents |
| --- | --- |
| [collection/](collection/README.md) | Minecraft trajectory collection, recording, and camera control |
| [quality/](quality/README.md) | Dataset audits, cleanup, and trajectory quality reports |
| [models/](models/README.md) | Separate adapters and setup guides for five world models |
| [evaluation/](evaluation/README.md) | SGCS evaluation, task alignment, and controlled perturbations |
| [benchmark/](benchmark/README.md) | Fixed sample selection, input preparation, and archived results |
| [docs/](docs/README.md) | Dataset download/format, collector changes, and source provenance |

Every directory has a README describing its files and usage. Commands in the collection, quality, evaluation, and benchmark guides run from the repository root unless stated otherwise. Model commands run from their corresponding upstream checkout.

## Dataset and benchmark

LoopNav contains **250 hours of Minecraft navigation** with three complementary tasks:

| Task | Trajectory | Tests |
| --- | --- | --- |
| TURN | Rotate away and back | Consistency under changes in viewing direction |
| ABA | A → B → A | Reconstruction after retracing a route |
| ABCA | A → B → C → A | Reconstruction after revisiting via a different route |

The public dataset names fixed-return trajectories `ABA`; the code and results use `ABA_FIXED`. Navigation ranges are 5, 15, 30, and 50 blocks. See the [dataset guide](docs/dataset.md) for downloads, splits, and annotations.

The released benchmark uses **720 fixed trajectories** and includes **3,600 per-video scores** across Open OASIS, MineWorld, NWM, DIAMOND, and WorldMem. View the [results summary](benchmark/results/README.md) or follow the [reproduction guide](benchmark/README.md). Fine-tuned weights and generated videos are obtained separately.

```bash
git clone https://github.com/Kevin-lkw/LoopNav.git
cd LoopNav
python benchmark/verify_results.py
```

The command above checks sample identities and recomputes every archived summary statistic without model weights. Collection and evaluation tests are documented in their respective folders.

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
