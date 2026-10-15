# Archived benchmark results

[summary.csv](summary.csv) contains 60 task/range/model rows plus 15 pooled `ALL` rows. JSON files are grouped by model:

| Directory | Model |
| --- | --- |
| [oasis/](oasis/README.md) | Open OASIS |
| [mineworld/](mineworld/README.md) | MineWorld |
| [nwm/](nwm/README.md) | NWM |
| [diamond/](diamond/README.md) | DIAMOND |
| [worldmem/](worldmem/README.md) | WorldMem |

Each model has 12 JSON files, one for each task/range, with 60 per-video means per file. All models use the same [720 sample identities](../manifests/README.md). Scores were archived from the audited local runs; generated videos, masks, and fine-tuned weights are not bundled.

Means average per-video scores. Standard deviations use `ddof=0`; `ALL` pools the 240 video means for that task/model. Validate the complete archive from the repository root:

```bash
python benchmark/verify_results.py
```
