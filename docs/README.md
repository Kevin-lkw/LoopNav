# Documentation and provenance

| File | Purpose |
| --- | --- |
| [dataset.md](dataset.md) | Download the Hugging Face release, extract location archives, and interpret videos/annotations |
| [collector_changes.md](collector_changes.md) | Changes made to the Mineflayer, Pathfinder, and Prismarine Viewer forks |
| [release_sources.json](release_sources.json) | Pinned upstream revisions and hashes of the research source and packaged files |

`release_sources.json` records each released file relative to the repository root. `source_sha256` identifies the original local research source; `release_sha256` identifies the current packaged file. The original `source` labels remain unchanged for traceability. Model-specific setup instructions live under [models/](../models/README.md).

## Directory migration

| Previous location | Current location |
| --- | --- |
| Root collection scripts and `package.json` | [collection/](../collection/README.md) |
| Root audit/cleanup scripts | [quality/](../quality/README.md) |
| `baseline/baseline/*-release/` | [models/](../models/README.md), one directory per model |
| `baseline/metrics/` | [evaluation/](../evaluation/README.md) |
| `baseline/benchmark/` | [benchmark/](../benchmark/README.md) |
| Flat benchmark result JSONs | `benchmark/results/<model>/` |
| `Changes.md`, `baseline/release_sources.json` | This directory |

The TURN entrypoints are now `collection/run_turn.py` and `collection/sender_turn.js` (formerly `run_trun.py` and `sender_trun.js`). Legacy utilities are documented in their own subdirectories. Update local commands and scripts to these paths; old root entrypoints are no longer present.
