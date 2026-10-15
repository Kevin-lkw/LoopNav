# Open OASIS results

This directory contains 12 audited result files: four navigation ranges for ABA_FIXED, four for ABCA, and four TURN angle buckets. Each file contains 60 video scores, covering the same frozen sample identities as the other models.

Filenames follow `oasis_<task>_<range>.json`. Each JSON records `mean_sgcs`, `std_sgcs`, sampled-frame counts, per-video identities/scores, and the original result file's `source_sha256`. Its `selection_manifest` link points to the [frozen manifest](../../manifests/minecraft.json).

See the [combined CSV](../summary.csv), [result conventions](../README.md), and [Open OASIS model guide](../../../models/oasis/README.md). Run `python benchmark/verify_results.py` from the repository root to validate all models together.
