# Dataset quality checks and cleanup

These tools inspect the raw layout `<data-root>/<task>/<location>/<range>/<stem>.avi` and `.json`. Run commands from the repository root. Install `opencv-python` and `numpy` in the active Python environment.

## Audit first

```bash
python quality/audit_dataset_quality.py --data_root ./output \
  --tasks TURN ABA_FIXED ABCA --output results/quality_audit.json

python quality/qc_aba_variance.py --root ./output --nvtype ABA_FIXED --check_video
```

The public dataset uses `ABA` as its fixed-return directory name; pass that task name when auditing downloaded data. Reports do not change recordings. Use the exact paths written into a report when applying a later cleanup.

| Script | Purpose |
| --- | --- |
| [audit_dataset_quality.py](audit_dataset_quality.py) | Inventory, video/metadata consistency, and trajectory-quality reports |
| [audit_camera_discontinuities.py](audit_camera_discontinuities.py) | Camera discontinuity CSV and summary JSON |
| [qc_aba_variance.py](qc_aba_variance.py) | Fixed-return trajectory variation and optional video frame counts |
| [remove_quality_audit_issues.py](remove_quality_audit_issues.py) | Select pairs from a quality report and record a cleanup manifest |
| [remove_camera_discontinuities.py](remove_camera_discontinuities.py) | Select trajectories flagged by the camera audit |
| [remove_zero_byte_orphans.py](remove_zero_byte_orphans.py) | Select empty AVI files without matching metadata |
| [trim_excess_trajectories.py](trim_excess_trajectories.py) | Select excess pairs above a per-configuration quota |
| [legacy/](legacy/README.md) | Original in-place cleanup script |

## Review a cleanup selection

```bash
python quality/remove_zero_byte_orphans.py \
  --data_root ./output --task ABA_FIXED --max_village_id 16 \
  --manifest results/zero_byte_orphans.json
```

The current cleanup scripts default to a dry run and write a manifest. Adding `--delete` applies deletion. Run each script with `--help` to select report paths, task names, thresholds, and expected counts. The legacy `process.py` has different behavior and is documented separately.

The camera audit imports shared angle handling from [collection/trajectory_validation.py](../collection/trajectory_validation.py). Collection-time checks and their tests are described in the [collection guide](../collection/README.md).
