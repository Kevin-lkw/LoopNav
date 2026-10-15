# Legacy cleanup

[process.py](process.py) is the original cleanup/counting utility. It deletes invalid pairs and orphaned videos in place, then trims excess trajectories. It has no dry-run mode and uses the village target list even when another target group is requested.

Use the current [quality tools](../README.md) for report-driven cleanup. If reproducing the earlier workflow, work on a copy and invoke `python quality/legacy/process.py --output_path /path/to/copy --target village` from the repository root. Its shared target definitions live in [collection/target.py](../../collection/target.py).
