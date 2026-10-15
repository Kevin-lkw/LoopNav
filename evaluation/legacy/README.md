# Legacy video metrics

[calcmetric.py](calcmetric.py) computes SSIM, LPIPS, and FVD through the external `common_metrics_on_video_quality` package. It retains the earlier batch workflow and requires local input/output path configuration.

Use the [current SGCS entrypoints](../README.md) for the released benchmark. For controlled visual changes with optional SSIM or sim-LPIPS, use [controlled_perturbation.py](../controlled_perturbation.py).
