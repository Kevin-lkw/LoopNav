# WorldMem LoopNav adaptation

`data.py` implements native-10-FPS samples, camera actions, and causal memory. `model.py` defines the LoRA adapters; `train.py` uses `config_p1.yaml` for training. `rollout_adapters.py` and `batch_rollout_adapters.py` generate individual or manifest-selected trajectories. `validate_adapters.py` checks adapters; `eval_sgcs_manifest.py` evaluates native-cadence predictions.

Run these modules from the upstream WorldMem root after installing the overlay, for example `python -m loopnav_finetune_v2.batch_rollout_adapters --help`. The [WorldMem guide](../README.md) covers setup, training, generation, and SGCS. Unit and optional dataset checks live in [tests/](tests/README.md).
