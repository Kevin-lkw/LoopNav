# MineWorld model configurations

These YAML files define 300M, 700M, and 1.2B models with the supported context lengths. The released benchmark uses `1200M_32f.yaml`. Match the model configuration to its checkpoint, and set the tokenizer checkpoint path under `checkpoints/vae/`. Pass the chosen file with `--config`; see the [MineWorld guide](../README.md).
