# WorldMem adapter tests

From the upstream WorldMem root after installing the overlay:

```bash
python -B -m loopnav_finetune_v2.tests.run_tests
```

The default checks cover camera action/pose consistency, button aggregation, and LoRA initialization/gradients. They need the WorldMem environment, but no checkpoints. Use `LOOPNAV_DATA_ROOT=/path/to/LoopNav` and add `--with-data` for the full training inventory and real-data causal-memory checks. See the [validation guide](../../README.md#validation).
