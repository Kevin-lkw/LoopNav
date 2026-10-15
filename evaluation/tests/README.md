# SGCS protocol tests

In the SGCS environment, from the repository root:

```bash
python -B -m unittest discover -s evaluation/tests -q
```

The tests use tiny synthetic videos and masks to verify return-point/tail alignment, unique return markers, audited SAM prompts, TURN segment bounds, and inclusion of zero-area frames in the mean. They do not load SAM weights or run a complete GPU benchmark.
