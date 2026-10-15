# Input preparation tests

From the repository root:

```bash
python -B -m unittest discover -s benchmark/tests -q
```

The tests use a temporary dataset to check missing/conflicting return markers, preservation of raw data, repeatable preparation, dry-run behavior, refusal to overwrite unrelated files, and dataset-relative path validation. No actual dataset or model weights are needed. Archived score validation is handled separately by [verify_results.py](../verify_results.py).
