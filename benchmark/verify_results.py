#!/usr/bin/env python3
"""Verify released sample identities and all benchmark summary statistics."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).parent
MODELS = {"oasis", "mineworld", "nwm", "diamond", "worldmem"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row):
    return (row["task"], str(row.get("nav_range", row.get("length"))), row["village"], row["stem"])


def validate(root=ROOT):
    manifest = json.loads((root / "manifests/minecraft.json").read_text())
    samples = {identity(s): s for s in manifest["samples"]}
    check(len(samples) == len(manifest["samples"]) == 720, "Expected 720 unique samples")
    buckets = Counter(i[:3] for i in samples)
    check(len(buckets) == 144 and set(buckets.values()) == {5}, "Expected 5 samples per setting")
    groups = defaultdict(list)
    seen = set()
    for path in sorted((root / "results").glob("*/*.json")):
        result = json.loads(path.read_text())
        check((path.parent / result["selection_manifest"]).resolve() ==
              (root / "manifests/minecraft.json").resolve(), f"Manifest link mismatch: {path}")
        key = (result["baseline"], result["task"], str(result["length"]))
        check(key not in seen and key[0] in MODELS, f"Invalid/duplicate group: {path}")
        seen.add(key)
        check(result["sgcs_pipeline_version"] == manifest["sgcs_pipeline_version"], f"Pipeline mismatch: {path}")
        check(not result["failures"], f"Failed evaluations: {path}")
        expected = {i for i in samples if i[:2] == key[1:]}
        videos = result["videos"]
        check(len(videos) == len(expected) == 60 and {identity(v) for v in videos} == expected,
              f"Selection mismatch: {path}")
        for video in videos:
            check(video["baseline"] == key[0], f"Model mismatch: {path}")
            check(video["source_json"] == samples[identity(video)]["source_json"], f"Source mismatch: {path}")
            check(0 <= video["mean_sgcs"] <= 1 and video["num_sampled_frames"] > 0, f"Invalid score: {path}")
        check_summary(result, videos, str(path))
        groups[key] = videos
        groups[(key[0], key[1], "ALL")].extend(videos)
    check(len(seen) == 60, "Expected 60 model/task/range groups")
    check({k[0] for k in seen} == MODELS, "Missing model")
    with (root / "results/summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) == len(groups) == 75, "Expected 75 summary rows")
    csv_keys = set()
    for row in rows:
        key = (row["baseline"], row["task"], row["length"])
        check(key not in csv_keys and key in groups, f"Invalid CSV group: {key}")
        csv_keys.add(key)
        check_summary(row, groups[key], str(key))
    return "PASS: 720 samples, 3,600 video scores, 60 groups, 75 summary rows"


def check_summary(summary, videos, label):
    scores = [v["mean_sgcs"] for v in videos]
    check(int(summary["num_videos"]) == len(videos), f"Video count mismatch: {label}")
    check(int(summary["num_sampled_frames"]) == sum(v["num_sampled_frames"] for v in videos),
          f"Frame count mismatch: {label}")
    for name, value in (("mean_sgcs", statistics.mean(scores)), ("std_sgcs", statistics.pstdev(scores))):
        check(math.isclose(float(summary[name]), value, rel_tol=1e-12, abs_tol=1e-12),
              f"{name} mismatch: {label}")


if __name__ == "__main__":
    print(validate())
