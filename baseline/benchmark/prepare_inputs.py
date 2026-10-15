#!/usr/bin/env python3
"""Prepare the frozen benchmark subset without changing downloaded data."""

import argparse
from collections import Counter
import json
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).parent / "manifests/minecraft.json"
TASKS = ("TURN", "ABA_FIXED", "ABCA")


def resolve_source(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a dataset-relative path: {relative}")
    path = root / relative
    # The public dataset calls fixed-return ABA simply ABA.
    if not path.is_file() and relative.parts[0] == "ABA":
        path = root / "ABA_FIXED" / Path(*relative.parts[1:])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def prepare(manifest_path, data_root, output_root, tasks=TASKS, dry_run=False):
    payload = json.loads(Path(manifest_path).read_text())
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    selected = [s for s in payload["samples"] if s["task"] in tasks]
    if not selected:
        raise ValueError("No samples selected")
    planned = []
    identities = set()
    restored = 0
    for sample in selected:
        identity = tuple(str(sample[k]) for k in ("task", "nav_range", "village", "stem"))
        if identity in identities:
            raise ValueError(f"Duplicate sample: {identity}")
        identities.add(identity)
        if any(Path(part).name != part or part in (".", "..") for part in identity):
            raise ValueError(f"Invalid sample identity: {identity}")
        source_json = resolve_source(data_root, sample["source_json"])
        source_video = resolve_source(data_root, sample["source_video"])
        records = json.loads(source_json.read_text())
        if not isinstance(records, list) or len(records) != sample["raw_frames"]:
            raise ValueError(f"Metadata length differs from frozen selection: {source_json}")
        point = sample["raw_return_point"]
        if not 0 <= point < len(records):
            raise ValueError(f"Invalid return point: {source_json}")
        if sample["task"] in ("TURN", "ABA_FIXED"):
            points = [i for i, row in enumerate(records) if row.get("return_point") is True]
            if points and points != [point]:
                raise ValueError(f"Return point differs from frozen selection: {source_json}")
            if not points:
                records[point]["return_point"] = True
                restored += 1
        task, length, village, stem = identity
        dest = output_root / "gt" / task / length / village / stem
        metadata = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
        runtime = dict(sample, index=len(planned), source_json=str(dest.with_suffix(".json")),
                       source_video=str(dest.with_suffix(".avi")), raw_source_json=str(source_json))
        links = [(dest.with_suffix(".avi"), source_video)]
        if task == "TURN":
            turn_dest = output_root / "turn_gt" / village / length / stem
            links.extend([(turn_dest.with_suffix(".avi"), source_video),
                          (turn_dest.with_suffix(".json"), dest.with_suffix(".json"))])
        planned.append((runtime, dest.with_suffix(".json"), metadata, links))

    # Check every destination before writing. Reruns accept only identical outputs.
    for _, dest, metadata, links in planned:
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() or not dest.is_file() or dest.read_text() != metadata:
                raise FileExistsError(f"Prepared metadata differs; use a new output root: {dest}")
        for link, target in links:
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != target.resolve():
                    raise FileExistsError(f"Refusing to replace existing path: {link}")
    runtime_payload = dict(payload, data_root=str(data_root), path_base="absolute",
                           total_samples=len(planned),
                           counts=dict(Counter(s["task"] for s in selected)),
                           samples=[p[0] for p in planned])
    runtime_text = json.dumps(runtime_payload, indent=2) + "\n"
    runtime_path = output_root / "selection_manifest.json"
    if runtime_path.exists() and runtime_path.read_text() != runtime_text:
        raise FileExistsError(f"Prepared selection differs; use a new output root: {runtime_path}")
    if not dry_run:
        for _, dest, metadata, links in planned:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_text(metadata)
            for link, target in links:
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.is_symlink():
                    link.symlink_to(target)
        runtime_path.write_text(runtime_text)
    return {"samples": len(planned), "return_points_restored": restored,
            "dry_run": dry_run, "manifest": str(runtime_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.data_root, args.output_root,
                             args.tasks, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
