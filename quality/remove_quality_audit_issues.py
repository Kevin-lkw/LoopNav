import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Manifest and optionally delete trajectories selected from a quality audit"
)
parser.add_argument("--audit_report", type=Path, required=True)
parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("data/LoopNav"),
)
parser.add_argument("--tasks", nargs="+", required=True)
parser.add_argument("--issue_type", default="camera")
parser.add_argument(
    "--expect",
    action="append",
    default=[],
    help="Expected selected count in TASK=COUNT form; may be repeated",
)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--delete", action="store_true")
args = parser.parse_args()


def parse_expected_counts():
    expected = {}
    for value in args.expect:
        task, separator, count = value.partition("=")
        if not separator:
            raise ValueError(f"Invalid --expect value: {value}")
        expected[task] = int(count)
    return expected


def resolve_under_root(relative_path, root):
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing path outside data root: {path}") from exc
    return path


def main():
    report_bytes = args.audit_report.read_bytes()
    report = json.loads(report_bytes)
    data_root = args.data_root.resolve()
    selected = []
    seen = set()

    for item in report["issues"]:
        task = item.get("kind", item.get("task"))
        if task not in args.tasks:
            continue
        matching_issues = [
            issue for issue in item["issues"] if issue["type"] == args.issue_type
        ]
        if not matching_issues:
            continue
        if item["path"] in seen:
            raise RuntimeError(f"Duplicate audit path: {item['path']}")
        seen.add(item["path"])

        json_path = resolve_under_root(item["path"], data_root)
        avi_path = json_path.with_suffix(".avi")
        if json_path.suffix != ".json":
            raise ValueError(f"Unexpected metadata suffix: {json_path}")
        if not json_path.is_file() or not avi_path.is_file():
            raise FileNotFoundError(f"Missing JSON/AVI pair: {json_path} {avi_path}")

        selected.append(
            {
                "kind": task,
                "json_path": str(json_path.relative_to(data_root)),
                "avi_path": str(avi_path.relative_to(data_root)),
                "frames": item["frames"],
                "issues": matching_issues,
                "json_bytes": json_path.stat().st_size,
                "avi_bytes": avi_path.stat().st_size,
            }
        )

    counts = Counter(item["kind"] for item in selected)
    expected = parse_expected_counts()
    for task, expected_count in expected.items():
        actual_count = counts[task]
        if actual_count != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} {task} trajectories, found {actual_count}"
            )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_audit_report": str(args.audit_report.resolve()),
        "source_audit_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "data_root": str(data_root),
        "selection": {
            "tasks": args.tasks,
            "issue_type": args.issue_type,
        },
        "trajectory_count": len(selected),
        "file_count": 2 * len(selected),
        "total_bytes": sum(
            item["json_bytes"] + item["avi_bytes"] for item in selected
        ),
        "by_task": dict(sorted(counts.items())),
        "trajectories": sorted(selected, key=lambda item: item["json_path"]),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, args.manifest)

    print(
        json.dumps(
            {
                "mode": "delete" if args.delete else "dry_run",
                "manifest": str(args.manifest),
                "trajectory_count": len(selected),
                "file_count": 2 * len(selected),
                "total_bytes": manifest["total_bytes"],
                "by_task": manifest["by_task"],
            },
            indent=2,
        ),
        flush=True,
    )

    if args.delete:
        deleted_files = 0
        deleted_bytes = 0
        for item in selected:
            for field, size_field in (
                ("avi_path", "avi_bytes"),
                ("json_path", "json_bytes"),
            ):
                path = resolve_under_root(item[field], data_root)
                path.unlink()
                deleted_files += 1
                deleted_bytes += item[size_field]
        print(
            f"deleted_files={deleted_files} deleted_bytes={deleted_bytes}",
            flush=True,
        )


if __name__ == "__main__":
    main()
