import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Manifest and optionally delete trajectories with camera jumps"
)
parser.add_argument(
    "--events_csv",
    type=Path,
    default=Path(
        ""
        "loopnav_finetune_v2/audits/camera_discontinuities_train/"
        "events_over_15deg.csv"
    ),
)
parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("data/LoopNav"),
)
parser.add_argument("--threshold_degrees", type=float, default=30.0)
parser.add_argument("--fps", type=int, choices=(10, 20), default=10)
parser.add_argument("--expect_trajectories", type=int, default=None)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument(
    "--delete",
    action="store_true",
    help="Delete the manifested JSON/AVI pairs; the default is a dry run",
)
args = parser.parse_args()


def ensure_under_root(path, root):
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing path outside data root: {resolved}") from exc
    return resolved


def load_trajectories():
    selected = {}
    with args.events_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            magnitude = float(row["max_abs_delta_deg"])
            if magnitude <= args.threshold_degrees:
                continue
            metadata_path = Path(row["metadata_path"])
            item = selected.setdefault(
                str(metadata_path),
                {
                    "task": row["task"],
                    "metadata_path": str(metadata_path),
                    "video_path": row["video_path"],
                    "event_count": 0,
                    "max_abs_delta_deg": 0.0,
                    "native_transition_indices": [],
                },
            )
            item["event_count"] += 1
            item["max_abs_delta_deg"] = max(
                item["max_abs_delta_deg"],
                magnitude,
            )
            transition_index = row.get(
                "frame_transition_index",
                row.get("native_transition_index"),
            )
            item["native_transition_indices"].append(int(transition_index))
    return list(selected.values())


def validate_and_measure(trajectories):
    data_root = args.data_root.resolve()
    total_bytes = 0
    for item in trajectories:
        metadata_path = ensure_under_root(Path(item["metadata_path"]), data_root)
        video_path = ensure_under_root(Path(item["video_path"]), data_root)
        if metadata_path.suffix != ".json":
            raise ValueError(f"Unexpected metadata suffix: {metadata_path}")
        if video_path != metadata_path.with_suffix(".avi"):
            raise ValueError(
                f"JSON/AVI stems do not match: {metadata_path} {video_path}"
            )
        if not metadata_path.is_file() or not video_path.is_file():
            raise FileNotFoundError(
                f"Missing JSON/AVI pair: {metadata_path} {video_path}"
            )

        metadata_bytes = metadata_path.stat().st_size
        video_bytes = video_path.stat().st_size
        item["metadata_bytes"] = metadata_bytes
        item["video_bytes"] = video_bytes
        total_bytes += metadata_bytes + video_bytes
    return total_bytes


def write_manifest(trajectories, total_bytes):
    counts = Counter(item["task"] for item in trajectories)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_events_csv": str(args.events_csv.resolve()),
        "data_root": str(args.data_root.resolve()),
        "criterion": (
            "max(abs(wrapped pitch delta), abs(wrapped yaw delta)) "
            f"> {args.threshold_degrees} degrees at {args.fps} FPS"
        ),
        "fps": args.fps,
        "threshold_degrees": args.threshold_degrees,
        "trajectory_count": len(trajectories),
        "file_count": 2 * len(trajectories),
        "total_bytes": total_bytes,
        "by_task": dict(sorted(counts.items())),
        "trajectories": sorted(
            trajectories,
            key=lambda item: item["metadata_path"],
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, args.manifest)


def delete_trajectories(trajectories):
    deleted_files = 0
    deleted_bytes = 0
    for item in trajectories:
        for field, size_field in (
            ("video_path", "video_bytes"),
            ("metadata_path", "metadata_bytes"),
        ):
            path = Path(item[field])
            path.unlink()
            deleted_files += 1
            deleted_bytes += item[size_field]
    return deleted_files, deleted_bytes


def main():
    trajectories = load_trajectories()
    if (
        args.expect_trajectories is not None
        and len(trajectories) != args.expect_trajectories
    ):
        raise RuntimeError(
            f"Expected {args.expect_trajectories} trajectories, "
            f"found {len(trajectories)}"
        )
    total_bytes = validate_and_measure(trajectories)
    write_manifest(trajectories, total_bytes)

    counts = Counter(item["task"] for item in trajectories)
    print(
        json.dumps(
            {
                "mode": "delete" if args.delete else "dry_run",
                "manifest": str(args.manifest),
                "trajectory_count": len(trajectories),
                "file_count": 2 * len(trajectories),
                "total_bytes": total_bytes,
                "by_task": dict(sorted(counts.items())),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.delete:
        deleted_files, deleted_bytes = delete_trajectories(trajectories)
        print(
            f"deleted_files={deleted_files} deleted_bytes={deleted_bytes}",
            flush=True,
        )


if __name__ == "__main__":
    main()
