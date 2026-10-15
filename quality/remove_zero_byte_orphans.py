import argparse
import json
from datetime import datetime
from pathlib import Path


def numeric_suffix(value):
    try:
        return int(value.rsplit("_", 1)[-1])
    except ValueError:
        return None


parser = argparse.ArgumentParser(
    description="Remove zero-byte AVI files that have no matching JSON"
)
parser.add_argument("--data_root", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--max_village_id", type=int, default=16)
parser.add_argument("--expect_files", type=int)
parser.add_argument("--manifest", required=True)
parser.add_argument("--delete", action="store_true")
args = parser.parse_args()

data_root = Path(args.data_root).resolve()
task_root = (data_root / args.task).resolve()
task_root.relative_to(data_root)

selected = []
for location_path in sorted(task_root.glob("*_village_*")):
    village_id = numeric_suffix(location_path.name)
    if village_id is None or village_id > args.max_village_id:
        continue
    for video_path in sorted(location_path.glob("*/*.avi")):
        if video_path.stat().st_size != 0:
            continue
        metadata_path = video_path.with_suffix(".json")
        if metadata_path.exists():
            continue
        video_path = video_path.resolve()
        video_path.relative_to(data_root)
        selected.append(str(video_path))

if args.expect_files is not None and len(selected) != args.expect_files:
    raise RuntimeError(
        f"Expected {args.expect_files} zero-byte orphans, selected {len(selected)}"
    )

manifest = {
    "created_at": datetime.now().astimezone().isoformat(),
    "mode": "delete" if args.delete else "dry_run",
    "reason": "Zero-byte AVI files with no matching JSON",
    "data_root": str(data_root),
    "task": args.task,
    "file_count": len(selected),
    "files": selected,
}
manifest_path = Path(args.manifest)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {key: value for key, value in manifest.items() if key != "files"},
        indent=2,
    )
)

if args.delete:
    for value in selected:
        Path(value).unlink()
    print(f"deleted_files={len(selected)} deleted_bytes=0")
