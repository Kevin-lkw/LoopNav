import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


RANGES = (5, 15, 30, 50)


def numeric_suffix(value):
    try:
        return int(value.rsplit("_", 1)[-1])
    except ValueError:
        return None


def is_below(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


parser = argparse.ArgumentParser(
    description="Trim newest complete trajectory pairs above a per-bucket target"
)
parser.add_argument("--data_root", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--target_count", type=int, default=20)
parser.add_argument("--max_village_id", type=int, default=16)
parser.add_argument("--expect_pairs", type=int)
parser.add_argument("--manifest", required=True)
parser.add_argument("--delete", action="store_true")
args = parser.parse_args()

data_root = Path(args.data_root).resolve()
task_root = (data_root / args.task).resolve()
if not is_below(task_root, data_root):
    raise ValueError(f"Task root escapes data root: {task_root}")

selected = []
for location_path in sorted(task_root.glob("*_village_*")):
    village_id = numeric_suffix(location_path.name)
    if village_id is None or village_id > args.max_village_id:
        continue
    for nvrange in RANGES:
        bucket = location_path / str(nvrange)
        json_by_stem = {path.stem: path for path in bucket.glob("*.json")}
        avi_by_stem = {path.stem: path for path in bucket.glob("*.avi")}
        complete_stems = sorted(json_by_stem.keys() & avi_by_stem.keys())
        excess = len(complete_stems) - args.target_count
        if excess <= 0:
            continue
        newest_first = sorted(
            complete_stems,
            key=lambda stem: max(
                json_by_stem[stem].stat().st_mtime_ns,
                avi_by_stem[stem].stat().st_mtime_ns,
            ),
            reverse=True,
        )
        for stem in newest_first[:excess]:
            json_path = json_by_stem[stem].resolve()
            avi_path = avi_by_stem[stem].resolve()
            if not is_below(json_path, data_root) or not is_below(avi_path, data_root):
                raise ValueError(f"Selected path escapes data root: {stem}")
            selected.append(
                {
                    "task": args.task,
                    "location": location_path.name,
                    "range": nvrange,
                    "stem": stem,
                    "json_path": str(json_path),
                    "video_path": str(avi_path),
                    "bytes": json_path.stat().st_size + avi_path.stat().st_size,
                }
            )

if args.expect_pairs is not None and len(selected) != args.expect_pairs:
    raise RuntimeError(
        f"Expected {args.expect_pairs} excess pairs, selected {len(selected)}"
    )

manifest = {
    "created_at": datetime.now().astimezone().isoformat(),
    "mode": "delete" if args.delete else "dry_run",
    "data_root": str(data_root),
    "task": args.task,
    "target_count": args.target_count,
    "pair_count": len(selected),
    "file_count": len(selected) * 2,
    "total_bytes": sum(item["bytes"] for item in selected),
    "by_location": dict(Counter(item["location"] for item in selected)),
    "pairs": selected,
}
manifest_path = Path(args.manifest)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({key: value for key, value in manifest.items() if key != "pairs"}, indent=2))

if args.delete:
    deleted_files = 0
    deleted_bytes = 0
    for item in selected:
        for key in ("json_path", "video_path"):
            path = Path(item[key])
            deleted_bytes += path.stat().st_size
            path.unlink()
            deleted_files += 1
    print(f"deleted_files={deleted_files} deleted_bytes={deleted_bytes}")
