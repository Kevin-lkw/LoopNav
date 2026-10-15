import argparse
import csv
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from trajectory_validation import wrapped_angle_delta


parser = argparse.ArgumentParser(
    description="Audit consecutive source-frame camera transitions"
)
parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("data/LoopNav"),
)
parser.add_argument(
    "--tasks",
    nargs="+",
    default=("ABA_FIXED", "ABCA"),
)
parser.add_argument("--max_village_id", type=int, default=16)
parser.add_argument("--threshold_degrees", type=float, default=30.0)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--output_csv", type=Path, required=True)
parser.add_argument("--summary_json", type=Path, required=True)
args = parser.parse_args()


def village_id(path):
    return int(path.parts[-3].rsplit("_", 1)[-1])


def scan_one(item):
    task, path_text, threshold = item
    path = Path(path_text)
    records = json.loads(path.read_text(encoding="utf-8"))
    events = []
    for current_index in range(1, len(records)):
        previous = records[current_index - 1]
        current = records[current_index]
        delta_pitch = math.degrees(
            wrapped_angle_delta(
                float(current["pitch"]),
                float(previous["pitch"]),
            )
        )
        delta_yaw = math.degrees(
            wrapped_angle_delta(
                float(current["yaw"]),
                float(previous["yaw"]),
            )
        )
        magnitude = max(abs(delta_pitch), abs(delta_yaw))
        if magnitude <= threshold:
            continue
        events.append(
            {
                "task": task,
                "metadata_path": str(path),
                "video_path": str(path.with_suffix(".avi")),
                "village": path.parts[-3],
                "bucket_or_range": path.parts[-2],
                "frame_transition_index": current_index,
                "previous_frame_index": current_index - 1,
                "current_frame_index": current_index,
                "delta_pitch_deg": delta_pitch,
                "delta_yaw_deg": delta_yaw,
                "max_abs_delta_deg": magnitude,
            }
        )
    return {
        "task": task,
        "metadata_path": str(path),
        "frame_count": len(records),
        "transition_count": max(0, len(records) - 1),
        "events": events,
    }


def discover():
    inventory = {}
    for task in args.tasks:
        root = args.data_root / task
        paths = sorted(
            path
            for path in root.rglob("*.json")
            if village_id(path) <= args.max_village_id
            and path.with_suffix(".avi").is_file()
        )
        inventory[task] = paths
    return inventory


def write_csv(events):
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "metadata_path",
        "video_path",
        "village",
        "bucket_or_range",
        "frame_transition_index",
        "previous_frame_index",
        "current_frame_index",
        "delta_pitch_deg",
        "delta_yaw_deg",
        "max_abs_delta_deg",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def main():
    inventory = discover()
    work = [
        (task, str(path), args.threshold_degrees)
        for task, paths in inventory.items()
        for path in paths
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(scan_one, work, chunksize=8))

    events = [event for result in results for event in result["events"]]
    write_csv(events)

    by_task = {}
    for task in args.tasks:
        task_results = [result for result in results if result["task"] == task]
        task_events = [event for event in events if event["task"] == task]
        by_task[task] = {
            "trajectories": len(task_results),
            "source_frames": sum(result["frame_count"] for result in task_results),
            "source_transitions": sum(
                result["transition_count"] for result in task_results
            ),
            "events": len(task_events),
            "affected_trajectories": len(
                {event["metadata_path"] for event in task_events}
            ),
            "by_village": dict(
                sorted(Counter(event["village"] for event in task_events).items())
            ),
        }
    summary = {
        "definition": (
            "An event is a consecutive 20 FPS source-frame transition where "
            "either absolute wrapped pitch or yaw delta is greater than "
            f"{args.threshold_degrees} degrees."
        ),
        "threshold_degrees": args.threshold_degrees,
        "total_trajectories": len(results),
        "total_events": len(events),
        "affected_trajectories": len(
            {event["metadata_path"] for event in events}
        ),
        "by_task": by_task,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
