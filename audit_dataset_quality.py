import argparse
import json
import math
import os
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2


def wrapped_delta(current, previous):
    return (current - previous + math.pi) % (2 * math.pi) - math.pi


def percentile_nearest(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def audit_one(work):
    data_root_text, task, relative_path_text = work
    data_root = Path(data_root_text)
    relative_path = Path(relative_path_text)
    json_path = data_root / relative_path
    avi_path = json_path.with_suffix(".avi")
    result = {
        "task": task,
        "path": relative_path.as_posix(),
        "frames": 0,
        "max_speed_bps": 0.0,
        "max_camera_stride_1_deg": 0.0,
        "max_camera_stride_2_deg": 0.0,
        "closure_blocks": None,
        "issues": [],
    }

    try:
        records = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError("metadata is not a non-empty list")
        result["frames"] = len(records)

        required = ("frame_count", "x", "z", "yaw", "pitch")
        for index, record in enumerate(records):
            for field in required:
                value = float(record[field])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {field} at record {index}")

        for index in range(1, len(records)):
            previous = records[index - 1]
            current = records[index]
            delta_frame = float(current["frame_count"]) - float(
                previous["frame_count"]
            )
            if delta_frame != 1:
                result["issues"].append(
                    {
                        "type": "frame_sequence",
                        "indices": [index - 1, index],
                        "delta_frame": delta_frame,
                    }
                )
                break

            speed = (
                math.hypot(
                    float(current["x"]) - float(previous["x"]),
                    float(current["z"]) - float(previous["z"]),
                )
                * 20.0
                / delta_frame
            )
            result["max_speed_bps"] = max(result["max_speed_bps"], speed)

            camera = max(
                abs(
                    math.degrees(
                        wrapped_delta(
                            float(current["yaw"]),
                            float(previous["yaw"]),
                        )
                    )
                ),
                abs(
                    math.degrees(
                        wrapped_delta(
                            float(current["pitch"]),
                            float(previous["pitch"]),
                        )
                    )
                ),
            )
            result["max_camera_stride_1_deg"] = max(
                result["max_camera_stride_1_deg"], camera
            )

        for index in range(2, len(records), 2):
            previous = records[index - 2]
            current = records[index]
            camera = max(
                abs(
                    math.degrees(
                        wrapped_delta(
                            float(current["yaw"]),
                            float(previous["yaw"]),
                        )
                    )
                ),
                abs(
                    math.degrees(
                        wrapped_delta(
                            float(current["pitch"]),
                            float(previous["pitch"]),
                        )
                    )
                ),
            )
            result["max_camera_stride_2_deg"] = max(
                result["max_camera_stride_2_deg"], camera
            )

        if result["frames"] > 2500:
            result["issues"].append(
                {"type": "long", "value": result["frames"]}
            )
        if result["max_speed_bps"] >= 15.0:
            result["issues"].append(
                {"type": "speed", "value": result["max_speed_bps"]}
            )
        if result["max_camera_stride_1_deg"] > 30.0001:
            result["issues"].append(
                {
                    "type": "camera_stride_1",
                    "value": result["max_camera_stride_1_deg"],
                }
            )
        if result["max_camera_stride_2_deg"] > 30.0001:
            result["issues"].append(
                {
                    "type": "camera_stride_2",
                    "value": result["max_camera_stride_2_deg"],
                }
            )

        if task in {"ABA", "ABCA"}:
            result["closure_blocks"] = math.hypot(
                float(records[-1]["x"]) - float(records[0]["x"]),
                float(records[-1]["z"]) - float(records[0]["z"]),
            )
            if result["closure_blocks"] >= 2.0:
                result["issues"].append(
                    {"type": "closure", "value": result["closure_blocks"]}
                )

        capture = cv2.VideoCapture(str(avi_path))
        try:
            if not capture.isOpened():
                result["issues"].append({"type": "avi_open"})
            else:
                avi_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                avi_fps = float(capture.get(cv2.CAP_PROP_FPS))
                avi_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
                avi_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                decoded, frame = capture.read()
                if not decoded or frame is None:
                    result["issues"].append({"type": "avi_first_frame"})
                if avi_frames != result["frames"]:
                    result["issues"].append(
                        {
                            "type": "avi_frame_count",
                            "json_frames": result["frames"],
                            "avi_frames": avi_frames,
                        }
                    )
                if abs(avi_fps - 20.0) > 0.01:
                    result["issues"].append(
                        {"type": "avi_fps", "value": avi_fps}
                    )
                if (avi_width, avi_height) != (640, 360):
                    result["issues"].append(
                        {
                            "type": "avi_resolution",
                            "value": [avi_width, avi_height],
                        }
                    )
        finally:
            capture.release()
    except Exception as exc:
        result["issues"].append(
            {"type": "fatal", "message": f"{type(exc).__name__}: {exc}"}
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Audit LoopNav dataset quality")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/LoopNav"),
    )
    parser.add_argument("--tasks", nargs="+", default=("TURN", "ABA", "ABCA"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    inventory = {}
    work = []
    for task in args.tasks:
        task_root = data_root / task
        json_paths = sorted(task_root.rglob("*.json"))
        avi_paths = sorted(task_root.rglob("*.avi"))
        json_stems = {path.relative_to(task_root).with_suffix("") for path in json_paths}
        avi_stems = {path.relative_to(task_root).with_suffix("") for path in avi_paths}
        inventory[task] = {
            "json_count": len(json_paths),
            "avi_count": len(avi_paths),
            "paired_count": len(json_stems & avi_stems),
            "orphan_json": sorted(str(path) for path in json_stems - avi_stems),
            "orphan_avi": sorted(str(path) for path in avi_stems - json_stems),
        }
        for json_path in json_paths:
            if json_path.with_suffix(".avi").is_file():
                work.append((str(data_root), task, str(json_path.relative_to(data_root))))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(audit_one, work, chunksize=8))

    summary = {}
    for task in args.tasks:
        task_results = [item for item in results if item["task"] == task]
        frames = [item["frames"] for item in task_results]
        issue_counts = Counter(
            issue["type"]
            for item in task_results
            for issue in item["issues"]
        )

        def maximum(field):
            item = max(task_results, key=lambda value: value[field])
            return {"value": item[field], "path": item["path"]}

        closure_results = [
            item for item in task_results if item["closure_blocks"] is not None
        ]
        closure_max = None
        if closure_results:
            item = max(closure_results, key=lambda value: value["closure_blocks"])
            closure_max = {
                "value": item["closure_blocks"],
                "path": item["path"],
            }

        summary[task] = {
            "trajectory_count": len(task_results),
            "frame_stats": {
                "min": min(frames),
                "median": statistics.median(frames),
                "p95": percentile_nearest(frames, 0.95),
                "max": max(frames),
            },
            "max_speed_bps": maximum("max_speed_bps"),
            "max_camera_stride_1_deg": maximum("max_camera_stride_1_deg"),
            "max_camera_stride_2_deg": maximum("max_camera_stride_2_deg"),
            "max_closure_blocks": closure_max,
            "issue_counts": dict(sorted(issue_counts.items())),
        }

    issue_results = [item for item in results if item["issues"]]
    payload = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "tasks": args.tasks,
        "criteria": {
            "fps": 20.0,
            "resolution": [640, 360],
            "max_frames": 2500,
            "max_horizontal_speed_bps": 15.0,
            "max_camera_degrees": 30.0,
            "camera_strides": [1, 2],
            "max_loop_closure_blocks": 2.0,
        },
        "inventory": inventory,
        "summary": summary,
        "issue_trajectory_count": len(issue_results),
        "issues": issue_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, args.output)
    print(json.dumps({
        "output": str(args.output),
        "audited_trajectories": len(results),
        "issue_trajectory_count": len(issue_results),
        "inventory": inventory,
        "summary": summary,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
