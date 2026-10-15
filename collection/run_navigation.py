import argparse
import glob
import os
import random
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

from target import biome_targets, structure_targets, village_targets


DEFAULT_BOT_NAMES = [f"Bot{index}" if index > 1 else "Bot" for index in range(1, 17)]
NAVIGATION_TYPES = ("ABA", "ABCA", "ABA_FIXED", "ABACA_FIXED")
RANGES = (5, 15, 30, 50)
SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Collect LoopNav trajectories with parallel workers and validation"
)
parser.add_argument("--nvtype", required=True, choices=NAVIGATION_TYPES)
parser.add_argument(
    "--output_path",
    default="data/LoopNav",
    help="Dataset root",
)
parser.add_argument(
    "--target",
    default="village",
    choices=("village", "biome", "structure"),
)
parser.add_argument("--episodes_per_location", type=int, default=20)
parser.add_argument("--location", action="append", default=None)
parser.add_argument(
    "--exclude_location",
    action="append",
    default=None,
    help="Skip a location; may be supplied multiple times",
)
parser.add_argument("--nvrange", action="append", type=int, choices=RANGES)
parser.add_argument(
    "--max_village_id",
    type=int,
    default=None,
    help="Only collect village targets whose numeric suffix is at most this value",
)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--bot_names", default="")
parser.add_argument("--mc_ports", default="8105")
parser.add_argument("--start_port", type=int, default=8089)
parser.add_argument("--retry_limit", type=int, default=6)
parser.add_argument(
    "--max_frames",
    type=int,
    default=2500,
    help="Maximum recorded 20 FPS frames before treating a sample as stuck",
)
parser.add_argument(
    "--max_camera_degrees",
    "--max_native_camera_degrees",
    dest="max_camera_degrees",
    type=float,
    default=30.0,
)
parser.add_argument(
    "--max_horizontal_speed_bps",
    "--max_speed_bps",
    dest="max_horizontal_speed_bps",
    type=float,
    default=15.0,
    help="Reject and retry samples with adjacent horizontal speed at/above this value",
)
parser.add_argument(
    "--max_loop_closure_distance",
    type=float,
    default=2.0,
    help="Reject and retry loops whose final XZ distance from the start reaches this value",
)
parser.add_argument("--end_hold_ms", type=int, default=0)
parser.add_argument("--post_tp_wait_ms", type=int, default=5000)
parser.add_argument("--no_shuffle", action="store_true")
args = parser.parse_args()
args.output_path = str(Path(args.output_path).expanduser().resolve())

print_lock = threading.Lock()


def log(*items):
    with print_lock:
        print(*items, flush=True)


def count_complete_pairs(location, nvrange):
    directory = Path(args.output_path) / args.nvtype / location / str(nvrange)
    json_stems = {Path(path).stem for path in glob.glob(str(directory / "*.json"))}
    avi_stems = {Path(path).stem for path in glob.glob(str(directory / "*.avi"))}
    return len(json_stems & avi_stems)


def receiver_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
            return True
        except OSError:
            return False


def terminate_process(process, label):
    if process is None or process.poll() is not None:
        return
    log(f"terminating {label} pid={process.pid}")
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_one_round(task, worker_index, bot_name, receiver_port, minecraft_port):
    location = task["location"]
    nvrange = task["nvrange"]
    round_number = task["round_number"]
    log(
        f"\n[worker {worker_index} {bot_name}] "
        f"{args.nvtype} {location} range {nvrange} round {round_number + 1}"
    )
    if not receiver_port_available(receiver_port):
        raise RuntimeError(f"Receiver port {receiver_port} is unavailable")

    receiver = None
    sender = None
    started = time.time()
    sample_id = f"{time.strftime('%m-%d_%H-%M-%S')}-{uuid.uuid4().hex[:12]}"
    sample_dir = Path(args.output_path) / args.nvtype / location / str(nvrange)
    try:
        receiver = subprocess.Popen(
            [
                sys.executable,
                "receiver.py",
                "--location",
                location,
                "--nvtype",
                args.nvtype,
                "--port",
                str(receiver_port),
                "--nvrange",
                str(nvrange),
                "--output_path",
                args.output_path,
                "--sample_id",
                sample_id,
                "--max_frames",
                str(args.max_frames),
                "--max_camera_degrees",
                str(args.max_camera_degrees),
                "--max_horizontal_speed_bps",
                str(args.max_horizontal_speed_bps),
                "--max_loop_closure_distance",
                str(args.max_loop_closure_distance),
            ],
            cwd=SCRIPT_DIR,
        )

        environment = os.environ.copy()
        node_path = "node_modules"
        environment["NODE_PATH"] = (
            node_path
            if not environment.get("NODE_PATH")
            else node_path + os.pathsep + environment["NODE_PATH"]
        )
        sender_command = [
            "node",
            "senderhl.js",
            "--location",
            location,
            "--nvtype",
            args.nvtype,
            "--port",
            str(receiver_port),
            "--nvrange",
            str(nvrange),
            "--name",
            bot_name,
            "--mc-port",
            str(minecraft_port),
            "--end-hold-ms",
            str(args.end_hold_ms),
            "--post-tp-wait-ms",
            str(args.post_tp_wait_ms),
        ]
        # The published ABA directory is the corrected ABA_FIXED dataset under
        # its final name, so replacement ABA samples must use the same
        # reverse-waypoint return strategy.
        if args.nvtype in {"ABA", "ABA_FIXED", "ABACA_FIXED"}:
            sender_command.append("--symmetric-return")
        sender = subprocess.Popen(
            sender_command,
            cwd=SCRIPT_DIR,
            env=environment,
        )

        while True:
            receiver_returncode = receiver.poll()
            if receiver_returncode is not None:
                break

            sender_returncode = sender.poll()
            if sender_returncode is not None and sender_returncode != 0:
                log(
                    f"[worker {worker_index}] sender exited before receiver "
                    f"with rc={sender_returncode}"
                )
                terminate_process(receiver, "receiver")
                receiver_returncode = sender_returncode
                break
            time.sleep(0.2)

        sender_returncode = sender.poll()
        if sender_returncode not in (None, 0):
            receiver_returncode = sender_returncode
        if receiver_returncode != 0:
            removed = []
            for suffix in (".avi", ".json"):
                failed_path = sample_dir / f"{sample_id}{suffix}"
                if failed_path.exists():
                    failed_path.unlink()
                    removed.append(failed_path.name)
            if removed:
                log(
                    f"[worker {worker_index}] removed failed sample files: "
                    f"{', '.join(removed)}"
                )
        log(
            f"[worker {worker_index}] finished rc={receiver_returncode} "
            f"elapsed={time.time() - started:.1f}s"
        )
        return receiver_returncode
    finally:
        terminate_process(receiver, "receiver")
        terminate_process(sender, "sender")


def numeric_suffix(value):
    try:
        return int(value.rsplit("_", 1)[-1])
    except ValueError:
        return None


def choose_locations():
    if args.location:
        locations = list(args.location)
    elif args.target == "village":
        locations = list(village_targets)
    elif args.target == "biome":
        locations = list(biome_targets)
    else:
        locations = list(structure_targets)

    excluded_locations = set(args.exclude_location or ())
    locations = [
        location for location in locations if location not in excluded_locations
    ]
    if args.max_village_id is not None:
        locations = [
            location
            for location in locations
            if numeric_suffix(location) is not None
            and numeric_suffix(location) <= args.max_village_id
        ]
    if not args.no_shuffle:
        random.shuffle(locations)
    return locations


def build_tasks(locations):
    ranges = args.nvrange or RANGES
    tasks = []
    for location in locations:
        for nvrange in ranges:
            have = count_complete_pairs(location, nvrange)
            need = max(0, args.episodes_per_location - have)
            if need:
                log(
                    f"{args.nvtype} {location} range {nvrange}: "
                    f"have {have}, need {need}"
                )
            for offset in range(need):
                tasks.append(
                    {
                        "location": location,
                        "nvrange": nvrange,
                        "round_number": have + offset,
                    }
                )
    if not args.no_shuffle:
        random.shuffle(tasks)
    return tasks


def parse_bot_names():
    names = (
        [item.strip() for item in args.bot_names.split(",") if item.strip()]
        if args.bot_names
        else DEFAULT_BOT_NAMES
    )
    if args.workers > len(names):
        raise ValueError(f"Need {args.workers} bot names, got {len(names)}")
    return names[: args.workers]


def parse_minecraft_ports():
    ports = [int(item.strip()) for item in args.mc_ports.split(",") if item.strip()]
    if not ports:
        raise ValueError("At least one Minecraft port is required")
    return ports


def worker_loop(worker_index, bot_name, minecraft_port, task_queue):
    receiver_port = args.start_port + worker_index
    while True:
        try:
            task = task_queue.get_nowait()
        except Empty:
            return
        try:
            failures = 0
            while True:
                try:
                    returncode = run_one_round(
                        task,
                        worker_index,
                        bot_name,
                        receiver_port,
                        minecraft_port,
                    )
                except Exception as exc:
                    log(f"[worker {worker_index}] exception: {exc}")
                    returncode = 1
                if returncode == 0:
                    break
                failures += 1
                if failures > args.retry_limit:
                    raise RuntimeError(
                        f"{args.nvtype} {task['location']} range "
                        f"{task['nvrange']} exceeded retry limit"
                    )
                log(
                    f"[worker {worker_index}] retry {failures}/"
                    f"{args.retry_limit}"
                )
                time.sleep(2)
        finally:
            task_queue.task_done()


def main():
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    receiver_ports = set(range(args.start_port, args.start_port + args.workers))
    minecraft_ports = parse_minecraft_ports()
    overlapping_ports = receiver_ports & set(minecraft_ports)
    if overlapping_ports:
        raise ValueError(
            "Receiver ports overlap configured Minecraft server ports: "
            f"{sorted(overlapping_ports)}"
        )

    tasks = build_tasks(choose_locations())
    log(f"Total pending {args.nvtype} samples: {len(tasks)}")
    if not tasks:
        return

    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)

    bot_names = parse_bot_names()
    log(
        f"Running {args.workers} workers with bots {bot_names} "
        f"and Minecraft ports {minecraft_ports}"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                worker_loop,
                worker_index,
                bot_name,
                minecraft_ports[worker_index % len(minecraft_ports)],
                task_queue,
            )
            for worker_index, bot_name in enumerate(bot_names)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
