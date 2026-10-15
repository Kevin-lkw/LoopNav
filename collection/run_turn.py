import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import argparse
import glob
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

from target import village_targets

start_port = 8089
max_port = 9089
nv_type = 'TURN'
angle_bins = [
    ('60-90', 60.0, 90.0),
    ('90-120', 90.0, 120.0),
    ('120-150', 120.0, 150.0),
    ('150-180', 150.0, 180.0),
]

parser = argparse.ArgumentParser(description='Collect TURN view-rotation data')
parser.add_argument('--name', type=str, default='Bot', help='Bot name')
parser.add_argument('--output_path', type=str, default='./output', help='Output path')
parser.add_argument('--episodes_per_location', type=int, default=20, help='Total episodes per location')
parser.add_argument('--max_locations', type=int, default=None, help='Optional limit for sampling/debugging')
parser.add_argument('--location', action='append', default=None, help='Collect only specific location(s)')
parser.add_argument('--spawn_range', type=int, default=20, help='Random spawn range around target location')
parser.add_argument('--no_shuffle', action='store_true', help='Do not shuffle target order')
parser.add_argument('--workers', type=int, default=1, help='Concurrent bot/receiver workers')
parser.add_argument('--bot_names', type=str, default='', help='Comma-separated bot names for workers')
parser.add_argument('--mc_ports', type=str, default='8105', help='Comma-separated Minecraft server ports')
parser.add_argument('--start_port', type=int, default=start_port, help='First receiver port')
parser.add_argument('--retry_limit', type=int, default=6, help='Retries per failed sample')
args = parser.parse_args()
print_lock = threading.Lock()

def log(*items):
    with print_lock:
        print(*items, flush=True)

def count_bin_files(location, bin_name, min_angle, max_angle):
    path = f'{args.output_path}/{nv_type}/{location}/{bin_name}'
    if not os.path.exists(path):
        return 0
    count = 0
    for json_path in glob.glob(os.path.join(path, '*.json')):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if not data:
                continue
            extra = data[0].get('extra_info', {})
            angle = extra.get('turn_angle')
            if (
                isinstance(angle, (int, float))
                and min_angle <= angle <= max_angle
            ):
                count += 1
        except Exception:
            continue
    return count

def run_one_round(task, worker_idx, bot_name, receiver_port, mc_port):
    location = task['location']
    turn_order = task['turn_order']
    turn_angle = task['turn_angle']
    bin_name = task['bin_name']
    round_num = task['round_num']

    log(f'\n[worker {worker_idx} {bot_name}] {location} {bin_name} {turn_order} round {round_num + 1} angle {turn_angle:.2f}')
    t0 = time.time()
    receiver = subprocess.Popen([
        sys.executable, str(SCRIPT_DIR / 'receiver.py'),
        '--location', location,
        '--nvtype', nv_type,
        '--port', str(receiver_port),
        '--nvrange', bin_name,
        '--output_path', args.output_path,
        '--turn_angle', f'{turn_angle:.6f}',
        '--turn_order', turn_order,
    ])
    log(f'[worker {worker_idx}] receiver.py started on port {receiver_port}')

    env = os.environ.copy()
    node_path = str(SCRIPT_DIR / 'node_modules')
    env['NODE_PATH'] = (
        node_path if not env.get('NODE_PATH')
        else node_path + os.pathsep + env['NODE_PATH']
    )
    sender = subprocess.Popen([
        'node', str(SCRIPT_DIR / 'sender_turn.js'),
        '--location', location,
        '--nvtype', nv_type,
        '--port', str(receiver_port),
        '--nvrange', bin_name,
        '--name', bot_name,
        '--mc-port', str(mc_port),
        '--turn-angle', f'{turn_angle:.6f}',
        '--turn-order', turn_order,
        '--spawn-range', str(args.spawn_range),
    ], env=env)
    log(f'[worker {worker_idx}] sender_turn.js started as {bot_name} on MC port {mc_port}')

    receiver.wait()
    returncode = receiver.returncode
    log(f'[worker {worker_idx}] receiver.py finished {returncode}')

    if sender.poll() is None:
        log(f'[worker {worker_idx}] terminating sender_turn.js')
        sender.terminate()
        time.sleep(2)
        if sender.poll() is None:
            sender.kill()
    else:
        log(f'[worker {worker_idx}] sender_turn.js exited early {sender.returncode}')

    log(f'[worker {worker_idx}] round time: {time.time() - t0:.2f}s')
    return returncode

def build_tasks(targets):
    if args.episodes_per_location % len(angle_bins) != 0:
        raise ValueError('--episodes_per_location must be divisible by the number of angle bins')
    per_bin = args.episodes_per_location // len(angle_bins)
    tasks = []

    for location in targets:
        for bin_name, min_angle, max_angle in angle_bins:
            have = count_bin_files(location, bin_name, min_angle, max_angle)
            need = max(0, per_bin - have)
            log(f'{location} {bin_name}: have {have}, need {need}')
            for idx in range(need):
                tasks.append({
                    'location': location,
                    'bin_name': bin_name,
                    'turn_order': random.choice(['left_right', 'right_left']),
                    'turn_angle': random.uniform(min_angle, max_angle),
                    'round_num': have + idx,
                })
    if not args.no_shuffle:
        random.shuffle(tasks)
    return tasks

def worker_loop(worker_idx, bot_name, mc_port, task_queue):
    receiver_port = args.start_port + worker_idx
    if receiver_port > max_port:
        raise RuntimeError('Not enough receiver ports for requested workers')

    while True:
        try:
            task = task_queue.get_nowait()
        except Empty:
            return
        try:
            retries = 0
            while True:
                res = run_one_round(task, worker_idx, bot_name, receiver_port, mc_port)
                if res == 0:
                    break
                retries += 1
                if retries > args.retry_limit:
                    location = task['location']
                    bin_name = task['bin_name']
                    raise RuntimeError(f'{location} {bin_name} failed too many times')
                log(f"[worker {worker_idx}] {task['location']} {task['bin_name']} failed, retry {retries}")
        finally:
            task_queue.task_done()

def parse_bot_names():
    if args.bot_names:
        names = [item.strip() for item in args.bot_names.split(',') if item.strip()]
    elif args.workers == 1:
        names = [args.name]
    else:
        names = [
            'Bot', 'Bot2', 'Bot3', 'Bot4', 'Bot5', 'Bot6', 'Bot7', 'Bot8',
            'Bot9', 'Bot10', 'Bot11', 'Bot12', 'Bot13', 'Bot14', 'Bot15', 'Bot16',
        ]
    if args.workers > len(names):
        raise ValueError(f'Need at least {args.workers} bot names, got {len(names)}')
    return names[:args.workers]

def parse_mc_ports():
    ports = [int(item.strip()) for item in args.mc_ports.split(',') if item.strip()]
    if not ports:
        raise ValueError('Need at least one Minecraft server port')
    return ports

def main():
    if args.workers < 1:
        raise ValueError('--workers must be >= 1')
    if args.location:
        targets = args.location
    else:
        targets = list(village_targets)
    if not args.no_shuffle:
        random.shuffle(targets)
    if args.max_locations is not None:
        targets = targets[:args.max_locations]

    tasks = build_tasks(targets)
    log(f'Total pending TURN samples: {len(tasks)}')
    if not tasks:
        return

    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)

    bot_names = parse_bot_names()
    mc_ports = parse_mc_ports()
    log(f'Running {args.workers} workers with bots {bot_names} and MC ports {mc_ports}')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for worker_idx, bot_name in enumerate(bot_names):
            mc_port = mc_ports[worker_idx % len(mc_ports)]
            futures.append(executor.submit(worker_loop, worker_idx, bot_name, mc_port, task_queue))
        for future in futures:
            future.result()

if __name__ == '__main__':
    main()
