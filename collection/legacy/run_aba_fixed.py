import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import argparse
import glob
import os
import random
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

from target import biome_targets, structure_targets, village_targets

start_port = 8089
max_port = 9089
nv_type = 'ABA_FIXED'
nvranges = [5, 15, 30, 50]

parser = argparse.ArgumentParser(description='Collect ABA with reverse-waypoint return')
parser.add_argument('--name', type=str, default='Bot', help='Bot name for single-worker runs')
parser.add_argument('--output_path', type=str, default='data/LoopNav', help='Output path')
parser.add_argument('--target', type=str, default='village', choices=['village', 'biome', 'structure'], help='target set')
parser.add_argument('--episodes_per_location', type=int, default=20, help='Episodes per location/range')
parser.add_argument('--location', action='append', default=None, help='Collect only specific location(s)')
parser.add_argument('--nvrange', action='append', type=int, default=None, help='Collect only specific range(s)')
parser.add_argument('--max_locations', type=int, default=None, help='Optional target limit for debugging')
parser.add_argument('--no_shuffle', action='store_true', help='Do not shuffle targets/tasks')
parser.add_argument('--workers', type=int, default=1, help='Concurrent bot/receiver workers')
parser.add_argument('--bot_names', type=str, default='', help='Comma-separated bot names for workers')
parser.add_argument('--mc_ports', type=str, default='8105', help='Comma-separated Minecraft server ports')
parser.add_argument('--start_port', type=int, default=start_port, help='First receiver port')
parser.add_argument('--retry_limit', type=int, default=6, help='Retries per failed sample')
parser.add_argument('--end_hold_ms', type=int, default=0, help='Extra static hold after return; default records no extra hold')
args = parser.parse_args()

print_lock = threading.Lock()


def log(*items):
    with print_lock:
        print(*items, flush=True)


def count_json_files(location, nvrange):
    path = f'{args.output_path}/{nv_type}/{location}/{nvrange}'
    if not os.path.exists(path):
        return 0
    return len(glob.glob(os.path.join(path, '*.json')))


def receiver_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('localhost', port))
            return True
        except socket.error:
            return False


def run_one_round(task, worker_idx, bot_name, receiver_port, mc_port):
    location = task['location']
    nvrange = task['nvrange']
    round_num = task['round_num']
    log(f'\n[worker {worker_idx} {bot_name}] {location} range {nvrange} round {round_num + 1}')
    t0 = time.time()

    if not receiver_port_available(receiver_port):
        raise RuntimeError(f'Receiver port {receiver_port} is not available')

    receiver = subprocess.Popen([
        sys.executable, str(SCRIPT_DIR / 'receiver.py'),
        '--location', location,
        '--nvtype', nv_type,
        '--port', str(receiver_port),
        '--nvrange', str(nvrange),
        '--output_path', args.output_path,
    ])
    log(f'[worker {worker_idx}] receiver.py started on port {receiver_port}')

    env = os.environ.copy()
    node_path = str(SCRIPT_DIR / 'node_modules')
    env['NODE_PATH'] = (
        node_path if not env.get('NODE_PATH')
        else node_path + os.pathsep + env['NODE_PATH']
    )
    sender = subprocess.Popen([
        'node', str(SCRIPT_DIR / 'senderhl.js'),
        '--location', location,
        '--nvtype', nv_type,
        '--port', str(receiver_port),
        '--nvrange', str(nvrange),
        '--name', bot_name,
        '--mc-port', str(mc_port),
        '--end-hold-ms', str(args.end_hold_ms),
        '--symmetric-return',
    ], env=env)
    log(f'[worker {worker_idx}] senderhl.js started as {bot_name} on MC port {mc_port}')

    receiver.wait()
    returncode = receiver.returncode
    log(f'[worker {worker_idx}] receiver.py finished {returncode}')

    sender_returncode = None
    try:
        sender_returncode = sender.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass

    if sender_returncode is None:
        log(f'[worker {worker_idx}] terminating senderhl.js')
        sender.terminate()
        try:
            sender_returncode = sender.wait(timeout=2)
        except subprocess.TimeoutExpired:
            sender.kill()
            sender_returncode = sender.wait()
        log(f'[worker {worker_idx}] senderhl.js terminated {sender_returncode}')
    else:
        log(f'[worker {worker_idx}] senderhl.js exited {sender_returncode}')
        if sender_returncode != 0:
            returncode = sender_returncode

    log(f'[worker {worker_idx}] round time: {time.time() - t0:.2f}s')
    return returncode


def choose_targets():
    if args.location:
        targets = args.location
    elif args.target == 'village':
        targets = list(village_targets)
    elif args.target == 'biome':
        targets = list(biome_targets)
    else:
        targets = list(structure_targets)

    if not args.no_shuffle:
        random.shuffle(targets)
    if args.max_locations is not None:
        targets = targets[:args.max_locations]
    return targets


def build_tasks(targets):
    ranges = args.nvrange if args.nvrange else nvranges
    tasks = []
    for location in targets:
        for nvrange in ranges:
            have = count_json_files(location, nvrange)
            need = max(0, args.episodes_per_location - have)
            log(f'{nv_type}, location {location}, range {nvrange}: have {have}, need {need}')
            for idx in range(need):
                tasks.append({
                    'location': location,
                    'nvrange': nvrange,
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
                try:
                    res = run_one_round(task, worker_idx, bot_name, receiver_port, mc_port)
                except Exception as exc:
                    log(f"[worker {worker_idx}] {task['location']} range {task['nvrange']} exception: {exc}")
                    res = 1
                if res == 0:
                    break
                retries += 1
                if retries > args.retry_limit:
                    raise RuntimeError(f"{task['location']} range {task['nvrange']} failed too many times")
                log(f"[worker {worker_idx}] {task['location']} range {task['nvrange']} failed, retry {retries}")
                time.sleep(2)
        finally:
            task_queue.task_done()


def parse_bot_names():
    if args.bot_names:
        names = [item.strip() for item in args.bot_names.split(',') if item.strip()]
    elif args.workers == 1:
        names = [args.name]
    else:
        names = ['Bot', 'Bot2', 'Bot3', 'Bot4', 'Bot5', 'Bot6', 'Bot7', 'Bot8', 'Bot9']
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
    targets = choose_targets()
    tasks = build_tasks(targets)
    log(f'Total pending {nv_type} samples: {len(tasks)}')
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
