import socket
import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import time
import argparse
from queue import Queue
from threading import Thread
import fcntl  # 用于获取socket缓冲区信息
import array
from queue import Empty
import math

from trajectory_validation import (
    validate_horizontal_loop_closure,
    validate_horizontal_speed_transitions,
    validate_native_camera_transitions,
)

HOST = ''
PORT = 8089

def angle_delta(yaw, start_yaw):
    delta = yaw - start_yaw
    while delta <= -math.pi:
        delta += 2 * math.pi
    while delta > math.pi:
        delta -= 2 * math.pi
    return delta

def annotate_trun(action_data):
    if args.nvtype not in ('TRUN', 'TURN') or not action_data:
        return

    turn_order = args.turn_order
    direction = 1 if turn_order == 'right_left' else -1
    target_rad = math.radians(args.turn_angle)
    start_yaw = action_data[0].get('initial_yaw', action_data[0].get('yaw', 0.0))

    progress = [
        direction * angle_delta(item.get('yaw', start_yaw), start_yaw)
        for item in action_data
    ]
    max_progress = max(progress)
    target_threshold = min(target_rad, max_progress) - math.radians(2)
    candidates = [
        idx for idx, value in enumerate(progress)
        if value >= target_threshold
    ]

    stable_candidates = []
    for idx in candidates:
        camera = action_data[idx].get('action', {}).get('camera', [0, 0])
        if abs(camera[0]) < 1e-3 and abs(camera[1]) < 1e-3:
            stable_candidates.append(idx)

    if stable_candidates:
        return_idx = stable_candidates[0]
    else:
        return_idx = max(range(len(progress)), key=lambda idx: progress[idx])

    return_frame = action_data[return_idx].get('frame_count', return_idx)
    for idx, item in enumerate(action_data):
        item['return_point'] = idx == return_idx
        if idx < return_idx:
            item['turn_phase'] = 'outbound'
        elif idx == return_idx:
            item['turn_phase'] = 'hold'
        else:
            item['turn_phase'] = 'return'
        item['turn_order'] = turn_order
        item['turn_angle'] = round(args.turn_angle, 3)
        item['turn_direction'] = 'right' if turn_order == 'right_left' else 'left'
        item['turn_progress'] = round(math.degrees(progress[idx]), 3)
        item['extra_info']['turn_order'] = turn_order
        item['extra_info']['turn_angle'] = round(args.turn_angle, 3)
        item['extra_info']['turn_range'] = args.nvrange
        item['extra_info']['initial_yaw'] = round(start_yaw, 3)
        item['extra_info']['return_frame'] = return_frame

def annotate_aba_return_point(action_data):
    if args.nvtype != 'ABA_FIXED' or not action_data:
        return

    outbound_indices = [
        idx for idx, item in enumerate(action_data)
        if item.get('route_phase') == 'outbound'
    ]
    return_indices = [
        idx for idx, item in enumerate(action_data)
        if item.get('route_phase') == 'return'
    ]
    if not outbound_indices or not return_indices:
        return

    first_return_idx = min(return_indices)
    boundary_candidates = [
        idx for idx in outbound_indices
        if idx < first_return_idx
    ]
    if not boundary_candidates:
        return

    return_idx = max(boundary_candidates)
    return_frame = action_data[return_idx].get('frame_count', return_idx)
    return_position = {
        'x': action_data[return_idx].get('x'),
        'y': action_data[return_idx].get('y'),
        'z': action_data[return_idx].get('z'),
    }

    for idx, item in enumerate(action_data):
        is_return_point = idx == return_idx
        item['return_point'] = is_return_point
        item['construct_phase'] = is_return_point
        item['extra_info']['return_frame'] = return_frame
        item['extra_info']['return_index'] = return_idx
        item['extra_info']['return_position'] = return_position
        item['extra_info']['return_point_phase'] = 'construct'

def annotate_abaca_events(action_data):
    if args.nvtype != 'ABACA_FIXED' or not action_data:
        return

    phase_events = [
        ('exploration_1', 'turnaround_B'),
        ('evaluation_1', 'home_revisit_1'),
        ('exploration_2', 'turnaround_C'),
        ('evaluation_2', 'home_revisit_2'),
    ]
    event_indices = {}
    event_positions = {}

    for phase, event_name in phase_events:
        indices = [
            idx for idx, item in enumerate(action_data)
            if item.get('route_phase') == phase
        ]
        if not indices:
            continue
        event_idx = max(indices)
        event_indices[event_name] = event_idx
        event_positions[event_name] = {
            'x': action_data[event_idx].get('x'),
            'y': action_data[event_idx].get('y'),
            'z': action_data[event_idx].get('z'),
        }

    for idx, item in enumerate(action_data):
        phase = item.get('route_phase')
        item['return_point'] = False
        item['construct_phase'] = False
        item['event_type'] = None
        item['home_revisit'] = False
        item['memory_eval_phase'] = phase in ('evaluation_1', 'evaluation_2')
        item['extra_info']['abaca_event_indices'] = event_indices
        item['extra_info']['abaca_event_positions'] = event_positions

    for phase, event_name in phase_events:
        event_idx = event_indices.get(event_name)
        if event_idx is None:
            continue
        item = action_data[event_idx]
        item['event_type'] = event_name
        item['extra_info']['event_type'] = event_name
        if event_name in ('turnaround_B', 'turnaround_C'):
            item['return_point'] = True
            item['construct_phase'] = True
            item['extra_info']['return_point_phase'] = 'construct'
        else:
            item['home_revisit'] = True
            item['extra_info']['return_point_phase'] = 'home_revisit'

def validate_aba_fixed(action_data):
    return_points = [
        idx for idx, item in enumerate(action_data)
        if item.get('return_point')
    ]
    if len(return_points) != 1:
        print(f'Validation failed: bad_return_point_count {len(return_points)}')
        return False

    return_idx = return_points[0]
    if action_data[return_idx].get('route_phase') != 'outbound':
        print(f"Validation failed: return_point_phase {action_data[return_idx].get('route_phase')}")
        return False
    if not action_data[return_idx].get('construct_phase'):
        print('Validation failed: return_point_not_construct_phase')
        return False
    if not any(item.get('route_phase') == 'return' for item in action_data[return_idx + 1:]):
        print('Validation failed: no_return_after_return_point')
        return False
    return True

def validate_abaca_fixed(action_data):
    expected_phases = ['exploration_1', 'evaluation_1', 'exploration_2', 'evaluation_2']
    phase_indices = {}
    for phase in expected_phases:
        indices = [
            idx for idx, item in enumerate(action_data)
            if item.get('route_phase') == phase
        ]
        if not indices:
            print(f'Validation failed: missing_phase {phase}')
            return False
        phase_indices[phase] = (min(indices), max(indices))

    previous_end = -1
    for phase in expected_phases:
        phase_start, phase_end = phase_indices[phase]
        if phase_start <= previous_end:
            print(f'Validation failed: phase_order {phase}')
            return False
        previous_end = phase_end

    expected_events = {
        'turnaround_B': 'exploration_1',
        'home_revisit_1': 'evaluation_1',
        'turnaround_C': 'exploration_2',
        'home_revisit_2': 'evaluation_2',
    }
    for event_name, phase in expected_events.items():
        event_items = [
            (idx, item) for idx, item in enumerate(action_data)
            if item.get('event_type') == event_name
        ]
        if len(event_items) != 1:
            print(f'Validation failed: bad_event_count {event_name} {len(event_items)}')
            return False
        event_idx, event_item = event_items[0]
        if event_item.get('route_phase') != phase:
            print(f"Validation failed: event_phase {event_name} {event_item.get('route_phase')}")
            return False
        if event_idx != phase_indices[phase][1]:
            print(f'Validation failed: event_not_phase_end {event_name}')
            return False

    return_points = [
        idx for idx, item in enumerate(action_data)
        if item.get('return_point')
    ]
    if len(return_points) != 2:
        print(f'Validation failed: bad_return_point_count {len(return_points)}')
        return False
    for idx in return_points:
        if action_data[idx].get('route_phase') not in ('exploration_1', 'exploration_2'):
            print(f"Validation failed: return_point_phase {action_data[idx].get('route_phase')}")
            return False
        if not action_data[idx].get('construct_phase'):
            print('Validation failed: return_point_not_construct_phase')
            return False

    memory_eval = [
        item for item in action_data
        if item.get('memory_eval_phase')
    ]
    if not memory_eval:
        print('Validation failed: no_memory_eval_phase')
        return False
    return True

def validate_completed_sample(json_path):
    if not os.path.exists(json_path):
        print(f'Validation failed: missing json {json_path}')
        return False

    try:
        with open(json_path, 'r') as f:
            action_data = json.load(f)
    except Exception as exc:
        print(f'Validation failed: cannot read json {exc}')
        return False

    # Enforce both the adjacent-frame limit and the dataset audit limit over
    # two 20 FPS records (0.1 seconds). Either failure rejects the sample.
    for camera_stride in (1, 2):
        camera_valid, camera_error = validate_native_camera_transitions(
            action_data,
            source_stride=camera_stride,
            max_degrees=args.max_camera_degrees,
        )
        if not camera_valid:
            print(
                'Validation failed: '
                f'camera_stride={camera_stride} {camera_error}'
            )
            return False

    speed_valid, speed_error = validate_horizontal_speed_transitions(
        action_data,
        fps=20.0,
        max_speed_bps=args.max_horizontal_speed_bps,
    )
    if not speed_valid:
        print(f'Validation failed: {speed_error}')
        return False

    if args.nvtype in ('ABA', 'ABCA', 'ABA_FIXED', 'ABACA_FIXED'):
        closure_valid, closure_error = validate_horizontal_loop_closure(
            action_data,
            max_distance_blocks=args.max_loop_closure_distance,
        )
        if not closure_valid:
            print(f'Validation failed: {closure_error}')
            return False

    if args.nvtype == 'ABA_FIXED':
        return validate_aba_fixed(action_data)
    if args.nvtype == 'ABACA_FIXED':
        return validate_abaca_fixed(action_data)
    return True

def process_frame_worker(frame_queue, output_path, time_now):
    """后台处理帧的worker函数"""
    out = cv2.VideoWriter(f'{output_path}/{time_now}.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'), 20, (640, 360))
    action_data = []
    
    while True:
        try:
            # 从队列获取数据，设置超时以避免无限等待
            data = frame_queue.get(timeout=5)
            if data is None:  # 结束信号
                break
                
            img, pos = data
            img_count = pos.get('frame_count', 0)
            
            if img_count < 20:  # 跳过前20帧
                continue
                
            # x,y,z,yaw,pitch 保留三位小数
            pos['x'] = round(pos['x'], 3)
            pos['y'] = round(pos['y'], 3)
            pos['z'] = round(pos['z'], 3)
            pos['yaw'] = round(pos['yaw'], 3)
            pos['pitch'] = round(pos['pitch'], 3)
            
            # 添加位置信息
            try:
                nvrange = int(args.nvrange)
            except ValueError:
                nvrange = args.nvrange
            pos['extra_info'] = {
                'seed': 42,
                'location': args.location,
                'nvtype': args.nvtype,
                'nvrange': nvrange,
            }
            for key in [
                'route_phase',
                'return_strategy',
                'waypoint_index',
                'return_waypoint_total',
                'outbound_waypoint_count',
                'phase_index',
                'memory_eval_phase',
                'abaca_cycle',
                'abaca_min_angle',
                'target_label',
                'target_x',
                'target_y',
                'target_z',
            ]:
                if key in pos:
                    pos['extra_info'][key] = pos[key]
            action_data.append(pos)
            
            # 写入视频
            if img is None:
                print(f"Error: Received None image at frame {img_count}")
                continue
            if not isinstance(img, np.ndarray):
                print(f"Error: Invalid image type at frame {img_count}: {type(img)}")
                continue
            if img.size == 0:
                print(f"Error: Empty image at frame {img_count}")
                continue
                
            out.write(img)
        except Empty:
            # print("Frame queue timeout (no new frames in 5 seconds)")
            continue 
        except Exception as e:
            print(f"Error processing frame {img_count if 'img_count' in locals() else 'unknown'}:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print(f"  Data type: {type(data) if 'data' in locals() else 'unknown'}")
            if 'img' in locals():
                print(f"  Image type: {type(img)}")
                print(f"  Image shape: {img.shape if hasattr(img, 'shape') else 'no shape'}")
            continue
            
    # 清理工作
    out.release()
    annotate_trun(action_data)
    annotate_aba_return_point(action_data)
    annotate_abaca_events(action_data)
    print('Total frames processed:', len(action_data))
    with open(os.path.join(output_path, f'{time_now}.json'), 'w') as f:
        json.dump(action_data, f)

def recvall(sock, count):
    buf = b''
    total_received = 0
    while count:
        try:
            newbuf = sock.recv(count)
            if not newbuf: 
                return None
            received = len(newbuf)
            total_received += received
            buf += newbuf
            count -= received
        except socket.error as e:
            return None
    return buf

def recvint(sock): 
    return int.from_bytes(recvall(sock, 4), byteorder='little')

def get_recv_buffer_used(sock):
    buf = array.array('i', [0])
    fcntl.ioctl(sock.fileno(), 0x541B, buf)  # FIONREAD
    return buf[0]

# 解析命令行参数
argparser = argparse.ArgumentParser(description='Receiver script')
argparser.add_argument('--location', type=str, default='default_location', help='minecraft location(biome)')
argparser.add_argument('--nvtype', type=str, default='normal', help='ABA, ABCA or centered')
argparser.add_argument('--port', type=int, default=8089, help='Port number')
argparser.add_argument('--nvrange', type=str, default='30', help='range')
argparser.add_argument('--teaser', action='store_true', help='teaser mode')
argparser.add_argument('--output_path', type=str, required=True, help='output path')
argparser.add_argument('--turn_angle', type=float, default=90.0, help='TURN turn angle in degrees')
argparser.add_argument('--turn_order', type=str, default='right_left', choices=['right_left', 'left_right'], help='TURN order')
argparser.add_argument(
    '--sample_id',
    type=str,
    default=None,
    help='Exact output stem supplied by the collection coordinator',
)
argparser.add_argument(
    '--max_frames',
    type=int,
    default=2500,
    help='Maximum recorded 20 FPS frames before treating the sample as stuck',
)
argparser.add_argument(
    '--max_camera_degrees',
    '--max_native_camera_degrees',
    dest='max_camera_degrees',
    type=float,
    default=30.0,
    help='Reject a sample when a consecutive 20 FPS camera transition exceeds this threshold',
)
argparser.add_argument(
    '--max_horizontal_speed_bps',
    '--max_speed_bps',
    dest='max_horizontal_speed_bps',
    type=float,
    default=15.0,
    help='Reject a sample when adjacent 20 FPS positions reach this horizontal speed',
)
argparser.add_argument(
    '--max_loop_closure_distance',
    type=float,
    default=2.0,
    help='Reject a loop whose final XZ position is this many blocks from its start',
)

args = argparser.parse_args()
PORT = args.port

# 创建输出目录
time_now = args.sample_id or datetime.now().strftime("%m-%d_%H-%M-%S-%f")
if not time_now or Path(time_now).name != time_now:
    argparser.error('--sample_id must be a non-empty file stem without directories')
output_path = f'{args.output_path}/{args.nvtype}/{args.location}/{args.nvrange}'

if args.teaser:
    output_path = f'./teaser_data'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 创建帧处理队列
frame_queue = Queue()

# 启动后台处理进程
processor = Thread(target=process_frame_worker, 
                  args=(frame_queue, output_path, time_now))
processor.daemon = True
processor.start()

# 设置socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

# 设置连接超时时间，16 worker 并发启动时 Mineflayer 登录和初始寻路会更慢
s.settimeout(120)
try:
    conn, addr = s.accept()
except socket.timeout:
    print('No connection received within 120 seconds, exiting...')
    s.close()
    exit(1)

conn.settimeout(10)
conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
print('Socket connected')

img_count = 0
retcode = 0
try:
    while True:
        t0 = time.time()
        pos_length = recvint(conn)
        if pos_length == 0: 
            print("recv 0 length, normal end.")
            retcode = 0
            break
            
        pos_data = recvall(conn, pos_length)
        if pos_data is None:
            print("Error receiving position data")
            retcode = 1
            break
            
        pos = json.loads(pos_data.decode('utf-8'))
        pos['frame_count'] = img_count
        length = recvint(conn)
        if length == 0:
            print("ERROR! recv 0 image length")
            retcode = 1
            break

        stringData = recvall(conn, int(length))
        if stringData is None:
            print("[Error] Received None instead of valid image data")
            retcode = 1
            break

        img_count += 1
        if img_count > args.max_frames:
            print(f"img_count > {args.max_frames}, probably stuck, quit")
            retcode = 1
            break
        img = cv2.imdecode(np.frombuffer(stringData, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if args.teaser:
            # directly save the image
            if img_count % 5 == 0:
                cv2.imwrite(os.path.join(output_path, f'{img_count}.png'), img)
        else:
            try:
                frame_queue.put((img, pos), timeout=1)
            except Queue.Full:
                print("Queue full, dropping frame")
            continue
            
        t1 = time.time()
        # print(f"Processed in {(t1 - t0)*1000:.2f}ms")
        
except socket.timeout:
    print('Socket timeout')
    retcode = 1
except Exception as e:
    print(f'Error: {e}')
    retcode = 1
finally:
    avi_path = os.path.join(output_path, f'{time_now}.avi')
    json_path = os.path.join(output_path, f'{time_now}.json')
    if retcode == 0:
        frame_queue.put(None)
        processor.join(timeout=50)
        if not validate_completed_sample(json_path):
            retcode = 1
    if retcode != 0:
        if os.path.exists(avi_path):
            os.remove(avi_path)
        if os.path.exists(json_path):
            os.remove(json_path)
    conn.close()
    exit(retcode)
