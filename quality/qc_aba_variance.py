import argparse
import glob
import json
import math
import os
import random
from statistics import mean

import cv2


parser = argparse.ArgumentParser(description='QC forward/return path variance for ABA-like trajectories')
parser.add_argument('--root', type=str, default='data/LoopNav', help='Dataset root')
parser.add_argument('--nvtype', type=str, default='ABA_FIXED', help='Navigation type folder')
parser.add_argument('--limit', type=int, default=None, help='Optional number of files to sample')
parser.add_argument('--check_video', action='store_true', help='Check AVI frame counts')
args = parser.parse_args()


def distance_point_to_path(point, path):
    px, pz = point
    return min(math.hypot(px - x, pz - z) for x, z in path)


def path_length(path):
    return sum(
        math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        for i in range(1, len(path))
    )


def loop_area(outbound, ret):
    polygon = outbound + ret
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for i, (x1, z1) in enumerate(polygon):
        x2, z2 = polygon[(i + 1) % len(polygon)]
        area += x1 * z2 - x2 * z1
    return abs(area) / 2.0


def compact_positions(items):
    path = []
    for item in items:
        point = (float(item['x']), float(item['z']))
        if not path or math.hypot(point[0] - path[-1][0], point[1] - path[-1][1]) > 0.05:
            path.append(point)
    return path


def split_by_phase(data):
    outbound = [item for item in data if item.get('route_phase') == 'outbound']
    ret = [item for item in data if item.get('route_phase') == 'return']
    if outbound and ret:
        return outbound, ret, 'route_phase'

    goals = [item.get('goal', {}) for item in data]
    return_start = None
    for idx, goal in enumerate(goals):
        if isinstance(goal, dict) and 'rangeSq' in goal:
            return_start = idx
            break
    if return_start is None:
        return [], [], 'unknown'

    outbound = data[:return_start]
    ret = data[return_start:]
    moving_outbound = [item for item in outbound if item.get('action', {}).get('forward') or item.get('action', {}).get('jump')]
    moving_return = [item for item in ret if item.get('action', {}).get('forward') or item.get('action', {}).get('jump')]
    return moving_outbound or outbound, moving_return or ret, 'goal_inference'


def qc_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not data:
        return {'path': path, 'error': 'empty_json'}

    avi_path = path[:-5] + '.avi'
    if args.check_video:
        cap = cv2.VideoCapture(avi_path)
        avi_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
        cap.release()
        if avi_frames != len(data):
            return {'path': path, 'error': f'frame_mismatch json={len(data)} avi={avi_frames}'}

    extra = data[0].get('extra_info', {})
    if extra.get('nvtype') != args.nvtype:
        return {'path': path, 'error': f"nvtype_mismatch {extra.get('nvtype')}"}

    return_point_indices = [idx for idx, item in enumerate(data) if item.get('return_point')]
    if args.nvtype == 'ABA_FIXED':
        if len(return_point_indices) != 1:
            return {'path': path, 'error': f'bad_return_point_count {len(return_point_indices)}'}
        rp = return_point_indices[0]
        if data[rp].get('route_phase') != 'outbound':
            return {'path': path, 'error': f"return_point_phase {data[rp].get('route_phase')}"}
        next_return = next(
            (idx for idx in range(rp + 1, len(data)) if data[idx].get('route_phase') == 'return'),
            None
        )
        if next_return is None:
            return {'path': path, 'error': 'no_return_after_return_point'}
        if not data[rp].get('construct_phase'):
            return {'path': path, 'error': 'return_point_not_construct_phase'}

    outbound_items, return_items, split_method = split_by_phase(data)
    outbound = compact_positions(outbound_items)
    ret = compact_positions(return_items)
    if len(outbound) < 2 or len(ret) < 2:
        return {'path': path, 'error': f'cannot_split split={split_method} outbound={len(outbound)} return={len(ret)}'}

    return_to_outbound = [distance_point_to_path(point, outbound) for point in ret]
    outbound_to_return = [distance_point_to_path(point, ret) for point in outbound]
    area = loop_area(outbound, ret)
    total_len = path_length(outbound) + path_length(ret)
    strategy = data[-1].get('return_strategy') or extra.get('return_strategy')

    return {
        'path': path,
        'split_method': split_method,
        'frames': len(data),
        'outbound_points': len(outbound),
        'return_points': len(ret),
        'return_point_index': return_point_indices[0] if return_point_indices else None,
        'return_strategy': strategy,
        'mean_return_to_outbound': mean(return_to_outbound),
        'max_return_to_outbound': max(return_to_outbound),
        'mean_symmetric_distance': mean(return_to_outbound + outbound_to_return),
        'max_symmetric_distance': max(return_to_outbound + outbound_to_return),
        'loop_area': area,
        'normalized_area': area / total_len if total_len > 0 else 0.0,
    }


def main():
    paths = sorted(glob.glob(os.path.join(args.root, args.nvtype, '*', '*', '*.json')))
    if args.limit is not None and len(paths) > args.limit:
        random.seed(0)
        paths = random.sample(paths, args.limit)

    results = [qc_file(path) for path in paths]
    errors = [item for item in results if 'error' in item]
    valid = [item for item in results if 'error' not in item]

    print('QC_ABA_VARIANCE')
    print('root', args.root)
    print('nvtype', args.nvtype)
    print('checked', len(results))
    print('valid', len(valid))
    print('errors', len(errors))
    if errors:
        print('FIRST_ERRORS')
        for item in errors[:20]:
            print(item)
    if valid:
        for key in [
            'mean_return_to_outbound',
            'max_return_to_outbound',
            'mean_symmetric_distance',
            'max_symmetric_distance',
            'loop_area',
            'normalized_area',
        ]:
            values = [item[key] for item in valid]
            print(key, 'min', round(min(values), 4), 'avg', round(mean(values), 4), 'max', round(max(values), 4))
        strategies = {}
        for item in valid:
            strategies[item['return_strategy']] = strategies.get(item['return_strategy'], 0) + 1
        print('return_strategies', strategies)


if __name__ == '__main__':
    main()
