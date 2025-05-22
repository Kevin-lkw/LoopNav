import subprocess
import time
from target import village_targets, biome_targets, structure_targets
import argparse
import socket
import random
import os
import glob

start_port = 8089
max_port = 9089

parser = argparse.ArgumentParser(description='Run script')
parser.add_argument('--name', type=str, default='Bot', help='Bot name')
parser.add_argument('--output_path', type=str, default='./output', help='output path')
parser.add_argument('--target', type=str, default='village', help='target or biome or structure')
args = parser.parse_args()

total_eps_number = 20

def count_json_files(location,nv_type,nvrange):
    """检查指定位置是否已经有足够的JSON文件"""
    path = f'{args.output_path}/{nv_type}/{location}/{nvrange}'
    if not os.path.exists(path):
        return 0
    json_files = glob.glob(os.path.join(path, '*.json'))
    return len(json_files)

def find_free_port():
    port = start_port
    while port <= max_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('localhost', port))  # 尝试绑定端口
                sock.close()  # 立即关闭以释放端口
                print("using free port", port)
                return port
            except socket.error as e:
                port += 1
def run_one_round(round_num, location, nv_type, nvrange):
    print(f"\n--- 第 {round_num + 1} 轮开始 ---")
    t0 = time.time()
    port = find_free_port()
    # 1. 启动 receiver.py
    receiver = subprocess.Popen(['python', 'receiver.py', '--location', location,
         '--nvtype', nv_type, '--port', str(port), '--nvrange', str(nvrange), '--output_path', args.output_path])
    print("receiver.py 已启动")

    # 2. 启动 senderhl.js
    sender = subprocess.Popen(['node', 'senderhl.js','--location', location, 
        '--nvtype', nv_type, '--port', str(port), '--nvrange', str(nvrange), '--name', args.name])
    print("senderhl.js 已启动")

    # 3. 等待 receiver.py 完成
    receiver.wait()
    print("receiver.py 已完成", receiver.returncode)
    returncode = receiver.returncode
    # 4. 杀掉 senderhl.js
    if sender.poll() is None:
        print("杀掉 senderhl.js")
        try:
            sender.terminate()  # 尝试优雅退出
            time.sleep(2)
            if sender.poll() is None:
                sender.kill()  # 如果还活着，强制杀
        except Exception as e:
            print(f"杀 senderhl.js 出错: {e}")
    else:
        print("senderhl.js 已经提前退出")
    t1 = time.time()
    print(f"第 {round_num + 1} 轮耗时: {t1 - t0:.2f} 秒")
    return returncode
def collect(lo, nv_type, nvrange):
    print("开始收集位置:", lo)
    print("开始收集类型:", nv_type)
    print("开始收集范围:", nvrange)

    # 检查是否已经有足够的JSON文件
    json_count = count_json_files(lo,nv_type,nvrange)
    print(f"{nv_type}, 位置 {lo}, 范围{nvrange} 已经有 {json_count} 个JSON文件，需要执行 {total_eps_number - json_count} 轮")
    if json_count >= total_eps_number:
        return
    collect_number = total_eps_number - json_count
    for i in range(collect_number):
        res = run_one_round(i, lo, nv_type, nvrange)
        json_count = count_json_files(lo,nv_type,nvrange)
        if json_count >= total_eps_number:
            break
        retries = 0
        while res != 0:
            print(f"第 {i + 1} 轮执行失败, 重新执行")
            res = run_one_round(i, lo, nv_type, nvrange)
            retries += 1
            if retries > 6:
                print("重试次数过多, 异常退出")
                exit(1)
        print(f"第 {i + 1} 轮执行成功")
    print(f"\n全部{collect_number}轮执行完成。")
    
def main():
    if args.target == 'village':
        targets = village_targets
    elif args.target == 'biome':
        targets = biome_targets
    elif args.target == 'structure':
        targets = structure_targets
    else:
        raise ValueError(f"Invalid target: {args.target}")
    random.shuffle(targets)
    for lo in targets:
        for nv_type in ['ABA','ABCA']:
            for nvrange in [5,15,30,50]:
                collect(lo,nv_type,nvrange)
if __name__ == "__main__":
    main()
    