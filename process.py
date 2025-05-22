import subprocess
import time
from target import village_targets,biome_targets,structure_targets
import argparse
import socket
import random
import os
import glob
import json 

target_number = 25
parser = argparse.ArgumentParser(description='Run script')
parser.add_argument('--output_path', type=str, required=True, help='output path')
parser.add_argument('--target', type=str, required=True, help='target')
args = parser.parse_args()

targets = village_targets


def count_json_files(nv_type, location, nvrange):
    """检查指定位置是否已经有足够的JSON文件"""
    path = f'{args.output_path}/{nv_type}/{location}/{nvrange}'
    if not os.path.exists(path):
        return 0
    json_files = glob.glob(os.path.join(path, '*.json'))
    return len(json_files)

def del_orphaned_avi_files(nv_type, location, nvrange, delete=False):
    """处理没有对应JSON文件的AVI文件
    如果delete=True，则删除这些文件
    返回没有对应JSON的AVI文件数量
    """
    path = f'{args.output_path}/{nv_type}/{location}/{nvrange}'
    if not os.path.exists(path):
        return 0
    
    # 获取所有AVI和JSON文件
    avi_files = glob.glob(os.path.join(path, '*.avi'))
    json_files = set(os.path.splitext(f)[0] for f in glob.glob(os.path.join(path, '*.json')))
    
    # 找出没有对应JSON文件的AVI文件
    orphaned_files = []
    for avi_file in avi_files:
        base_name = os.path.splitext(avi_file)[0]
        if base_name not in json_files:
            orphaned_files.append(avi_file)
    
    # 如果需要删除文件
    if delete and orphaned_files:
        for file in orphaned_files:
            try:
                print(f"删除文件: {file}")
                os.remove(file)
                print(f"已删除: {file}")
            except Exception as e:
                print(f"删除文件 {file} 时出错: {str(e)}")
    
    return len(orphaned_files)
def del_json_files(nv_type, location,nvrange):
    path = f'{args.output_path}/{nv_type}/{location}/{nvrange}'

    json_files = glob.glob(os.path.join(path, '*.json'))    
    
    number = len(json_files)
    if number <= target_number:
        print(f"位置 {location} 的JSON文件数量为 {number}，<= {target_number} 个，跳过")
        return
    if number > target_number:
        more = number - target_number
        print(f"位置 {location} 的JSON文件数量为 {number}，多余 {more} 个，删除多余文件")
    # 列出所有json文件的长度
    jslen = []  
    for fn in json_files:
        with open(fn, 'r') as f:
            file = json.load(f)
            length = len(file)
            jslen.append((fn,length))
    # 按照长度排序
    jslen.sort(key=lambda x: x[1],reverse=True)
    
    # 删除最长的more个
    more = len(jslen) - target_number
    for i in range(more):
        fn,length = jslen[i]
        # import ipdb;ipdb.set_trace()
        os.remove(fn)
        os.remove(fn.replace('.json','.avi'))
        print(fn)

def del_json():
    for nv_type in ['ABA','ABCA']:
        for nvrange in [5,15,30,50]:
            for lo in targets:
                del_json_files(nv_type,lo,nvrange)

def del_orphaned_avi():
    for nv_type in ['ABA','ABCA']:
        for nvrange in [5,15,30,50]:
            for lo in targets:
                del_orphaned_avi_files(nv_type,lo,nvrange,delete=True)

def count_json():
    unfin = 0
    sum_more = 0
    for nv_type in ['ABA','ABCA']:
        for nvrange in [5,15,30,50]:
            for lo in targets:
                # 检查是否已经有足够的JSON文件
                json_count = count_json_files(nv_type,lo,nvrange)            
                if json_count >= target_number:
                    more = json_count - target_number
                    sum_more += more
                else:
                    print(f"{nv_type} 位置 {nvrange} {lo} 的JSON文件数量为 {json_count}，不足 {target_number} 个")
                    unfin += 1
    print(f"未完成的位置有 {unfin} 个")
    print(f"more JSON文件有 {sum_more} 个")

def get_offset_actions(path):
    assert path.endswith(".json")
    actions = json.load(open(path))
    length = len(actions)
    # print(f"length: {length}")
    if length == 0:
        return None
    start_pos ={"x":actions[0]["x"],"z":actions[0]["z"]}

    for i in reversed(range(length)):
        # print(i)
        if actions[i]["goal"] is not None:
            gx = actions[i]["goal"]["x"]
            gz = actions[i]["goal"]["z"]
            # print(f"gx: {gx}, gz: {gz}")
            if abs(gx - start_pos["x"]) > 2 or abs(gz - start_pos["z"]) > 2:
                return i
    return None
def del_illegal_json_files(nv_type,location,nvrange):
    path = f'{args.output_path}/{nv_type}/{location}/{nvrange}'
    json_files = glob.glob(os.path.join(path, '*.json'))
    for fn in json_files:
        with open(fn, 'r') as f:
            file = json.load(f)
            jslen = len(file)
        offset = get_offset_actions(fn)
        if jslen == 0:
            os.remove(fn)
            os.remove(fn.replace('.json','.avi'))
            print("jslen==0, delete")
            continue
        if offset is None:
            os.remove(fn)
            os.remove(fn.replace('.json','.avi'))
            print("offset==0, delete")
            continue
        if jslen == offset + 1:
            os.remove(fn)
            os.remove(fn.replace('.json','.avi'))
            print("jslen==offset+1, delete")
            continue
def del_illegal_json():
    for nv_type in ['ABA','ABCA']:
        for nvrange in [5,15,30,50]:
            for lo in targets:
                del_illegal_json_files(nv_type,lo,nvrange)
if __name__ == "__main__":
    
    del_illegal_json()

    del_json()
    
    del_orphaned_avi()
    
    count_json()
