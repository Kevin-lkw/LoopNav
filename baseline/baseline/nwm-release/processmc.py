import os
import cv2
import json
import pickle
import numpy as np
import shutil
from tqdm import tqdm

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    cnt_jpg = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_path = os.path.join(output_dir, f"{frame_id:05d}.jpg")
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        # compare the size of the two files
        # print(os.path.getsize(img_path), os.path.getsize(png_path))
        cnt_jpg += os.path.getsize(img_path)
        frame_id += 1
    cap.release()
    # cnt_avi = os.path.getsize(video_path)
    # print(cnt_avi, cnt_jpg, cnt_jpg/cnt_avi * 100)
    return frame_id

def extract_traj(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
        position = np.array([[step["x"], step["z"]] for step in data], dtype=np.float32)
        yaw = np.array([step["yaw"] for step in data], dtype=np.float32)

    # Convert to expected format
    traj_data = {
        "position": position,
        "yaw": yaw
    }
    return traj_data

def process_dataset(input_dir, output_dir):
    # if os.path.exists(output_dir):
    #     print(f"✅ {output_dir} already exists")
    #     return
    # os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".avi")]
    for avi_file in tqdm(files, desc="Processing trajectories"):
        base_name = os.path.splitext(avi_file)[0]
        json_file = f"{base_name}.json"

        avi_path = os.path.join(input_dir, avi_file)
        json_path = os.path.join(input_dir, json_file)
        out_path = os.path.join(output_dir, base_name)
        os.makedirs(out_path, exist_ok=True)

        # Step 1: Extract frames
        num_frames = extract_frames(avi_path, out_path)

        # Step 2: Extract trajectory
        traj_data = extract_traj(json_path)

        if len(traj_data["position"]) != num_frames:
            print(f"[Warning] Frame count mismatch for {base_name}: {num_frames} frames vs {len(traj_data['position'])} positions")

        # # Step 3: Save traj_data.pkl
        # traj_pkl_path = os.path.join(out_path, "traj_data.pkl")
        # with open(traj_pkl_path, "wb") as f:
        #     pickle.dump(traj_data, f)
        # cp json file to out_path
        shutil.copy(json_path, os.path.join(out_path, json_file))

    print(f"✅ {input_dir} Dataset processing complete.")

if __name__ == "__main__":
    location_list = [
        "plains_village",
        "desert_village",
        "snowy_village",
        "taiga_village",
        "savanna_village",
        "zombie_village",
    ]
    all_location = []
    for i in range(len(location_list)):
        for j in range(1,21):
            all_location.append(f"{location_list[i]}_{j}")
    for nvtype in ["ABA", "ABCA"]:
        for nvrange in [5,15,30,50]:
            for location in all_location:
                input_dir = f"./data/{nvtype}/{location}/{nvrange}"      # <-- 修改为你的原始数据路径
                output_dir = f"./data_img/{nvtype}/{location}/{nvrange}"   # <-- 修改为输出路径
                process_dataset(input_dir, output_dir)
