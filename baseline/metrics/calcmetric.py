import torch
from torchvision.io import read_video
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from common_metrics_on_video_quality.calculate_lpips import calculate_lpips
from common_metrics_on_video_quality.calculate_ssim import calculate_ssim
import os
import json
from datetime import datetime
import cv2

def calculate_video_metrics(real_video, generated_video, device):
    """Calculate all video metrics.
    
    Args:
        real_video: Tensor of shape [B, T, H, W, C]
        generated_video: Tensor of shape [B, T, H, W, C]
    
    Returns:
        Dictionary containing all metrics
    """
    assert real_video.shape == generated_video.shape
    metrics = {}
    
    # Calculate SSIM
    metrics['ssim'] = calculate_ssim(real_video, generated_video, only_final=True)
    
    # Calculate LPIPS
    metrics['lpips'] = calculate_lpips(real_video, generated_video, device, only_final=True)
    
    # # Calculate FVD
    real_video = real_video.to(device)
    generated_video = generated_video.to(device)
    metrics['fvd'] = calculate_fvd(real_video, generated_video, device, method='styleganv', only_final=True)
    
    return metrics

def main():
    # Create output directory for metrics
    model = "mineworld"
    model_path = model + '/' + "output_32"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = "metrics_output" + model
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, f"metrics_{timestamp}.txt")
    all_avg = []
    with open(metrics_file, 'w') as f:
        f.write(f"Metrics Log - {timestamp}\n")
        f.write("="*50 + "\n\n")
        
        for nvtype in ["ABA", "ABCA"]:
            for length in [5,15,30,50]:
                length_metrics = {
                    'ssim': [],
                    'lpips': [],
                    'fvd': []
                }
                print(f"Calculating metrics for length {length}")
                for loc in ["snowy_village_20", "desert_village_20", "plains_village_20", 
                            "taiga_village_20", "savanna_village_20", "zombie_village_20"]:
                    path = "test_data/" + nvtype + "/" + str(length) + "/" + loc
                    file_list = os.listdir(path)
                    file_list.sort()
                    for i in range(0,len(file_list),2):
                        real_video = read_video(os.path.join(path, file_list[i]), pts_unit="sec")[0]
                        gen_path = model_path + "/" + nvtype + "/" + str(length) + "/" + loc + "/" + f"{file_list[i].split('.')[0]}.mp4"
                        if not os.path.exists(gen_path):
                            print(f"ERROR: File {gen_path} does not exist")
                            continue
                        generated_video = read_video(gen_path, pts_unit="sec")[0]
                        H,W = generated_video.shape[1], generated_video.shape[2]

                        real_video = real_video[:generated_video.shape[0]]

                        if real_video.shape[1] != H or real_video.shape[2] != W:
                            resized = []
                            for j in range(len(real_video)):
                                resized.append(cv2.resize(real_video[j].numpy(), (W, H)))
                            real_video = torch.tensor(resized)
                        # import ipdb; ipdb.set_trace()
                        # normalize to [0, 1]
                        real_video = real_video / 255.0
                        generated_video = generated_video / 255.0
                        real_video = real_video.unsqueeze(0)
                        generated_video = generated_video.unsqueeze(0)
                        real_video = real_video.permute(0, 1, 4, 2, 3)
                        generated_video = generated_video.permute(0, 1, 4, 2, 3)
                        
                        metrics = calculate_video_metrics(real_video, generated_video, device = "cuda:0")
                        # Convert metrics to float values if they are tensors
                        metrics = {k: float(v) if torch.is_tensor(v) else v for k, v in metrics.items()}
                        
                        # 收集当前length的所有指标
                        for key in length_metrics:
                            length_metrics[key].append(metrics[key]['value'][0])
                        print(f"Metrics for nvtype {nvtype} length {length} and location {loc}, file {file_list[i]}: {metrics}")
                        # 写入文件
                        f.write(f"Metrics for nvtype {nvtype} length {length} and location {loc}, file {file_list[i]}: {metrics}\n")
                
                # 计算并保存当前length的平均值
                avg_metrics = {key: sum(values)/len(values) for key, values in length_metrics.items()}
                all_avg.append(avg_metrics)
                # 打印到控制台
                print(f"Average metrics for nvtype {nvtype} length {length}: {avg_metrics}")
                # 写入文件
                f.write(f"Average metrics for nvtype {nvtype} length {length}: {avg_metrics}\n")
                f.write("-"*50 + "\n")  # 添加分隔线使输出更清晰
        index = 0
        f.write("#"*50 + "\n")
        for nvtype in ["ABA", "ABCA"]:
            for length in [5,15,30,50]:
                print(f"Average metrics for nvtype {nvtype} and length {length}: {all_avg[index]}")
                f.write(f"Average metrics for nvtype {nvtype} and length {length}: {all_avg[index]}\n")
                index += 1
if __name__ == "__main__":
    main()