import cv2
import json
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset

from typing import Dict, List
from .episode import Episode
from .segment import Segment, SegmentId

import numpy as np
import attr
#### vpt camera quantization
class QuantizationScheme:
    LINEAR = "linear"
    MU_LAW = "mu_law"


@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.

    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_([QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy


### end of vpt camera quantization

class MinecraftDataset(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root)
        print("loading MC dataset... It takes a while...")
        self.video_json_pairs, self.total_length = self._find_all_pairs(self.root)
        self.num_episodes = len(self.video_json_pairs)
        self.num_steps = self.total_length
        self.id_to_name = {i: pair["id"] for i, pair in enumerate(self.video_json_pairs)}
        self.name_to_idx = {v: k for k, v in self.id_to_name.items()}
        self.quantizer = CameraQuantizer(
            camera_maxval=0.1,
            camera_binsize=0.02,
            quantization_scheme=QuantizationScheme.MU_LAW,
            mu=5
        )
    def _get_length(self, json_path: Path) -> int:
        with open(json_path) as f:
            actions = json.load(f)
        return len(actions)

    def _find_all_pairs(self, root: Path) -> List[Dict]:
        pairs = []
        total_length = 0
        lengths = []
        for village in root.iterdir():
            for trajlen in village.iterdir():
                if trajlen.name != "5" and trajlen.name != "15":  
                    continue
                for video_file in trajlen.glob("*.avi"):
                    json_file = video_file.with_suffix(".json")
                    if json_file.exists():
                        length = self._get_length(json_file)
                        pairs.append({
                            "video": video_file,
                            "json": json_file,
                            "id": f"{village.name}/{trajlen.name}/{video_file.stem}",
                            "length": length
                        })
                        total_length += length
        return sorted(pairs, key=lambda x: x["id"]), total_length

    def __len__(self):
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        
        id = segment_id.episode_id
        pair = self.video_json_pairs[id]
        assert segment_id.start < pair["length"] and segment_id.stop > 0 and segment_id.start < segment_id.stop        

        pad_len_right = max(0, segment_id.stop - pair["length"])
        pad_len_left = max(0, -segment_id.start)

        start = max(0, segment_id.start)
        stop = min(pair["length"], segment_id.stop)
        mask_padding = torch.cat((
            torch.zeros(pad_len_left),
            torch.ones(stop - start),
            torch.zeros(pad_len_right)
        )).bool()

        obs = self._read_video(pair["video"], start, stop)
        act = self._read_actions(pair["json"], start, stop)

        def pad(x):
            right = F.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [pad_len_right]) if pad_len_right > 0 else x
            return F.pad(right, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0]) if pad_len_left > 0 else right

        obs = pad(obs)
        act = pad(act)
        rew = torch.zeros(obs.size(0))
        end = torch.zeros(obs.size(0), dtype=torch.uint8)
        trunc = torch.zeros(obs.size(0), dtype=torch.uint8)

        return Segment(obs, act, rew, end, trunc, mask_padding, info={}, id=segment_id)

    def _get_video_length(self, video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return length

    def _read_video(self, path: Path, start: int, stop: int) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(stop - start):
            ret, frame = cap.read()
            if not ret:
                break
            # downsample from (360, 640, 3) to (180, 320, 3)
            frame = cv2.resize(frame, (320, 180))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0 * 2 - 1

            frames.append(frame)
            # after proces: RGB,[-1,1],(3,180,320)
        cap.release()
        return torch.stack(frames)
    def _convert_to_tensor(self, action:dict) -> torch.Tensor:
        action = action["action"]
        ls = []
        ls.append(1 if action["forward"] else 0)
        ls.append(1 if action["jump"] else 0)
        yaw = action["camera"][0]
        yaw = self.quantizer.discretize(yaw)
        # convert to one hot vector
        yaw_one_hot = F.one_hot(torch.tensor(yaw), num_classes=11)
        ls.extend(yaw_one_hot)
        pitch = action["camera"][1]
        pitch = self.quantizer.discretize(pitch)
        pitch_one_hot = F.one_hot(torch.tensor(pitch), num_classes=11)
        ls.extend(pitch_one_hot)
        return torch.tensor(ls)

    def _read_actions(self, path: Path, start: int, stop: int) -> torch.Tensor:
        with open(path) as f:
            actions = json.load(f)

        # return torch.tensor(actions[start:stop])  # assumes list of scalars or vectors
        # return actions[start:stop] # assume failed
        tensor_actions = []
        for i in range(start, stop):
            tensor_actions.append(self._convert_to_tensor(actions[i]))
        return torch.stack(tensor_actions)

    def load_episode(self, episode_id: str) -> Episode:
        seg = self[SegmentId(episode_id, 0, self._length_one_episode)]
        return Episode(seg.obs, seg.action, seg.reward, seg.end, seg.trunc, seg.info)


if __name__ == "__main__":
    dataset = MinecraftDataset(Path("./data/ABA"))