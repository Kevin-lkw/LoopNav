from pathlib import Path
import os

import numpy as np

from loopnav_finetune_v2.data import (
    CorrectedLoopNavTrainingDataset,
    action_pose_mismatch_degrees,
    convert_native_10fps,
    discover_training_trajectories,
)


DATA_ROOT = Path(os.environ.get("LOOPNAV_DATA_ROOT", "data/LoopNav"))


def _record(yaw, pitch=0.0, **actions):
    action = {
        "forward": False,
        "back": False,
        "left": False,
        "right": False,
        "jump": False,
        "sneak": False,
        "sprint": False,
    }
    action.update(actions)
    return {
        "x": 0.0,
        "y": 64.0,
        "z": 0.0,
        "pitch": pitch,
        "yaw": yaw,
        "action": action,
        "goal": {"x": 0, "z": 0},
    }


def test_pose_consistent_continuous_camera_action():
    records = [_record(index * 0.1) for index in range(6)]
    indices, poses, actions = convert_native_10fps(records)
    assert indices.tolist() == [0, 2, 4]
    assert actions.shape == (3, 25)
    assert np.all(actions[0] == 0)
    pitch_error, yaw_error = action_pose_mismatch_degrees(poses, actions)
    assert pitch_error < 1e-5
    assert yaw_error < 1e-5
    assert actions[1, 16] < 0


def test_pitch_sign_and_button_aggregation():
    records = [
        _record(0.0, pitch=0.0),
        _record(0.0, pitch=0.05, forward=True),
        _record(0.0, pitch=0.10),
    ]
    _indices, poses, actions = convert_native_10fps(records)
    assert poses[1, 3] < poses[0, 3] or poses[1, 3] > 300
    assert actions[1, 15] < 0
    assert actions[1, 11] == 1


def test_training_inventory_is_split_and_complete():
    inventory = discover_training_trajectories(DATA_ROOT)
    assert {task: len(paths) for task, paths in inventory.items()} == {
        "TURN": 7680,
        "ABA_FIXED": 7680,
        "ABCA": 7680,
    }
    for paths in inventory.values():
        assert all(int(path.parts[-3].rsplit("_", 1)[-1]) <= 16 for path in paths)


def test_real_samples_have_expected_shapes_and_causal_memory():
    dataset = CorrectedLoopNavTrainingDataset(
        data_root=DATA_ROOT,
        virtual_epoch_size=3,
        memory_retrieval_dropout=0.0,
    )
    for index, expected_task in enumerate(("TURN", "ABA_FIXED", "ABCA")):
        (video, action, pose, timestamp), debug = dataset.build_sample(index)
        assert debug.task == expected_task
        assert video.shape == (16, 3, 360, 640)
        assert action.shape == (16, 25)
        assert pose.shape == (16, 5)
        assert timestamp.shape == (16,)
        assert video.min() >= 0 and video.max() <= 1
        assert max(debug.memory_indices) < max(1, debug.observed_stop)
