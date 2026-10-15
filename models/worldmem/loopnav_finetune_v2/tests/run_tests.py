"""Run unit checks; opt into dataset checks with --with-data."""

import argparse

from loopnav_finetune_v2.tests.test_data import (
    test_pitch_sign_and_button_aggregation,
    test_pose_consistent_continuous_camera_action,
    test_real_samples_have_expected_shapes_and_causal_memory,
    test_training_inventory_is_split_and_complete,
)
from loopnav_finetune_v2.tests.test_model import (
    test_lora_is_initially_identity_and_has_gradients,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-data", action="store_true")
    args = parser.parse_args()
    tests = [
        test_pose_consistent_continuous_camera_action,
        test_pitch_sign_and_button_aggregation,
        test_lora_is_initially_identity_and_has_gradients,
    ]
    if args.with_data:
        tests.extend([test_training_inventory_is_split_and_complete,
                      test_real_samples_have_expected_shapes_and_causal_memory])
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    print(f"PASS all {len(tests)} tests", flush=True)


if __name__ == "__main__":
    main()
