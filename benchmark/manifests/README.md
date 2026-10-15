# Frozen sample selection

[minecraft.json](minecraft.json) defines 720 trajectories: 240 each for TURN, ABA_FIXED, and ABCA. Each task/range/location setting contains five samples across six village biomes and village IDs 19/20.

Source video and annotation paths are relative to the extracted dataset root. `raw_frames` and `raw_return_point` describe the source sequence; the `*_10fps` fields define WorldMem context and prediction lengths. `output_relative_path` fixes the generated filename/layout shared by the runtime tools.

Use [prepare_inputs.py](../prepare_inputs.py) to resolve paths, prepare metadata, and create the runtime manifest. See the [preparation guide](../README.md#prepare-the-evaluation-subset). Do not edit this frozen selection when comparing archived scores; use a separate manifest for a different experiment.
