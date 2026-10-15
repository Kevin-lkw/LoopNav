# DIAMOND on LoopNav

This directory overlays [diamond](https://github.com/eloialonso/diamond) at revision `851cefb497733d27f1b85c804104638765860fca`. Install upstream dependencies and obtain the required model checkpoints separately.

## Setup

From the LoopNav repository root:

```bash
export LOOPNAV_REPO="$PWD"
git clone https://github.com/eloialonso/diamond.git ../diamond-loopnav
git -C ../diamond-loopnav checkout 851cefb497733d27f1b85c804104638765860fca
cp -a models/diamond/. ../diamond-loopnav/
cd ../diamond-loopnav
```

After [preparing benchmark inputs](../../benchmark/README.md#prepare-the-evaluation-subset), set:

```bash
export LOOPNAV_INPUTS="$LOOPNAV_REPO/benchmark/prepared"
export LOOPNAV_PREDICTIONS="$LOOPNAV_REPO/predictions"
```

## Run the model

Commands below run from this upstream checkout. The navigation example uses ABA_FIXED; repeat it with ABCA. Input preparation limits each location/range to the five frozen benchmark samples.

The overlay includes training/data changes, the world-model environment, and both inference entrypoints. Use a trained run directory containing `checkpoints/agent_versions/agent_epoch_00200.pt`.

```bash
cd src
python inference.py --checkpoint_dir /path/to/diamond-run --ckpt 00200 \
  --data_root "$LOOPNAV_INPUTS/gt" --nvtype ABA_FIXED --village_ids 19 20 \
  --output_dir "$LOOPNAV_PREDICTIONS/diamond"
cd ..

python src/evaluate_turn.py --run-dir /path/to/diamond-run --epoch 00200 \
  --data-root "$LOOPNAV_INPUTS/turn_gt" --village-ids 19,20 \
  --output-root "$LOOPNAV_PREDICTIONS/diamond"
```

DIAMOND saves both navigation and TURN outputs below `<output>/00200/`. Point SGCS at that epoch directory, or its `TURN` child for TURN evaluation. Configure data paths in `config/env/csgo.yaml` before using the upstream training entrypoint.

## Evaluate

Follow the [benchmark evaluation guide](../../benchmark/README.md#evaluate-generated-videos). Keep each source trajectory's stem in the generated filename. This overlay's upstream license is included in the directory.
