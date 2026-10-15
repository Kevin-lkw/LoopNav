# MineWorld on LoopNav

This directory overlays [mineworld](https://github.com/microsoft/mineworld) at revision `b1b5a385aeef20aed0cc1e0d5f25f74776b17fdc`. Install upstream dependencies and obtain the required model checkpoints separately.

## Setup

From the LoopNav repository root:

```bash
export LOOPNAV_REPO="$PWD"
git clone https://github.com/microsoft/mineworld.git ../mineworld-loopnav
git -C ../mineworld-loopnav checkout b1b5a385aeef20aed0cc1e0d5f25f74776b17fdc
cp -a models/mineworld/. ../mineworld-loopnav/
cd ../mineworld-loopnav
```

After [preparing benchmark inputs](../../benchmark/README.md#prepare-the-evaluation-subset), set:

```bash
export LOOPNAV_INPUTS="$LOOPNAV_REPO/benchmark/prepared"
export LOOPNAV_PREDICTIONS="$LOOPNAV_REPO/predictions"
```

## Run the model

Commands below run from this upstream checkout. The navigation example uses ABA_FIXED; repeat it with ABCA. Input preparation limits each location/range to the five frozen benchmark samples.

Use the 1.2B, 32-frame checkpoint and corresponding configuration. Set `--model_ckpt` to the checkpoint **file**. The navigation entry requires `--frames` and `--config`.

```bash
python inference.py \
  --config configs/1200M_32f.yaml --model_ckpt /path/to/mineworld.ckpt \
  --frames 32 --top_p 0.8 \
  --data_root "$LOOPNAV_INPUTS/gt" --nvtype ABA_FIXED --village_ids 19 20 \
  --output_dir "$LOOPNAV_PREDICTIONS/mineworld"

python evaluate_turn.py \
  --config configs/1200M_32f.yaml --checkpoint /path/to/mineworld.ckpt \
  --data-root "$LOOPNAV_INPUTS/turn_gt" --village-ids 19,20 \
  --output-root "$LOOPNAV_PREDICTIONS/mineworld"
```

## Evaluate

Follow the [benchmark evaluation guide](../../benchmark/README.md#evaluate-generated-videos). Keep each source trajectory's stem in the generated filename. This overlay's upstream license is included in the directory.
