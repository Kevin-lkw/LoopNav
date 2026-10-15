# Open OASIS on LoopNav

This directory overlays [open-oasis](https://github.com/etched-ai/open-oasis) at revision `f59deef2c019c212bd0c5a3a5b986a51f3701847`. Install upstream dependencies and obtain the required model checkpoints separately.

## Setup

From the LoopNav repository root:

```bash
export LOOPNAV_REPO="$PWD"
git clone https://github.com/etched-ai/open-oasis.git ../open-oasis-loopnav
git -C ../open-oasis-loopnav checkout f59deef2c019c212bd0c5a3a5b986a51f3701847
cp -a models/oasis/. ../open-oasis-loopnav/
cd ../open-oasis-loopnav
```

After [preparing benchmark inputs](../../benchmark/README.md#prepare-the-evaluation-subset), set:

```bash
export LOOPNAV_INPUTS="$LOOPNAV_REPO/benchmark/prepared"
export LOOPNAV_PREDICTIONS="$LOOPNAV_REPO/predictions"
```

## Run the model

Commands below run from this upstream checkout. The navigation example uses ABA_FIXED; repeat it with ABCA. Input preparation limits each location/range to the five frozen benchmark samples.

Download `oasis500m.safetensors` and `vit-l-20.safetensors` using the upstream instructions. The overlay includes its action loader and VAE changes, including chunked decoding for long trajectories.

```bash
python evaluate.py \
  --data-root "$LOOPNAV_INPUTS/gt" --nvtype ABA_FIXED \
  --village-ids 19 20 --n-prompt-frames 32 --ddim-steps 10 \
  --oasis-ckpt oasis500m.safetensors --vae-ckpt vit-l-20.safetensors \
  --output-path "$LOOPNAV_PREDICTIONS/oasis"

python evaluate_turn.py \
  --data-root "$LOOPNAV_INPUTS/turn_gt" --village-ids 19,20 \
  --context-frames 32 --ddim-steps 10 \
  --output-root "$LOOPNAV_PREDICTIONS/oasis"
```

## Evaluate

Follow the [benchmark evaluation guide](../../benchmark/README.md#evaluate-generated-videos). Keep each source trajectory's stem in the generated filename. This overlay's upstream license is included in the directory.
