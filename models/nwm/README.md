# NWM on LoopNav

This directory overlays [nwm](https://github.com/facebookresearch/nwm) at revision `16bec8770d73ff9e82531bad36a8229cd1a02554`. Install upstream dependencies and obtain the required model checkpoints separately.

## Setup

From the LoopNav repository root:

```bash
export LOOPNAV_REPO="$PWD"
git clone https://github.com/facebookresearch/nwm ../nwm-loopnav
git -C ../nwm-loopnav checkout 16bec8770d73ff9e82531bad36a8229cd1a02554
cp -a models/nwm/. ../nwm-loopnav/
cd ../nwm-loopnav
```

After [preparing benchmark inputs](../../benchmark/README.md#prepare-the-evaluation-subset), set:

```bash
export LOOPNAV_INPUTS="$LOOPNAV_REPO/benchmark/prepared"
export LOOPNAV_PREDICTIONS="$LOOPNAV_REPO/predictions"
```

## Run the model

Commands below run from this upstream checkout. The navigation example uses ABA_FIXED; repeat it with ABCA. Input preparation limits each location/range to the five frozen benchmark samples.

After installing the upstream environment, run `python -m pip install -r requirements-loopnav.txt` for the metric dependencies imported by the training utilities.

The adapted configuration uses 32 context frames. Navigation inference locates the checkpoint using `results_dir`, `run_name`, and `--ckp`: by default `logs/nwm_cdit_b_32/checkpoints/0200000.pth.tar`. TURN accepts a checkpoint path directly.

```bash
python inference.py --exp config/nwm_cdit_b.yaml --ckp 0200000 \
  --data_root "$LOOPNAV_INPUTS/gt" --nvtype ABA_FIXED --village_ids 19 20 \
  --output_dir "$LOOPNAV_PREDICTIONS/nwm"

python evaluate_turn.py --checkpoint /path/to/0200000.pth.tar \
  --data-root "$LOOPNAV_INPUTS/turn_gt" --village-ids 19,20 \
  --output-root "$LOOPNAV_PREDICTIONS/nwm"
```

For training, `processmc.py --data-root /path/to/LoopNav --output-root data/LoopNav_img --tasks ABA ABCA` extracts JPEG frames and `traj_data.pkl` annotations expected by the upstream loader. Set `datasets.mc.data_folder` in `config/nwm_cdit_b.yaml` to the chosen processed task. The loader uses villages 1–16 for training and 17–18 for validation; keep villages 19–20 for benchmark evaluation. Use the adapted `train.py` with the upstream distributed-training setup.

## Evaluate

Follow the [benchmark evaluation guide](../../benchmark/README.md#evaluate-generated-videos). Keep each source trajectory's stem in the generated filename. This overlay's upstream license is included in the directory.
