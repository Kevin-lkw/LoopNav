# Minecraft data collection

Collect paired 640 × 360 videos at 20 FPS and per-frame JSON annotations using Mineflayer, Pathfinder, and Prismarine Viewer. Run the following commands from the LoopNav repository root.

## Install and configure

```bash
npm --prefix collection install
python -m pip install opencv-python numpy
```

Use Linux, Node.js/npm, and Java compatible with your Minecraft server. Canvas/WebGL also requires its native headless-rendering dependencies. Start the server with its usual Java command, for example `java -Xmx2G -Xms2G -jar server.jar nogui`.

| Setting | Where |
| --- | --- |
| Target coordinates | [target.js](target.js) |
| Location groups | [target.py](target.py); keep names consistent with the JavaScript coordinates |
| Minecraft ports | `--mc_ports`, default `8105` |
| Bot usernames | `--bot_names`; grant each bot operator permission on the server |
| Recorded world seed | `extra_info.seed` in [receiver.py](receiver.py) |

`senderhl.js` accepts `collection/targets.js` as a local coordinate override and falls back to `target.js`. Video receiver ports are separate from Minecraft server ports and start at `8089`. See [fork changes](../docs/collector_changes.md) for the navigation/rendering modifications.

## Collect trajectories

```bash
python collection/run_navigation.py --nvtype ABA_FIXED \
  --output_path ./output --workers 1 --bot_names Bot --mc_ports 8105

python collection/run_navigation.py --nvtype ABCA \
  --output_path ./output --workers 1 --bot_names Bot --mc_ports 8105

python collection/run_turn.py --output_path ./output \
  --workers 1 --bot_names Bot --mc_ports 8105
```

The navigation driver also supports ABA and ABACA_FIXED. Defaults target 20 accepted recordings per configuration; restrict runs with `--location` and `--nvrange`. Use multiple bot names and Minecraft ports with `--workers` for parallel collection. `--help` lists all options. Output paths are resolved relative to the directory from which you invoke the command.

Navigation validation rejects excessive camera discontinuities, horizontal speeds, and loop-closure errors. Default limits are 30 degrees per native frame, 15 blocks/s, and 2 blocks, respectively.

```text
output/<task>/<location>/<range>/<timestamp>.avi
output/<task>/<location>/<range>/<timestamp>.json
```

The receiver records actions, pose, navigation metadata, and task-specific return markers. Use JSON list positions to align saved frames; `frame_count` includes the initial skipped frames. See the [dataset format](../docs/dataset.md#format).

## Files

| File or directory | Role |
| --- | --- |
| [run_navigation.py](run_navigation.py) | Navigation orchestration, retries, and acceptance checks |
| [run_turn.py](run_turn.py) | TURN collection and worker scheduling |
| [senderhl.js](senderhl.js), [sender_turn.js](sender_turn.js) | Minecraft bot control and rendering |
| [receiver.py](receiver.py) | Save videos and frame annotations |
| [smooth_camera.js](smooth_camera.js) | Interpolate camera motion |
| [trajectory_validation.py](trajectory_validation.py) | Shared camera, speed, and closure checks |
| [tests/](tests/README.md) | Unit checks without a running Minecraft server |
| [legacy/](legacy/README.md) | Earlier collection drivers and bird's-eye viewer |

After recording, run the [quality tools](../quality/README.md). Use the [benchmark preparation script](../benchmark/README.md#prepare-the-evaluation-subset) when evaluating the frozen public subset.
