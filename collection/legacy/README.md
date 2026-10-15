# Legacy collection utilities

Use [run_navigation.py](../run_navigation.py) and [run_turn.py](../run_turn.py) for current collection. This directory retains earlier workflows for reference.

| File | Purpose |
| --- | --- |
| [run.py](run.py) | Original ABA/ABCA collection loop |
| [run_aba_fixed.py](run_aba_fixed.py) | Earlier dedicated ABA_FIXED worker |
| [run_abaca_fixed.py](run_abaca_fixed.py) | Earlier dedicated ABACA_FIXED worker |
| [bird.js](bird.js) | Bird's-eye path visualization in Minecraft |

Python drivers locate the shared receiver, sender, and target files in their parent directory. Run `python collection/legacy/<script>.py --help` from the repository root for options. Install dependencies with `npm --prefix collection install`. The viewer uses the included target coordinates; edit its location/server/data settings before running `node collection/legacy/bird.js`.
