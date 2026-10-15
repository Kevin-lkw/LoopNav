# Collection tests

From the repository root:

```bash
python -B -m unittest discover -s collection/tests -q
node --test collection/tests/smooth_camera.test.js
```

`test_trajectory_validation.py` covers camera discontinuities, horizontal speed, and loop closure. `smooth_camera.test.js` checks per-step camera limits and replacement of an in-flight camera target. No Minecraft server or model checkpoint is required.

`test_launch_paths.py` launches the driver functions with mocked child processes from a different working directory, checking receiver/sender resolution and preservation of caller-relative output paths.
