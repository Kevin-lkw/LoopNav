"""Ensure moving the drivers does not change caller-relative output paths."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


COLLECTION = Path(__file__).resolve().parents[1]


class LaunchPathTest(unittest.TestCase):
    def check_driver(self, filename, cli, task, sender):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            caller = Path(directory)
            try:
                os.chdir(caller)
                with patch.object(sys, "argv", [filename, "--output_path", "recordings", *cli]), \
                     patch.object(sys, "path", [str(COLLECTION), *sys.path]):
                    module = runpy.run_path(str(COLLECTION / filename), run_name="driver_test")
                process = Mock(returncode=0)
                process.poll.return_value = 0
                process.wait.return_value = 0
                run = module["run_one_round"]
                with patch.dict(run.__globals__, receiver_port_available=lambda port: True), \
                     patch("subprocess.Popen", return_value=process) as launch, \
                     contextlib.redirect_stdout(io.StringIO()):
                    run(task, 0, "Bot", 8089, 8105)
                self.assertEqual(launch.call_count, 2)
                for call, expected in zip(launch.call_args_list, ("receiver.py", sender)):
                    command = call.args[0]
                    child_cwd = Path(call.kwargs.get("cwd", caller))
                    self.assertEqual((child_cwd / command[1]).resolve(), COLLECTION / expected)
                    if "--output_path" in command:
                        output = command[command.index("--output_path") + 1]
                        self.assertEqual((child_cwd / output).resolve(), caller / "recordings")
            finally:
                os.chdir(previous)

    def test_navigation_from_another_directory(self):
        self.check_driver("run_navigation.py", ["--nvtype", "ABA_FIXED"],
                          {"location": "desert_village_19", "nvrange": 5, "round_number": 0},
                          "senderhl.js")

    def test_turn_from_another_directory(self):
        self.check_driver("run_turn.py", [],
                          {"location": "desert_village_19", "turn_order": "left-right",
                           "turn_angle": 75, "bin_name": "60-90", "round_num": 0},
                          "sender_turn.js")


if __name__ == "__main__":
    unittest.main()
