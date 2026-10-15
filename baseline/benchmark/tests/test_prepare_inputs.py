import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("prepare_inputs", Path(__file__).parents[1] / "prepare_inputs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrepareInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.output = self.root / "prepared"
        self.source = self.data / "ABA/desert_village_19/5/sample.json"
        self.source.parent.mkdir(parents=True)
        self.source.write_text(json.dumps([{}, {}, {}, {}]))
        self.source.with_suffix(".avi").write_bytes(b"fixture")
        self.sample = dict(task="ABA_FIXED", nav_range="5", village="desert_village_19",
                           stem="sample", source_json="ABA/desert_village_19/5/sample.json",
                           source_video="ABA/desert_village_19/5/sample.avi",
                           raw_frames=4, raw_return_point=2)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"samples": [self.sample]}))

    def test_restores_marker_without_touching_source_and_is_repeatable(self):
        original = self.source.read_bytes()
        result = module.prepare(self.manifest, self.data, self.output)
        self.assertEqual(result["return_points_restored"], 1)
        manifest = json.loads((self.output / "selection_manifest.json").read_text())
        prepared = manifest["samples"][0]
        self.assertTrue(json.loads(Path(prepared["source_json"]).read_text())[2]["return_point"])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(Path(prepared["source_video"]).resolve(), self.source.with_suffix(".avi"))
        self.assertEqual(module.prepare(self.manifest, self.data, self.output), result)

    def test_conflicting_return_marker_is_rejected_before_writing(self):
        self.source.write_text(json.dumps([{}, {"return_point": True}, {}, {}]))
        with self.assertRaisesRegex(ValueError, "Return point differs"):
            module.prepare(self.manifest, self.data, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_input_does_not_leave_partial_output(self):
        self.source.with_suffix(".avi").unlink()
        with self.assertRaises(FileNotFoundError):
            module.prepare(self.manifest, self.data, self.output)
        self.assertFalse(self.output.exists())

    def test_dry_run_has_no_writes(self):
        module.prepare(self.manifest, self.data, self.output, dry_run=True)
        self.assertFalse(self.output.exists())

    def test_refuses_to_replace_unrelated_video(self):
        target = self.output / "gt/ABA_FIXED/5/desert_village_19/sample.avi"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"unrelated")
        with self.assertRaises(FileExistsError):
            module.prepare(self.manifest, self.data, self.output)
        self.assertEqual(target.read_bytes(), b"unrelated")

    def test_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            module.resolve_source(self.data, "../manifest.json")


if __name__ == "__main__":
    unittest.main()
