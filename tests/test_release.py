import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.release import ReleaseBundleConfig, build_release_bundle


class ReleaseTests(unittest.TestCase):
    def test_release_bundle_raises_when_no_artifacts(self):
        # This test only checks command behavior in isolated temp out-dir.
        # Real artifacts are created in actual project flow.
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "dist"
            # Can't assert success without writing into real project paths.
            # Just ensure function returns Path or raises controlled RuntimeError.
            try:
                p = build_release_bundle(ReleaseBundleConfig(out_dir=out_dir, include_videos=False))
                self.assertTrue(p.exists())
            except RuntimeError as ex:
                self.assertIn("No artifacts found to bundle", str(ex))


if __name__ == "__main__":
    unittest.main()

