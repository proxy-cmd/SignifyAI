import sys
from pathlib import Path
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.external_data import import_zip_dataset


class ExternalDataTests(unittest.TestCase):
    def test_import_zip_dataset_extracts_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "data.zip"
            out_dir = tmp_path / "out"

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("class_a/sample.txt", "ok")
                zf.writestr("class_b/sample.txt", "ok")

            count = import_zip_dataset(zip_path, out_dir)
            self.assertGreaterEqual(count, 2)
            self.assertTrue((out_dir / "class_a" / "sample.txt").exists())


if __name__ == "__main__":
    unittest.main()
