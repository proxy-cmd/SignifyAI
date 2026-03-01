import sys
from pathlib import Path
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.external_data import import_zip_dataset, _validate_remote_url


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

    def test_import_zip_dataset_blocks_path_traversal_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "data.zip"
            out_dir = tmp_path / "out"

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../escape.txt", "nope")
                zf.writestr("safe/in.txt", "ok")

            count = import_zip_dataset(zip_path, out_dir)
            self.assertEqual(count, 1)
            self.assertTrue((out_dir / "safe" / "in.txt").exists())
            self.assertFalse((tmp_path / "escape.txt").exists())

    def test_import_zip_dataset_rejects_oversized_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "data.zip"
            out_dir = tmp_path / "out"

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("big.bin", b"a" * 20000)

            with self.assertRaises(RuntimeError):
                import_zip_dataset(zip_path, out_dir, max_member_bytes=1024)

    def test_validate_remote_url_rejects_local_hosts(self):
        bad_urls = [
            "file:///tmp/data.zip",
            "http://localhost/data.zip",
            "http://127.0.0.1/data.zip",
            "http://192.168.0.2/data.zip",
        ]
        for url in bad_urls:
            with self.assertRaises(ValueError):
                _validate_remote_url(url)

    def test_validate_remote_url_accepts_public_https(self):
        _validate_remote_url("https://example.com/data.zip")


if __name__ == "__main__":
    unittest.main()
