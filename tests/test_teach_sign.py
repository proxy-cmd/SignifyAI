import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.teach_sign import TeachSignConfig, normalize_label, run_teach_sign


class TeachSignTests(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(normalize_label(" Hello World "), "hello_world")
        with self.assertRaises(ValueError):
            normalize_label("   ")

    @patch("signifyai.teach_sign.run_collection", return_value=42)
    @patch("signifyai.teach_sign.run_training", return_value=0.91)
    @patch("signifyai.teach_sign.set_phrase")
    def test_run_teach_sign_writes_summary(self, mock_set_phrase, mock_train, mock_collect):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "teach_summary.json"
            cfg = TeachSignConfig(
                label="hello",
                phrase_text="Hello there",
                samples=50,
                dataset_csv=root / "dataset.csv",
                model_path=root / "model.joblib",
                labels_path=root / "labels.json",
                metadata_path=root / "meta.json",
                summary_path=summary_path,
                run_deep=False,
                run_temporal=False,
            )
            out = run_teach_sign(cfg)
            self.assertEqual(out, summary_path)
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["label"], "hello")
            self.assertEqual(payload["samples_saved"], 42)
            self.assertAlmostEqual(payload["frame_accuracy"], 0.91)
            mock_set_phrase.assert_called_once()
            mock_collect.assert_called_once()
            mock_train.assert_called_once()


if __name__ == "__main__":
    unittest.main()

