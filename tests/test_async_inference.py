import sys
from pathlib import Path
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.async_inference import LatestFrameWorker


class AsyncInferenceTests(unittest.TestCase):
    def test_worker_processes_latest_item(self):
        def process(x: int) -> int:
            time.sleep(0.01)
            return x * 2

        w = LatestFrameWorker(process)
        w.start()
        try:
            for i in range(1, 8):
                w.submit(i)
            # give worker time to drain to latest
            time.sleep(0.2)
            out = w.poll_latest_result()
            self.assertIsNotNone(out)
            self.assertEqual(out, 14)
            self.assertGreaterEqual(w.stats.submitted, 7)
            self.assertGreaterEqual(w.stats.processed, 1)
        finally:
            w.close()

    def test_worker_records_errors(self):
        def process(_x: int) -> int:
            raise RuntimeError("boom")

        w = LatestFrameWorker(process)
        w.start()
        try:
            w.submit(1)
            time.sleep(0.05)
            self.assertGreaterEqual(w.stats.errors, 1)
            self.assertIn("boom", w.last_error() or "")
        finally:
            w.close()

    def test_worker_result_id_changes_when_new_result_arrives(self):
        def process(x: int) -> int:
            return x + 1

        w = LatestFrameWorker(process)
        w.start()
        try:
            out0, rid0 = w.poll_latest_result_with_id()
            self.assertIsNone(out0)
            self.assertEqual(rid0, 0)

            w.submit(10)
            time.sleep(0.05)
            out1, rid1 = w.poll_latest_result_with_id()
            self.assertEqual(out1, 11)
            self.assertGreaterEqual(rid1, 1)

            w.submit(20)
            time.sleep(0.05)
            out2, rid2 = w.poll_latest_result_with_id()
            self.assertEqual(out2, 21)
            self.assertGreater(rid2, rid1)
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
