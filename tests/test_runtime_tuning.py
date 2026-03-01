import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.runtime_tuning import (
    HardwareInfo,
    classify_hardware_tier,
    preset_for_tier,
)


class RuntimeTuningTests(unittest.TestCase):
    def test_classify_low(self):
        tier = classify_hardware_tier(HardwareInfo(cpu_cores=4, ram_gb=8.0, cpu_freq_ghz=2.3))
        self.assertEqual(tier, "low")

    def test_classify_mid(self):
        tier = classify_hardware_tier(HardwareInfo(cpu_cores=8, ram_gb=16.0, cpu_freq_ghz=2.8))
        self.assertEqual(tier, "mid")

    def test_classify_high(self):
        tier = classify_hardware_tier(HardwareInfo(cpu_cores=12, ram_gb=32.0, cpu_freq_ghz=3.5))
        self.assertEqual(tier, "high")

    def test_presets_are_reasonable(self):
        low = preset_for_tier("low")
        mid = preset_for_tier("mid")
        high = preset_for_tier("high")
        self.assertLessEqual(low.width, mid.width)
        self.assertLessEqual(mid.width, high.width)
        self.assertLessEqual(low.target_fps, mid.target_fps)
        self.assertLessEqual(mid.target_fps, high.target_fps)


if __name__ == "__main__":
    unittest.main()
