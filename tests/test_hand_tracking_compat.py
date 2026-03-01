import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google.protobuf import message_factory, symbol_database
from signifyai import hand_tracking  # noqa: F401  # importing applies compat patch


class HandTrackingCompatTests(unittest.TestCase):
    def test_protobuf_getprototype_compat_exists(self):
        self.assertTrue(hasattr(symbol_database.SymbolDatabase, "GetPrototype"))
        self.assertTrue(hasattr(message_factory.MessageFactory, "GetPrototype"))


if __name__ == "__main__":
    unittest.main()
