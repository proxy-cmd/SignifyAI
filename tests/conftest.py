import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def add_src_to_python_path():
    src_path = str(SRC)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


add_src_to_python_path()
