import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model.replay_eval import run_replay_eval


def make_parser():
    p = argparse.ArgumentParser(description="Replay evaluation on saved landmark clips")
    p.add_argument("--version", required=True, help="Dataset version name under data/landmarks/versions")
    p.add_argument("--model-name", default="custom")
    p.add_argument("--out-dir", default="data/models")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--min-conf", type=float, default=0.0, help="Below this confidence, prediction is treated as unknown")
    return p


def main():
    args = make_parser().parse_args()
    version_dir = Path("data/landmarks/versions") / str(args.version)
    result = run_replay_eval(
        version_dir=version_dir,
        model_name=str(args.model_name),
        out_dir=Path(args.out_dir),
        splits=list(args.splits),
        min_conf=float(args.min_conf),
    )
    print("\n=== Replay Evaluation ===")
    print(f"Model: {result['model_name']} | seq_len={result['seq_len']} | min_conf={result['min_conf']:.2f}")
    for split, stats in result["splits"].items():
        print(
            f"{split}: samples={stats['samples']} | acc={stats['accuracy']*100:.2f}% | "
            f"unknown={stats['unknown_rate']*100:.2f}% | avg_conf={stats['avg_conf']:.3f}"
        )
    print("\nJSON result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
