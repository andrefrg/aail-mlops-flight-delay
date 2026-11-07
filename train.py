"""Part 1: prepare, train, evaluate and save the model; no service or infrastructure setup."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from flight_delay.data.prepare import prepare
from flight_delay.models.training import train, evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "prepare", "train", "evaluate"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Explicitly repeat training and overwrite local model/results",
    )
    args = parser.parse_args()
    if args.phase in ["all", "prepare"]:
        output = ROOT / "data/processed"
        if (output / "manifest.json").exists() and not args.force_prepare:
            print("Using existing prepared data.", flush=True)
        else:
            prepare(args.data_dir, output, args.config, force=args.force_prepare)
    if args.phase in ["all", "train"]:
        train(args.config, retrain=args.retrain)
    if args.phase in ["all", "evaluate"]:
        evaluated = evaluate_model(args.config)
        result = evaluated["baseline"]
        print(f"model.joblib: accuracy={result['accuracy']:.6f}; macro-F1={result['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
