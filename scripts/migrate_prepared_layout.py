"""Move legacy prepared partitions to monthly/ without changing their contents."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def migrate_layout(data, dry_run=False):
    data = Path(data)
    if not (data / "manifest.json").is_file():
        raise ValueError("Expected a prepared dataset with manifest.json")
    moves = [
        (source, data / "monthly" / source.name)
        for folder in ["train", "validation", "test"]
        for source in sorted((data / folder).glob("month=*"))
        if source.is_dir()
    ]
    targets = [target for _, target in moves]
    if len(set(targets)) != len(targets) or any(target.exists() for target in targets):
        raise FileExistsError("Conflicting monthly partitions; no files were moved")
    if not dry_run:
        if moves:
            (data / "monthly").mkdir(exist_ok=True)
            for source, target in moves:
                source.rename(target)
        for folder in ["train", "validation", "test"]:
            directory = data / folder
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    return len(moves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    count = migrate_layout(args.data, dry_run=args.dry_run)
    print(
        f"{'Would move' if args.dry_run else 'Moved'} {count} month partitions to {args.data / 'monthly'}"
    )
    print("File contents, sample and original manifest are preserved; folds remain date-based.")


if __name__ == "__main__":
    main()
