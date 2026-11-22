"""Verify migration preserves files and refuses conflicting destinations."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from migrate_prepared_layout import migrate_layout


class MigrationTests(unittest.TestCase):
    def test_preserves_bytes_dry_run_and_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_bytes(b"original manifest")
            (root / "monthly_sample.parquet").write_bytes(b"original sample")
            for folder, month in [
                ("train", "2022-11"),
                ("validation", "2023-01"),
                ("test", "2023-07"),
            ]:
                path = root / folder / f"month={month}" / "part.parquet"
                path.parent.mkdir(parents=True)
                path.write_bytes(month.encode())
            self.assertEqual(migrate_layout(root, dry_run=True), 3)
            self.assertFalse((root / "monthly").exists())
            self.assertEqual(migrate_layout(root), 3)
            self.assertEqual(migrate_layout(root), 0)
            for month in ["2022-11", "2023-01", "2023-07"]:
                self.assertEqual(
                    (root / "monthly" / f"month={month}" / "part.parquet").read_bytes(),
                    month.encode(),
                )
            self.assertEqual((root / "manifest.json").read_bytes(), b"original manifest")
            self.assertEqual((root / "monthly_sample.parquet").read_bytes(), b"original sample")
            self.assertFalse((root / "train").exists())

    def test_conflict_does_not_move_other_months(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").touch()
            (root / "train/month=2022-01").mkdir(parents=True)
            (root / "train/month=2022-02").mkdir(parents=True)
            (root / "monthly/month=2022-02").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                migrate_layout(root)
            self.assertTrue((root / "train/month=2022-01").exists())
            self.assertFalse((root / "monthly/month=2022-01").exists())
