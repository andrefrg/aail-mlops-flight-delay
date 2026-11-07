"""Boundary and cross-chunk checks for the audit; no real dataset needed."""

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.data.audit import (
    AuditState,
    candidate_keys,
    eligible_mask,
    make_target,
    run_audit,
    valid_hhmm,
)


def sample():
    return pd.DataFrame(
        {
            "FlightDate": ["2022-01-01"] * 5,
            "Reporting_Airline": ["AA"] * 5,
            "Origin": ["JFK"] * 5,
            "Dest": ["LAX"] * 5,
            "Flight_Number_Reporting_Airline": ["10", "10.0", "11", "12", "13"],
            "CRSDepTime": ["0900"] * 5,
            "CRSArrTime": ["1200"] * 5,
            "CRSElapsedTime": ["180"] * 5,
            "Distance": ["1000"] * 5,
            "Cancelled": ["0", "0", "1", "0", "0"],
            "Diverted": ["0", "0", "0", "1", "0"],
            "ArrDelay": ["-4", "15", "60", "70", ""],
            "Flights": ["1"] * 5,
        }
    )


class AuditTests(unittest.TestCase):
    def test_target_boundaries_and_invalid(self):
        values = pd.Series(["-10", "0", "14.99", "15", "59.99", "60", "", "bad", "inf"])
        result = make_target(values)
        self.assertEqual(
            result.iloc[:6].tolist(), ["on_time"] * 3 + ["moderate_delay"] * 2 + ["severe_delay"]
        )
        self.assertTrue(result.iloc[6:].isna().all())

    def test_eligibility_does_not_impute_missing_target(self):
        self.assertEqual(eligible_mask(sample()).tolist(), [True, True, False, False, False])

    def test_hhmm_handles_midnight_and_impossible_values(self):
        values = pd.Series(["0000", "2359", "2400", "2360", "2401", "1261", "-1", "9.5", ""])
        self.assertEqual(
            valid_hhmm(values).tolist(),
            [True, True, True, False, False, False, False, False, False],
        )

    def test_accumulation_is_invariant_to_chunk_boundaries(self):
        full, split = AuditState(), AuditState()
        data = sample()
        data.loc[0, "Distance"] = " "
        full.consume(data, 2022)
        split.consume(data.iloc[:2], 2022)
        split.consume(data.iloc[2:], 2022)
        self.assertEqual(full.rows, 5)
        self.assertEqual(full.eligible, 2)
        self.assertEqual(full.classes, split.classes)
        self.assertEqual(full.missing, split.missing)
        self.assertEqual(full.values, split.values)
        self.assertEqual(full.checks, split.checks)
        self.assertEqual(full.eligible_missing["Distance"], 1)

    def test_candidate_key_normalizes_number_but_preserves_distinct_flights(self):
        keys = candidate_keys(sample())
        self.assertEqual(keys.iloc[0], keys.iloc[1])
        self.assertNotEqual(keys.iloc[1], keys.iloc[2])

    def test_complete_run_counts_keys_across_chunks_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            for year in [2022, 2023]:
                data = sample()
                data["FlightDate"] = f"{year}-01-01"
                data.to_csv(data_dir / f"flights_{year}.csv", index=False)
            policy = root / "policy.csv"
            pd.DataFrame(
                {
                    "feature": list(sample().columns),
                    "role": "candidate",
                    "initial_baseline": "false",
                    "reason": "test",
                }
            ).to_csv(policy, index=False)
            before = {p.name: p.read_bytes() for p in data_dir.iterdir()}
            summary = run_audit(data_dir, root / "reports", policy, chunksize=1)
            self.assertEqual(summary["rows_raw"], 10)
            self.assertEqual(summary["rows_eligible"], 4)
            self.assertEqual(summary["duplicates"]["repeated_key_groups"], 2)
            self.assertEqual(summary["duplicates"]["extra_rows_with_repeated_key"], 2)
            self.assertEqual(before, {p.name: p.read_bytes() for p in data_dir.iterdir()})
            self.assertTrue((root / "reports/feature_review.csv").exists())


if __name__ == "__main__":
    unittest.main()
