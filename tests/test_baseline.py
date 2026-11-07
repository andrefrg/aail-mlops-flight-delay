"""Leakage, preparation, time separation, metrics and batch inference contracts."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.data.audit import BASE_FEATURES
from flight_delay.data.prepare import prepare, priority
from flight_delay.features.build import FlightFeatures, canonical_inputs
from flight_delay.models.evaluation import metrics
from flight_delay.models.inference import classify_payload
from flight_delay.models.pipeline import CLASSES, make_pipeline
from flight_delay.models.training import before


def fixture(year=2022):
    count = 18
    return pd.DataFrame(
        {
            "FlightDate": [f"{year}-01-{day:02d}" for day in range(1, count + 1)],
            "Reporting_Airline": ["AA", "UA", "DL"] * 6,
            "Origin": ["JFK", "LAX", "SFO"] * 6,
            "Dest": ["LAX", "SFO", "JFK"] * 6,
            "CRSDepTime": ["0900", "1300", "2400"] * 6,
            "CRSArrTime": ["1200"] * count,
            "CRSElapsedTime": ["180"] * count,
            "Distance": ["1000"] * count,
            "Flight_Number_Reporting_Airline": [str(i) for i in range(count)],
            "Cancelled": ["0"] * count,
            "Diverted": ["0"] * count,
            "ArrDelay": ["-5", "15", "60"] * 6,
            "target": ["on_time", "moderate_delay", "severe_delay"] * 6,
        }
    )


class BaselineTests(unittest.TestCase):
    def test_priority_is_independent_of_chunking(self):
        ids = np.arange(1000, dtype=np.uint64)
        self.assertTrue(
            np.array_equal(
                priority(ids, 42), np.concatenate([priority(ids[:17], 42), priority(ids[17:], 42)])
            )
        )

    def test_preparation_keeps_sources_and_is_chunk_invariant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for year in [2022, 2023]:
                data = fixture(year)
                data.loc[0, "CRSElapsedTime"] = "-5"
                data.loc[1, "Cancelled"] = "1"
                data.loc[2, "Diverted"] = "1"
                data.loc[3, "ArrDelay"] = ""
                data.loc[4, "FlightDate"] = "invalid"
                data.to_csv(raw / f"flights_{year}.csv", index=False)
            content = {p.name: p.read_bytes() for p in raw.iterdir()}
            config = {
                "years": [2022, 2023],
                "features": BASE_FEATURES,
                "chunksize": 3,
                "sample_per_month": 5,
                "seed": 42,
                "validation_start": "2023-01-01",
                "holdout_start": "2023-07-01",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            one = prepare(raw, root / "one", config_path)
            config["chunksize"] = 7
            config_path.write_text(json.dumps(config))
            two = prepare(raw, root / "two", config_path)
            a = pd.read_parquet(root / "one/monthly_sample.parquet")
            b = pd.read_parquet(root / "two/monthly_sample.parquet")
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(one["counts"], two["counts"])
            self.assertEqual(one["counts"]["eligible"], 30)
            self.assertEqual(one["counts"]["prepared"], 28)
            self.assertEqual(one["counts"]["invalid_duration_to_missing"], 2)
            self.assertEqual(content, {p.name: p.read_bytes() for p in raw.iterdir()})

    def test_outcomes_cannot_change_features(self):
        data = fixture()
        builder = FlightFeatures(include_route=True)
        original = builder.transform(data)
        data["ArrDelay"] = "999999"
        data["DepDelay"] = "-999"
        data["Cancelled"] = "1"
        pd.testing.assert_frame_equal(original, builder.transform(data))
        self.assertNotIn("FlightDate", original)
        self.assertNotIn("Year", original)

    def test_negative_duration_is_missing_and_2400_is_midnight(self):
        data = fixture()
        data.loc[0, "CRSElapsedTime"] = "-5"
        result = FlightFeatures().transform(data)
        self.assertTrue(pd.isna(result.loc[0, "CRSElapsedTime"]))
        self.assertEqual(result.loc[2, "departure_hour"], "0")
        data.loc[0, "CRSDepTime"] = "2360"
        with self.assertRaises(ValueError):
            FlightFeatures().transform(data)

    def test_temporal_embargo(self):
        data = pd.DataFrame(
            {"FlightDate": ["2023-06-20", "2023-06-24", "2023-06-30", "2023-07-01"]}
        )
        selected, cutoff = before(data, "2023-07-01", 7)
        self.assertEqual(cutoff, "2023-06-24")
        self.assertEqual(selected["FlightDate"].tolist(), ["2023-06-20"])

    def test_unknown_categories_batch_order_and_alias(self):
        data = fixture()
        config = {"seed": 42, "model": {"max_iter": 200}}
        model = make_pipeline({"name": "logistic_C1.0", "C": 1.0}, config)
        model.fit(data[BASE_FEATURES], data["target"])
        records = data[BASE_FEATURES].iloc[:2].to_dict("records")
        records[1]["Origin"] = "UNSEEN"
        records[1]["FL_DATE"] = records[1].pop("FlightDate")
        expected = [model.predict(pd.DataFrame([record]))[0] for record in records]
        self.assertEqual(
            classify_payload(model, {"instances": records})["classifications"], expected
        )
        self.assertEqual(classify_payload(model, {"instances": []}), {"classifications": []})
        with self.assertRaises(ValueError):
            classify_payload(model, {"instances": [123]})
        with self.assertRaises(ValueError):
            canonical_inputs(pd.DataFrame([{"FlightDate": "2022-01-01"}]))

    def test_streaming_confusion_metrics_match_sklearn(self):
        truth = ["on_time", "on_time", "moderate_delay", "severe_delay"]
        predicted = ["on_time", "moderate_delay", "moderate_delay", "on_time"]
        matrix = confusion_matrix(truth, predicted, labels=CLASSES)
        result = metrics(matrix)
        self.assertAlmostEqual(
            result["macro_f1"],
            f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0),
        )

    def test_imputer_statistics_do_not_change_on_prediction(self):
        data = fixture()
        data.loc[0, "CRSElapsedTime"] = "-5"
        config = {"seed": 42, "model": {"max_iter": 200}}
        model = make_pipeline({"name": "logistic_C1.0", "C": 1.0}, config).fit(
            data[BASE_FEATURES], data.target
        )
        imputer = model["encode"].named_transformers_["numeric"]["imputer"]
        original = imputer.statistics_.copy()
        future = data[BASE_FEATURES].copy()
        future["CRSElapsedTime"] = "999999"
        model.predict(future)
        np.testing.assert_array_equal(imputer.statistics_, original)
        self.assertEqual(original[1], 180)

    def test_conflicting_date_aliases_and_missing_fields_are_rejected(self):
        record = fixture()[BASE_FEATURES].iloc[0].to_dict()
        record["FL_DATE"] = "2023-01-01"
        with self.assertRaises(ValueError):
            canonical_inputs([record])
        del record["FL_DATE"]
        del record["Origin"]
        with self.assertRaises(ValueError):
            classify_payload(None, {"instances": [record]})


if __name__ == "__main__":
    unittest.main()
