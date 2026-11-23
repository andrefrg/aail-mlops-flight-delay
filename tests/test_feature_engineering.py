"""Train/serve parity and point-in-time edge cases for additional features."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from flight_delay.data.audit import BASE_FEATURES
from flight_delay.features.build import FlightFeatures, categorical_columns, numeric_columns
from flight_delay.features.recent import recent_departure_features
from flight_delay.models.inference import classify_payload
from flight_delay.models.pipeline import make_pipeline
from test_baseline import fixture


class EngineeredFeaturesTests(unittest.TestCase):
    def test_frequency_is_fit_only_and_ignores_outcomes(self):
        train = fixture().iloc[:6].copy()
        features = FlightFeatures(include_engineered=True).fit(train, train.target)
        before = {k: dict(v) for k, v in features.training_frequencies_.items()}
        queries = fixture().iloc[:2].copy()
        queries.loc[queries.index[0], "Origin"] = "NEW"
        queries["ArrDelay"] = 999999
        values = features.transform(queries)
        self.assertEqual(values.iloc[0].origin_training_frequency, 0)
        self.assertAlmostEqual(values.iloc[1].origin_training_frequency, 1 / 3)
        pd.testing.assert_frame_equal(values, features.transform(queries.assign(ArrDelay=-999999)))
        self.assertEqual(features.training_frequencies_, before)
        alone = features.transform(queries.iloc[[1]])
        pd.testing.assert_frame_equal(alone, values.iloc[[1]])

    def test_cyclic_midnight_leap_year_and_missing_values(self):
        frame = fixture().iloc[:4].copy()
        frame["CRSDepTime"] = ["2359", "0001", "2400", ""]
        frame["FlightDate"] = ["2024-02-29"] * 4
        frame.loc[frame.index[-1], ["Distance", "CRSElapsedTime"]] = ["", "0"]
        values = FlightFeatures(include_engineered=True).fit_transform(frame)
        self.assertAlmostEqual(values.iloc[0].departure_time_cos, values.iloc[1].departure_time_cos)
        self.assertAlmostEqual(
            values.iloc[0].departure_time_sin, -values.iloc[1].departure_time_sin
        )
        self.assertAlmostEqual(values.iloc[2].departure_time_cos, 1)
        self.assertTrue(np.isnan(values.iloc[3].departure_time_sin))
        self.assertTrue(np.isnan(values.iloc[3].distance_per_scheduled_minute))
        self.assertAlmostEqual(values.iloc[0].year_day_sin, np.sin(2 * np.pi * 59 / 366))
        self.assertEqual(
            set(values.columns),
            set(
                categorical_columns(include_engineered=True)
                + numeric_columns(include_engineered=True)
            ),
        )

    def test_legacy_transformer_still_predicts(self):
        features = FlightFeatures()
        del features.include_engineered  # Simulate a previously serialized baseline.
        self.assertEqual(len(features.transform(fixture())), 18)

    def test_enriched_model_serving_serialization_and_unknowns(self):
        import skops.io as sio
        from scripts.app import create_app

        frame = fixture()
        for estimator in ["LinearSVC", "ExtraTreesClassifier", "RandomForestClassifier"]:
            with self.subTest(estimator=estimator):
                variant = {
                    "name": "engineered",
                    "estimator": estimator,
                    "feature_set": "engineered",
                    "C": 0.1,
                    "n_estimators": 3,
                    "n_jobs": 1,
                    "min_samples_leaf": 1,
                }
                model = make_pipeline(variant, {"seed": 42}).fit(frame[BASE_FEATURES], frame.target)
                future = frame[BASE_FEATURES].iloc[:2].copy()
                future["Origin"] = "NEW"
                future["FlightDate"] = "2025-05-15"
                payload = {"instances": future.to_dict("records")}
                expected = classify_payload(model, payload)
                trusted = ["flight_delay.features.build.FlightFeatures", "numpy.dtype"]
                if estimator != "LinearSVC":
                    trusted.append("sklearn.tree._tree.Tree")
                restored = sio.loads(sio.dumps(model), trusted=trusted)
                self.assertEqual(classify_payload(restored, payload), expected)
                self.assertEqual(
                    classify_payload(restored, {"instances": payload["instances"][:1]})[
                        "classifications"
                    ],
                    expected["classifications"][:1],
                )
                response = create_app(model=restored).test_client().post("/classify", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), expected)
                alias_payload = {
                    "instances": [
                        {
                            **{k: v for k, v in row.items() if k != "FlightDate"},
                            "FL_DATE": row["FlightDate"],
                        }
                        for row in payload["instances"]
                    ]
                }
                self.assertEqual(classify_payload(restored, alias_payload), expected)

    def test_missing_numeric_inputs_and_overflow_are_imputable(self):
        frame = fixture()
        model = make_pipeline(
            {"name": "e", "estimator": "LinearSVC", "feature_set": "engineered"}, {}
        )
        model.fit(frame[BASE_FEATURES], frame.target)
        query = frame[BASE_FEATURES].iloc[:1].copy()
        query["Distance"] = "1e308"
        query["CRSElapsedTime"] = "1e-308"
        features = model["features"].transform(query)
        self.assertTrue(np.isnan(features.iloc[0].distance_per_scheduled_minute))
        query["Distance"] = ""
        query["CRSElapsedTime"] = "0"
        query["CRSDepTime"] = ""
        query["CRSArrTime"] = ""
        self.assertEqual(len(model.predict(query)), 1)


class RecentContextTests(unittest.TestCase):
    def events(self):
        return pd.DataFrame(
            [
                {
                    "flight_id": "old",
                    "Origin": "JFK",
                    "departed_at": "2023-01-02T13:59:00Z",
                    "available_at": "2023-01-02T13:59:00Z",
                    "DepDelay": 100,
                },
                {
                    "flight_id": "a",
                    "Origin": "JFK",
                    "departed_at": "2023-01-02T14:00:00Z",
                    "available_at": "2023-01-02T14:01:00Z",
                    "DepDelay": 10,
                },
                {
                    "flight_id": "b",
                    "Origin": "JFK",
                    "departed_at": "2023-01-02T14:20:00Z",
                    "available_at": "2023-01-02T14:25:00Z",
                    "DepDelay": 30,
                },
                {
                    "flight_id": "late",
                    "Origin": "JFK",
                    "departed_at": "2023-01-02T14:10:00Z",
                    "available_at": "2023-01-02T14:31:00Z",
                    "DepDelay": 90,
                },
                {
                    "flight_id": "self",
                    "Origin": "JFK",
                    "departed_at": "2023-01-02T14:15:00Z",
                    "available_at": "2023-01-02T14:16:00Z",
                    "DepDelay": 300,
                },
                {
                    "flight_id": "equal",
                    "Origin": "JFK",
                    "departed_at": "2023-01-02T14:30:00Z",
                    "available_at": "2023-01-02T14:30:00Z",
                    "DepDelay": 200,
                },
            ]
        )

    def queries(self):
        return pd.DataFrame(
            [{"flight_id": "self", "Origin": "JFK", "prediction_at": "2023-01-02T14:30:00Z"}]
        )

    def calculate(self, events=None, queries=None, **kwargs):
        return recent_departure_features(
            self.events() if events is None else events,
            self.queries() if queries is None else queries,
            coverage_start=kwargs.get("coverage_start", "2023-01-02T00:00:00Z"),
            observed_through=kwargs.get("observed_through", "2023-01-03T00:00:00Z"),
        )

    def test_boundaries_late_events_and_self_exclusion(self):
        result = self.calculate().iloc[0]
        self.assertEqual(result.origin_departure_30m_mean_delay, 20)
        self.assertEqual(result.origin_departure_30m_valid_count, 2)
        self.assertEqual(result.origin_departure_30m_delayed_fraction, 0.5)
        self.assertEqual(result.origin_departure_30m_covered, 1)
        self.assertAlmostEqual(result.origin_departure_60m_mean_delay, 140 / 3)

    def test_missing_history_is_not_zero_delay(self):
        queries = self.queries().assign(Origin="NEW")
        result = self.calculate(queries=queries).iloc[0]
        self.assertTrue(np.isnan(result.origin_departure_30m_mean_delay))
        self.assertEqual(result.origin_departure_30m_valid_count, 0)
        self.assertEqual(result.origin_departure_30m_covered, 1)
        empty = self.events().iloc[:0]
        self.assertEqual(self.calculate(events=empty).iloc[0].origin_departure_30m_valid_count, 0)

    def test_missing_snapshot_and_warmup_are_distinct(self):
        result = self.calculate(
            queries=self.queries().assign(prediction_at="2023-01-04T00:00:00Z")
        ).iloc[0]
        self.assertEqual(result.origin_departure_30m_covered, 0)
        self.assertTrue(np.isnan(result.origin_departure_30m_valid_count))
        events = self.events().loc[lambda f: f.flight_id.ne("old")]
        result = self.calculate(events=events, coverage_start="2023-01-02T14:00:00Z").iloc[0]
        self.assertEqual(result.origin_departure_30m_covered, 1)
        self.assertEqual(result.origin_departure_60m_covered, 0)

    def test_invalid_delay_duplicates_and_revisions(self):
        events = self.events()
        events["DepDelay"] = events["DepDelay"].astype(float)
        events.loc[events.flight_id.eq("b"), "DepDelay"] = np.inf
        events = pd.concat([events, events.iloc[[1]]], ignore_index=True)
        result = self.calculate(events=events).iloc[0]
        self.assertEqual(result.origin_departure_30m_event_count, 2)
        self.assertEqual(result.origin_departure_30m_valid_count, 1)
        self.assertEqual(result.origin_departure_30m_mean_delay, 10)
        conflict = events.iloc[[1]].assign(DepDelay=999)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.calculate(events=pd.concat([events, conflict]))

    def test_order_batch_and_future_outcome_independence(self):
        events = self.events()
        expected = self.calculate(events=events)
        events.loc[events.flight_id.isin(["equal", "late"]), "DepDelay"] = 999999
        pd.testing.assert_frame_equal(self.calculate(events=events.sample(frac=1)), expected)
        queries = pd.concat(
            [self.queries().assign(Origin="NEW"), self.queries()], ignore_index=True
        )
        pd.testing.assert_series_equal(
            self.calculate(queries=queries).iloc[1], expected.iloc[0], check_names=False
        )

    def test_midnight_timezones_and_negative_delays(self):
        event = self.events().iloc[[0]].copy()
        event["departed_at"] = "2023-01-01T18:50:00-05:00"
        event["available_at"] = "2023-01-01T23:55:00Z"
        event["DepDelay"] = -5
        query = self.queries().assign(prediction_at="2023-01-02T00:10:00Z")
        result = self.calculate(
            events=event, queries=query, coverage_start="2023-01-01T00:00:00Z"
        ).iloc[0]
        self.assertEqual(result.origin_departure_30m_mean_delay, -5)
        self.assertEqual(result.origin_departure_30m_delayed_fraction, 0)

    def test_ambiguous_times_and_impossible_availability_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            self.calculate(queries=self.queries().assign(prediction_at="2023-01-02 14:30"))
        with self.assertRaisesRegex(ValueError, "before departure"):
            self.calculate(events=self.events().assign(available_at="2023-01-02T00:00:00Z"))

    def test_empty_queries_and_all_invalid_delays(self):
        self.assertEqual(self.calculate(queries=self.queries().iloc[:0]).shape, (0, 10))
        result = self.calculate(events=self.events().assign(DepDelay=None)).iloc[0]
        self.assertEqual(result.origin_departure_30m_event_count, 2)
        self.assertEqual(result.origin_departure_30m_valid_count, 0)
        self.assertTrue(np.isnan(result.origin_departure_30m_mean_delay))


if __name__ == "__main__":
    unittest.main()
