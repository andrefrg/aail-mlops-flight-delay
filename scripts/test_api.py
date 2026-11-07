import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import create_app, load_model
from flight_delay.data.audit import BASE_FEATURES
from flight_delay.models.pipeline import make_pipeline

TRAIN_DATA_PATH = ROOT / "data" / "processed" / "monthly_sample.parquet"
TRAINED_MODEL_PATH = ROOT / "models" / "model.joblib"
TRAIN_CONFIG_PATH = ROOT / "configs" / "baseline.json"


def flight(**overrides):
    record = {
        "FlightDate": "2025-05-15",
        "Reporting_Airline": "AA",
        "Origin": "JFK",
        "Dest": "LAX",
        "CRSDepTime": "0800",
        "CRSArrTime": "1200",
        "CRSElapsedTime": 240,
        "Distance": 2475,
    }
    record.update(overrides)
    return record


class FakeModel:
    """Classifica pela distância; permite verificar a ordem do batch."""

    feature_names_in_ = np.array(["Origin", "Distance"])

    def predict(self, frame):
        def label(distance):
            if distance >= 2000:
                return "severe_delay"
            if distance >= 500:
                return "moderate_delay"
            return "on_time"

        return [label(d) for d in frame["Distance"]]


class RecordingModel(FakeModel):
    def predict(self, frame):
        self.frame = frame
        return ["on_time"] * len(frame)


class IntegerOutputModel(FakeModel):
    def predict(self, frame):
        return np.array([0, 1, 2][: len(frame)] + [0] * max(0, len(frame) - 3))


class FailingModel(FakeModel):
    def predict(self, frame):
        raise ValueError("internal detail")


class WrongLengthModel(FakeModel):
    def predict(self, frame):
        return ["on_time", "on_time"]


class InvalidLabelModel(FakeModel):
    def predict(self, frame):
        return ["setosa"] * len(frame)


class ClassifyApiTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(FakeModel()).test_client()

    def post(self, payload, client=None):
        return (client or self.client).post("/classify", json=payload)

    def test_batch_preserves_order_and_size(self):
        instances = [
            flight(Distance=2500),
            flight(Distance=100),
            flight(Distance=900),
            flight(Distance=3000),
        ]
        response = self.post({"instances": instances})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"classifications": ["severe_delay", "on_time", "moderate_delay", "severe_delay"]},
        )

    def test_single_instance_returns_list_with_one_item(self):
        response = self.post({"instances": [flight(Distance=900)]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"classifications": ["moderate_delay"]})

    def test_integer_model_output_is_mapped_to_categories(self):
        client = create_app(IntegerOutputModel()).test_client()
        response = self.post({"instances": [flight(), flight(), flight()]}, client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["classifications"],
            ["on_time", "moderate_delay", "severe_delay"],
        )

    def test_missing_extra_and_null_fields_are_handled(self):
        model = RecordingModel()
        client = create_app(model).test_client()
        response = self.post(
            {
                "instances": [
                    {"Origin": "JFK", "DEP_DELAY": 55, "CAMPO_NOVO": 1},
                    {"Origin": "ATL", "Distance": None},
                ]
            },
            client,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(model.frame.columns), ["Origin", "Distance"])
        self.assertTrue(model.frame["Distance"].isna().all())
        self.assertEqual(len(response.get_json()["classifications"]), 2)

    def test_rejects_invalid_payloads(self):
        invalid_payloads = [
            {},
            [],
            "texto",
            {"instances": []},
            {"instances": "x"},
            {"instances": [1, 2]},
            {"instances": [{}]},
            {"instances": [flight(), "nao-e-objeto"]},
            {"instances": [{"Origin": {"nested": 1}}]},
            {"instances": [{"Origin": ["JFK"]}]},
            {"features": [1, 2]},  # formato antigo
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.post(payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_rejects_batch_above_maximum_size(self):
        client = create_app(FakeModel(), max_batch_size=2).test_client()
        response = self.post({"instances": [flight(), flight(), flight()]}, client)
        self.assertEqual(response.status_code, 400)
        self.assertIn("maximum batch size", response.get_json()["error"])

    def test_rejects_malformed_json(self):
        response = self.client.post("/classify", data="{", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_old_routes_are_gone(self):
        for route in ("/predict", "/prediction-batch"):
            with self.subTest(route=route):
                response = self.client.post(route, json={"instances": [flight()]})
                self.assertEqual(response.status_code, 404)
                self.assertIn("error", response.get_json())

    def test_wrong_method_returns_json_error(self):
        response = self.client.get("/classify")
        self.assertEqual(response.status_code, 405)
        self.assertIn("error", response.get_json())

    def test_prediction_error_is_reported_without_internal_details(self):
        client = create_app(FailingModel()).test_client()
        response = self.post({"instances": [flight()]}, client)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "Model prediction failed."})

    def test_wrong_number_of_predictions_is_reported(self):
        client = create_app(WrongLengthModel()).test_client()
        response = self.post({"instances": [flight()]}, client)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.get_json(),
            {"error": "Model returned an invalid number of predictions."},
        )

    def test_invalid_category_is_never_returned(self):
        client = create_app(InvalidLabelModel()).test_client()
        response = self.post({"instances": [flight()]}, client)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "Model returned an invalid category."})

    def test_response_has_request_id_header(self):
        response = self.post({"instances": [flight()]})
        self.assertTrue(response.headers.get("X-Request-ID"))


class UnknownCategoriesTests(unittest.TestCase):
    """Exercise the API with the model and prepared sample produced by train.py."""

    @classmethod
    def setUpClass(cls):
        if TRAINED_MODEL_PATH.is_file():
            model = joblib.load(TRAINED_MODEL_PATH)
        elif TRAIN_DATA_PATH.is_file():
            sample = pd.read_parquet(TRAIN_DATA_PATH)
            training = sample.loc[sample["split"].eq("train")]
            if training.empty:
                raise unittest.SkipTest("Prepared training sample has no training rows.")
            config = json.loads(TRAIN_CONFIG_PATH.read_text(encoding="utf-8"))
            model = make_pipeline(config["variants"][0], config)
            model.fit(training[BASE_FEATURES], training["target"].astype(str))
        else:
            raise unittest.SkipTest(
                "Run train.py to create the trained model or prepared training sample."
            )
        cls.client = create_app(model).test_client()

    def test_unseen_airline_and_airports_do_not_fail(self):
        response = self.client.post(
            "/classify",
            json={
                "instances": [
                    flight(),
                    flight(Reporting_Airline="ZZ", Origin="XXX", Dest="???"),
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        classifications = response.get_json()["classifications"]
        self.assertEqual(len(classifications), 2)
        self.assertTrue(set(classifications) <= {"on_time", "moderate_delay", "severe_delay"})

    def test_missing_fields_are_imputed(self):
        response = self.client.post("/classify", json={"instances": [{"Origin": "JFK"}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["classifications"]), 1)


class ModelAvailabilityTests(unittest.TestCase):
    def test_health_reports_model_version(self):
        with mock.patch.dict(os.environ, {"MODEL_VERSION": "v7"}):
            client = create_app(FakeModel()).test_client()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok", "model_version": "v7"})

    def test_app_starts_without_model_and_reports_503(self):
        with mock.patch.dict(os.environ, {"MODEL_PATH": "missing-model-for-test.pkl"}):
            client = create_app().test_client()
        health = client.get("/health")
        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.get_json(), {"status": "model_not_loaded"})
        response = client.post("/classify", json={"instances": [flight()]})
        self.assertEqual(response.status_code, 503)
        self.assertIn("error", response.get_json())

    def test_missing_model_file_fails_with_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "Unable to load model"):
            load_model(Path("missing-model-for-test.pkl"))


if __name__ == "__main__":
    unittest.main()