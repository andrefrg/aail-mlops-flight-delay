import json
import os
import time
import unittest
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


FLIGHTS_FILE = Path(__file__).resolve().parents[1] / "examples" / "classify_request.json"
VALID_CLASSES = ("on_time", "moderate_delay", "severe_delay")


def load_example_flights():
    payload = json.loads(FLIGHTS_FILE.read_text(encoding="utf-8"))
    return payload["instances"]


def wait_for_api(base_url, timeout=60):
    """Esperar pelo arranque da API; falhar se não ficar disponível."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass  # A API pode ainda estar a arrancar.

        time.sleep(2)

    raise RuntimeError("A API não ficou disponível em /health")


class FlightAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ.get("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
        cls.flights = load_example_flights()
        wait_for_api(cls.base_url)

    def post_classify(self, payload):
        """Enviar um pedido JSON para /classify e ler a resposta."""
        request = Request(
            f"{self.base_url}/classify",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=30) as response:
            self.assertEqual(response.status, 200)
            return json.load(response)

    def classify(self, flights):
        """Classificar voos e confirmar o formato e as classes devolvidas."""
        result = self.post_classify({"instances": flights})

        self.assertIsInstance(result, dict)
        self.assertIn("classifications", result)

        predictions = result["classifications"]
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(flights))

        for prediction in predictions:
            self.assertIn(prediction, VALID_CLASSES)

        return predictions

    def test_health(self):
        with urlopen(f"{self.base_url}/health", timeout=5) as response:
            self.assertEqual(response.status, 200)

    def test_single_flight(self):
        self.classify([self.flights[0]])

    def test_batch_preserves_individual_predictions_and_order(self):
        batch_predictions = self.classify(self.flights)
        individual_predictions = []

        for flight in self.flights:
            prediction = self.classify([flight])[0]
            individual_predictions.append(prediction)

        self.assertEqual(
            batch_predictions,
            individual_predictions,
            "As previsões em lote devem manter a ordem dos voos.",
        )


if __name__ == "__main__":
    unittest.main()
