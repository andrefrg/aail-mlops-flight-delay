"""Model/reference contract only; no HTTP API tests."""

from pathlib import Path
import sys
import unittest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flight_delay.models.pipeline import make_pipeline


class ClassroomTests(unittest.TestCase):
    def test_estimator_matches_exact_classroom_constructor(self):
        for C in [0.1, 1.0, 10.0]:
            pipeline = make_pipeline({"name": f"logistic_C{C}", "C": C}, {})
            expected = LogisticRegression(max_iter=200, C=C)
            self.assertIsInstance(pipeline[-1], LogisticRegression)
            self.assertEqual(pipeline[-1].get_params(), expected.get_params())
            self.assertFalse(pipeline["features"].include_route)


if __name__ == "__main__":
    unittest.main()
