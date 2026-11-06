"""Testar promoções sem contactar o servidor MLflow real."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.promote_model import promote_model


class PromoteModelTest(unittest.TestCase):
    def test_promotes_the_version_validated_for_the_commit(self):
        client = Mock()
        client.get_model_version_by_alias.return_value = SimpleNamespace(version="17")
        version = promote_model(client, "model", "staging", "production", "a" * 40)
        self.assertEqual(version, "17")
        client.set_registered_model_alias.assert_called_once_with(
            name="model", alias="production", version="17"
        )

    def test_refuses_to_promote_a_different_commit(self):
        client = Mock()
        client.get_model_version_by_alias.side_effect = [
            SimpleNamespace(version="18"), SimpleNamespace(version="17")
        ]
        with self.assertRaises(ValueError):
            promote_model(client, "model", "staging", "production", "a" * 40)
        client.set_registered_model_alias.assert_not_called()

    def test_check_only_does_not_change_the_registry(self):
        client = Mock()
        client.get_model_version_by_alias.return_value = SimpleNamespace(version="17")
        promote_model(client, "model", "production", "production", "a" * 40, check_only=True)
        client.set_registered_model_alias.assert_not_called()

    def test_a_missing_alias_stops_promotion(self):
        client = Mock()
        client.get_model_version_by_alias.side_effect = RuntimeError("Alias inexistente")
        with self.assertRaises(RuntimeError):
            promote_model(client, "model", "staging", "production", "a" * 40)
        client.set_registered_model_alias.assert_not_called()


if __name__ == "__main__":
    unittest.main()
