"""Testar promoções sem contactar o servidor MLflow real."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from model_promotion.promote_model import promote_model
from model_promotion.reuse_pr_model import TAG, record_validation, reuse_model, validation_key


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


class ReusePrModelTest(unittest.TestCase):
    def test_records_validation_on_the_version_tested_for_the_pr_commit(self):
        client = Mock()
        client.get_model_version_by_alias.return_value = SimpleNamespace(version="17")

        self.assertEqual(record_validation(client, "model", "a" * 40, "matching-key"), "17")
        client.get_model_version_by_alias.assert_called_once_with("model", "a" * 40)
        client.set_model_version_tag.assert_called_once_with("model", "17", TAG, "matching-key")

    def test_reuses_the_latest_ready_version_of_the_same_model(self):
        client = Mock()
        client.search_model_versions.return_value = [
            SimpleNamespace(name="model", version="9", status="READY"),
            SimpleNamespace(name="model", version="17", status="READY"),
            SimpleNamespace(name="model", version="18", status="PENDING_REGISTRATION"),
            SimpleNamespace(name="other", version="19", status="READY"),
        ]

        self.assertEqual(reuse_model(client, "model", "b" * 40, "matching-key"), "17")
        client.search_model_versions.assert_called_once_with(f"tags.{TAG} = 'matching-key'")
        client.set_registered_model_alias.assert_called_once_with("model", "b" * 40, "17")

    def test_reuse_without_a_match_leaves_the_commit_alias_available_for_training(self):
        client = Mock()
        client.search_model_versions.return_value = []

        self.assertIsNone(reuse_model(client, "model", "b" * 40, "unmatched-key"))
        client.set_registered_model_alias.assert_not_called()

    def test_key_changes_when_the_merge_or_training_dependencies_change(self):
        packages = [SimpleNamespace(metadata={"Name": "scikit-learn"}, version="1.7.2")]
        with patch("model_promotion.reuse_pr_model.distributions", return_value=packages):
            reference = validation_key("base:head:tree")
            self.assertEqual(validation_key("base:head:tree"), reference)
            self.assertNotEqual(validation_key("other-base:head:tree"), reference)
            self.assertNotEqual(validation_key("base:head:other-tree"), reference)
            packages[0].version = "1.8.0"
            self.assertNotEqual(validation_key("base:head:tree"), reference)


if __name__ == "__main__":
    unittest.main()
