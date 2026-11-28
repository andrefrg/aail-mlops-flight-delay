import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import distributions

from model_promotion.promote_model import required_variable


TAG = "validated_pr_merge"


def validation_key(merge_key):
    environment = [
        platform.python_version(),
        platform.machine(),
        platform.libc_ver(),
        sorted((package.metadata["Name"], package.version) for package in distributions()),
    ]
    return hashlib.sha256(json.dumps([merge_key, environment]).encode()).hexdigest()


def record_validation(client, model_name, commit_sha, key):
    version = client.get_model_version_by_alias(model_name, commit_sha)
    client.set_model_version_tag(model_name, version.version, TAG, key)
    return version.version


def reuse_model(client, model_name, commit_sha, key):
    versions = client.search_model_versions(f"tags.{TAG} = '{key}'")
    version = max(
        (item for item in versions if item.name == model_name and item.status == "READY"),
        key=lambda item: int(item.version),
        default=None,
    )
    if version is None:
        return None
    client.set_registered_model_alias(model_name, commit_sha, version.version)
    return version.version


def main():
    from mlflow.tracking import MlflowClient

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("record", "reuse"))
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=required_variable("MLFLOW_TRACKING_URI"))
    model_name = required_variable("MLFLOW_MODEL_NAME")
    commit_sha = required_variable("COMMIT_SHA")
    key = validation_key(required_variable("PR_MERGE_KEY"))
    if args.action == "record":
        version = record_validation(client, model_name, commit_sha, key)
        print(f"Modelo {model_name}, versão {version}: validação do PR registada.")
        return

    version = reuse_model(client, model_name, commit_sha, key)
    if version is None:
        print("Sem modelo validado para este merge. Vai ser feito o treino.", file=sys.stderr)
        print("reused=false")
    else:
        print(f"A reutilizar o modelo {model_name}, versão {version}, validado no PR.", file=sys.stderr)
        print("reused=true")


if __name__ == "__main__":
    main()
