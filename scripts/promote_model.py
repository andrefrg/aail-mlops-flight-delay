"""Mover um alias MLflow, seguindo o script de promoção do tutorial do docente."""

import argparse
import os


def promote_model(client, model_name, from_alias, to_alias, commit_sha, check_only=False):
    source = client.get_model_version_by_alias(model_name, from_alias)
    expected = client.get_model_version_by_alias(model_name, commit_sha)
    if source.version != expected.version:
        raise ValueError(f"O alias {from_alias} já não corresponde ao commit {commit_sha}")

    if not check_only:
        client.set_registered_model_alias(
            name=model_name, alias=to_alias, version=source.version
        )
        print(f"Modelo {model_name}, versão {source.version}: {from_alias} -> {to_alias}")
    else:
        print(f"Alias {from_alias} verificado: modelo {model_name}, versão {source.version}")
    return source.version


def required_variable(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Definir a variável {name}")
    return value


def main():
    from mlflow.tracking import MlflowClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    client = MlflowClient(tracking_uri=required_variable("MLFLOW_TRACKING_URI"))
    promote_model(
        client,
        required_variable("MLFLOW_MODEL_NAME"),
        required_variable("FROM_ALIAS"),
        required_variable("TO_ALIAS"),
        required_variable("COMMIT_SHA"),
        check_only=args.check_only,
    )


if __name__ == "__main__":
    main()
