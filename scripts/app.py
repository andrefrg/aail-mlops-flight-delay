import logging
import os
import pickle
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException
from flight_delay.models.inference import prepare_instances


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "data" / "model.pkl"
DEFAULT_MAX_BATCH_SIZE = 10000

# Se o modelo devolver inteiros (0/1/2) em vez de texto, este é o mapeamento.
# Tem de coincidir com o usado no treino.
LABEL_BY_INDEX = {0: "on_time", 1: "moderate_delay", 2: "severe_delay"}
VALID_LABELS = frozenset(LABEL_BY_INDEX.values())


def _configure_logging():
    # Sob Gunicorn o logger raiz não tem handlers: sem isto, os INFO não aparecem.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def load_model(model_path=None):
    path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    try:
        model = joblib.load(path)
    except (OSError, EOFError, ValueError, ImportError, pickle.UnpicklingError) as exc:
        LOGGER.exception("Unable to load model from %s", path)
        raise RuntimeError(f"Unable to load model from {path}") from exc
    if not callable(getattr(model, "predict", None)):
        raise RuntimeError(f"Model loaded from {path} does not provide predict().")
    return model


def _normalize_value(value):
    """None e floats não finitos passam a NaN (valor em falta, imputado no Pipeline)."""
    if value is None:
        return float("nan")
    if isinstance(value, float) and not np.isfinite(value):
        return float("nan")
    return value


def _parse_instances(payload, max_batch_size):
    if not isinstance(payload, dict) or "instances" not in payload:
        return None, "JSON body must be an object containing an 'instances' array."

    instances = payload["instances"]
    if not isinstance(instances, list) or not instances:
        return None, "'instances' must be a non-empty array."
    if len(instances) > max_batch_size:
        return None, f"'instances' exceeds the maximum batch size of {max_batch_size}."

    cleaned = []
    for index, record in enumerate(instances):
        if not isinstance(record, dict) or not record:
            return None, f"instances[{index}] must be a non-empty object."
        row = {}
        for name, value in record.items():
            if isinstance(value, (dict, list)):
                return None, f"instances[{index}].{name} must be a scalar value."
            row[name] = _normalize_value(value)
        cleaned.append(row)
    return cleaned, None


def _build_frame(frame, model):
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        # Colunas em falta -> NaN; colunas extra (ex.: DEP_DELAY) -> ignoradas.
        frame = frame.reindex(columns=list(expected))
    # Coluna totalmente em falta fica float64; passa a object para que imputers e
    # encoders categóricos (strings) não falhem. Os numéricos convertem-se de volta.
    for column in frame.columns[frame.isna().all()]:
        frame[column] = frame[column].astype(object)
    return frame


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_label(value):
    value = _json_value(value)
    if isinstance(value, str) and value in VALID_LABELS:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in LABEL_BY_INDEX:
            return LABEL_BY_INDEX[value]
    raise ValueError(f"unsupported model output: {value!r}")


def create_app(model=None, max_batch_size=None):
    _configure_logging()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    app.config["MAX_BATCH_SIZE"] = int(
        max_batch_size or os.environ.get("MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE)
    )
    app.config["MODEL_VERSION"] = os.environ.get("MODEL_VERSION", "unknown")

    if model is None:
        try:
            model = load_model()
        except RuntimeError:
            # A API arranca na mesma: /health e /classify devolvem 503.
            LOGGER.error("Starting without a model; /health and /classify return 503.")
            model = None
    app.config["MODEL"] = model

    @app.before_request
    def _start_timer():
        g.start = time.perf_counter()
        g.request_id = uuid.uuid4().hex[:8]

    @app.after_request
    def _log_request(response):
        elapsed_ms = (time.perf_counter() - getattr(g, "start", time.perf_counter())) * 1000
        level = logging.DEBUG if request.path == "/health" else logging.INFO
        LOGGER.log(
            level,
            "rid=%s %s %s status=%d instances=%s ms=%.1f",
            getattr(g, "request_id", "-"),
            request.method,
            request.path,
            response.status_code,
            getattr(g, "n_instances", "-"),
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
        return response

    @app.errorhandler(Exception)
    def _handle_error(exc):
        if isinstance(exc, HTTPException):  # 404, 405, 413, ...
            return jsonify({"error": exc.description}), exc.code
        LOGGER.exception("Unhandled error")
        return jsonify({"error": "Internal server error."}), 500

    @app.get("/health")
    def health():
        if app.config["MODEL"] is None:
            return jsonify({"status": "model_not_loaded"}), 503
        return jsonify({"status": "ok", "model_version": app.config["MODEL_VERSION"]})

    @app.post("/classify")
    def classify():
        current_model = app.config["MODEL"]
        if current_model is None:
            return jsonify({"error": "Model is not available."}), 503

        payload = request.get_json(silent=True, force=True)
        instances, error = _parse_instances(payload, app.config["MAX_BATCH_SIZE"])
        if error:
            return jsonify({"error": error}), 400
        g.n_instances = len(instances)

        try:
            frame = prepare_instances(instances)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            frame = _build_frame(frame, current_model)
            raw = np.asarray(current_model.predict(frame))
        except Exception:
            LOGGER.exception("Model prediction failed (n=%d)", len(instances))
            return jsonify({"error": "Model prediction failed."}), 500

        if raw.ndim != 1 or len(raw) != len(instances):
            LOGGER.error(
                "Model returned an invalid prediction shape %s for %d instances",
                raw.shape,
                len(instances),
            )
            return jsonify({"error": "Model returned an invalid number of predictions."}), 500

        try:
            classifications = [_to_label(value) for value in raw]
        except ValueError:
            LOGGER.error(
                "Model returned labels outside the valid categories: %s",
                sorted({str(value) for value in raw})[:5],
            )
            return jsonify({"error": "Model returned an invalid category."}), 500

        return jsonify({"classifications": classifications})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
