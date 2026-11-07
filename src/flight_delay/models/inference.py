"""Batch contract for the teammate implementing /classify."""

import pandas as pd

from flight_delay.data.audit import BASE_FEATURES
from flight_delay.features.build import FlightFeatures, serving_frame
from flight_delay.models.pipeline import CLASSES


def prepare_instances(instances):
    """Validate flight records using the same rules as the model."""
    for index, record in enumerate(instances):
        if not isinstance(record, dict):
            raise ValueError("Each instance must be a flight record object")
        available = set(record)
        if "FL_DATE" in available:
            available.add("FlightDate")
        missing = sorted(set(BASE_FEATURES) - available)
        if missing:
            raise ValueError(f"Instance {index} is missing required fields: {missing}")
    frame = serving_frame(pd.DataFrame(instances))
    FlightFeatures().transform(frame)
    return frame


def classify_payload(model, payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        raise ValueError("Expected a JSON object with an instances array")
    instances = payload["instances"]
    if not instances:
        return {"classifications": []}
    frame = prepare_instances(instances)
    predictions = model.predict(frame).tolist()
    if any(prediction not in CLASSES for prediction in predictions):
        raise ValueError("Model returned an unsupported prediction category")
    return {"classifications": predictions}
