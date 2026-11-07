"""Batch contract for the teammate implementing /classify."""

import pandas as pd

from flight_delay.data.audit import BASE_FEATURES
from flight_delay.features.build import serving_frame
from flight_delay.models.pipeline import CLASSES


def classify_payload(model, payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        raise ValueError("Expected a JSON object with an instances array")
    instances = payload["instances"]
    if not all(isinstance(record, dict) for record in instances):
        raise ValueError("Each instance must be a flight record object")
    for index, record in enumerate(instances):
        available = set(record)
        if "FL_DATE" in available:
            available.add("FlightDate")
        missing = sorted(set(BASE_FEATURES) - available)
        if missing:
            raise ValueError(f"Instance {index} is missing required fields: {missing}")
    if not instances:
        return {"classifications": []}
    predictions = model.predict(serving_frame(pd.DataFrame(instances))).tolist()
    if any(prediction not in CLASSES for prediction in predictions):
        raise ValueError("Model returned an unsupported prediction category")
    return {"classifications": predictions}
