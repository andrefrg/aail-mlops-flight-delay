"""Explicit feature allowlist: outcomes and identifiers never enter the estimator."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from flight_delay.data.audit import BASE_FEATURES, numeric, valid_hhmm


def canonical_inputs(records):
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if "FL_DATE" in frame:
        if "FlightDate" in frame:
            primary = frame["FlightDate"]
            alias = frame["FL_DATE"]
            present = primary.notna() & primary.astype(str).str.strip().ne("")
            overlap = present & alias.notna() & alias.astype(str).str.strip().ne("")
            if (primary.loc[overlap].astype(str) != alias.loc[overlap].astype(str)).any():
                raise ValueError("Conflicting FlightDate and FL_DATE fields")
            frame["FlightDate"] = primary.where(present, alias)
        else:
            frame["FlightDate"] = frame["FL_DATE"]
    missing = sorted(set(BASE_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    # Selection is explicit, even if a historical row contains ArrDelay or other outcomes.
    return frame[BASE_FEATURES].copy()


class FlightFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, include_route=False):
        self.include_route = include_route

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        frame = canonical_inputs(X)
        dates = pd.to_datetime(frame["FlightDate"], format="%Y-%m-%d", errors="coerce")
        if dates.isna().any():
            raise ValueError("FlightDate must be a valid YYYY-MM-DD date")
        output = pd.DataFrame(index=frame.index)
        for field in ["Reporting_Airline", "Origin", "Dest"]:
            output[field] = (
                frame[field].fillna("").astype(str).str.strip().replace("", "__missing__")
            )
        output["month"] = dates.dt.month.astype(str)
        output["weekday"] = dates.dt.dayofweek.astype(str)
        output["day_of_month"] = dates.dt.day.astype(float)
        for field, destination in [
            ("CRSDepTime", "departure_hour"),
            ("CRSArrTime", "arrival_hour"),
        ]:
            strings = frame[field].fillna("").astype(str).str.strip()
            valid = valid_hhmm(strings)
            if (strings.ne("") & ~valid).any():
                raise ValueError(f"{field} must be HHMM (0000..2359, or 2400)")
            hours = (numeric(strings).floordiv(100) % 24).astype("Int64").astype(str)
            output[destination] = hours.where(valid, "__missing__")
        for field in ["CRSElapsedTime", "Distance"]:
            values = numeric(frame[field])
            output[field] = values.where(values.gt(0), np.nan).astype(float)
        if self.include_route:
            output["route"] = output["Origin"] + "->" + output["Dest"]
        return output


def categorical_columns(include_route=False):
    columns = [
        "Reporting_Airline",
        "Origin",
        "Dest",
        "month",
        "weekday",
        "departure_hour",
        "arrival_hour",
    ]
    return columns + (["route"] if include_route else [])


NUMERIC_COLUMNS = ["day_of_month", "CRSElapsedTime", "Distance"]


def serving_frame(records):
    """Canonical raw schema for sklearn and MLflow's typed pyfunc input signature."""
    frame = canonical_inputs(records)
    for field in ["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime"]:
        frame[field] = frame[field].fillna("").astype(str)
    for field in ["CRSElapsedTime", "Distance"]:
        frame[field] = numeric(frame[field]).astype("float64")
    return frame
