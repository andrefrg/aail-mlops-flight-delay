"""Explicit feature allowlist: outcomes and identifiers never enter the estimator."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from flight_delay.data.audit import BASE_FEATURES, numeric, valid_hhmm


LEADERBOARD_INPUT_NAMES = {
    "FlightDate": "FL_DATE",
    "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "Origin": "ORIGIN",
    "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME",
    "CRSArrTime": "CRS_ARR_TIME",
    "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "Distance": "DISTANCE",
}


def canonical_inputs(records):
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    for field, leaderboard_name in LEADERBOARD_INPUT_NAMES.items():
        if leaderboard_name not in frame:
            continue
        alternative = frame[leaderboard_name]
        if field not in frame:
            frame[field] = alternative
            continue
        primary = frame[field]
        present = primary.notna() & primary.astype(str).str.strip().ne("")
        if field == "FlightDate":
            overlap = present & alternative.notna() & alternative.astype(str).str.strip().ne("")
            if (primary.loc[overlap].astype(str) != alternative.loc[overlap].astype(str)).any():
                raise ValueError("Conflicting FlightDate and FL_DATE fields")
        frame[field] = primary.where(present, alternative)
    missing = sorted(set(BASE_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    # Selection is explicit, even if a historical row contains ArrDelay or other outcomes.
    return frame[BASE_FEATURES].copy()


class FlightFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, include_route=False, include_engineered=False):
        self.include_route = include_route
        self.include_engineered = include_engineered

    def fit(self, X, y=None):
        if getattr(self, "include_engineered", False):
            # Validate and normalize with the baseline transformer. No outcomes are used.
            base = FlightFeatures().transform(X)
            if base.empty:
                raise ValueError("Cannot learn feature frequencies from an empty training set")
            self.training_frequencies_ = {
                name: {
                    str(key): float(value)
                    for key, value in keys.value_counts(normalize=True).items()
                }
                for name, keys in frequency_keys(base).items()
            }
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
        if self.include_route or getattr(self, "include_engineered", False):
            output["route"] = output["Origin"] + "->" + output["Dest"]
        if getattr(self, "include_engineered", False):
            check_is_fitted(self, "training_frequencies_")
            output["airline_origin"] = output["Reporting_Airline"] + "->" + output["Origin"]
            output["origin_departure_hour"] = output["Origin"] + "->" + output["departure_hour"]
            for field, prefix in [("CRSDepTime", "departure"), ("CRSArrTime", "arrival")]:
                values = numeric(frame[field])
                minutes = (values.floordiv(100) * 60 + values.mod(100)).mod(1440)
                for suffix, function in [("sin", np.sin), ("cos", np.cos)]:
                    output[f"{prefix}_time_{suffix}"] = function(2 * np.pi * minutes / 1440)
            for name, values, period in [
                ("weekday", dates.dt.dayofweek, 7),
                ("year_day", dates.dt.dayofyear - 1, np.where(dates.dt.is_leap_year, 366, 365)),
            ]:
                output[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
                output[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
            output["is_weekend"] = dates.dt.dayofweek.ge(5).astype(float)
            output["log_distance"] = np.log1p(output["Distance"])
            output["log_scheduled_duration"] = np.log1p(output["CRSElapsedTime"])
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                ratio = output["Distance"] / output["CRSElapsedTime"]
            output["distance_per_scheduled_minute"] = ratio.replace([np.inf, -np.inf], np.nan)
            for name, keys in frequency_keys(output).items():
                output[f"{name}_training_frequency"] = keys.map(
                    self.training_frequencies_[name]
                ).fillna(0.0)
        return output


def frequency_keys(frame):
    return {
        "origin": frame["Origin"],
        "route": frame["Origin"] + "->" + frame["Dest"],
        "airline_origin": frame["Reporting_Airline"] + "->" + frame["Origin"],
    }


def categorical_columns(include_route=False, include_engineered=False):
    columns = [
        "Reporting_Airline",
        "Origin",
        "Dest",
        "month",
        "weekday",
        "departure_hour",
        "arrival_hour",
    ]
    columns += ["route"] if include_route or include_engineered else []
    return columns + (["airline_origin", "origin_departure_hour"] if include_engineered else [])


NUMERIC_COLUMNS = ["day_of_month", "CRSElapsedTime", "Distance"]


def numeric_columns(include_engineered=False):
    return NUMERIC_COLUMNS + (
        [
            "departure_time_sin",
            "departure_time_cos",
            "arrival_time_sin",
            "arrival_time_cos",
            "weekday_sin",
            "weekday_cos",
            "year_day_sin",
            "year_day_cos",
            "is_weekend",
            "log_distance",
            "log_scheduled_duration",
            "distance_per_scheduled_minute",
            "origin_training_frequency",
            "route_training_frequency",
            "airline_origin_training_frequency",
        ]
        if include_engineered
        else []
    )


def serving_frame(records):
    """Canonical raw schema for sklearn and MLflow's typed pyfunc input signature."""
    frame = canonical_inputs(records)
    for field in ["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime"]:
        frame[field] = frame[field].fillna("").astype(str)
    for field in ["CRSElapsedTime", "Distance"]:
        frame[field] = numeric(frame[field]).astype("float64")
    return frame
