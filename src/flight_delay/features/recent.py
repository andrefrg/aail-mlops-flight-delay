"""Optional point-in-time departure context; requires an external event history.

This is not called by the deployed model: BTS requests do not provide this feed.
Times must have explicit offsets, and snapshot coverage is supplied by its owner.
"""

import numpy as np
import pandas as pd


def utc_timestamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("History timestamps must be valid and timezone-aware")
    return value.tz_convert("UTC")


def recent_departure_features(
    events, queries, *, coverage_start, observed_through, windows=(30, 60)
):
    """Return context for queries with flight_id, Origin and prediction_at.

    Events require flight_id, Origin, departed_at, available_at and DepDelay.
    Each flight may have one immutable event (exact duplicates are removed).
    Corrections require explicit version handling upstream; conflicts are rejected.
    Means preserve signed BTS departure delays, and only finite delays contribute.
    Snapshot coverage is an explicit completeness claim, not inferred from flights.
    """
    if (
        not windows
        or len(set(windows)) != len(windows)
        or any(isinstance(w, bool) or not isinstance(w, int) or w <= 0 for w in windows)
    ):
        raise ValueError("Windows must be unique positive integer minutes")
    start, through = utc_timestamp(coverage_start), utc_timestamp(observed_through)
    if start > through:
        raise ValueError("Snapshot coverage_start must not exceed observed_through")
    required_events = {"flight_id", "Origin", "departed_at", "available_at", "DepDelay"}
    required_queries = {"flight_id", "Origin", "prediction_at"}
    if not required_events.issubset(events) or not required_queries.issubset(queries):
        raise ValueError("Missing history event or query columns")
    history = events[list(sorted(required_events))].drop_duplicates().copy()
    requests = queries[list(sorted(required_queries))].copy()
    for frame in [history, requests]:
        for column in ["flight_id", "Origin"]:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"History {column} must be nonempty")
            frame[column] = frame[column].astype(str).str.strip()
    history = history.drop_duplicates()
    if history.flight_id.duplicated().any():
        raise ValueError("Conflicting duplicate history events; resolve revisions upstream")
    for column in ["departed_at", "available_at"]:
        history[column] = pd.to_datetime(history[column].map(utc_timestamp), utc=True)
    requests["prediction_at"] = pd.to_datetime(requests.prediction_at.map(utc_timestamp), utc=True)
    if (history.available_at < history.departed_at).any():
        raise ValueError("A final departure delay cannot be available before departure")
    if (history.departed_at < start).any() or (history.available_at > through).any():
        raise ValueError("History events fall outside the declared snapshot coverage")
    history["DepDelay"] = pd.to_numeric(history.DepDelay, errors="coerce")
    groups = {
        origin: frame.sort_values("departed_at") for origin, frame in history.groupby("Origin")
    }
    records = []
    for request in requests.itertuples(index=False):
        record = {}
        time = request.prediction_at
        group = groups.get(request.Origin)
        for window in windows:
            prefix = f"origin_departure_{window}m"
            lower = time - pd.Timedelta(minutes=window)
            covered = lower >= start and time <= through
            record[f"{prefix}_covered"] = float(covered)
            record[f"{prefix}_mean_delay"] = np.nan
            record[f"{prefix}_delayed_fraction"] = np.nan
            # Outside the snapshot, zero would falsely claim no departures.
            known = start <= time <= through
            record[f"{prefix}_valid_count"] = 0.0 if known else np.nan
            record[f"{prefix}_event_count"] = 0.0 if known else np.nan
            if not known or group is None:
                continue
            left = group.departed_at.searchsorted(lower, side="left")
            right = group.departed_at.searchsorted(time, side="left")
            candidates = group.iloc[left:right]
            candidates = candidates.loc[
                candidates.available_at.le(time) & candidates.flight_id.ne(request.flight_id)
            ]
            values = candidates.DepDelay.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            record[f"{prefix}_event_count"] = float(len(candidates))
            record[f"{prefix}_valid_count"] = float(len(values))
            if len(values):
                record[f"{prefix}_mean_delay"] = float(values.mean())
                record[f"{prefix}_delayed_fraction"] = float((values >= 15).mean())
        records.append(record)
    columns = [
        f"origin_departure_{w}m_{suffix}"
        for w in windows
        for suffix in ["covered", "mean_delay", "delayed_fraction", "valid_count", "event_count"]
    ]
    return pd.DataFrame(records, index=queries.index, columns=columns, dtype=float)
