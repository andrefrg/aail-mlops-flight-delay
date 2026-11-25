"""Store leaderboard predictions and outcomes by mode and request ID."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


def batch_key(payload):
    if not isinstance(payload, dict):
        return None
    mode = payload.get("mode")
    request_id = payload.get("request_id")
    if (
        mode not in ("sandbox", "live")
        or not isinstance(request_id, str)
        or not request_id.strip()
    ):
        return None
    return mode, request_id


def store_batch(database, key, *, instances=None, predictions=None, labels=None, outcomes=None):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    values = [
        json.dumps(value) if value is not None else None
        for value in (instances, predictions, labels, outcomes)
    ]

    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard_batches (
                mode TEXT NOT NULL,
                request_id TEXT NOT NULL,
                instances TEXT,
                predictions TEXT,
                labels TEXT,
                outcomes TEXT,
                PRIMARY KEY (mode, request_id)
            )
        """)
        connection.execute("""
            INSERT INTO leaderboard_batches (mode, request_id, instances, predictions, labels, outcomes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (mode, request_id) DO UPDATE SET
                instances = COALESCE(excluded.instances, leaderboard_batches.instances),
                predictions = COALESCE(excluded.predictions, leaderboard_batches.predictions),
                labels = COALESCE(excluded.labels, leaderboard_batches.labels),
                outcomes = COALESCE(excluded.outcomes, leaderboard_batches.outcomes)
        """, (*key, *values))
