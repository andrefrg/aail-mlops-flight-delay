"""Read-only, chunked audit of BTS flight CSVs. No model is fitted here."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_FEATURES = [
    "FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime",
    "CRSArrTime", "CRSElapsedTime", "Distance",
]
KEY_FIELDS = [
    "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "CRSDepTime",
]
REQUIRED = set(BASE_FEATURES + KEY_FIELDS + ["Cancelled", "Diverted", "ArrDelay"])


def numeric(series: pd.Series) -> pd.Series:
    """Non-numeric and non-finite values become missing; never replace with zero."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def strip_values(series: pd.Series) -> pd.Series:
    # Most BTS columns repeat a small vocabulary. Strip each distinct string once.
    return series.map({value: value.strip() for value in series.unique()})


def eligible_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        numeric(frame["Cancelled"]).eq(0)
        & numeric(frame["Diverted"]).eq(0)
        & numeric(frame["ArrDelay"]).notna()
    )


def make_target(delay: pd.Series) -> pd.Series:
    values = numeric(delay)
    target = pd.Series(pd.NA, index=delay.index, dtype="string")
    target.loc[values.lt(15)] = "on_time"
    target.loc[values.ge(15) & values.lt(60)] = "moderate_delay"
    target.loc[values.ge(60)] = "severe_delay"
    return target


def valid_hhmm(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return (
        values.notna() & values.mod(1).eq(0)
        & ((values.between(0, 2359) & values.mod(100).lt(60)) | values.eq(2400))
    )


def candidate_keys(frame: pd.DataFrame) -> pd.Series:
    """Exact, separator-safe candidate operation keys; not proof of duplicate flights."""
    values = frame[KEY_FIELDS].copy()
    dates = pd.to_datetime(values["FlightDate"], format="%Y-%m-%d", errors="coerce")
    number = numeric(values["Flight_Number_Reporting_Airline"])
    complete = (
        values.ne("").all(axis=1) & dates.notna()
        & valid_hhmm(values["CRSDepTime"])
        & number.notna() & number.ge(0) & number.mod(1).eq(0)
    )
    values = values.loc[complete].copy()
    values["Flight_Number_Reporting_Airline"] = number.loc[complete].astype("int64").astype(str)
    values["CRSDepTime"] = numeric(values["CRSDepTime"]).astype("int64").astype(str)
    separator = "\x1f"
    if any(values[column].str.contains(separator, regex=False).any() for column in KEY_FIELDS):
        raise ValueError("Unexpected separator in operation key; exact key encoding is unsafe.")
    result = values[KEY_FIELDS[0]]
    for column in KEY_FIELDS[1:]:
        result = result + separator + values[column]
    return result


class AuditState:
    def __init__(self) -> None:
        self.rows = 0
        self.eligible = 0
        self.missing = Counter()
        self.eligible_missing = Counter()
        self.present = Counter()
        self.eligible_present = Counter()
        self.values: dict[str, Counter] = defaultdict(Counter)
        self.months = Counter()
        self.classes = Counter()
        self.month_classes = Counter()
        self.subgroups = Counter()
        self.stages = Counter()
        self.checks: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.numeric_stats: dict[str, list[float]] = {}

    def check(self, name: str, scope: pd.Series, violation: pd.Series) -> None:
        self.checks[name][0] += int(scope.sum())
        self.checks[name][1] += int((scope & violation).sum())

    def consume(self, frame: pd.DataFrame, expected_year: int) -> None:
        # Explicit strings preserve airport/airline codes and avoid mixed-type inference.
        # Whitespace-only cells are missing; literal strings such as "NA" are not.
        frame = frame.apply(strip_values)
        n = len(frame)
        self.rows += n
        eligible = eligible_mask(frame)
        self.eligible += int(eligible.sum())
        for column in frame:
            series = frame[column]
            missing = series.eq("")
            self.present[column] += n
            self.eligible_present[column] += int(eligible.sum())
            self.missing[column] += int(missing.sum())
            self.eligible_missing[column] += int((missing & eligible).sum())
            self.values[column].update(series.loc[~missing].value_counts().to_dict())

        cancelled = numeric(frame["Cancelled"])
        diverted = numeric(frame["Diverted"])
        delay = numeric(frame["ArrDelay"])
        not_cancelled = cancelled.eq(0)
        not_diverted = not_cancelled & diverted.eq(0)
        self.stages["01_raw"] += n
        self.stages["02_cancelled_eq_0"] += int(not_cancelled.sum())
        self.stages["03_also_diverted_eq_0"] += int(not_diverted.sum())
        self.stages["04_also_finite_arrdelay"] += int(eligible.sum())

        dates = pd.to_datetime(frame["FlightDate"], format="%Y-%m-%d", errors="coerce")
        month = dates.dt.to_period("M").astype(str).where(dates.notna(), "invalid_date")
        self.months.update(month.value_counts().to_dict())
        labels = make_target(frame["ArrDelay"])
        self.classes.update(labels.loc[eligible].value_counts().to_dict())
        for field in ["ArrDelay", "DepDelay", "Distance", "CRSElapsedTime", "ActualElapsedTime", "AirTime", "TaxiIn", "TaxiOut"]:
            if field not in frame:
                continue
            values = numeric(frame[field]).dropna()
            count = len(values)
            if not count:
                continue
            mean = float(values.mean())
            m2 = float(values.var(ddof=0) * count)
            low, high = float(values.min()), float(values.max())
            if field in self.numeric_stats:
                previous_n, previous_mean, previous_m2, previous_low, previous_high = self.numeric_stats[field]
                delta = mean - previous_mean
                total = previous_n + count
                self.numeric_stats[field] = [total, previous_mean + delta * count / total,
                    previous_m2 + m2 + delta * delta * previous_n * count / total,
                    min(previous_low, low), max(previous_high, high)]
            else:
                self.numeric_stats[field] = [count, mean, m2, low, high]
        groups = pd.DataFrame({"month": month.loc[eligible], "class": labels.loc[eligible]})
        self.month_classes.update(groups.value_counts().to_dict())
        for field in ["Reporting_Airline", "Origin"]:
            groups = pd.DataFrame({"group": frame.loc[eligible, field], "class": labels.loc[eligible]})
            for (group, label), count in groups.value_counts().items():
                self.subgroups[(field, group or "<missing>", label)] += int(count)

        all_rows = pd.Series(True, index=frame.index)
        self.check("FlightDate_missing_or_invalid", all_rows, dates.isna())
        self.check("FlightDate_outside_file_year", dates.notna(), dates.dt.year.ne(expected_year))
        self.check("eligible_invalid_FlightDate", eligible, dates.isna())
        for field, values in [("Cancelled", cancelled), ("Diverted", diverted)]:
            self.check(field + "_missing_or_not_binary", all_rows, ~values.isin([0, 1]))
        self.check("ArrDelay_nonempty_invalid_numeric", frame["ArrDelay"].ne(""), delay.isna())
        for field in ["CRSDepTime", "CRSArrTime"]:
            self.check(field + "_missing_or_invalid_HHMM_eligible", eligible, ~valid_hhmm(frame[field]))
        for field in ["CRSElapsedTime", "Distance"]:
            values = numeric(frame[field])
            self.check(field + "_missing_or_nonpositive_eligible", eligible, values.isna() | values.le(0))

        calendar = {
            "Year": dates.dt.year, "Month": dates.dt.month,
            "Quarter": dates.dt.quarter, "DayofMonth": dates.dt.day,
            "DayOfWeek": dates.dt.dayofweek + 1,
        }
        for field, expected in calendar.items():
            if field in frame:
                observed = numeric(frame[field])
                self.check(field + "_inconsistent_with_FlightDate", dates.notna(), observed.ne(expected))
        for prefix in ["Arr", "Dep"]:
            delay_field, minutes_field, indicator_field = prefix + "Delay", prefix + "DelayMinutes", prefix + "Del15"
            if all(field in frame for field in [delay_field, minutes_field, indicator_field]):
                values = numeric(frame[delay_field])
                scope = values.notna()
                self.check(minutes_field + "_inconsistent", scope, numeric(frame[minutes_field]).ne(values.clip(lower=0)))
                self.check(indicator_field + "_inconsistent", scope, numeric(frame[indicator_field]).ne(values.ge(15).astype(int)))

    def tables(self) -> dict[str, pd.DataFrame]:
        profiles = []
        top_values = []
        for column, count in self.present.items():
            counts = self.values[column]
            profiles.append({
                "column": column, "rows_raw": count, "missing_raw": self.missing[column],
                "missing_pct_raw": 100 * self.missing[column] / count if count else None,
                "rows_eligible": self.eligible_present[column],
                "missing_eligible": self.eligible_missing[column],
                "missing_pct_eligible": 100 * self.eligible_missing[column] / self.eligible_present[column] if self.eligible_present[column] else None,
                "distinct_nonmissing_raw": len(counts),
                "constant_nonmissing_raw": len(counts) == 1,
            })
            top_values.extend({"column": column, "value": value, "count": int(n)} for value, n in counts.most_common(5))
        monthly = pd.DataFrame([{"month": month, "rows_raw": n} for month, n in sorted(self.months.items())])
        classes = pd.DataFrame([
            {"class": label, "count": self.classes[label], "pct_eligible": 100 * self.classes[label] / self.eligible if self.eligible else None}
            for label in ["on_time", "moderate_delay", "severe_delay"]
        ])
        return {
            "numeric_summary": pd.DataFrame([
                {"column": field, "finite_count_raw": int(n), "mean_raw": mean,
                 "std_raw": (m2 / (n - 1)) ** 0.5 if n > 1 else None, "min_raw": low, "max_raw": high}
                for field, (n, mean, m2, low, high) in self.numeric_stats.items()
            ]),
            "columns": pd.DataFrame(profiles),
            "top_values": pd.DataFrame(top_values),
            "monthly_coverage": monthly,
            "eligibility": pd.DataFrame([{"stage": stage, "rows": n} for stage, n in sorted(self.stages.items())]),
            "classes": classes,
            "classes_by_month": pd.DataFrame([{"month": month, "class": label, "count": n} for (month, label), n in sorted(self.month_classes.items())]),
            "classes_by_subgroup": pd.DataFrame([{"field": field, "group": group, "class": label, "count": n} for (field, group, label), n in sorted(self.subgroups.items())]),
            "checks": pd.DataFrame([{"check": name, "rows_checked": checked, "violations": violations, "violation_pct": 100 * violations / checked if checked else None} for name, (checked, violations) in sorted(self.checks.items())]),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_audit(data_dir: Path, output_dir: Path, policy_path: Path, chunksize: int = 100_000,
              max_chunks: int | None = None, duplicates: bool = True) -> dict:
    if chunksize <= 0 or (max_chunks is not None and max_chunks <= 0):
        raise ValueError("chunksize and max_chunks must be positive")
    inputs = [data_dir / "flights_2022.csv", data_dir / "flights_2023.csv"]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Guard original data, even when an output path was supplied by mistake.
    output_dir = output_dir.resolve()
    if output_dir == data_dir.resolve() or data_dir.resolve() in output_dir.parents:
        raise ValueError("Write reports outside the source dataset directory.")
    policy = pd.read_csv(policy_path, dtype=str, keep_default_na=False)
    expected_columns = set(policy["feature"])
    started = time.monotonic()
    state = AuditState()
    identities = []
    schemas = []
    duplicate_summary = {"checked": duplicates, "scope": "raw rows with a valid complete candidate key", "key_fields": KEY_FIELDS,
                         "note": "Repeated candidate keys are not proof of identical records; no records were removed. Exact full-row duplication is not assessed."}
    with tempfile.TemporaryDirectory(prefix="laia-audit-") as work:
        connection = sqlite3.connect(str(Path(work) / "keys.sqlite")) if duplicates else None
        if connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute("CREATE TABLE keys (key TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID")
        rows_with_key = 0
        for path in inputs:
            before = path.stat()
            year = int(path.stem.split("_")[-1])
            print(f"Reading {path.name} ({before.st_size / 1e9:.2f} GB)", flush=True)
            file_rows = 0
            reader = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, chunksize=chunksize, encoding="utf-8-sig")
            try:
                for index, frame in enumerate(reader):
                    if index == 0:
                        missing_required = REQUIRED - set(frame.columns)
                        if missing_required:
                            raise ValueError(f"{path.name}: missing required fields: {sorted(missing_required)}")
                        schemas.append({"file": path.name, "columns": list(frame.columns),
                                        "missing_policy_columns": sorted(expected_columns - set(frame.columns)),
                                        "unexpected_columns": sorted(set(frame.columns) - expected_columns)})
                    state.consume(frame, year)
                    file_rows += len(frame)
                    if connection:
                        keys = candidate_keys(frame[KEY_FIELDS].apply(strip_values))
                        counts = keys.value_counts()
                        rows_with_key += len(keys)
                        connection.executemany("INSERT INTO keys(key, n) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET n=n+excluded.n",
                                               ((str(key), int(count)) for key, count in counts.items()))
                        connection.commit()
                    if (index + 1) % 5 == 0:
                        print(f"  {file_rows:,} rows read; total {state.rows:,}", flush=True)
                    if max_chunks is not None and index + 1 >= max_chunks:
                        break
            finally:
                reader.close()
            # Full-file hashes identify the source even for a partial audit.
            print(f"Hashing source {path.name}", flush=True)
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Input changed during audit: {path}")
            identities.append({"file": path.name, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns, "sha256": digest, "rows_audited": file_rows})
        if connection:
            unique, groups, extras = connection.execute("SELECT COUNT(*), COALESCE(SUM(n > 1),0), COALESCE(SUM(n-1),0) FROM keys").fetchone()
            duplicate_summary.update({"rows_with_key": rows_with_key, "rows_without_valid_key": state.rows - rows_with_key,
                                      "unique_keys": unique, "repeated_key_groups": groups, "extra_rows_with_repeated_key": extras})
            connection.close()

    summary = {
        "audit_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "complete" if max_chunks is None else "partial_first_chunks_not_representative",
        "max_chunks_per_file": max_chunks, "chunksize": chunksize,
        "rows_raw": state.rows, "rows_eligible": state.eligible,
        "rows_excluded": state.rows - state.eligible,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "audit_code_sha256": sha256(Path(__file__)), "feature_policy_sha256": sha256(policy_path),
        "sources": identities, "schemas": schemas, "duplicates": duplicate_summary,
        "eligibility_rule": "Cancelled == 0 and Diverted == 0 and finite numeric ArrDelay",
        "missing_rule": "Empty or whitespace-only cells. Literal NA/NULL strings are retained; numeric/date checks detect malformed values.",
        "scope": "Descriptive quality audit only; no fitting, feature selection or model evaluation. Raw/eligible scopes are explicit. Holdout class summaries must not guide model decisions.",
        "limitations": ["No proof of when historical fields became available before departure.",
                        "No model importance or causal conclusions.", "No exact full-row duplicate check.",
                        "Candidate operation keys are not guaranteed unique flight identifiers.",
                        "Numeric ranges are checks, not automatic deletion rules.",
                        "Exact cardinality counters grow with the number of distinct values; only CSV reading is bounded by chunksize."],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = state.tables()
    feature_table = policy.merge(tables["columns"], left_on="feature", right_on="column", how="left", validate="one_to_one")
    tables["feature_review"] = feature_table
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Done: {state.rows:,} raw; {state.eligible:,} eligible. Reports: {output_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, help="Default: reports/generated/data_audit (or data_audit_smoke for a partial run).")
    parser.add_argument("--policy", type=Path, default=PROJECT_ROOT / "configs/feature_policy.csv")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-chunks", type=int, help="Smoke test only: read first N chunks per year, never a representative sample.")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip exact candidate-key counting on temporary SQLite storage.")
    args = parser.parse_args()
    output = args.output_dir or PROJECT_ROOT / "reports/generated" / ("data_audit_smoke" if args.max_chunks else "data_audit")
    run_audit(args.data_dir, output, args.policy, args.chunksize, args.max_chunks, not args.skip_duplicates)


if __name__ == "__main__":
    main()
