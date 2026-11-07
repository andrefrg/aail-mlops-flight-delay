"""Prepare immutable monthly Parquet partitions and a deterministic monthly sample."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from flight_delay.data.audit import (
    BASE_FEATURES,
    eligible_mask,
    make_target,
    numeric,
    sha256,
    strip_values,
)

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def preparation_signature(config):
    keys = [
        "seed",
        "years",
        "features",
        "sample_per_month",
        "validation_start",
        "holdout_start",
        "end_exclusive",
    ]
    values = {key: config[key] for key in keys if key in config}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def priority(row_ids, seed):
    """SplitMix64 ranking: deterministic, independent of labels and chunk boundaries."""
    with np.errstate(over="ignore"):
        values = np.asarray(row_ids, dtype=np.uint64) ^ np.uint64(seed)
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> 27)) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> 31)


def split_name(date, config):
    if date < config["validation_start"]:
        return "train"
    if date < config["holdout_start"]:
        return "validation"
    return "test"


def prepare(data_dir, output, config_path, force=False):
    config = json.loads(Path(config_path).read_text())
    if config["features"] != BASE_FEATURES:
        raise ValueError("Configuration must match the implemented raw feature allowlist")
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if output == data_dir or data_dir in output.parents or output in data_dir.parents:
        raise ValueError("Prepared output must be separate from raw source data")
    if output.exists() and (not force or not (output / "manifest.json").is_file()):
        raise FileExistsError(
            f"{output} exists; --force only replaces an existing prepared dataset"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, monthly, sample, invalid_examples = Counter(), Counter(), {}, []
    sources = []
    required = BASE_FEATURES + [
        "Flight_Number_Reporting_Airline",
        "Cancelled",
        "Diverted",
        "ArrDelay",
    ]
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output.parent) as directory:
        stage = Path(directory) / "dataset"
        stage.mkdir()
        for year in config["years"]:
            path = data_dir / f"flights_{year}.csv"
            before = path.stat()
            source_rows = 0
            print(f"Preparing {path.name}", flush=True)
            with pd.read_csv(
                path,
                usecols=required,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                chunksize=config["chunksize"],
            ) as reader:
                for block, raw in enumerate(reader):
                    raw = raw.apply(strip_values)
                    counts["raw"] += len(raw)
                    source_rows += len(raw)
                    mask = eligible_mask(raw)
                    counts["eligible"] += int(mask.sum())
                    frame = raw.loc[
                        mask, BASE_FEATURES + ["Flight_Number_Reporting_Airline"]
                    ].copy()
                    dates = pd.to_datetime(frame["FlightDate"], format="%Y-%m-%d", errors="coerce")
                    valid_date = dates.notna() & dates.dt.year.eq(year)
                    counts["quarantined_invalid_date"] += int((~valid_date).sum())
                    frame = frame.loc[valid_date].copy()
                    if frame.empty:
                        continue
                    frame["FlightDate"] = dates.loc[frame.index].dt.date.astype(str)
                    frame["row_id"] = np.uint64(year * 100_000_000) + frame.index.to_numpy(
                        dtype="uint64"
                    )
                    frame["target"] = make_target(raw.loc[frame.index, "ArrDelay"])
                    frame["ArrDelay"] = numeric(raw.loc[frame.index, "ArrDelay"]).astype("float32")
                    duration = numeric(frame["CRSElapsedTime"])
                    invalid_duration = duration.isna() | duration.le(0)
                    counts["invalid_duration_to_missing"] += int(invalid_duration.sum())
                    if invalid_duration.any():
                        examples = frame.loc[
                            invalid_duration,
                            ["row_id", "FlightDate", "Origin", "Dest", "CRSElapsedTime"],
                        ]
                        invalid_examples.extend(
                            examples.head(max(0, 20 - len(invalid_examples))).to_dict("records")
                        )
                    frame["CRSElapsedTime"] = duration.where(~invalid_duration, np.nan).astype(
                        "float32"
                    )
                    distance = numeric(frame["Distance"])
                    invalid_distance = distance.isna() | distance.le(0)
                    counts["invalid_distance_to_missing"] += int(invalid_distance.sum())
                    frame["Distance"] = distance.where(~invalid_distance, np.nan).astype("float32")
                    frame["month"] = dates.loc[frame.index].dt.to_period("M").astype(str)
                    frame["split"] = frame["FlightDate"].map(lambda date: split_name(date, config))
                    frame["sample_priority"] = priority(frame["row_id"], config["seed"])
                    for month, group in frame.groupby("month", sort=True):
                        split = group["split"].iloc[0]
                        target = stage / split / f"month={month}"
                        target.mkdir(parents=True, exist_ok=True)
                        group.drop(columns="sample_priority").to_parquet(
                            target / f"part-{year}-{block:04d}.parquet", index=False
                        )
                        monthly[(split, month)] += len(group)
                        counts["prepared"] += len(group)
                        combined = (
                            pd.concat([sample[month], group], ignore_index=True)
                            if month in sample
                            else group
                        )
                        sample[month] = combined.nsmallest(
                            config["sample_per_month"], "sample_priority"
                        )
                    if (block + 1) % 10 == 0:
                        print(f"  {source_rows:,} source rows", flush=True)
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed during preparation: {path}")
            sources.append(
                {"file": path.name, "bytes": before.st_size, "sha256": digest, "rows": source_rows}
            )
        sampled = pd.concat([sample[key] for key in sorted(sample)], ignore_index=True)
        sampled = sampled.sort_values("row_id").drop(columns="sample_priority")
        sampled.to_parquet(stage / "monthly_sample.parquet", index=False)
        manifest = {
            "schema_version": 1,
            "preparation_config_sha256": preparation_signature(config),
            "config_sha256": sha256(config_path),
            "sources": sources,
            "counts": dict(counts),
            "sample_rows": len(sampled),
            "seed": config["seed"],
            "sample_per_month": config["sample_per_month"],
            "monthly_counts": [
                {"split": split, "month": month, "rows": n}
                for (split, month), n in sorted(monthly.items())
            ],
            "invalid_duration_examples": invalid_examples,
            "rules": {
                "eligibility": "Cancelled == 0; Diverted == 0; finite ArrDelay",
                "duration_distance": "nonpositive/nonfinite -> missing; train-only median imputation",
                "invalid_date": "quarantine; count separately",
                "sample": "smallest SplitMix64 priorities per month; no target-based selection",
            },
        }
        write_json(stage / "manifest.json", manifest)
        if output.exists():
            backup = Path(directory) / "previous"
            output.rename(backup)
            try:
                stage.rename(output)
            except Exception:
                backup.rename(output)
                raise
        else:
            stage.rename(output)
    print(f"Prepared {counts['prepared']:,} rows; sample {len(sampled):,}; {output}", flush=True)
    return manifest
