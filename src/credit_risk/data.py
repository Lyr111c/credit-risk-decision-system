"""Download, load, and validate the UCI credit-card default dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.api.types import is_integer_dtype

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
ARCHIVE_NAME = "default_of_credit_card_clients.zip"
RAW_XLS_NAME = "default of credit card clients.xls"
ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
RAW_XLS_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"

COLUMN_MAP = {
    "ID": "customer_id",
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marital_status",
    "AGE": "age",
    "PAY_0": "repayment_status_sep",
    "PAY_2": "repayment_status_aug",
    "PAY_3": "repayment_status_jul",
    "PAY_4": "repayment_status_jun",
    "PAY_5": "repayment_status_may",
    "PAY_6": "repayment_status_apr",
    "BILL_AMT1": "bill_amount_sep",
    "BILL_AMT2": "bill_amount_aug",
    "BILL_AMT3": "bill_amount_jul",
    "BILL_AMT4": "bill_amount_jun",
    "BILL_AMT5": "bill_amount_may",
    "BILL_AMT6": "bill_amount_apr",
    "PAY_AMT1": "payment_amount_sep",
    "PAY_AMT2": "payment_amount_aug",
    "PAY_AMT3": "payment_amount_jul",
    "PAY_AMT4": "payment_amount_jun",
    "PAY_AMT5": "payment_amount_may",
    "PAY_AMT6": "payment_amount_apr",
    "default payment next month": "default_next_month",
}
EXPECTED_COLUMNS = tuple(COLUMN_MAP.values())
REPAYMENT_STATUS_COLUMNS = tuple(
    column for column in EXPECTED_COLUMNS if column.startswith("repayment_status_")
)


@dataclass(frozen=True)
class DownloadResult:
    archive_path: Path
    data_path: Path
    archive_sha256: str
    data_sha256: str


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "credit-risk-project/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def download_dataset(
    destination: str | Path = Path("data/raw"),
    *,
    expected_archive_sha256: str = ARCHIVE_SHA256,
    expected_data_sha256: str | None = RAW_XLS_SHA256,
    fetcher: Callable[[str], bytes] = _fetch_bytes,
) -> DownloadResult:
    """Download and safely extract the exact UCI artifact after checksum validation."""
    destination = Path(destination)
    archive_bytes = fetcher(DATASET_URL)
    archive_sha256 = _sha256(archive_bytes)
    if archive_sha256 != expected_archive_sha256:
        raise ValueError(
            "Archive SHA-256 mismatch: "
            f"expected {expected_archive_sha256}, got {archive_sha256}"
        )

    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != 1 or members[0].filename != RAW_XLS_NAME:
            names = [member.filename for member in members]
            raise ValueError(f"Unexpected archive members: {names}")
        data_bytes = archive.read(members[0])

    data_sha256 = _sha256(data_bytes)
    if expected_data_sha256 is not None and data_sha256 != expected_data_sha256:
        raise ValueError(
            "Data SHA-256 mismatch: "
            f"expected {expected_data_sha256}, got {data_sha256}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / ARCHIVE_NAME
    data_path = destination / RAW_XLS_NAME
    archive_path.write_bytes(archive_bytes)
    data_path.write_bytes(data_bytes)
    return DownloadResult(archive_path, data_path, archive_sha256, data_sha256)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with raw UCI headers mapped to descriptive project names."""
    return frame.rename(columns=COLUMN_MAP).copy()


def validate_dataset(frame: pd.DataFrame, *, expected_rows: int = 30_000) -> None:
    """Raise ValueError when the normalized dataset violates its core contract."""
    actual_columns = tuple(frame.columns)
    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected columns: "
            f"expected {list(EXPECTED_COLUMNS)}, got {list(actual_columns)}"
        )
    if len(frame) != expected_rows:
        raise ValueError(f"Unexpected row count: expected {expected_rows}, got {len(frame)}")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values")
    non_integer_columns = [
        column for column in frame.columns if not is_integer_dtype(frame[column])
    ]
    if non_integer_columns:
        raise ValueError(f"Expected integer columns, got: {non_integer_columns}")
    if not frame["customer_id"].is_unique:
        raise ValueError("customer_id must be unique")
    if set(frame["default_next_month"].unique()) - {0, 1}:
        raise ValueError("target must contain only 0 and 1")
    if set(frame["sex"].unique()) - {1, 2}:
        raise ValueError("sex contains unsupported codes")
    if set(frame["education"].unique()) - set(range(7)):
        raise ValueError("education contains unsupported codes")
    if set(frame["marital_status"].unique()) - set(range(4)):
        raise ValueError("marital_status contains unsupported codes")
    for column in REPAYMENT_STATUS_COLUMNS:
        if set(frame[column].unique()) - set(range(-2, 10)):
            raise ValueError(f"{column} contains unsupported repayment codes")
    if (frame["credit_limit"] <= 0).any():
        raise ValueError("credit_limit must be positive")
    if (frame["age"] <= 0).any():
        raise ValueError("age must be positive")
    payment_columns = [
        column for column in EXPECTED_COLUMNS if column.startswith("payment_amount_")
    ]
    if (frame[payment_columns] < 0).any().any():
        raise ValueError("payment amounts must be non-negative")


def load_dataset(
    path: str | Path = Path("data/raw") / RAW_XLS_NAME,
    *,
    verify_sha256: bool = True,
) -> pd.DataFrame:
    """Load the two-header-row UCI workbook and enforce the normalized schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m credit_risk.data download`."
        )
    if verify_sha256:
        actual_sha256 = _sha256(path.read_bytes())
        if actual_sha256 != RAW_XLS_SHA256:
            raise ValueError(
                "Data SHA-256 mismatch: "
                f"expected {RAW_XLS_SHA256}, got {actual_sha256}"
            )
    frame = normalize_columns(pd.read_excel(path, sheet_name="Data", header=1, engine="xlrd"))
    validate_dataset(frame)
    return frame


def audit_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Return a JSON-serializable summary without altering observed source values."""
    features_without_id = frame.drop(columns="customer_id")
    repayment_undocumented = {
        column: sorted(set(frame[column].unique()) & {-2, 0})
        for column in REPAYMENT_STATUS_COLUMNS
    }
    repayment_undocumented = {
        column: values for column, values in repayment_undocumented.items() if values
    }
    target_counts = frame["default_next_month"].value_counts().sort_index()
    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_rows_without_id": int(features_without_id.duplicated().sum()),
        "unique_customer_ids": int(frame["customer_id"].nunique()),
        "target_counts": {str(key): int(value) for key, value in target_counts.items()},
        "target_rate": float(frame["default_next_month"].mean()),
        "undocumented_codes": {
            "education": sorted(set(frame["education"].unique()) & {0, 5, 6}),
            "marital_status": sorted(set(frame["marital_status"].unique()) & {0}),
            "repayment_status": repayment_undocumented,
        },
    }


def format_audit_report(frame: pd.DataFrame) -> str:
    """Render the observed audit results as a concise Markdown report."""
    audit = audit_dataset(frame)
    target_counts = audit["target_counts"]
    undocumented = audit["undocumented_codes"]
    missing_statement = (
        "No missing cells were detected after parsing the second header row."
        if audit["missing_cells"] == 0
        else f"The dataset contains {audit['missing_cells']:,} missing cells."
    )
    repayment_lines = "\n".join(
        f"  - `{column}`: {values}"
        for column, values in undocumented["repayment_status"].items()
    )
    return f"""# Data Quality Report

Generated from the verified UCI workbook by `credit_risk.data`.

## Structural checks

- Parsed shape: {audit['shape']['rows']:,} rows and {audit['shape']['columns']} columns.
- Unique customer IDs: {audit['unique_customer_ids']:,}.
- Exact duplicate rows including ID: {audit['duplicate_rows']:,}.
- Duplicate rows after dropping ID: {audit['duplicate_rows_without_id']:,}. These are not automatically removed because distinct IDs can represent different clients with identical observed values.
- {missing_statement}

## Target

- Non-default (`0`): {target_counts.get('0', 0):,}.
- Default next month (`1`): {target_counts.get('1', 0):,}.
- Observed default rate: {audit['target_rate']:.2%}.

The target is moderately imbalanced. Accuracy alone is not an adequate evaluation metric; later stages will include ROC-AUC, PR-AUC, KS, Brier score, and calibration.

## Undocumented source codes

- `education`: {undocumented['education']}.
- `marital_status`: {undocumented['marital_status']}.
- Repayment status fields:
{repayment_lines or '  - None observed.'}

These values pass the project schema because they occur in the official artifact, but the UCI description does not define them. They remain distinct until a documented and tested recoding decision is made.

## Range observations

- Credit limits and ages are positive in all records.
- Previous payment amounts are non-negative in all records.
- Bill statement amounts include negative values. They are retained because credits or accounting adjustments are plausible and the source does not establish that they are errors.

## Leakage and availability review

- `customer_id` is an identifier and must not be used as a predictor.
- The target is the following month's default event; it is not an input feature.
- Billing, payment, and repayment-status fields precede the target and are not direct target leakage for this account-level prediction task.
- Those behavioral fields are generally unavailable for new-to-bank applicants. Results must be described as existing-customer behavioral risk, not as a faithful reconstruction of a new application decision.

## Validation limitation

The dataset does not expose repeated target snapshots or a reliable development/validation time axis. A stratified random split may be used later, but it must not be labeled out-of-time validation.
"""


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="download and verify the official UCI data")
    audit_parser = subparsers.add_parser("audit", help="print a JSON quality summary")
    audit_parser.add_argument(
        "--path", type=Path, default=Path("data/raw") / RAW_XLS_NAME
    )
    audit_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "download":
        result = download_dataset()
        print(f"Archive: {result.archive_path} ({result.archive_sha256})")
        print(f"Data: {result.data_path} ({result.data_sha256})")
    else:
        frame = load_dataset(args.path)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(format_audit_report(frame), encoding="utf-8")
            print(f"Report: {args.output}")
        else:
            print(json.dumps(audit_dataset(frame), indent=2))


if __name__ == "__main__":
    _main()
