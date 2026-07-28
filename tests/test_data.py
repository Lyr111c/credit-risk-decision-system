from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data import (
    EXPECTED_COLUMNS,
    RAW_XLS_NAME,
    audit_dataset,
    download_dataset,
    format_audit_report,
    load_dataset,
    normalize_columns,
    validate_dataset,
)


def make_valid_frame() -> pd.DataFrame:
    rows = []
    for customer_id, target in ((1, 0), (2, 1)):
        row = {column: 0 for column in EXPECTED_COLUMNS}
        row.update(
            {
                "customer_id": customer_id,
                "credit_limit": 20_000 * customer_id,
                "sex": customer_id,
                "education": customer_id,
                "marital_status": customer_id,
                "age": 23 + customer_id,
                "default_next_month": target,
            }
        )
        for column in EXPECTED_COLUMNS:
            if column.startswith("repayment_status_"):
                row[column] = -1
        rows.append(row)
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_normalize_columns_maps_raw_headers() -> None:
    raw = pd.DataFrame(
        [[1, 20_000, 0]],
        columns=["ID", "LIMIT_BAL", "default payment next month"],
    )

    normalized = normalize_columns(raw)

    assert normalized.columns.tolist() == [
        "customer_id",
        "credit_limit",
        "default_next_month",
    ]


def test_validate_dataset_accepts_expected_schema() -> None:
    frame = make_valid_frame()

    validate_dataset(frame, expected_rows=2)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns="age"), "columns"),
        (lambda frame: frame.assign(customer_id=[1, 1]), "customer_id"),
        (lambda frame: frame.assign(default_next_month=[0, 2]), "target"),
        (lambda frame: frame.assign(age=[24, None]), "missing"),
        (lambda frame: frame.assign(sex=[1, 3]), "sex"),
        (lambda frame: frame.assign(bill_amount_sep=["100", "200"]), "integer"),
    ],
)
def test_validate_dataset_rejects_broken_constraints(mutator, message: str) -> None:
    frame = mutator(make_valid_frame())

    with pytest.raises(ValueError, match=message):
        validate_dataset(frame, expected_rows=2)


def test_audit_dataset_reports_observed_quality() -> None:
    frame = make_valid_frame()
    duplicate = frame.iloc[[0]].assign(customer_id=3)
    frame = pd.concat([frame, duplicate], ignore_index=True)

    audit = audit_dataset(frame)

    assert audit["shape"] == {"rows": 3, "columns": 25}
    assert audit["missing_cells"] == 0
    assert audit["duplicate_rows"] == 0
    assert audit["duplicate_rows_without_id"] == 1
    assert audit["target_counts"] == {"0": 2, "1": 1}
    assert audit["undocumented_codes"] == {
        "education": [],
        "marital_status": [],
        "repayment_status": {},
    }


def test_format_audit_report_contains_key_findings() -> None:
    frame = make_valid_frame()

    report = format_audit_report(frame)

    chinese_title = "# 数据质量报告（中文）"
    english_title = "# Data Quality Report (English)"
    assert report.startswith(chinese_title)
    assert report.index(chinese_title) < report.index(english_title)

    chinese_report, english_report = report.split(english_title, maxsplit=1)
    assert "2 行、25 列" in chinese_report
    assert "50.00%" in chinese_report
    assert "未检测到缺失单元格" in chinese_report
    assert "不会自动删除" in chinese_report

    assert "2 rows and 25 columns" in english_report
    assert "50.00%" in english_report
    assert "No missing cells were detected" in english_report
    assert "not automatically removed" in english_report


def test_download_dataset_verifies_archive_and_extracts_expected_file(
    tmp_path: Path,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(RAW_XLS_NAME, b"test workbook")
    archive_bytes = archive_buffer.getvalue()
    expected_hash = hashlib.sha256(archive_bytes).hexdigest()

    def fetcher(_: str) -> bytes:
        return archive_bytes

    result = download_dataset(
        tmp_path,
        expected_archive_sha256=expected_hash,
        expected_data_sha256=None,
        fetcher=fetcher,
    )

    assert result.archive_path.exists()
    assert result.data_path.read_bytes() == b"test workbook"
    assert result.archive_sha256 == expected_hash


def test_download_dataset_rejects_checksum_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        download_dataset(
            tmp_path,
            expected_archive_sha256="0" * 64,
            fetcher=lambda _: b"not the expected archive",
        )


def test_download_dataset_rejects_unexpected_archive_members(tmp_path: Path) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("../unexpected.txt", b"unsafe")
    archive_bytes = archive_buffer.getvalue()

    with pytest.raises(ValueError, match="archive members"):
        download_dataset(
            tmp_path,
            expected_archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
            fetcher=lambda _: archive_bytes,
        )


def test_load_dataset_reads_verified_local_workbook() -> None:
    path = Path("data/raw") / RAW_XLS_NAME
    if not path.exists():
        pytest.skip("official UCI workbook is not downloaded")

    frame = load_dataset(path)

    assert frame.shape == (30_000, 25)
    assert frame["customer_id"].is_unique
    assert frame["default_next_month"].value_counts().sort_index().to_dict() == {
        0: 23_364,
        1: 6_636,
    }


def test_load_dataset_rejects_tampered_workbook(tmp_path: Path) -> None:
    path = tmp_path / RAW_XLS_NAME
    path.write_bytes(b"not the official workbook")

    with pytest.raises(ValueError, match="Data SHA-256"):
        load_dataset(path)
