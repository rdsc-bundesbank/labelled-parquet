"""Tests for labelled Parquet writing."""

import json

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from labelled_parquet import (
    STATA_METADATA_KEY,
    update_labelled_parquet_metadata,
    write_labelled_parquet,
)


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame({"id": [1, 2], "status": [0, 1]}),
        pl.DataFrame({"id": [1, 2], "status": [0, 1]}),
        {"id": [1, 2], "status": [0, 1]},
    ],
)
def test_writes_stata_compatible_metadata(tmp_path, dataframe):
    path = tmp_path / "labelled.parquet"
    write_labelled_parquet(
        path,
        dataframe,
        labels={
            "id": {"column_label": "Person identifier"},
            "status": {
                "column_label": "Current status",
                "value_labels": {0: "Inactive", 1: "Active"},
            },
        },
    )

    metadata = pq.ParquetFile(path).metadata.metadata
    assert metadata is not None
    assert json.loads(metadata[STATA_METADATA_KEY]) == {
        "version": 1,
        "variables": {
            "id": {"label": "Person identifier"},
            "status": {"label": "Current status", "value_label": "status"},
        },
        "value_labels": {"status": {"0": "Inactive", "1": "Active"}},
    }


def test_rejects_labels_for_unknown_columns(tmp_path):
    with pytest.raises(ValueError, match="unknown columns: missing"):
        write_labelled_parquet(
            tmp_path / "labelled.parquet",
            pd.DataFrame({"status": [0]}),
            labels={"missing": {"column_label": "Missing"}},
        )


def test_allows_columns_without_label_specifications(tmp_path):
    path = tmp_path / "labelled.parquet"
    write_labelled_parquet(
        path,
        {"id": [1], "status": [0]},
        labels={"status": {"column_label": "Current status"}},
    )

    metadata = pq.ParquetFile(path).metadata.metadata
    assert metadata is not None
    assert json.loads(metadata[STATA_METADATA_KEY])["variables"] == {
        "status": {"label": "Current status"}
    }


def test_rejects_incomplete_value_labels_by_default(tmp_path):
    message = (
        "value_labels for 'status' are missing codes: 2 (50.0% of non-null values are unlabelled)"
    )
    with pytest.raises(ValueError) as exception:
        write_labelled_parquet(
            tmp_path / "labelled.parquet",
            {"status": [1, 2]},
            labels={"status": {"value_labels": {1: "Active"}}},
        )
    assert str(exception.value) == message


def test_reports_all_values_missing_from_value_labels(tmp_path):
    message = (
        "value_labels for 'status' are missing codes: 1, 2 "
        "(100.0% of non-null values are unlabelled)"
    )
    with pytest.raises(ValueError) as exception:
        write_labelled_parquet(
            tmp_path / "labelled.parquet",
            {"status": [1, 2, None]},
            labels={"status": {"value_labels": {3: "Unknown"}}},
        )
    assert str(exception.value) == message


def test_allows_incomplete_value_labels_when_explicitly_enabled(tmp_path):
    path = tmp_path / "labelled.parquet"
    write_labelled_parquet(
        path,
        {"status": [1, 2]},
        labels={"status": {"value_labels": {1: "Active"}}},
        allow_incomplete_value_labels=True,
    )

    metadata = pq.ParquetFile(path).metadata.metadata
    assert metadata is not None
    assert json.loads(metadata[STATA_METADATA_KEY])["variables"] == {
        "status": {"value_label": "status"}
    }


def test_reuses_identical_value_labels(tmp_path):
    path = tmp_path / "labelled.parquet"
    write_labelled_parquet(
        path,
        pd.DataFrame({"employment_status": [1], "partner_status": [2]}),
        labels={
            "employment_status": {
                "value_labels": {1: "Employed", 2: "Not employed"},
            },
            "partner_status": {
                "value_labels": {2: "Not employed", 1: "Employed"},
            },
        },
    )

    metadata = pq.ParquetFile(path).metadata.metadata
    assert metadata is not None
    envelope = json.loads(metadata[STATA_METADATA_KEY])
    assert envelope["variables"] == {
        "employment_status": {"value_label": "employment_status"},
        "partner_status": {"value_label": "employment_status"},
    }
    assert envelope["value_labels"] == {
        "employment_status": {"1": "Employed", "2": "Not employed"}
    }


def test_rejects_non_integer_value_label_codes(tmp_path):
    with pytest.raises(ValueError, match="must be an integer"):
        write_labelled_parquet(
            tmp_path / "labelled.parquet",
            pd.DataFrame({"status": [0]}),
            labels={"status": {"value_labels": {"0": "Inactive"}}},
        )


def test_rejects_invalid_stata_type(tmp_path):
    with pytest.raises(ValueError, match="stata_type for 'status' is not supported"):
        write_labelled_parquet(
            tmp_path / "labelled.parquet",
            {"status": [1]},
            labels={"status": {"stata_type": "string"}},
        )


def test_writes_advanced_stata_metadata_verbatim(tmp_path):
    path = tmp_path / "multilingual.parquet"
    stata_metadata = {
        "version": 2,
        "label_languages": {
            "en": {"dataset_label": "Household survey"},
            "de": {"dataset_label": "Haushaltsbefragung"},
        },
    }

    write_labelled_parquet(path, {"status": [0, 1]}, stata_metadata=stata_metadata)

    metadata = pq.ParquetFile(path).metadata.metadata
    assert metadata is not None
    assert json.loads(metadata[STATA_METADATA_KEY]) == stata_metadata


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("labels", {"status": {"column_label": "Status"}}),
        ("dataset_label", "Household survey"),
        ("dataset_notes", ["Preliminary release."]),
        ("allow_incomplete_value_labels", True),
    ],
)
def test_rejects_advanced_metadata_combined_with_convenience_metadata(tmp_path, keyword, value):
    with pytest.raises(ValueError, match="stata_metadata cannot be combined"):
        write_labelled_parquet(
            tmp_path / "labelled.parquet",
            {"status": [0, 1]},
            stata_metadata={"version": 2},
            **{keyword: value},
        )


def test_rejects_non_mapping_advanced_stata_metadata(tmp_path):
    with pytest.raises(TypeError, match="stata_metadata must be a mapping"):
        write_labelled_parquet(
            tmp_path / "labelled.parquet",
            {"status": [0, 1]},
            stata_metadata=["not", "a", "mapping"],  # type: ignore[bad-argument-type]
        )


def test_updates_metadata_without_rewriting_data(tmp_path):
    path = tmp_path / "ordinary.parquet"
    table = pa.table({"status": [0, 1], "id": [1, 2]})
    pq.write_table(table, path, metadata_collector=[])
    original = path.read_bytes()
    original_footer_start = len(original) - pq.ParquetFile(path).metadata.serialized_size - 8

    update_labelled_parquet_metadata(
        path,
        labels={"status": {"value_labels": {0: "Inactive", 1: "Active"}}},
    )

    result = pq.read_table(path)
    assert result == table
    metadata = pq.ParquetFile(path).metadata.metadata
    assert metadata is not None
    assert json.loads(metadata[STATA_METADATA_KEY])["value_labels"] == {
        "status": {"0": "Inactive", "1": "Active"}
    }
    assert path.read_bytes()[:original_footer_start] == original[:original_footer_start]


def test_update_optionally_validates_value_label_coverage(tmp_path):
    path = tmp_path / "ordinary.parquet"
    pq.write_table(pa.table({"status": [0, 1, 2]}), path)

    update_labelled_parquet_metadata(
        path, labels={"status": {"value_labels": {0: "Inactive", 1: "Active"}}}
    )

    with pytest.raises(ValueError, match="missing codes: 2"):
        update_labelled_parquet_metadata(
            path,
            labels={"status": {"value_labels": {0: "Inactive", 1: "Active"}}},
            validate_value_label_coverage=True,
        )
