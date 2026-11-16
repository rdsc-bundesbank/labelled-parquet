"""Tests for labelled Parquet reading."""

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from labelled_parquet import (
    LabelledParquetMetadataNotFoundError,
    read_labelled_parquet,
    update_labelled_parquet_metadata,
    validate_parquet_metadata,
    write_labelled_parquet,
)


@pytest.fixture
def labelled_path(tmp_path):
    path = tmp_path / "labelled.parquet"
    write_labelled_parquet(
        path,
        {"status": [0, 1], "partner_status": [1, 0], "score": [2.5, 3.0]},
        labels={
            "status": {
                "column_label": "Current status",
                "value_labels": {0: "No", 1: "Yes"},
                "variable_notes": ["Collected during the first interview."],
                "stata_format": "%9.0g",
                "stata_type": "byte",
            },
            "partner_status": {"value_labels": {0: "No", 1: "Yes"}},
            "score": {"column_label": "Assessment score"},
        },
        dataset_label="Household survey",
        dataset_notes=["Preliminary release."],
    )
    return path


def test_reads_arrow_table_and_resolves_shared_value_labels(labelled_path):
    result = read_labelled_parquet(labelled_path)

    assert isinstance(result.data, pa.Table)
    assert result.data.to_pydict() == {
        "status": [0, 1],
        "partner_status": [1, 0],
        "score": [2.5, 3.0],
    }
    assert result.labels == {
        "status": {
            "column_label": "Current status",
            "value_labels": {0: "No", 1: "Yes"},
            "variable_notes": ["Collected during the first interview."],
            "stata_format": "%9.0g",
            "stata_type": "byte",
        },
        "partner_status": {"value_labels": {0: "No", 1: "Yes"}},
        "score": {"column_label": "Assessment score"},
    }
    assert result.stata_metadata["version"] == 1
    assert result.dataset_label == "Household survey"
    assert result.dataset_notes == ["Preliminary release."]
    assert result.stata_metadata["variables"]["partner_status"]["value_label"] == "status"


def test_read_labels_can_be_written_again(labelled_path, tmp_path):
    result = read_labelled_parquet(labelled_path)
    round_trip_path = tmp_path / "round-trip.parquet"

    write_labelled_parquet(
        round_trip_path,
        result.data,
        labels=result.labels,
        dataset_label=result.dataset_label,
        dataset_notes=result.dataset_notes,
    )

    round_trip_result = read_labelled_parquet(round_trip_path)
    assert round_trip_result.data == result.data
    assert round_trip_result.labels == result.labels
    assert round_trip_result.dataset_label == result.dataset_label
    assert round_trip_result.dataset_notes == result.dataset_notes


@pytest.mark.parametrize(
    ("backend", "expected_type"), [("pandas", pd.DataFrame), ("polars", pl.DataFrame)]
)
def test_reads_requested_dataframe_backend(labelled_path, backend, expected_type):
    result = read_labelled_parquet(labelled_path, backend=backend)

    assert isinstance(result.data, expected_type)


def test_rejects_file_without_label_metadata(tmp_path):
    path = tmp_path / "ordinary.parquet"
    pq.write_table(pa.table({"status": [0]}), path)

    with pytest.raises(
        LabelledParquetMetadataNotFoundError, match="has no org.stata.pq.labels.v1"
    ):
        read_labelled_parquet(path)


def test_rejects_unknown_backend(labelled_path):
    with pytest.raises(ValueError, match="backend must be one of"):
        read_labelled_parquet(labelled_path, backend="dictionary")


def test_rejects_unsupported_metadata_version(tmp_path):
    path = tmp_path / "future-metadata.parquet"
    write_labelled_parquet(
        path,
        {"status": [0, 1]},
        stata_metadata={
            "version": 2,
            "label_languages": {"en": {"dataset_label": "Household survey"}},
        },
    )

    with pytest.raises(ValueError, match="must have version 1"):
        read_labelled_parquet(path)


def test_validates_metadata_and_value_label_coverage(labelled_path):
    assert validate_parquet_metadata(labelled_path) is None


def test_validator_rejects_incomplete_value_labels(tmp_path):
    path = tmp_path / "incomplete.parquet"
    write_labelled_parquet(
        path,
        {"status": [0, 1, 2]},
        labels={"status": {"value_labels": {0: "No", 1: "Yes"}}},
        allow_incomplete_value_labels=True,
    )

    with pytest.raises(ValueError, match="missing codes: 2"):
        validate_parquet_metadata(path)


def test_validator_rejects_invalid_v1_value_label_reference(tmp_path):
    path = tmp_path / "invalid.parquet"
    write_labelled_parquet(path, {"status": [0, 1]})
    update_labelled_parquet_metadata(
        path,
        stata_metadata={
            "version": 1,
            "variables": {"status": {"value_label": "missing"}},
        },
    )

    with pytest.raises(ValueError, match="undefined value label 'missing'"):
        validate_parquet_metadata(path)
