"""Shared parsing, resolution, and validation for Stata ``pq`` metadata."""

import json
from collections.abc import Mapping
from typing import Any

import pyarrow.parquet as pq

STATA_METADATA_KEY = b"org.stata.pq.labels.v1"
"""Parquet footer key used by Stata's ``pq`` package."""


class LabelledParquetMetadataNotFoundError(ValueError):
    """Raised when a Parquet file has no Stata-compatible label metadata."""


def read_stata_metadata(parquet_file: pq.ParquetFile) -> dict[str, Any]:
    """Read and parse a Stata metadata envelope from an open Parquet file."""
    footer_metadata = parquet_file.metadata.metadata or {}
    raw_metadata = footer_metadata.get(STATA_METADATA_KEY)
    if raw_metadata is None:
        key = STATA_METADATA_KEY.decode("utf-8")
        raise LabelledParquetMetadataNotFoundError(f"Parquet file has no {key} metadata")
    try:
        stata_metadata = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Stata label metadata is not valid JSON") from error
    if not isinstance(stata_metadata, dict):
        raise ValueError("Stata label metadata must be a JSON object")
    return stata_metadata


def resolve_labels(
    stata_metadata: Mapping[str, Any], column_names: set[str]
) -> dict[str, dict[str, Any]]:
    """Convert named value-label definitions into the writer's label shape."""
    variables = stata_metadata.get("variables", {})
    definitions = stata_metadata.get("value_labels", {})
    if not isinstance(variables, dict) or not isinstance(definitions, dict):
        raise ValueError("Stata label metadata has invalid variables or value_labels entries")
    labels: dict[str, dict[str, Any]] = {}
    for column_name, variable in variables.items():
        if column_name not in column_names:
            continue
        if not isinstance(variable, dict):
            raise ValueError(f"Stata metadata for {column_name!r} must be an object")
        label = {
            target: variable[source]
            for source, target in {
                "label": "column_label",
                "notes": "variable_notes",
                "format": "stata_format",
                "stata_type": "stata_type",
            }.items()
            if source in variable
        }
        value_label_name = variable.get("value_label")
        if value_label_name is not None:
            try:
                definition = definitions[value_label_name]
            except KeyError as error:
                raise ValueError(
                    f"Stata metadata for {column_name!r} refers to undefined value label "
                    f"{value_label_name!r}"
                ) from error
            if not isinstance(definition, dict):
                raise ValueError(f"value label {value_label_name!r} must be an object")
            label["value_labels"] = {int(code): text for code, text in definition.items()}
        labels[column_name] = label
    return labels


def validate_v1_metadata(
    stata_metadata: Mapping[str, Any], column_names: set[str]
) -> dict[str, dict[str, str]]:
    """Validate the supported pq v1 schema and return value-label definitions."""
    if stata_metadata.get("version") != 1:
        raise ValueError("Stata label metadata must have version 1")
    _validate_optional_string(stata_metadata, "dataset_label", "Stata dataset_label metadata")
    _validate_string_list(stata_metadata.get("dataset_notes", []), "Stata dataset_notes metadata")
    variables = stata_metadata.get("variables", {})
    if not isinstance(variables, dict):
        raise ValueError("Stata label metadata variables must be an object")
    unknown_columns = set(variables) - column_names
    if unknown_columns:
        raise ValueError(
            f"Stata metadata refers to unknown columns: {', '.join(sorted(unknown_columns))}"
        )
    value_labels = _validate_value_label_definitions(stata_metadata.get("value_labels", {}))
    for column_name, variable in variables.items():
        if not isinstance(column_name, str) or not isinstance(variable, dict):
            raise ValueError(f"Stata metadata for {column_name!r} must be an object")
        _validate_optional_string(variable, "label", f"Stata label for {column_name!r}")
        _validate_optional_string(variable, "format", f"Stata format for {column_name!r}")
        _validate_string_list(variable.get("notes", []), f"Stata notes for {column_name!r}")
        if variable.get("stata_type") not in {
            None,
            "byte",
            "int",
            "long",
            "float",
            "double",
            "date",
            "time",
            "datetime",
        }:
            raise ValueError(
                f"Stata type for {column_name!r} is not supported: {variable['stata_type']!r}"
            )
        value_label_name = variable.get("value_label")
        if value_label_name is not None:
            if not isinstance(value_label_name, str):
                raise ValueError(f"Stata value label for {column_name!r} must be a string")
            if value_label_name not in value_labels:
                raise ValueError(
                    f"Stata metadata for {column_name!r} refers to undefined value label "
                    f"{value_label_name!r}"
                )
    return value_labels


def validate_value_label_coverage(
    parquet_file: pq.ParquetFile,
    variables: Mapping[str, Mapping[str, Any]],
    value_labels: Mapping[str, Mapping[str, str]],
) -> None:
    """Batch-scan value-labelled columns and ensure every observed code is defined."""
    for column_name, variable in variables.items():
        value_label_name = variable.get("value_label")
        if value_label_name is None:
            continue
        _validate_value_label_values(
            (
                value
                for batch in parquet_file.iter_batches(columns=[column_name])
                for value in batch.column(0)
            ),
            column_name,
            value_labels[value_label_name],
        )


def validate_value_label_array_coverage(
    column: Any, column_name: str, value_labels: Mapping[str, str]
) -> None:
    """Validate value-label coverage for an in-memory Arrow chunked array."""
    _validate_value_label_values(
        (value for chunk in column.chunks for value in chunk), column_name, value_labels
    )


def _validate_value_label_values(
    values: Any, column_name: str, definition: Mapping[str, str]
) -> None:
    """Ensure the supplied Arrow scalar values all have a label definition."""
    missing_codes: set[str] = set()
    non_null_count = missing_count = 0
    for value in values:
        if value.is_valid:
            non_null_count += 1
            code = str(value.as_py())
            if code not in definition:
                missing_count += 1
                missing_codes.add(code)
    if missing_codes:
        codes = ", ".join(sorted(missing_codes))
        raise ValueError(
            f"value_labels for {column_name!r} are missing codes: {codes} "
            f"({missing_count / non_null_count * 100:.1f}% of non-null values are unlabelled)"
        )


def _validate_value_label_definitions(value_labels: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value_labels, dict):
        raise ValueError("Stata label metadata value_labels must be an object")
    validated: dict[str, dict[str, str]] = {}
    for name, definition in value_labels.items():
        if not isinstance(name, str) or not isinstance(definition, dict):
            raise ValueError(f"value label {name!r} must be an object")
        for code, text in definition.items():
            if not isinstance(code, str):
                raise ValueError(f"value-label code in {name!r} must be a string")
            try:
                int(code)
            except ValueError as error:
                raise ValueError(
                    f"value-label code {code!r} in {name!r} must be an integer"
                ) from error
            if not isinstance(text, str):
                raise ValueError(f"value-label text for {name!r} must be a string")
        validated[name] = definition
    return validated


def _validate_optional_string(mapping: Mapping[str, Any], key: str, description: str) -> None:
    if mapping.get(key) is not None and not isinstance(mapping[key], str):
        raise ValueError(f"{description} must be a string")


def _validate_string_list(values: Any, description: str) -> None:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{description} must be a list of strings")
