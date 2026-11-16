"""Write Parquet files with Stata-compatible labels."""

import json
from collections.abc import Mapping
from numbers import Integral
from os import PathLike
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from labelled_parquet.metadata import (
    STATA_METADATA_KEY,
    validate_v1_metadata,
    validate_value_label_array_coverage,
)


def write_labelled_parquet(
    path: str | PathLike[str],
    dataframe: Any,
    *,
    labels: Mapping[str, Mapping[str, Any]] | None = None,
    dataset_label: str | None = None,
    dataset_notes: list[str] | None = None,
    stata_metadata: Mapping[str, Any] | None = None,
    allow_incomplete_value_labels: bool = False,
    **parquet_options: Any,
) -> None:
    """Write supported tabular data with Stata-compatible labels.

    Labels are stored in the Parquet footer under
    ``org.stata.pq.labels.v1``, the format read by Stata's ``pq use`` command.
    Each labelled column may specify ``column_label``, ``value_labels``,
    ``variable_notes``, ``stata_format``, and/or ``stata_type``. Use
    ``stata_metadata`` to write an advanced, versioned metadata envelope
    directly, such as a future ``pq`` label-language schema.

    Args:
        path: Destination Parquet file.
        dataframe: A :class:`pyarrow.Table`, :class:`pandas.DataFrame`,
            :class:`polars.DataFrame`, or dictionary mapping column names to
            lists of values.
        labels: Optional metadata by physical column name. A column can have a
            column label, value labels, both, or neither.
        dataset_label: Optional Stata dataset label.
        dataset_notes: Optional Stata dataset notes.
        stata_metadata: Advanced Stata metadata envelope written verbatim to
            the Parquet footer. Cannot be combined with ``labels``,
            ``dataset_label``, ``dataset_notes``, or
            ``allow_incomplete_value_labels``.
        allow_incomplete_value_labels: Allow values that have no corresponding
            value-label code. Defaults to false.
        **parquet_options: Options accepted by :func:`pyarrow.parquet.write_table`.

    Raises:
        TypeError: If the dataframe or labels have an unsupported type.
        ValueError: If declared value labels do not cover data values, labels
            refer to unknown columns, or contain invalid value-label codes.
    """
    table = _to_arrow_table(dataframe)
    if stata_metadata is not None:
        _validate_advanced_metadata_arguments(
            labels=labels,
            dataset_label=dataset_label,
            dataset_notes=dataset_notes,
            allow_incomplete_value_labels=allow_incomplete_value_labels,
        )
        if not isinstance(stata_metadata, Mapping):
            raise TypeError("stata_metadata must be a mapping")
        envelope = dict(stata_metadata)
    else:
        envelope = _build_envelope(
            labels or {},
            table,
            dataset_label=dataset_label,
            dataset_notes=dataset_notes,
            allow_incomplete_value_labels=allow_incomplete_value_labels,
        )
    metadata = dict(table.schema.metadata or {})
    metadata[STATA_METADATA_KEY] = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path, **parquet_options)


def _validate_advanced_metadata_arguments(
    *,
    labels: Mapping[str, Mapping[str, Any]] | None,
    dataset_label: str | None,
    dataset_notes: list[str] | None,
    allow_incomplete_value_labels: bool,
) -> None:
    """Prevent ambiguous merging of convenience and raw metadata inputs."""
    if (
        labels is not None
        or dataset_label is not None
        or dataset_notes is not None
        or allow_incomplete_value_labels
    ):
        raise ValueError(
            "stata_metadata cannot be combined with labels, dataset_label, "
            "dataset_notes, or allow_incomplete_value_labels"
        )


def _to_arrow_table(dataframe: Any) -> pa.Table:
    """Convert supported input without mandatory pandas or Polars imports."""
    if isinstance(dataframe, pa.Table):
        return dataframe
    if isinstance(dataframe, dict):
        return pa.Table.from_pydict(dataframe)

    module_name = type(dataframe).__module__.split(".", maxsplit=1)[0]
    class_name = type(dataframe).__name__
    if module_name == "pandas" and class_name == "DataFrame":
        return pa.Table.from_pandas(dataframe, preserve_index=False)
    if module_name == "polars" and class_name == "DataFrame":
        return dataframe.to_arrow()
    raise TypeError(
        "dataframe must be a pyarrow.Table, pandas.DataFrame, polars.DataFrame, "
        "or dictionary of lists"
    )


def _build_envelope(
    labels: Mapping[str, Mapping[str, Any]],
    table: pa.Table,
    *,
    dataset_label: str | None,
    dataset_notes: list[str] | None,
    allow_incomplete_value_labels: bool,
) -> dict[str, Any]:
    """Build the versioned metadata envelope expected by Stata's ``pq``."""
    if not isinstance(labels, Mapping):
        raise TypeError("labels must be a mapping keyed by column name")

    for column_name in labels:
        if not isinstance(column_name, str):
            raise TypeError("label column names must be strings")

    column_names = set(table.column_names)
    unknown_columns = labels.keys() - column_names
    if unknown_columns:
        names = ", ".join(sorted(unknown_columns))
        raise ValueError(f"labels contain unknown columns: {names}")

    variables: dict[str, dict[str, str]] = {}
    value_labels: dict[str, dict[str, str]] = {}
    for column_name, specification in labels.items():
        if not isinstance(specification, Mapping):
            raise TypeError(f"labels for {column_name!r} must be a mapping")

        unexpected_keys = set(specification) - {
            "column_label",
            "value_labels",
            "variable_notes",
            "stata_format",
            "stata_type",
        }
        if unexpected_keys:
            names = ", ".join(sorted(unexpected_keys))
            raise ValueError(f"labels for {column_name!r} contain unsupported keys: {names}")

        variable: dict[str, Any] = {}
        column_label = specification.get("column_label")
        if column_label is not None:
            if not isinstance(column_label, str):
                raise TypeError(f"column_label for {column_name!r} must be a string")
            if column_label:
                variable["label"] = column_label

        variable_notes = specification.get("variable_notes")
        if variable_notes is not None:
            variable["notes"] = _validate_notes(
                variable_notes, f"variable_notes for {column_name!r}"
            )

        stata_format = specification.get("stata_format")
        if stata_format is not None:
            if not isinstance(stata_format, str):
                raise TypeError(f"stata_format for {column_name!r} must be a string")
            if stata_format:
                variable["format"] = stata_format

        stata_type = specification.get("stata_type")
        if stata_type is not None:
            if stata_type not in {
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
                    f"stata_type for {column_name!r} is not supported: {stata_type!r}"
                )
            variable["stata_type"] = stata_type

        codes = specification.get("value_labels")
        if codes is not None:
            normalised_codes = _normalise_value_labels(column_name, codes)
            value_label_name = _find_value_label_name(value_labels, normalised_codes)
            if value_label_name is None:
                value_label_name = column_name
                value_labels[value_label_name] = normalised_codes
            variable["value_label"] = value_label_name
        variables[column_name] = variable

    envelope: dict[str, Any] = {"version": 1}
    if dataset_label is not None:
        if not isinstance(dataset_label, str):
            raise TypeError("dataset_label must be a string")
        if dataset_label:
            envelope["dataset_label"] = dataset_label
    if dataset_notes is not None:
        envelope["dataset_notes"] = _validate_notes(dataset_notes, "dataset_notes")
    if variables:
        envelope["variables"] = variables
    if value_labels:
        envelope["value_labels"] = value_labels
    validated_value_labels = validate_v1_metadata(envelope, column_names)
    if not allow_incomplete_value_labels:
        for column_name, variable in variables.items():
            value_label_name = variable.get("value_label")
            if value_label_name is not None:
                validate_value_label_array_coverage(
                    table[column_name], column_name, validated_value_labels[value_label_name]
                )
    return envelope


def _validate_notes(notes: Any, name: str) -> list[str]:
    """Validate a Stata dataset or variable note collection."""
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise TypeError(f"{name} must be a list of strings")
    return notes


def _normalise_value_labels(column_name: str, codes: Any) -> dict[str, str]:
    """Validate and serialise integer Stata value-label codes."""
    if not isinstance(codes, Mapping):
        raise TypeError(f"value_labels for {column_name!r} must be a mapping")
    normalised: dict[str, str] = {}
    for code, text in codes.items():
        if isinstance(code, bool) or not isinstance(code, Integral):
            raise ValueError(f"value-label code {code!r} for {column_name!r} must be an integer")
        if not isinstance(text, str):
            raise TypeError(f"value-label text for {column_name!r} must be a string")
        normalised[str(code)] = text
    return normalised


def _find_value_label_name(
    value_labels: Mapping[str, Mapping[str, str]], codes: Mapping[str, str]
) -> str | None:
    """Return the existing label name for an equivalent value-label definition."""
    for name, existing_codes in value_labels.items():
        if existing_codes == codes:
            return name
    return None
