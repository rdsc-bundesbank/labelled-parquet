"""Update Stata metadata in Parquet footers without materialising table data."""

import json
from collections.abc import Mapping
from os import PathLike
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from labelled_parquet.footer import replace_key_value_metadata
from labelled_parquet.metadata import (
    validate_v1_metadata,
    validate_value_label_coverage as validate_file_value_label_coverage,
)
from labelled_parquet.writer import (
    STATA_METADATA_KEY,
    _build_envelope,
    _validate_advanced_metadata_arguments,
)


def update_labelled_parquet_metadata(
    path: str | PathLike[str],
    *,
    labels: Mapping[str, Mapping[str, Any]] | None = None,
    dataset_label: str | None = None,
    dataset_notes: list[str] | None = None,
    stata_metadata: Mapping[str, Any] | None = None,
    validate_value_label_coverage: bool = False,
) -> None:
    """Replace a Parquet file's Stata metadata without materialising its table.

    The existing ``org.stata.pq.labels.v1`` envelope is replaced as a whole.
    All other Parquet footer metadata and the encoded data pages are retained.

    Args:
        path: Existing local Parquet file to update.
        labels: Convenience metadata by physical column name.
        dataset_label: Optional Stata dataset label.
        dataset_notes: Optional Stata dataset notes.
        stata_metadata: Advanced Stata metadata envelope written verbatim. It
            cannot be combined with the convenience metadata arguments.
        validate_value_label_coverage: Scan labelled columns in record batches
            to ensure all observed non-null values have a value-label code.
            Defaults to false, so updates only read and rewrite file footers.
    """
    parquet_file = pq.ParquetFile(path)
    try:
        if stata_metadata is not None:
            _validate_advanced_metadata_arguments(
                labels=labels,
                dataset_label=dataset_label,
                dataset_notes=dataset_notes,
                allow_incomplete_value_labels=False,
            )
            if not isinstance(stata_metadata, Mapping):
                raise TypeError("stata_metadata must be a mapping")
            envelope = dict(stata_metadata)
        else:
            empty_table = pa.table(
                {field.name: pa.array([], type=field.type) for field in parquet_file.schema_arrow}
            )
            envelope = _build_envelope(
                labels or {},
                empty_table,
                dataset_label=dataset_label,
                dataset_notes=dataset_notes,
                allow_incomplete_value_labels=True,
            )
            if validate_value_label_coverage:
                value_labels = validate_v1_metadata(envelope, set(parquet_file.schema_arrow.names))
                validate_file_value_label_coverage(
                    parquet_file, envelope.get("variables", {}), value_labels
                )
    finally:
        parquet_file.close()

    replace_key_value_metadata(
        path, STATA_METADATA_KEY, json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    )
