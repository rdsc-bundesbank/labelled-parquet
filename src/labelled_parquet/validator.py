"""Validate Stata ``pq`` v1 metadata stored in Parquet files."""

from os import PathLike

import pyarrow.parquet as pq

from labelled_parquet.metadata import (
    read_stata_metadata,
    validate_v1_metadata,
    validate_value_label_coverage,
)


def validate_parquet_metadata(path: str | PathLike[str]) -> None:
    """Validate pq v1 JSON metadata, references, and value-label coverage.

    Data is scanned in record batches rather than materialised as a table or
    dataframe. Future metadata versions are rejected because their schemas are
    not yet known to this package.
    """
    parquet_file = pq.ParquetFile(path)
    try:
        envelope = read_stata_metadata(parquet_file)
        value_labels = validate_v1_metadata(envelope, set(parquet_file.schema_arrow.names))
        validate_value_label_coverage(parquet_file, envelope.get("variables", {}), value_labels)
    finally:
        parquet_file.close()
