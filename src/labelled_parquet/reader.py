"""Read Parquet files with Stata-compatible labels."""

from dataclasses import dataclass
from os import PathLike
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from labelled_parquet.metadata import (
    read_stata_metadata,
    resolve_labels,
    validate_v1_metadata,
)


@dataclass(frozen=True)
class LabelledParquet:
    """Data and labels read from a labelled Parquet file."""

    data: Any
    labels: dict[str, dict[str, Any]]
    dataset_label: str | None
    dataset_notes: list[str]
    stata_metadata: dict[str, Any]


def read_labelled_parquet(
    path: str | PathLike[str], *, backend: str = "pyarrow"
) -> LabelledParquet:
    """Read labelled Parquet data and resolve its shared value labels.

    Args:
        path: Source Parquet file.
        backend: Data representation to return. ``"pyarrow"`` is the default;
            ``"pandas"`` and ``"polars"`` require their respective optional
            dependencies.

    Raises:
        LabelledParquetMetadataNotFoundError: If the file has no Stata label
            metadata.
        ValueError: If the metadata is malformed, uses an unsupported metadata
            version, or ``backend`` is unsupported.
        ImportError: If the requested optional dataframe backend is unavailable.
    """
    parquet_file = pq.ParquetFile(path)
    stata_metadata = read_stata_metadata(parquet_file)
    validate_v1_metadata(stata_metadata, set(parquet_file.schema_arrow.names))
    table = parquet_file.read()
    labels = resolve_labels(stata_metadata, set(table.column_names))
    dataset_label = stata_metadata.get("dataset_label")
    dataset_notes = stata_metadata.get("dataset_notes", [])
    return LabelledParquet(
        data=_convert_table(table, backend),
        labels=labels,
        dataset_label=dataset_label,
        dataset_notes=dataset_notes,
        stata_metadata=stata_metadata,
    )


def _convert_table(table: pa.Table, backend: str) -> Any:
    """Convert an Arrow table to the requested output backend."""
    if backend == "pyarrow":
        return table
    if backend == "pandas":
        return table.to_pandas()
    if backend == "polars":
        try:
            import polars as pl
        except ImportError as error:
            raise ImportError("backend='polars' requires the polars extra") from error
        return pl.from_arrow(table)
    raise ValueError("backend must be one of: pyarrow, pandas, polars")
