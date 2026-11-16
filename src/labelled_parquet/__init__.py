"""labelled_parquet module."""

from labelled_parquet.metadata import LabelledParquetMetadataNotFoundError
from labelled_parquet.reader import LabelledParquet, read_labelled_parquet
from labelled_parquet.writer import STATA_METADATA_KEY, write_labelled_parquet
from labelled_parquet.updater import update_labelled_parquet_metadata
from labelled_parquet.validator import validate_parquet_metadata

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("labelled-parquet")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "LabelledParquet",
    "LabelledParquetMetadataNotFoundError",
    "STATA_METADATA_KEY",
    "read_labelled_parquet",
    "update_labelled_parquet_metadata",
    "validate_parquet_metadata",
    "write_labelled_parquet",
]
