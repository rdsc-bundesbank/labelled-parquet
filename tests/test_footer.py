"""Compatibility and corruption tests for the footer-only metadata updater."""

import importlib.metadata

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from labelled_parquet import STATA_METADATA_KEY, update_labelled_parquet_metadata
from labelled_parquet.footer import (
    _BINARY,
    _DOUBLE,
    _FALSE,
    _I32,
    _LIST,
    _MAP,
    _STRUCT,
    _TRUE,
    _read_struct,
    _write_struct,
    replace_key_value_metadata,
)


def _write_pyarrow(path):
    pq.write_table(pa.table({"id": [1, 2], "status": [0, 1]}), path)


def _write_polars(path):
    pl.DataFrame({"id": [1, 2], "status": [0, 1]}).write_parquet(path)


def _write_duckdb(path):
    path_sql = str(path).replace("'", "''")
    duckdb.execute(
        "COPY (SELECT 1 AS id, 0 AS status UNION ALL SELECT 2, 1) "
        f"TO '{path_sql}' (FORMAT PARQUET)"
    )


@pytest.mark.parametrize(
    ("producer", "write_file"),
    [
        (f"pyarrow-{importlib.metadata.version('pyarrow')}", _write_pyarrow),
        (f"polars-{importlib.metadata.version('polars')}", _write_polars),
        (f"duckdb-{importlib.metadata.version('duckdb')}", _write_duckdb),
    ],
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
def test_updates_footers_written_by_supported_producers(tmp_path, producer, write_file):
    """Retain data and unrelated metadata from independently-written files."""
    path = tmp_path / f"{producer}.parquet"
    write_file(path)
    original = path.read_bytes()
    footer_start = len(original) - pq.ParquetFile(path).metadata.serialized_size - 8

    update_labelled_parquet_metadata(path, stata_metadata={"version": 1, "variables": {}})

    assert pq.read_table(path).to_pydict() == {"id": [1, 2], "status": [0, 1]}
    metadata = pq.ParquetFile(path).metadata.metadata
    assert metadata is not None
    assert metadata[STATA_METADATA_KEY] == b'{"version": 1, "variables": {}}'
    assert path.read_bytes()[:footer_start] == original[:footer_start]


def test_compact_protocol_round_trip_preserves_explicit_ids_and_collection_booleans():
    """Exercise representations uncommon in ordinary PyArrow FileMetaData."""
    fields = [
        (1, _I32, -12),
        # The gap forces the compact protocol's explicit, zigzag-encoded field ID.
        (20, _BINARY, b"explicit field id"),
        (21, _LIST, (_TRUE, [True, False])),
        (22, _MAP, (_BINARY, _FALSE, [(b"enabled", False)])),
        (23, _STRUCT, [(1, _DOUBLE, b"\x00\x01\x02\x03\x04\x05\x06\x07")]),
    ]

    encoded = _write_struct(fields)
    decoded, offset = _read_struct(encoded, 0)

    assert offset == len(encoded)
    assert decoded == fields
    assert _write_struct(decoded) == encoded


@pytest.mark.parametrize(
    "footer",
    [
        b"\x18",  # binary field with no length
        b"\x15\x80",  # integer field with a truncated varint
        b"\x19\x18",  # list with no collection header
        b"\x1b\x01",  # map with no key/value type byte
        b"\x0c",  # struct with no STOP marker
    ],
)
def test_rejects_truncated_compact_footers_with_value_error(tmp_path, footer):
    path = tmp_path / "truncated.parquet"
    path.write_bytes(b"PAR1" + footer + len(footer).to_bytes(4, "little") + b"PAR1")

    with pytest.raises(ValueError, match="invalid Parquet footer|ended unexpectedly"):
        replace_key_value_metadata(path, b"key", b"value")


def test_rejects_unterminated_compact_varint_with_value_error(tmp_path):
    path = tmp_path / "invalid-varint.parquet"
    footer = b"\x15" + b"\x80" * 10
    path.write_bytes(b"PAR1" + footer + len(footer).to_bytes(4, "little") + b"PAR1")

    with pytest.raises(ValueError, match="invalid variable-length integer"):
        replace_key_value_metadata(path, b"key", b"value")
