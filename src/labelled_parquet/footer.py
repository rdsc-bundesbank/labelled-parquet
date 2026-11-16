"""Low-level, footer-only Parquet key-value metadata updates."""

import os
import tempfile
from os import PathLike
from pathlib import Path
from typing import Any

_PARQUET_MAGIC = b"PAR1"
_STOP = 0
_TRUE = 1
_FALSE = 2
_BYTE = 3
_I16 = 4
_I32 = 5
_I64 = 6
_DOUBLE = 7
_BINARY = 8
_LIST = 9
_SET = 10
_MAP = 11
_STRUCT = 12


def replace_key_value_metadata(path: str | PathLike[str], key: bytes, value: bytes) -> None:
    """Atomically replace one Parquet footer key-value entry without reading data pages."""
    source = Path(path)
    with source.open("rb") as input_file:
        input_file.seek(0, os.SEEK_END)
        file_size = input_file.tell()
        if file_size < 12:
            raise ValueError(f"{source} is not a valid Parquet file")
        input_file.seek(0)
        if input_file.read(4) != _PARQUET_MAGIC:
            raise ValueError(f"{source} is not a valid Parquet file")
        input_file.seek(-8, os.SEEK_END)
        footer_size = int.from_bytes(input_file.read(4), "little")
        if input_file.read(4) != _PARQUET_MAGIC or footer_size > file_size - 12:
            raise ValueError(f"{source} has an invalid Parquet footer")
        data_size = file_size - footer_size - 8
        input_file.seek(data_size)
        footer = input_file.read(footer_size)

        try:
            fields, offset = _read_struct(footer, 0)
        except (IndexError, OverflowError) as error:
            # Keep malformed input from exposing implementation details.  The
            # parser below performs its own bounds checks, but this is a final
            # boundary for malformed footer input.
            raise ValueError(f"{source} has an invalid Parquet footer") from error
        if offset != len(footer):
            raise ValueError(f"{source} has an invalid Parquet footer")
        _replace_key_value_field(fields, key, value)
        new_footer = _write_struct(fields)

        descriptor = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{source.name}.", suffix=".tmp", dir=source.parent
            )
            with os.fdopen(descriptor, "wb") as output_file:
                descriptor = None
                input_file.seek(0)
                _copy_bytes(input_file, output_file, data_size)
                output_file.write(new_footer)
                output_file.write(len(new_footer).to_bytes(4, "little"))
                output_file.write(_PARQUET_MAGIC)
                output_file.flush()
                os.fsync(output_file.fileno())
            input_file.close()
            os.replace(temporary_name, source)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if "temporary_name" in locals() and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def _copy_bytes(input_file: Any, output_file: Any, size: int) -> None:
    """Copy a fixed number of bytes without retaining the input in memory."""
    remaining = size
    while remaining:
        chunk = input_file.read(min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError("Parquet file ended unexpectedly")
        output_file.write(chunk)
        remaining -= len(chunk)


def _replace_key_value_field(fields: list[tuple[int, int, Any]], key: bytes, value: bytes) -> None:
    """Replace FileMetaData field 5 (key_value_metadata)."""
    entries: list[list[tuple[int, int, Any]]] = []
    for field_id, field_type, field_value in fields:
        if field_id != 5:
            continue
        if field_type != _LIST:
            raise ValueError("Parquet footer has invalid key-value metadata")
        element_type, existing_entries = field_value
        if element_type != _STRUCT:
            raise ValueError("Parquet footer has invalid key-value metadata")
        for entry in existing_entries:
            entry_key = next(
                (item[2] for item in entry if item[0] == 1 and item[1] == _BINARY), None
            )
            if entry_key != key:
                entries.append(entry)
    entries.append([(1, _BINARY, key), (2, _BINARY, value)])
    replacement = (5, _LIST, (_STRUCT, entries))
    insertion_index = next((i for i, field in enumerate(fields) if field[0] == 5), len(fields))
    fields[:] = [field for field in fields if field[0] != 5]
    fields.insert(insertion_index, replacement)


def _read_struct(data: bytes, offset: int) -> tuple[list[tuple[int, int, Any]], int]:
    """Read a Thrift compact-protocol struct into a generic representation."""
    fields: list[tuple[int, int, Any]] = []
    previous_field_id = 0
    while True:
        _require_bytes(data, offset, 1)
        header = data[offset]
        offset += 1
        field_type = header & 0x0F
        delta = header >> 4
        if field_type == _STOP:
            return fields, offset
        if delta:
            field_id = previous_field_id + delta
        else:
            encoded_field_id, offset = _read_varint(data, offset)
            field_id = _zigzag_decode(encoded_field_id)
            if field_id <= 0:
                raise ValueError("Parquet footer has an invalid Thrift field ID")
        field_value, offset = _read_value(data, offset, field_type)
        fields.append((field_id, field_type, field_value))
        previous_field_id = field_id


def _read_value(
    data: bytes, offset: int, field_type: int, *, boolean_in_collection: bool = False
) -> tuple[Any, int]:
    if field_type in {_TRUE, _FALSE}:
        if boolean_in_collection:
            _require_bytes(data, offset, 1)
            byte = data[offset]
            if byte not in {_TRUE, _FALSE}:
                raise ValueError("Parquet footer has an invalid boolean value")
            return byte == _TRUE, offset + 1
        return field_type == _TRUE, offset
    if field_type == _BYTE:
        _require_bytes(data, offset, 1)
        return data[offset], offset + 1
    if field_type in {_I16, _I32, _I64}:
        value, offset = _read_varint(data, offset)
        return (value >> 1) ^ -(value & 1), offset
    if field_type == _DOUBLE:
        _require_bytes(data, offset, 8)
        return data[offset : offset + 8], offset + 8
    if field_type == _BINARY:
        size, offset = _read_varint(data, offset)
        _require_bytes(data, offset, size)
        return data[offset : offset + size], offset + size
    if field_type == _STRUCT:
        return _read_struct(data, offset)
    if field_type in {_LIST, _SET}:
        _require_bytes(data, offset, 1)
        header = data[offset]
        offset += 1
        size = header >> 4
        element_type = header & 0x0F
        if size == 15:
            size, offset = _read_varint(data, offset)
        values = []
        for _ in range(size):
            item, offset = _read_value(data, offset, element_type, boolean_in_collection=True)
            values.append(item)
        return (element_type, values), offset
    if field_type == _MAP:
        size, offset = _read_varint(data, offset)
        if not size:
            return (None, None, []), offset
        _require_bytes(data, offset, 1)
        types = data[offset]
        offset += 1
        key_type, value_type = types >> 4, types & 0x0F
        entries = []
        for _ in range(size):
            map_key, offset = _read_value(data, offset, key_type, boolean_in_collection=True)
            map_value, offset = _read_value(data, offset, value_type, boolean_in_collection=True)
            entries.append((map_key, map_value))
        return (key_type, value_type, entries), offset
    raise ValueError("Parquet footer has an unsupported Thrift field type")


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Parquet footer ended unexpectedly")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("Parquet footer has an invalid variable-length integer")


def _require_bytes(data: bytes, offset: int, size: int) -> None:
    """Raise a stable error when a compact-protocol value is truncated."""
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError("Parquet footer ended unexpectedly")


def _zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _write_struct(fields: list[tuple[int, int, Any]]) -> bytes:
    result = bytearray()
    previous_field_id = 0
    for field_id, field_type, value in fields:
        delta = field_id - previous_field_id
        if 0 < delta <= 15:
            result.append((delta << 4) | field_type)
        else:
            result.append(field_type)
            result.extend(_write_varint(_zigzag_encode(field_id)))
        result.extend(_write_value(field_type, value))
        previous_field_id = field_id
    result.append(_STOP)
    return bytes(result)


def _write_value(field_type: int, value: Any, *, boolean_in_collection: bool = False) -> bytes:
    if field_type in {_TRUE, _FALSE}:
        return bytes([_TRUE if value else _FALSE]) if boolean_in_collection else b""
    if field_type == _BYTE:
        return bytes([value])
    if field_type in {_I16, _I32, _I64}:
        return _write_varint((value << 1) ^ (value >> 63))
    if field_type == _DOUBLE:
        return value
    if field_type == _BINARY:
        return _write_varint(len(value)) + value
    if field_type == _STRUCT:
        return _write_struct(value)
    if field_type in {_LIST, _SET}:
        element_type, values = value
        size = len(values)
        prefix = (
            bytes([(size << 4) | element_type])
            if size < 15
            else bytes([0xF0 | element_type]) + _write_varint(size)
        )
        return prefix + b"".join(
            _write_value(element_type, item, boolean_in_collection=True) for item in values
        )
    if field_type == _MAP:
        key_type, value_type, entries = value
        if not entries:
            return b"\0"
        result = bytearray(_write_varint(len(entries)))
        result.append((key_type << 4) | value_type)
        for map_key, map_value in entries:
            result.extend(_write_value(key_type, map_key, boolean_in_collection=True))
            result.extend(_write_value(value_type, map_value, boolean_in_collection=True))
        return bytes(result)
    raise ValueError("Parquet footer has an unsupported Thrift field type")


def _write_varint(value: int) -> bytes:
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 63)
