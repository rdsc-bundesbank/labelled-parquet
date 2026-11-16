# labelled-parquet

Read and write Parquet files that carry Stata-compatible dataset and variable
metadata. Files produced by this package can be imported by Stata's
[`pq`](https://github.com/jrothbaum/stata_parquet_io) package using
`pq use ..., statametadata`.

## Installation

```bash
uv add labelled-parquet
```

For pandas or Polars support, install the appropriate optional extra:

```bash
uv add 'labelled-parquet[pandas]'
# or
uv add 'labelled-parquet[polars]'
```

## Usage

`write_labelled_parquet` accepts PyArrow tables, pandas and Polars dataframes,
and dictionaries of lists, storing labels in the Stata-compatible
`org.stata.pq.labels.v1` Parquet footer. The resulting files can be imported
using Stata's `pq use ..., statametadata`.

```python
import pandas as pd

from labelled_parquet import write_labelled_parquet

dataframe = pd.DataFrame({"id": [1, 2], "status": [0, 1]})
write_labelled_parquet(
    "people.parquet",
    dataframe,
    labels={
        "id": {"column_label": "Person identifier"},
        "status": {
            "column_label": "Current status",
            "value_labels": {0: "Inactive", 1: "Active"},
        },
    },
    compression="zstd",
)
```

Dataset-level Stata metadata and additional variable metadata can also be
preserved. `stata_format` uses **Stata** format syntax (for example `%9.2f`),
not Python format-specification syntax.

```python
write_labelled_parquet(
    "people.parquet",
    {"income": [1200.50, 950.00]},
    labels={
        "income": {
            "column_label": "Monthly net income",
            "variable_notes": ["Reported in EUR."],
            "stata_format": "%12.2f",
            "stata_type": "double",
        }
    },
    dataset_label="2025 household survey",
    dataset_notes=["Preliminary release."],
)
```

Supported `stata_type` values are `byte`, `int`, `long`, `float`, `double`,
`date`, `time`, and `datetime`.

Value labels are optional. However, when a column declares `value_labels`, the
mapping must cover every non-null value in that column by default. To permit
unlabelled observed values, opt in explicitly:

```python
write_labelled_parquet(
    "people.parquet",
    {"id": [1, 2], "status": [0, 1, 2]},
    labels={"status": {"value_labels": {0: "Inactive", 1: "Active"}}},
    allow_incomplete_value_labels=True,
)
```

Value-label codes must be integers. They are serialised as strings as required
by the Stata metadata format. When multiple columns have equal `value_labels`,
the package writes the definition once and makes every relevant column refer to
it. The shared definition is named after the first labelled column.

## Reading

`read_labelled_parquet` returns a `LabelledParquet` result containing the data,
writer-compatible labels, and the complete original Stata metadata envelope.
Shared value-label definitions are resolved automatically for each column.

```python
from labelled_parquet import read_labelled_parquet

result = read_labelled_parquet("people.parquet", backend="pandas")
dataframe = result.data
labels = result.labels
dataset_label = result.dataset_label
dataset_notes = result.dataset_notes
```

The default backend is PyArrow. Use `backend="pandas"` or `backend="polars"`
when the corresponding optional extra is installed. Files without
`org.stata.pq.labels.v1` metadata raise `LabelledParquetMetadataNotFoundError`.

## Updating metadata

`update_labelled_parquet_metadata` adds or replaces Stata metadata on an
**existing local file** without materialising the table data. This is useful
when the data has already been written by another tool:

```python
from labelled_parquet import update_labelled_parquet_metadata

update_labelled_parquet_metadata(
    "people.parquet",
    labels={
        "status": {
            "column_label": "Current status",
            "value_labels": {0: "Inactive", 1: "Active"},
        }
    },
    dataset_label="2025 household survey",
)
```

For speed, an update does not scan data by default, so value-label mappings may
not cover every observed non-null value. Enable the potentially expensive scan
when complete coverage is required:

```python
update_labelled_parquet_metadata(
    "people.parquet",
    labels={"status": {"value_labels": {0: "Inactive", 1: "Active"}}},
    validate_value_label_coverage=True,
)
```

> [!WARNING]
> Updating mutates the path passed to it: it creates a temporary file in the
> same directory and atomically replaces the original file when complete. Keep
> a backup if the original must be retained, and ensure the directory is
> writable. The operation retains encoded data pages and other Parquet footer
> entries, but it **replaces the entire** `org.stata.pq.labels.v1` metadata
> envelope. Metadata already in that envelope is discarded unless it is
> included in the replacement supplied through `labels`, dataset arguments, or
> `stata_metadata`.

### Footer compatibility

The footer-only updater reads and writes the standard Apache Thrift Compact
Protocol representation of Parquet `FileMetaData`; it does not depend on a
particular data-page encoding. The automated compatibility suite exercises
files produced directly by the supported PyArrow, Polars, and DuckDB versions,
including non-delta Compact Protocol field IDs and collection boolean
representations. Malformed or truncated footers are rejected with `ValueError`
before the source file is replaced. Files from other producers (such as Spark
and Stata `pq`) are not yet covered by this compatibility suite and should be
validated for the relevant producer/version before production use.


## Validating metadata

`validate_parquet_metadata` checks the `pq` v1 metadata structure, references
to value-label definitions, and that every observed non-null value in labelled
columns has a label code:

```python
from labelled_parquet import validate_parquet_metadata

validate_parquet_metadata("people.parquet")
```

Validation reads labelled columns in record batches rather than materialising a
table or dataframe, but it does scan their data. It can therefore be costly for
large files; unlike metadata updates, coverage validation is always performed.

## Development

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run pyrefly check
```

## License

Distributed under the [MIT License](LICENSE). Copyright 2026 Bundesbank
Research Data and Service Centre.
