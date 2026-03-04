import json
from pathlib import Path
from typing import Iterator


def iter_parquet_batches(parquet_path: Path | str, batch_size: int, text_field: str | list[str] | None = None) -> Iterator['pandas.DataFrame']:
    """Stream batches of data from a parquet file."""
    if isinstance(parquet_path, str): parquet_path = Path(parquet_path)
    if isinstance(text_field, str): text_field = [text_field]

    try:
        import pandas
        from pyarrow import dataset
    except ImportError:
        raise ImportError('pandas and pyarrow are required for streaming parquet files. Please install them via `pip install pandas pyarrow`.')

    dataset_obj = dataset.dataset(source=parquet_path, format='parquet')
    for batch in dataset_obj.to_batches(columns=text_field, batch_size=batch_size):
        yield batch.to_pandas()


def iter_jsonl_rows(file_path: Path) -> Iterator[dict]:
    """ 流式读取jsonl文件 """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)
