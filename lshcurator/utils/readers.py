import json
from pathlib import Path
from typing import Iterator, Literal


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


def iter_corpus_texts(
    corpus_files_path: str | Path | list[str | Path],
    corpus_field_name: str | list[str],
    corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
    **kwargs
) -> Iterator[str]:
    """迭代语料文本内容的生成器，支持多文件和多字段"""
    if isinstance(corpus_field_name, str): corpus_field_name = [corpus_field_name]

    if corpus_file_format == 'parquet':
        for file_path in corpus_files_path:
            for batch in iter_parquet_batches(
                parquet_path=file_path,
                batch_size=kwargs.get('batch_size', 2048),
                text_field=corpus_field_name,
            ):
                for sample in batch.stack():
                    yield str(sample)
    elif corpus_file_format == 'jsonl':
        for file_path in corpus_files_path:
            for row in iter_jsonl_rows(file_path=file_path):
                for field in corpus_field_name:
                    yield str(row[field])
    else: raise ValueError(f"Unsupported file format: {corpus_file_format}")
