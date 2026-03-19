import json
from pathlib import Path
from typing import Iterator

import numpy

from .normalizations import path_normalize


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
    files_path: str | Path | list[str | Path],
    fields: str | list[str] | None = None,
    **kwargs
) -> Iterator[str | tuple[str, Path]]:
    """
    流式迭代语料文本内容，支持多文件、多字段，以及可选返回来源文件路径。

    Args:
        files_path (str | Path | list[str | Path]):
            语料文件路径，支持单个路径或路径列表；内部会统一规范化为 ``list[Path]`` 并按给定顺序迭代。
        fields (str | list[str] | None):
            需要提取的文本字段名，支持单个字段或字段列表。
            当传入多个字段时，会按文件内原始顺序依次展开各字段文本，而不是拼接为一条样本。
            当传入 None 时代表提取所有字段文本，适用于纯文本文件或需要全字段内容的场景。
        **kwargs:
            batch_size (int):
                仅对 ``parquet`` 生效，每次读取的批大小，默认 ``2048``。
            return_file_path (bool):
                是否同时返回文本来源文件路径。默认 ``False``。

    Yields:
        str | tuple[str, Path]:
            - 当 ``return_file_path=False`` 时，逐条产出文本 ``str``；
            - 当 ``return_file_path=True`` 时，逐条产出 ``(text, file_path)``。

    Notes:
        - 该函数会过滤空内容，以保持与 bucket key 计算阶段一致，避免样本行错位；
        - ``parquet`` 路径下会跳过空字符串、纯空白字符串和缺失值；
        - ``jsonl`` 路径下会对字段值执行 ``strip()``，空结果会被跳过；
        - 若 ``corpus_file_format`` 不受支持，将抛出 ``ValueError``。
    """
    if isinstance(fields, str): fields = [fields]
    files_path: list[Path] = path_normalize(path=files_path)
    batch_size = kwargs.pop('batch_size', 2048)
    return_file_path = kwargs.pop('return_file_path', False)

    for file_path in files_path:
        file_format = file_path.suffix.lstrip('.').lower()
        if file_format == 'parquet':
            for batch in iter_parquet_batches(
                parquet_path=file_path,
                batch_size=batch_size,
                text_field=fields,
            ):
                for sample in batch.stack().replace(r'^\s*$', numpy.nan, regex=True).dropna().reset_index(drop=True):
                    yield (str(sample), file_path) if return_file_path else str(sample)
        elif file_format == 'jsonl':
            for row in iter_jsonl_rows(file_path=file_path):
                fields = fields or row.keys()
                for field in fields:
                    content = row.get(field, '').strip()
                    if not content: continue
                    yield (str(content), file_path) if return_file_path else str(content)
        else: raise ValueError(f"Unsupported file format: {file_format}")
