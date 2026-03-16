# Copyright 2026 Morton Li. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import Literal, Iterator, cast

import numpy

from .config import CuratorConfig, DeduperConfig, BucketConfig, BucketWorkerManagerConfig
from .deduper import Deduper
from .utils.normalizations import path_normalize
from .utils.readers import iter_corpus_texts
from .utils.types import BucketKeyChunk
from .workers.bucket_worker import BucketWorkerManager


class Curator:
    def __init__(self, config: CuratorConfig):
        self.config = config

    def process_corpus(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        filter_low_freq_bucket_keys: int | None = None,
        **kwargs
    ) -> Iterator[tuple[str, bool]]:
        """
        处理语料的主流程接口
        Args:
            corpus_files_path: 语料文件路径，支持单个路径或路径列表
            corpus_field_name: 语料文本字段名称，支持单个字段或字段列表
            corpus_file_format: 语料文件格式，支持 'parquet' 和 'jsonl'
            filter_low_freq_bucket_keys: 是否过滤低频 bucket keys，传入频率阈值（例如 10）表示过滤掉出现次数低于该阈值的 bucket keys，保留高频 bucket keys 用于后续 deduplication
            kwargs:
                batch_size: 处理语料时每个批次的文本数量，仅在 corpus_file_format='parquet' 时有效，默认为 2048
        Returns:
            生成器逐条返回文本和去重结果的元组 (text, not_duplicated)，其中 not_duplicated 为 True 表示该文本被认为是唯一的，False 表示该文本被认为是重复的
        """
        if self.config.similarity_threshold is None: raise ValueError("similarity_threshold must be set in config for deduplication")
        if filter_low_freq_bucket_keys is not None and (not isinstance(filter_low_freq_bucket_keys, int) or filter_low_freq_bucket_keys < 0):
            raise ValueError("filter_low_freq_bucket_keys must be a non-negative integer or None")

        corpus_files_path: list[Path] = path_normalize(path=corpus_files_path)

        # 1. Compute bucket keys
        bucket_keys, file_bucket_pos_mapping = self.compute_bucket_keys(
            corpus_files_path=corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            key_layout='row_bands',  # 计算 bucket keys 时使用 'row_bands' 布局，得到 shape=(num_samples, bands) 的二维数组，方便后续按样本处理 bucket keys
            **kwargs
        )

        if bucket_keys.ndim != 2:
            raise ValueError(f"Expected bucket_keys to be a 2D array with shape (num_samples, bands), but got shape {bucket_keys.shape}. Ensure that compute_bucket_keys is called with key_layout='row_bands'.")
        n_row, n_bands = bucket_keys.shape
        if n_row == 0:
            print("No bucket keys were computed.")
            return # 没有 bucket keys，直接返回空迭代器
        print(f"Total bucket keys computed: {n_row * n_bands}")

        unique_keys, key_counts = numpy.unique(bucket_keys, return_counts=True)
        freq: int = filter_low_freq_bucket_keys if filter_low_freq_bucket_keys is not None else 1  # 频率阈值，默认为 1 表示过滤掉所有只出现一次的 bucket keys
        low_freq_keys = unique_keys[key_counts <= freq]  # 出现次数为 freq 的 bucket keys 的掩码
        should_dedupe_row_mask = ~numpy.isin(bucket_keys, low_freq_keys).all(axis=1)  # bucket_keys 中所有 band keys 都是 low_freq_keys 的 row 对应的掩码，这些行无需后续 deduplication，相反的掩码表示需要进行 deduplication 的行
        bucket_keys_for_deduper = bucket_keys[should_dedupe_row_mask]  # 把需要进行 deduplication 的行对应的 bucket keys 提取出来，形状仍然是 (num_dedupe_rows, bands)
        deduper_bucket_keys = bucket_keys_for_deduper.reshape(-1)  # [num_dedupe_rows * bands] 展平为一维数组，包含所有需要进行 deduplication 的 bucket keys，准备后续过滤和排序

        if deduper_bucket_keys.size == 0 or not should_dedupe_row_mask.any():
            for text in iter_corpus_texts(
                corpus_files_path=corpus_files_path,
                corpus_field_name=corpus_field_name,
                corpus_file_format=corpus_file_format,
                **kwargs
            ):
                yield text, True
            return

        # 2. 基于计算得到的 bucket keys 进行 deduplication，统计去重结果
        deduper = Deduper(
            bucket_keys=deduper_bucket_keys,
            config=DeduperConfig(
                bands=self.config.bands,
                rows_per_band=self.config.rows_per_band,
                shingle_k=self.config.shingle_k,
                shingle_step=self.config.shingle_step,
                similarity_threshold=self.config.similarity_threshold,
                compute_mode=self.config.compute_mode,
                max_representatives_per_bucket=self.config.max_representatives_per_bucket,
            )
        )

        for text in iter_corpus_texts(
            corpus_files_path=corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            **kwargs
        ):
            # 是否唯一
            not_duplicated: bool = deduper(text)
            yield text, not_duplicated  # 生成器逐条返回文本和去重结果

    def compute_bucket_keys(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> tuple[numpy.ndarray[numpy.uint64], dict[Path, list[BucketKeyChunk]]]:
        """
        计算 bucket keys
        Args:
            corpus_files_path (str | Path | list[str | Path]): 语料文件路径，支持单个路径或路径列表
            corpus_field_name (str | list[str]): 语料文本字段名称，支持单个字段或字段列表
            corpus_file_format (Literal['parquet', 'jsonl']): 语料文件格式，支持 'parquet' 和 'jsonl'
            kwargs:
                batch_size: 处理语料时每个批次的文本数量，仅在 corpus_file_format='parquet' 时有效，默认为 2048
                key_layout:
                    计算 bucket keys 时的存储布局，支持 'flat' 和 'row_bands' 两种模式，默认为 'row_bands'。
                    - 'flat' 模式将所有 bucket keys 存储在一个一维数组中，适合后续需要对所有 bucket keys 进行全局排序的场景；
                    - 'row_bands' 模式将 bucket keys 存储在一个二维数组中，每行对应一个样本的所有 band keys，适合后续需要按样本处理 bucket keys 的场景。
        Returns:
            numpy.ndarray[numpy.uint64]: 计算得到的 bucket keys 数组，类型为 numpy.uint64，形状取决于 key_layout 的选择。当 key_layout='flat' 时，返回 shape=(num_keys,) 的 1D 数组，每个元素是一个 bucket key；当 key_layout='row_bands' 时，返回 shape=(num_samples, bands) 的 2D数组，每行对应一个样本的所有 band keys，列数等于 bands。
            dict[Path, list[BucketKeyChunk]]: 文件到 bucket_keys 位置区间的映射；flat 下单位为 key，row_bands 下单位为 row。
        """
        key_layout = kwargs.pop('key_layout', 'row_bands')
        if key_layout not in {'flat', 'row_bands'}: raise ValueError(f"Invalid key_layout: {key_layout}")

        bucket_worker_manager = BucketWorkerManager(
            bucket_config=BucketConfig(
                shingle_k=self.config.shingle_k,
                shingle_step=self.config.shingle_step,
                bands=self.config.bands,
                rows_per_band=self.config.rows_per_band,
                compute_mode=self.config.compute_mode,
                key_layout=cast(Literal['flat', 'row_bands'], key_layout)
            ),
            bucket_worker_manager_config=BucketWorkerManagerConfig(
                max_workers=self.config.max_workers,
                chunk_elements=self.config.chunk_elements,  # 每次分配共享内存的元素数量
            )
        )
        return bucket_worker_manager.run(
            corpus_files_path=corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            **kwargs
        ), bucket_worker_manager.file_bucket_pos_mapping
