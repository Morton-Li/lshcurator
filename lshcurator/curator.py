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
from typing import Literal, Iterator

import numpy

from .config import CuratorConfig, DeduperConfig, BucketConfig, BucketWorkerManagerConfig
from .deduper import Deduper
from .utils.readers import iter_corpus_texts
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
        if self.config.similarity_threshold is None:
            raise ValueError("similarity_threshold must be set in config for deduplication")

        _corpus_files_path = corpus_files_path
        if isinstance(_corpus_files_path, str): _corpus_files_path = [Path(_corpus_files_path)]
        elif isinstance(_corpus_files_path, Path): _corpus_files_path = [_corpus_files_path]
        elif isinstance(_corpus_files_path, list):
            for idx, path in enumerate(_corpus_files_path):
                if isinstance(path, str): _corpus_files_path[idx] = Path(path)
        else: raise ValueError(f"Invalid corpus_files_path type: {type(corpus_files_path)}")

        # 1. Compute bucket keys
        bucket_keys = self.compute_bucket_keys(
            corpus_files_path=_corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            **kwargs
        )

        bucket_keys_count = len(bucket_keys)
        if bucket_keys_count == 0:
            print("No bucket keys were computed.")
            return # 没有 bucket keys，直接返回空迭代器
        print(f"Total bucket keys computed: {bucket_keys_count}")

        if filter_low_freq_bucket_keys is not None:
            unique_keys, key_counts = numpy.unique(bucket_keys, return_counts=True)  # unique 会返回有序结果无需 sort
            filtered_keys = unique_keys[key_counts >= filter_low_freq_bucket_keys]
            bucket_keys = filtered_keys  # 更新 bucket keys 为过滤后的结果
            if len(bucket_keys) == 0:
                print(f"No bucket keys meet the frequency threshold of {filter_low_freq_bucket_keys}. Consider lowering the threshold.")
                return # 没有 bucket keys，直接返回空迭代器
        else: bucket_keys.sort()  # 直接排序全局 bucket keys 数组，确保其有序且升序，满足后续 deduplication 中 searchsorted 的正确性要求

        # 2. 基于计算得到的 bucket keys 进行 deduplication，统计去重结果
        deduper = Deduper(
            bucket_keys=bucket_keys,
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
            corpus_files_path=_corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            **kwargs
        ):
            # 是否唯一
            not_duplicated: bool = deduper(text)
            yield text, not_duplicated  # 生成器逐条返回文本和去重结果

    def compute_bucket_keys(
        self,
        corpus_files_path: list[Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> numpy.ndarray[numpy.uint64]:
        """计算 bucket keys"""
        bucket_worker_manager = BucketWorkerManager(
            bucket_config=BucketConfig(
                shingle_k=self.config.shingle_k,
                shingle_step=self.config.shingle_step,
                bands=self.config.bands,
                rows_per_band=self.config.rows_per_band,
                compute_mode=self.config.compute_mode,
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
        )
