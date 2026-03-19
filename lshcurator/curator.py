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
from typing import Literal, Iterator, cast, Iterable

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

        self.deduper: Deduper

    def init_deduper(self, bucket_keys: numpy.ndarray[numpy.uint64]) -> Deduper:
        if bucket_keys.ndim == 2: bucket_keys = bucket_keys.reshape(-1)  # 展平为一维数组，包含所有 band keys
        elif bucket_keys.ndim != 1: raise ValueError(f"Expected bucket_keys to be either a 1D array with shape (num_keys,) or a 2D array with shape (num_samples, bands), but got shape {bucket_keys.shape}.")

        self.deduper = Deduper(
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
        return self.deduper

    @staticmethod
    def _select_deduper_bucket_keys(
        bucket_keys: numpy.ndarray[numpy.uint64],
        filter_freq: int = 1,
        sorted: bool = True,
    ) -> tuple[numpy.ndarray[numpy.uint64], numpy.ndarray[numpy.bool]]:
        """
        根据 bucket key 频率过滤掉低频 bucket keys，仅保留更高频的 bucket keys 用于后续 deduplication。
        Args:
            bucket_keys: 形状为 (num_samples, bands) ,每行对应一个样本的所有 band keys 的二维数组。
            filter_freq: 频率阈值，默认为 1，表示过滤掉所有出现次数小于等于该阈值的 bucket keys。必须为非负整数。
            sorted: 是否对保留的 bucket keys 进行升序排序，默认为 True。
        Returns:
            deduper_bucket_keys: 经过频率过滤后用于 deduplication 的 bucket keys集合，形状为 (num_deduper_keys,) 的一维数组。
            should_dedupe_row_mask: 形状为 (num_samples,) 的布尔数组，表示每行是否命中任一保留的 bucket key，True 表示该行需要进入后续 deduplication，False 表示该行不需要 deduplication。
        """
        if bucket_keys.ndim != 2: raise ValueError(f'Expected bucket_keys to be a 2D array with shape (num_samples, bands), but got shape {bucket_keys.shape}.')
        num_samples, bands = bucket_keys.shape
        if num_samples == 0: return numpy.empty(0, dtype=numpy.uint64), numpy.empty(0, dtype=numpy.bool)
        if filter_freq < 0: raise ValueError(f"filter_freq must be a non-negative integer, but got {filter_freq}.")

        unique_keys, inverse, key_counts = numpy.unique(bucket_keys, return_inverse=True, return_counts=True, sorted=sorted)
        keep_unique_mask = key_counts > filter_freq  # 标记需要保留的 bucket keys
        deduper_bucket_keys: numpy.ndarray[numpy.uint64] = unique_keys[keep_unique_mask].astype(numpy.uint64, copy=False)  # 仅保留通过频率筛选的 bucket keys，作为第二阶段 Deduper 的候选 key 集合
        should_dedupe_row_mask = keep_unique_mask[inverse].reshape(bucket_keys.shape).any(axis=-1)  # 只要一行命中任一保留的 bucket key，就进入后续 deduplication
        return deduper_bucket_keys, should_dedupe_row_mask

    def process_corpus(
        self,
        files_path: str | Path | list[str | Path] | None = None,
        iterable: Iterable | None = None,
        fields: str | list[str] | None = None,
        filter_low_freq_bucket_keys: int = 1,
        **kwargs
    ) -> Iterator[tuple[str, bool]]:
        """
        处理语料的主流程接口
        Args:
            files_path: 语料文件路径，支持单个路径或路径列表。
            iterable: 通用语料源，支持可重复迭代的 Iterable。
            fields: 语料文本字段名称，支持单个字段或字段列表；当 corpus_source 直接产出 str 时可省略。
            filter_low_freq_bucket_keys:
                低频 bucket key 过滤阈值，默认 1。
                会过滤掉出现次数小于等于该阈值的 bucket keys，仅保留更高频的 bucket keys 用于后续 deduplication。
            kwargs:
                batch_size: 处理语料时每个批次的文本数量，仅在 corpus_file_format='parquet' 时有效，默认为 2048
        Returns:
            生成器逐条返回文本和去重结果的元组 (text, not_duplicated)，其中 not_duplicated 为 True 表示该文本被认为是唯一的，False 表示该文本被认为是重复的
        """
        if self.config.similarity_threshold is None: raise ValueError("similarity_threshold must be set in config for deduplication")
        if not isinstance(filter_low_freq_bucket_keys, int) or filter_low_freq_bucket_keys < 0: raise ValueError("filter_low_freq_bucket_keys must be a non-negative integer.")

        if iterable is not None:
            if not isinstance(iterable, Iterable): raise ValueError("iterable is not an iterable.")
            raise NotImplementedError("iterable input is not yet supported in this version. Please use files_path and fields to specify the corpus.")
        elif files_path is not None: files_path: list[Path] = path_normalize(path=files_path)
        else: raise ValueError("files_path or iterable must be provided.")

        # 1. Compute bucket keys
        bucket_keys, file_bucket_pos_mapping = self.compute_bucket_keys(
            files_path=files_path,
            fields=fields,
            key_layout='row_bands',  # 计算 bucket keys 时使用 'row_bands' 布局，得到 shape=(num_samples, bands) 的二维数组，方便后续按样本处理 bucket keys
            **kwargs
        )

        deduper_bucket_keys, should_dedupe_row_mask = self._select_deduper_bucket_keys(
            bucket_keys=bucket_keys,
            filter_freq=filter_low_freq_bucket_keys,
        )

        if deduper_bucket_keys.size == 0 or not should_dedupe_row_mask.any():
            print("All bucket keys are low frequency and will be filtered out. No deduplication will be performed.")
            return # 没有需要进行 deduplication 的 bucket keys，直接返回空迭代器

        # 2. 基于计算得到的 bucket keys 进行 deduplication，统计去重结果
        self.init_deduper(bucket_keys=deduper_bucket_keys)

        if should_dedupe_row_mask.all():  # 所有样本都需要 deduplication，直接逐条处理无需额外的文件行位置映射逻辑
            for text in iter_corpus_texts(
                files_path=files_path,
                fields=fields,
                **kwargs
            ):
                yield text, self.deduper(text)
            return

        file_should_dedupe_masks: dict[Path, numpy.ndarray] = {}
        file_row_positions: dict[Path, int] = {}
        for file_path in files_path:
            bucket_key_chunks = file_bucket_pos_mapping.get(file_path, [])
            if len(bucket_key_chunks) > 0:
                file_should_dedupe_masks[file_path] = numpy.concatenate([
                    should_dedupe_row_mask[chunk.start_position:chunk.start_position + chunk.size]
                    for chunk in bucket_key_chunks
                ])
                file_row_positions[file_path] = 0  # 初始化每个文件的行位置计数器

        for text, file_path in iter_corpus_texts(
            files_path=files_path,
            fields=fields,
            return_file_path=True,
            **kwargs
        ):
            file_mask = file_should_dedupe_masks.get(file_path, None)
            if file_mask is None:  # 该文件没有任何需要 deduplication 的行，直接标记为唯一
                yield text, True
                continue

            row_position = file_row_positions[file_path]
            file_row_positions[file_path] = row_position + 1

            if file_mask[row_position]: yield text, self.deduper(text)  # 是否唯一
            else: yield text, True  # 该行没有命中任何需要 deduplication 的 bucket keys，直接标记为唯一

        for file_path, file_mask in file_should_dedupe_masks.items():
            if file_row_positions[file_path] != file_mask.size:
                raise ValueError(
                    f"Reader output is out of sync with computed bucket keys for file {file_path}. "
                    f"consumed_rows={file_row_positions[file_path]}, expected_rows={file_mask.size}"
                )

    def compute_bucket_keys(
        self,
        files_path: str | Path | list[str | Path],
        fields: str | list[str] | None = None,
        **kwargs
    ) -> tuple[numpy.ndarray[numpy.uint64], dict[Path, list[BucketKeyChunk]]]:
        """
        计算 bucket keys
        Args:
            files_path (str | Path | list[str | Path]): 语料文件路径，支持单个路径或路径列表
            fields (str | list[str] | None): 语料文本字段名称，支持单个字段或字段列表；None 代表所有字段。
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
            files_path=files_path,
            fields=fields,
            **kwargs
        ), bucket_worker_manager.file_bucket_pos_mapping
