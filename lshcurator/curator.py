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
from multiprocessing.connection import wait
from pathlib import Path
from time import sleep
from typing import Literal, Iterator

import numpy

from .config import CuratorConfig, DeduperConfig
from .deduper import Deduper
from .utils.readers import iter_corpus_texts


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
        self._compute_bucket_keys(
            corpus_files_path=_corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            **kwargs
        )

        if len(self.bucket_keys) == 0:
            print("No bucket keys were computed.")
            return # 没有 bucket keys，直接返回空迭代器
        bucket_keys = numpy.concatenate(self.bucket_keys)  # 将所有 bucket keys 合并成一个大数组
        print(f"Total bucket keys computed: {len(bucket_keys)}")
        self.bucket_keys.clear()  # 清空全局 bucket keys 列表，释放内存

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

    def _compute_bucket_keys(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> None:
        """计算 bucket keys"""


        # 创建多进程
        max_workers = min(self.config.max_workers, len(corpus_files_path))
        for file_idx, file_path in enumerate(corpus_files_path):
            while self.worker_count >= max_workers:
                if not self._is_running: return None
                # 先取镜像避免被报告处理进程 delete worker_slot 后访问空槽位导致 KeyError
                worker_slots = self.worker_slots(snapshot=True)
                sentinels = [slot.process.sentinel for slot in worker_slots]
                if sentinels: wait(sentinels)  # 阻塞等待任一 worker 进程完成
                sleep(1)  # 等待一段时间让报告处理线程处理完成报告，确保资源被正确回收


        # 确保所有进程均运行完毕后再返回
        while self.worker_count > 0: sleep(1)

