# lshcurator

lshcurator 是一款面向大规模语料的近重复（near-duplicate）挖掘与去重工具，本质上依旧是基于 MinHash + LSH 的实现。
主要改善了传统 LSH 去重在处理大规模语料上常见的性能瓶颈和硬件利用效率的问题。

## 背景

传统 LSH 去重的工程的步骤依次为：

1. 对每个文本片段计算 MinHash 签名；
2. 按 band/row 切分；
3. 将 band key 映射到桶；
4. 桶内维护所有签名或索引；
5. 查询候选并验证相似度。

会遇到的典型问题是：

- **内存瓶颈**：在大规模语料上维护大量桶占用大量内存，尤其是在大部分样本唯一的情况下；
- **计算瓶颈**：模板化的语料会制造大量热点桶，桶内候选急剧膨胀，导致效率下降；
- **硬件利用率低**：跨文件、多语料需要流式处理，难以利用多核 CPU 进行加速。

## 本项目的核心优化思路

### 1) 解偶 MinHash 计算和 LSH 桶维护的流程，专注高相似密集区

统计阶段只生成和收集 band key，不维护桶以及候选列表，去重阶段只对 hot keys 建桶和维护代表样本。

收益在于可充分利用多核 CPU 并行运算性能同步统计多份语料，同时桶结构规模从“全量 keys”降低到“热点 keys 子集”，避免了大规模唯一样本带来的性能瓶颈和内存压力。

### 2) 扁平化 band key

将传统 bytes key 压缩为无符号双精度整数（uint64）指纹，降低 key 对象和内容的维护开销。

> 指纹基于哈希算法不可避免存在极小的碰撞概率，本项目已做碰撞防误伤处理，如担心碰撞风险可通过提升 digest 位数（如 uint128）来进一步降低碰撞可能性，但会增加单 key 内存占用和计算开销，需根据实际需求权衡选择。

### 3) 有界代表元（Bounded Representatives）

通过限制每个桶的代表样本数量，避免了极热桶“无限增长”的问题。

### 4) 引入 numpy 进行高效的批量计算和数据处理

避免使用 Python 原生数据结构，节省海量 dict 和 list 的内存开销，同时提升性能和稳定性。


## 使用方式

### 生成和收集 band key

1. 单线程实现

    ```python
    from lshcurator import Bucket
    
    # 初始化 Bucket 实例
    bucket = Bucket(bands=16, rows_per_band=4, shingle_k=5, shingle_step=1, compute_mode='char')
    
    # 添加文本样本
    bucket.insert(text="This is a sample text for LSH de-duplication.")
    # 批量添加文本样本
    texts = ["Another sample text.", "Yet another example for testing."]
    bucket.batch_insert(texts=texts)
    
    # 导出大于或等于指定频次的 band keys
    hot_keys: numpy.ndarray = bucket.extract_keys(min_hit_count=2)
    ```

2. 多线程实现

    ```python
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from lshcurator import Bucket
    from pathlib import Path
    from typing import Literal


    def compute_hot_keys_for_corpus_file(
        corpus_file_path: Path,
        shingle_k: int,
        shingle_step: int,
        bands: int,
        rows_per_band: int,
        compute_mode: Literal['char', 'byte'],
    ):
        bucket = Bucket(
            shingle_k=shingle_k,
            shingle_step=shingle_step,
            bands=bands,
            rows_per_band=rows_per_band,
            compute_mode=compute_mode
        )

        for text in iter_corpus(file_path=corpus_file_path):  # 迭代读取语料文件中的文本样本（函数实现略）
            bucket.insert(text)

        hot_keys: numpy.ndarray = bucket.extract_keys(min_hit_count=2)  # 只导出出现两次以及以上的 band key
        # 后续步骤略


    # 主进程中使用 ProcessPoolExecutor 来并行处理多个语料文件
    # 假设 corpus_files 是一个包含多个语料文件路径的列表，max_jobs 是并行处理的最大进程数
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        futures = [
            executor.submit(
                compute_hot_keys_for_corpus_file,
                corpus_file_path=file_path,
                shingle_k=5,
                shingle_step=1,
                bands=16,
                rows_per_band=8,
                compute_mode='char',
            ) for file_path in corpus_files
        ]
        for future in as_completed(futures):
            future.result()
    ```

## License

本项目采用 Apache License 2.0 许可证，允许用户在遵守许可证条款的前提下自由使用、修改和分发代码。
