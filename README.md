# lshcurator

`lshcurator` 是一个面向**大规模文本语料近重复挖掘与去重**的 Python 工具库，核心仍然建立在 **MinHash + LSH** 之上，但工程实现上不再沿用“边算签名边维护全量桶状态”的传统路线，而是采用**两阶段（2-stage）管线**：

1. **先计算并收集全局 bucket key**；
2. **再只针对被筛选出来的高频 key 做定向去重**。

这种设计更适合多文件、流式、大规模语料场景，能够显著降低唯一样本占多数时的无效桶维护成本，并更充分地利用多核 CPU 的并行计算能力。

---

## 目录

- [设计目标](#设计目标)
- [相比传统 LSH 去重的改进](#相比传统-lsh-去重的改进)
- [适用场景](#适用场景)
- [项目结构与公开 API](#项目结构与公开-api)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细用法](#详细用法)
  - [1. 使用 Curator 直接处理语料](#1-使用-curator-直接处理语料)
  - [2. 仅执行第一阶段：计算 bucket keys](#2-仅执行第一阶段计算-bucket-keys)
  - [3. 手动初始化 Deduper](#3-手动初始化-deduper)
  - [4. 低层接口：Bucket](#4-低层接口bucket)
- [注意事项](#注意事项)
- [构建 wheel](#构建-wheel)
- [License](#license)

---

## 设计目标

`lshcurator` 的目标不是重新发明 MinHash/LSH，而是针对工程实践中的几个痛点做优化：

- **大规模语料下桶状态膨胀**：传统做法会为几乎所有 key 建桶，即使绝大多数样本根本没有重复；
- **热点桶拖慢整体性能**：模板化数据、结构化文本会形成极热 bucket，候选集合急剧增大；
- **多文件流式处理不友好**：一边读一边建复杂桶状态，既吃内存，也难并行；
- **CPU 利用不足**：签名计算本质上适合并行，但传统单阶段流水常把“计算”和“状态维护”耦合在一起。

因此，本项目更关注：

- 面向流式数据的可扩展性；
- 面向热点 key 的定向 dedupe；
- 有界代表元（Bounded Representatives）策略；
- 低开销数据处理能力；
- 多进程数据并行计算能力。

对于大量本身唯一的样本，最浪费的并不是算 MinHash，而是：

- 建桶；
- 维护桶内状态；
- 反复做候选比较。

`lshcurator` 的做法是先统计，再聚焦热点密集区域，从而让资源主要花在真正可能重复的样本上。

---

## 相比传统 LSH 去重的改进

| 维度      | 传统单阶段 LSH | `lshcurator` 当前方案 |
|---------|-----------|-------------------|
| 签名计算与建桶 | 耦合        | 解耦为两阶段            |
| 桶维护范围   | 全量 key    | 仅确实存在重复的 keys     |
| 语料读取方式  | 常需配合额外缓存  | 全流式读取、按需批量处理      |
| 唯一样本成本  | 仍可能参与桶维护  | 大量样本直接跳过          |
| 并行化     | 无法实现      | 通过并行有效提升效率        |
| 内存压力    | 容易随着桶数量膨胀 | 集中在热点区域           |
| 工程扩展性   | 复杂        | 更易做分阶段优化          |

---

## 适用场景

`lshcurator` 特别适合：

- 大规模语料清洗；
- 多来源文本近重复过滤；
- 模板化内容、新闻聚合、网页正文等语料的相似样本清理；
- 想要在保留 MinHash + LSH 思路的前提下，提高工程吞吐与可扩展性的项目。

---

## 项目结构与公开 API

当前从 `lshcurator/__init__.py` 公开导出的核心对象包括：

- `Bucket`
- `BucketConfig`
- `BucketWorkerManager`
- `BucketWorkerManagerConfig`
- `Curator`
- `CuratorConfig`
- `Deduper`
- `DeduperConfig`

其中推荐的使用层级如下：

### 推荐优先级

1. **高层接口：`Curator`**
   - 适合直接处理语料文件；
   - 自动执行两阶段流程；
   - 是当前最推荐的入口。

2. **中层接口：`Curator.compute_bucket_keys(...)` + `Curator.init_deduper(...)`**
   - 适合你想自行观察第一阶段结果，或做自定义调度时使用。

3. **低层接口：`Bucket` / `Deduper`**
   - 更适合实验、调试、局部组件复用；
   - 需要自行管理阶段边界和数据路由。

---

## 安装

### 环境要求

- Python `>= 3.12`

### 从 PyPI 安装

安装核心功能：

```bash
pip install lshcurator
```

如果你需要处理 Parquet：

```bash
pip install "lshcurator[pd]"
```

其中 `[pd]` extra 会额外安装：

- `pandas`
- `pyarrow`

### 从源码安装（开发模式）

仅安装核心能力：

```bash
pip install -e .
```

如果你需要处理 Parquet：

```bash
pip install -e ".[pd]"
```

`[pd]` extra 会额外安装：

- `pandas`
- `pyarrow`

如果你只是使用库本身，优先推荐直接从 PyPI 安装；如果你需要修改源码或参与开发，再使用 editable install。

---

## 快速开始

最简单的使用方式，是通过 `Curator` 直接对 `jsonl` 语料做去重：

```python
from lshcurator import Curator, CuratorConfig

curator = Curator(CuratorConfig(
    shingle_k=5,
    shingle_step=1,
    bands=8,
    rows_per_band=4,
    similarity_threshold=0.9,
    compute_mode='char',
    max_workers=4,
))

for text, keep in curator.process_corpus(
    files_path=["./data/a.jsonl", "./data/b.jsonl"],
    fields="text",
    filter_low_freq_bucket_keys=1,
):
    if keep:
        print(text)
```

其中：

- `keep=True` 表示该文本被认为应保留；
- `keep=False` 表示该文本被认为是重复样本，应丢弃；
- `filter_low_freq_bucket_keys=1` 表示过滤掉出现次数小于等于 1 的 key，也就是默认过滤 singleton key。

---

## 详细用法

### 1. 使用 Curator 直接处理语料

这是当前最推荐的方式。

整个处理过程保持**全流式读取**：JSONL 会逐行读取，Parquet 会按 batch 流式读取，`Curator` 不要求先把整份语料物化到内存中。

#### JSONL 示例

假设你的 `jsonl` 文件内容形如：

```json lines
{"text": "第一条文本"}
{"text": "第二条文本"}
{"text": "第二条文本"}
```

可以这样调用：

```python
from lshcurator import Curator, CuratorConfig

config = CuratorConfig(
    shingle_k=5,
    shingle_step=1,
    bands=8,
    rows_per_band=4,
    similarity_threshold=0.9,
    compute_mode='char',
    max_workers=2,
    chunk_elements=1_000_000,
    max_representatives_per_bucket=32,
)

curator = Curator(config)

results = curator.process_corpus(
    files_path="./corpus.jsonl",
    fields="text",
    filter_low_freq_bucket_keys=1,
)

for text, keep in results:
    print({"text": text, "keep": keep})
```

#### 多字段展开

如果一个 JSONL 记录中有多个文本字段，可以传列表：

```python
results = curator.process_corpus(
    files_path="./corpus.jsonl",
    fields=["title", "content"],
)
```

当前实现会**按字段顺序逐条展开文本**，而不是把多个字段拼成一条样本。

#### Parquet 示例

处理 Parquet 需要先安装对应 extra：

```bash
pip install "lshcurator[pd]"
```

如果你是在源码仓库中进行开发，也可以使用：

```bash
pip install -e ".[pd]"
```

然后可以这样调用：

```python
results = curator.process_corpus(
    files_path=["./data/part-0001.parquet", "./data/part-0002.parquet"],
    fields="text",
    batch_size=4096,
    filter_low_freq_bucket_keys=1,
)

for text, keep in results:
    if keep:
        pass
```

#### 参数说明

`CuratorConfig` 中常用参数：

- `shingle_k`：shingle 长度；
- `shingle_step`：滑窗步长；
- `bands`：LSH 的 band 数量；
- `rows_per_band`：每个 band 的行数；
- `compute_mode`：`'char'` 或 `'byte'`；
- `similarity_threshold`：MinHash 相似度判定阈值；
- `max_workers`：阶段 1 并行 worker 数；
- `chunk_elements`：共享内存块容量（按 `uint64` 元素个数计）；
- `max_representatives_per_bucket`：每个 bucket 最多保留的代表元数量。

`process_corpus(...)` 中常用参数：

- `files_path`：单文件路径或路径列表；
- `fields`：字段名或字段名列表；
- `filter_low_freq_bucket_keys`：低频 key 过滤阈值。

---

### 2. 仅执行第一阶段：计算 bucket keys

如果你想单独观察第一阶段产物，可以调用：

```python
from lshcurator import Curator, CuratorConfig

curator = Curator(CuratorConfig(
    shingle_k=5,
    shingle_step=1,
    bands=4,
    rows_per_band=4,
    similarity_threshold=0.9,
))

bucket_keys, file_mapping = curator.compute_bucket_keys(
    files_path=["./data/a.jsonl", "./data/b.jsonl"],
    fields="text",
    key_layout='row_bands',
)

print(bucket_keys.shape)      # 例如: (num_samples, bands)
print(bucket_keys.dtype)      # uint64
print(file_mapping)           # {Path(...): [BucketKeyChunk(...), ...]}
```

当前 `process_corpus(...)` 内部就是基于这一步的结果继续做筛选与第二阶段路由。

---

### 3. 手动初始化 Deduper

如果你已经自己拿到了筛选后的 key 集合，也可以手动初始化第二阶段：

```python
import numpy
from lshcurator import Curator, CuratorConfig

curator = Curator(CuratorConfig(
    shingle_k=5,
    shingle_step=1,
    bands=4,
    rows_per_band=4,
    similarity_threshold=0.9,
))

selected_keys = numpy.array([1001, 1002, 1003], dtype=numpy.uint64)
deduper = curator.init_deduper(selected_keys)

print(deduper("hello"))
print(deduper("hello"))
```

`Curator.init_deduper(...)` 当前支持传入：

- 1D `numpy.uint64` 数组：表示已经整理好的 key 列表；
- 2D `row_bands` 数组：会自动展平后传给 `Deduper`。

适合以下场景：

- 你想手动控制第一阶段筛选逻辑；
- 你想做自定义批处理或实验；
- 你希望独立复用 `Deduper` 而不是直接走 `process_corpus(...)`。

---

### 4. 低层接口：Bucket

`Bucket` 更适合小规模实验或调试 band key 生成行为。

```python
import numpy
from lshcurator import Bucket, BucketConfig

bucket = Bucket(BucketConfig(
    shingle_k=5,
    shingle_step=1,
    bands=4,
    rows_per_band=4,
    compute_mode='char',
    key_layout='row_bands',
))

bucket.insert("hello world")
bucket.insert("hello world")
bucket.insert("another text")

keys = bucket.extract_keys()
print(keys.shape)   # (num_samples, bands)
print(keys.dtype)   # uint64
```

如果你使用：

```python
key_layout='flat'
```

则 `extract_keys()` 返回 1D 数组；若使用：

```python
key_layout='row_bands'
```

则返回二维数组，每一行对应一个样本的所有 band key。

---

## 注意事项

以下内容与当前实现强相关，建议在正式使用前了解：

### 1. 空文本会被过滤

`iter_corpus_texts(...)` 会过滤空的、缺失的内容，这会影响样本行数和对齐关系，因此如果你自己扩展读取逻辑，必须保证和第一阶段一致的过滤/顺序规则。

### 2. `process_corpus(...)` 当前不会回放“全部保留”的空筛选结果

当前实现中，如果阶段 1 之后没有任何 key 通过筛选，`process_corpus(...)` 会直接返回**空迭代器**，因为此时没有任何样本需要被处理。

### 3. `filter_low_freq_bucket_keys=0` 与 `1` 当前等价

当 filter_low_freq_bucket_keys 设置为 `0` 时从数学上完全没意义，且等同于本项目退化到传统实现，不仅不会提升性能，反而会增加不必要的开销。

### 4. `Deduper.__call__` 的返回值语义

- `True`：保留该样本；
- `False`：认为是重复样本，应丢弃。

`Curator.process_corpus(...)` 中的第二个返回值与此保持一致。

---

## 构建 wheel

仓库中提供了：

- `scripts/build_wheel.py`

可在已激活虚拟环境后执行。

---

## License

本项目采用 Apache License 2.0 许可证，允许在遵守许可证条款的前提下自由使用、修改和分发代码。
