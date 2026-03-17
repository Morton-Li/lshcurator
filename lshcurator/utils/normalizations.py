from pathlib import Path


def path_normalize(path: str | Path | list[str | Path]) -> list[Path]:
    """
    对路径进行规范化处理，支持单个路径或路径列表输入，返回规范化后的路径列表。
    Args:
        path (str | Path | list[str | Path]): 输入路径，可以是单个字符串路径、单个 Path 对象，或包含字符串路径和 Path 对象的列表。
    Returns:
        list[Path]: 规范化后的路径列表，每个元素都是一个 Path 对象。
    Raises:
        ValueError: 如果输入类型不合法，或者列表中包含非字符串和非 Path 类型的元素。
    """
    if isinstance(path, str): return [Path(path)]
    elif isinstance(path, Path): return [path]
    elif isinstance(path, list):
        normalized_paths = []
        for idx, p in enumerate(path):
            if isinstance(p, str): normalized_paths.append(Path(p))
            elif isinstance(p, Path): normalized_paths.append(p)
            else: raise ValueError(f"Invalid path type at index {idx}: {type(p)}. Expected str or Path.")
        return normalized_paths
    else: raise ValueError(f"Invalid path type: {type(path)}. Expected str, Path, or list of str/Path.")
