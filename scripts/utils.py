import os
import sys
from pathlib import Path


def prepare_environment():
    """添加项目根路径到 sys.path，确保工具脚本可正确导入主包"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    sys.path.insert(0, str(project_root))


def detect_venv() -> str | None:
    """检测是否存在虚拟环境目录并返回其激活脚本路径"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    for name in ["venv", ".venv"]:
        venv_dir = project_root / name
        if venv_dir.exists() and venv_dir.is_dir():
            if os.name == "nt":
                activate = venv_dir / "Scripts" / "activate"
            else:
                activate = venv_dir / "bin" / "activate"
            if activate.exists() and activate.is_file():
                return str(activate.resolve())
    return None
