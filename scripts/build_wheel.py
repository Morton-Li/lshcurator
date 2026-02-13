#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path

# 如果用户在 scripts/ 目录下运行，自动切换回项目根路径。
if Path.cwd().resolve().name == "scripts":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    os.chdir(project_root)

from utils import prepare_environment, detect_venv


prepare_environment()


def clean_build_artifacts():
    """删除旧的构建产物，确保干净的构建环境"""
    for folder in ["dist", "build"]:
        shutil.rmtree(folder, ignore_errors=True)
    for path in Path(".").glob("*.egg-info"):
        if path.is_dir():
            shutil.rmtree(path)


def run_command(command, use_shell=True):
    subprocess.run(command, shell=use_shell, check=True)


def main():
    venv_activate = detect_venv()
    if not venv_activate:
        raise RuntimeError("未检测到虚拟环境，请先创建虚拟环境。")
    print(f'检测到虚拟环境：{Path(venv_activate).parent.parent}')

    print(f'清理旧构建产物...')
    clean_build_artifacts()

    # 在 shell 中激活虚拟环境并执行后续命令
    shell_command = f'''source "{venv_activate}" && python -m build'''
    print('开始构建：')
    run_command(shell_command)
    print('构建完成！请检查 dist/ 目录中的产物。')


if __name__ == "__main__":
    main()
