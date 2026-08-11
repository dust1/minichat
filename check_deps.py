# -*- coding: utf-8 -*-
"""检查当前 Python 环境是否满足项目依赖。用法: python check_deps.py"""

import importlib
import subprocess
import sys

STDLIB = {"os", "glob", "json", "sys", "time", "math", "shutil", "argparse", "subprocess"}
THIRD_PARTY = {
    "numpy": "numpy",
    "sentencepiece": "sentencepiece",
    "cupy": "cupy",  # 可选: 仅在 GPU 训练时使用
}


def check(name, pip_name, required=True):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "未知版本")
        print(f"[OK]   {name} {ver}")
        return True
    except ImportError:
        if required:
            print(f"[缺失] {name} (pip install {pip_name})")
            return False
        print(f"[可选] {name} 未安装 (GPU 训练时需要: pip install {pip_name})")
        return True


def main():
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"Python 路径: {sys.executable}")
    print("-" * 50)

    ok = True
    for mod in sorted(STDLIB):
        try:
            importlib.import_module(mod)
            print(f"[OK]   {mod} (标准库)")
        except ImportError:
            print(f"[缺失] {mod} (标准库, 不应缺失)")
            ok = False

    for mod, pip in sorted(THIRD_PARTY.items()):
        if not check(mod, pip, required=(mod != "cupy")):
            ok = False

    print("-" * 50)
    if ok:
        print("全部依赖已满足")
    else:
        print("存在缺失依赖, 可运行: pip install numpy sentencepiece")
        sys.exit(1)


if __name__ == "__main__":
    main()
