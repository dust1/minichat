# -*- coding: utf-8 -*-
"""检查当前 Python 环境是否满足项目依赖。用法: python check_deps.py"""

import importlib
import subprocess
import sys

STDLIB = {"os", "glob", "json", "sys", "time", "math", "shutil", "argparse", "subprocess"}
THIRD_PARTY = {
    "numpy": "numpy",
    "sentencepiece": "sentencepiece",
    "torch": "torch",                # 训练/推理核心框架
    "tensorboard": "tensorboard",    # 可选: 训练曲线日志
    "safetensors": "safetensors",    # 可选: 仅 export_hf.py 导出 HF 格式时需要
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
        print(f"[可选] {name} 未安装 (pip install {pip_name})")
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
        if not check(mod, pip, required=(mod not in ("tensorboard", "safetensors"))):
            ok = False

    print("-" * 50)
    if ok:
        print("全部依赖已满足")
    else:
        print("存在缺失依赖, 可运行: pip install numpy sentencepiece torch")
        sys.exit(1)


if __name__ == "__main__":
    main()
