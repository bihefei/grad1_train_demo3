import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """固定 Python / numpy / PyTorch 的全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id, seed):
    """固定DataLoader多进程加载的随机种子"""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_run_dir(output_root, config_name):
    """以配置文件名创建实验目录；若已有训练产物则追加时间戳，保证历史结果不被覆盖。"""
    run_dir = Path(output_root) / config_name
    if run_dir.exists() and any(run_dir.glob("adapter_model*")):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(output_root) / f"{config_name}-{stamp}"
        print(f"[run] 检测到历史产物，自动改用新目录: {run_dir.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run_dir(output_root, config_name=None):
    """测试/预测阶段自动选用实验目录，优先找该配置对应的最新目录，否则取全局最新。"""
    output_root = Path(output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"outputs 目录不存在: {output_root}，请先运行 train.py")

    candidates = []
    if config_name:
        candidates = [p for p in output_root.glob(f"{config_name}*") if p.is_dir()]
    if not candidates:
        candidates = [p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith("r_sweep")]
    if not candidates:
        raise FileNotFoundError(f"{output_root} 下还没有任何实验目录，请先运行 train.py")

    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def save_json(obj, path):
    """结果落盘工具utf-8"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
