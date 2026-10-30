import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common_utils import save_json
from src.config_utils import load_config_with_overrides

import train as train_module  # 复用train.py的main，保证与正式训练完全同一条流程


def plot_trend(rows, out_path):
    """画r-F1趋势图（双轴标注显存）"""
    try:
        import matplotlib

        matplotlib.use("Agg")  # 无界面环境也能保存
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sweep] 未安装 matplotlib，跳过画图（pip install matplotlib 后可启用）")
        return

    rs = [row["r"] for row in rows]
    f1s = [row["dev_f1"] for row in rows]
    mems = [row.get("peak_memory_gb") for row in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(rs, f1s, marker="o", color="#c0392b", linewidth=2, label="Dev F1")
    for r, f1 in zip(rs, f1s):
        ax1.annotate(f"{f1:.4f}", (r, f1), textcoords="offset points", xytext=(0, 8), ha="center")
    ax1.set_xlabel("LoRA rank r")
    ax1.set_ylabel("Dev F1", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_xticks(rs)

    if all(m is not None for m in mems):
        ax2 = ax1.twinx()
        ax2.plot(rs, mems, marker="s", color="#2c3e50", linewidth=1.5, linestyle="--", label="Peak VRAM")
        ax2.set_ylabel("Peak VRAM (GB)", color="#2c3e50")
        ax2.tick_params(axis="y", labelcolor="#2c3e50")

    plt.title("LoRA rank vs Dev F1 / VRAM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[sweep] 趋势图已保存: {out_path}")


def main():
    manager, args = load_config_with_overrides()

    # --r_list 与 --alpha_ratio 是扫描脚本自己的参数
    import argparse

    parser = argparse.ArgumentParser(description="r 参数扫描")
    parser.add_argument("--r_list", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--alpha_ratio", type=float, default=2.0, help="lora_alpha = ratio * r（默认 2）")
    sweep_args, _ = parser.parse_known_args()

    base_cfg = manager.to_dict()
    config_name = manager.config_name
    r_list = sweep_args.r_list
    print(f"[sweep] 配置: {config_name} | 路线: {base_cfg['mode']} | 待测 r: {r_list}")

    rows = []
    for r in r_list:
        print("\n" + "=" * 70)
        print(f"[sweep] 开始训练 r={r}（{r_list.index(r) + 1}/{len(r_list)}）")
        print("=" * 70)

        cfg = copy.deepcopy(base_cfg) # 复制一份基础配置
        cfg["lora"]["r"] = r          # 改r
        cfg["lora"]["lora_alpha"] = int(sweep_args.alpha_ratio * r)  # 保持alpha=2r的经验配比

        run_name = f"{config_name}-r{r}" # 目录名带r，区分每次
        run_dir = train_module.main(cfg, run_name)

        # 从train.py写下的摘要里收集F1与显存峰值
        summary_path = Path(run_dir) / "train_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        rows.append(
            {
                "r": r,
                "lora_alpha": cfg["lora"]["lora_alpha"],
                "dev_f1": summary.get("best_dev_f1"),
                "peak_memory_gb": summary.get("peak_memory_gb"), # 显存峰值
                "run_dir": str(run_dir),
            }
        )

    # 汇总输出
    sweep_dir = Path(base_cfg["output"]["root"]) / f"r_sweep_{config_name}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    save_json(rows, sweep_dir / "results.json")

    lines = ["r\talpha\tdev_f1\tpeak_vram_gb\trun_dir"]
    for row in rows:
        lines.append(
            f"{row['r']}\t{row['lora_alpha']}\t{row['dev_f1']}\t{row['peak_memory_gb']}\t{row['run_dir']}"
        )
    table = "\n".join(lines)
    (sweep_dir / "r_sweep.txt").write_text(table + "\n", encoding="utf-8")
    print("\n==== r 调参趋势汇总 ====")
    print(table)
    print(f"[sweep] 数据: {sweep_dir / 'results.json'}")

    plot_trend(rows, sweep_dir / "r_sweep.png")


if __name__ == "__main__":
    main()
