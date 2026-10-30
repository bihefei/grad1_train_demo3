import argparse
import json
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigManager:
    """读取一份JSON配置，提供命令行覆盖、保存、摘要打印功能"""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        if not self.config_path.exists():
            raise FileNotFoundError(f"找不到配置文件: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.config_name = self.config_path.stem  # 例如 qwen25-7b-qlora
        self.cfg["config_name"] = self.config_name
        self._resolve_paths()

    def _resolve_paths(self):
        """把配置里的相对路径统一解析成绝对路径（相对项目根目录），换机器也能跑。"""
        for key in ("train_path", "dev_path", "test_path", "labels_path"):
            p = self.cfg["data"].get(key)
            if p:
                p = Path(p)
                self.cfg["data"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)

        root = Path(self.cfg["output"]["root"])
        self.cfg["output"]["root"] = str(root if root.is_absolute() else PROJECT_ROOT / root)

    def override(self, **kwargs):
        """用命令行参数覆盖配置"""
        # 参数名 -> (配置节, 配置键)
        mapping = {
            "mode": (None, "mode"),
            "model_name": (None, "model_name"),
            "r": ("lora", "r"),
            "lora_alpha": ("lora", "lora_alpha"),
            "target_modules": ("lora", "target_modules"),
            "lr": ("train", "lr"),
            "epochs": ("train", "epochs"),
            "batch_size": ("train", "batch_size"),
            "grad_accum": ("train", "grad_accum"),
            "seed": ("train", "seed"),
            "max_len": (None, "max_len"),
        }
        for arg_name, value in kwargs.items():
            if value is None:
                continue
            section, key = mapping[arg_name]
            if section is None:
                self.cfg[key] = value
            else:
                self.cfg[section][key] = value

    def to_dict(self):
        return self.cfg

    def save(self, path):
        """把当前配置存档到实验目录"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2)

    def summary(self):
        """打印配置摘要"""
        cfg = self.cfg
        route = (
            "QLoRA（4-bit 量化底座，参考显存约 12G）"
            if cfg["mode"] == "qlora"
            else "LoRA（16-bit 底座，参考显存约 27G）"
        )
        return "\n".join(
            [
                f"配置文件 : {self.config_path}",
                f"实验名   : {self.config_name}",
                f"底座模型 : {cfg['model_name']}",
                f"微调路线 : {route}",
                f"LoRA参数 : r={cfg['lora']['r']}, alpha={cfg['lora']['lora_alpha']}, "
                f"target_modules={cfg['lora']['target_modules']}",
                f"训练超参 : epochs={cfg['train']['epochs']}, batch_size={cfg['train']['batch_size']}, "
                f"grad_accum={cfg['train']['grad_accum']}, lr={cfg['train']['lr']}, seed={cfg['train']['seed']}",
            ]
        )


def build_arg_parser():
    """命令行入口，--config选配置，其余参数可选覆盖"""
    parser = argparse.ArgumentParser(
        description="Demo3: Qwen2.5-7B + bc2gm 基因实体识别（指令微调 SFT）"
    )
    parser.add_argument(
        "--config",
        default="configs/qwen25-7b-qlora.json",
        help="配置文件路径，默认走 QLoRA 路线（显存约 12G）",
    )
    parser.add_argument("--mode", choices=["lora", "qlora"], default=None, help="覆盖微调路线")
    parser.add_argument("--model_name", default=None, help="覆盖底座模型")
    parser.add_argument("--r", type=int, default=None, help="LoRA rank，参数调优实验的主角")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA 缩放因子")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--epochs", type=float, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="单卡 batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--max_len", type=int, default=None, help="最大序列长度")
    parser.add_argument("--run_dir", default=None, help="指定实验目录（test/predict 用；默认自动选最新）")
    return parser


def load_config_with_overrides(argv=None):
    args = build_arg_parser().parse_args(argv) # 解析命令行
    manager = ConfigManager(args.config) # 读JSON配置
    manager.override(  # 命令行覆盖
        mode=args.mode,
        model_name=args.model_name,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        max_len=args.max_len,
    )
    return manager, args
