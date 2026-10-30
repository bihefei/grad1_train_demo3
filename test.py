import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.common_utils import find_latest_run_dir, set_seed
from src.config_utils import load_config_with_overrides
from src.data_utils import Bc2gmNerDataset
from src.evaluate_utils import NERGenerativeEvaluator, save_test_report
from src.model_utils import build_base_model, build_tokenizer


def load_trained_model(cfg, run_dir):
    """按实验目录加载底座 + LoRA 适配器"""
    run_dir = Path(run_dir)
    tokenizer = build_tokenizer(cfg["model_name"])
    # 如果实验目录里存过tokenizer，优先用它
    if (run_dir / "tokenizer_config.json").exists():
        tokenizer = build_tokenizer(str(run_dir))

    if (run_dir / "adapter_config.json").exists():
        print(f"[test] 底座: {cfg['model_name']} | 适配器: {run_dir}")
        model = build_base_model(cfg)
        model = PeftModel.from_pretrained(model, str(run_dir))
    else:
        print(f"[test] 直接加载完整模型: {run_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            str(run_dir), device_map="auto", trust_remote_code=True
        )
    return model, tokenizer


def main(cfg, run_dir=None):
    set_seed(cfg["train"]["seed"])

    if run_dir is None:
        run_dir = find_latest_run_dir(cfg["output"]["root"], cfg.get("config_name"))
    run_dir = Path(run_dir)
    print(f"[test] 实验目录: {run_dir}")

    # 先加载模型/tokenizer，再用同一个 tokenizer 构建数据集
    model, tokenizer = load_trained_model(cfg, run_dir)
    test_set = Bc2gmNerDataset(
        cfg["data"]["test_path"],
        tokenizer,
        max_len=cfg["max_len"],
        instruction=cfg.get("instruction"),
    )
    print(f"[test] 测试集 {len(test_set)} 条")

    # 评估
    evaluator = NERGenerativeEvaluator(
        model, tokenizer,
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        batch_size=cfg["eval"]["batch_size"],
        max_length=cfg["max_len"],
    )
    metrics, predictions = evaluator.evaluate(test_set, split_name="test")

    cfg_summary = (
        f"配置: {cfg.get('config_name')} | 路线: {cfg['mode'].upper()} | "
        f"r={cfg['lora']['r']} | 模型: {cfg['model_name']}"
    )
    report = save_test_report(metrics, predictions, run_dir, cfg_summary)
    print(report)
    print(f"[done] 报告已写入: {run_dir / 'test_result.txt'}")
    return metrics


if __name__ == "__main__":
    manager, args = load_config_with_overrides()
    print(manager.summary())
    print("-" * 60)
    main(manager.to_dict(), run_dir=args.run_dir)
