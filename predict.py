
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common_utils import find_latest_run_dir
from src.config_utils import load_config_with_overrides
from src.data_utils import build_prompt
from src.evaluate_utils import NERGenerativeEvaluator, parse_prediction
from src.model_utils import build_base_model, build_tokenizer

from peft import PeftModel
from transformers import AutoModelForCausalLM


def load_trained_model(cfg, run_dir):
    """与test.py相同的加载逻辑，底座+LoRA适配器。"""
    run_dir = Path(run_dir)
    tokenizer = build_tokenizer(cfg["model_name"])
    if (run_dir / "tokenizer_config.json").exists():
        tokenizer = build_tokenizer(str(run_dir))

    if (run_dir / "adapter_config.json").exists():
        print(f"[predict] 底座: {cfg['model_name']} | 适配器: {run_dir}")
        model = build_base_model(cfg)                          # 底座
        model = PeftModel.from_pretrained(model, str(run_dir)) # 挂LoRA
    else:
        print(f"[predict] 直接加载完整模型: {run_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            str(run_dir), device_map="auto", trust_remote_code=True
        )
    return model, tokenizer


def load_sentences(file_path):
    """从txt或json读取待预测句子。"""
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item["sentence"] for item in data]  # bc2gm 格式：取每条的 sentence
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]  # txt：每行一句，空行跳过


def main(cfg, run_dir=None, text=None, file=None):
    if not text and not file:
        raise ValueError("请通过 --text 或 --file 提供待预测的句子")

    if run_dir is None:
        run_dir = find_latest_run_dir(cfg["output"]["root"], cfg.get("config_name"))
    run_dir = Path(run_dir)
    print(f"[predict] 实验目录: {run_dir}")

    sentences = [text] if text else load_sentences(file)
    print(f"[predict] 待预测 {len(sentences)} 句")

    model, tokenizer = load_trained_model(cfg, run_dir)
    evaluator = NERGenerativeEvaluator(     # evaluate_utils的评估器
        model, tokenizer,
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        batch_size=cfg["eval"]["batch_size"],
        max_length=cfg["max_len"],
    )
    raw_outputs = evaluator.predict_texts(sentences)

    print("\n==== 预测结果 ====")
    results = []
    for sentence, raw in zip(sentences, raw_outputs):
        entities = parse_prediction(raw)
        results.append({"sentence": sentence, "entities": entities, "raw_output": raw})
        print(f"句子  : {sentence}")
        print(f"基因  : {entities if entities else '（无）'}")
        print("-" * 60)

    # 批量预测时把结果落盘，方便抽查
    if file:
        out_path = run_dir / "predict_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[done] 批量预测结果已写入: {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    manager, args = load_config_with_overrides()
    print(manager.summary())
    print("-" * 60)

    # 预测专用参数
    parser = argparse.ArgumentParser(description="bc2gm 基因实体识别预测")
    parser.add_argument("--text", default=None, help="单句预测")
    parser.add_argument("--file", default=None, help="批量预测文件（txt 每行一句 / json 同 bc2gm 格式）")
    predict_args, _ = parser.parse_known_args()

    main(
        manager.to_dict(),
        run_dir=args.run_dir,
        text=predict_args.text,
        file=predict_args.file,
    )
