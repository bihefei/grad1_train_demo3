import json
import re
from pathlib import Path

import torch

from src.data_utils import ANSWER_MARK, EMPTY_ANSWERS

# 多个实体之间的分隔符
_SPLIT_RE = re.compile(r"[,，;；\n]")


def normalize_entity(text):
    """实体归一化：小写、压缩连续空白、去首尾标点（bc2gm 采用大小写不敏感匹配）。"""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(" .;,，。；、")


def extract_answer(text):
    """从生成文本中取出答案部分"""
    if ANSWER_MARK in text:
        text = text.split(ANSWER_MARK)[-1]
    return text.strip()


def parse_prediction(text):
    """把模型生成的文本解析成实体列表或空列表"""
    answer = extract_answer(text)
    if answer.lower() in EMPTY_ANSWERS:
        return []
    entities = []
    for piece in _SPLIT_RE.split(answer):
        norm = normalize_entity(piece)
        if norm and norm not in EMPTY_ANSWERS:
            entities.append(norm)
    return entities


def entity_f1_from_texts(pred_texts, gold_lists):
    """精确率 / 召回率 / F1"""
    tp = fp = fn = 0
    exact = 0
    for text, gold in zip(pred_texts, gold_lists):
        pred_set = set(parse_prediction(text))
        gold_set = set(normalize_entity(g) for g in gold)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)   # 模型多找的（误报）
        fn += len(gold_set - pred_set)   # 模型漏找的（漏报）
        exact += int(pred_set == gold_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": round(exact / max(len(pred_texts), 1), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_samples": len(pred_texts),
    }


def make_compute_metrics(tokenizer, gold_lists):
    """Trainer在predict_with_generate=True处的接口"""

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        return entity_f1_from_texts(texts, gold_lists)

    return compute_metrics


class NERGenerativeEvaluator:
    """逐批生成 -> 解析 -> 统计"""

    def __init__(self, model, tokenizer, max_new_tokens=64, batch_size=8, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.max_length = max_length

    @torch.no_grad()
    def predict_texts(self, sentences):
        """对一批句子做生成预测，返回每句的原始生成文本。"""
        self.model.eval()
        raw_outputs = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.model.device)
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # 只取新生成的部分，去掉 prompt，再解码成文本
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            raw_outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return raw_outputs

    def evaluate(self, dataset, split_name="dev", max_samples=None):
        """在数据集（dev/test）上完整评估"""
        sentences = [item["sentence"] for item in dataset.raw_items]
        golds = dataset.gold_entities
        if max_samples:
            sentences, golds = sentences[:max_samples], golds[:max_samples]

        raw_outputs = self.predict_texts(sentences)
        metrics = entity_f1_from_texts(raw_outputs, golds)
        predictions = [
            {
                "sentence": s,
                "gold": g,
                "pred": parse_prediction(r),
                "raw_output": r,
            }
            for s, g, r in zip(sentences, golds, raw_outputs)
        ]
        print(
            f"[{split_name}] 样本 {metrics['num_samples']} | "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} EM={metrics['exact_match']:.4f}"
        )
        return metrics, predictions


def save_test_report(metrics, predictions, run_dir, cfg_summary=""):
    """把测试结果写成 test_result.txt + predictions.json + test_metrics.json。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "==== bc2gm 基因实体识别 测试结果 ====",
        cfg_summary,
        f"样本数      : {metrics['num_samples']}",
        f"Precision   : {metrics['precision']:.4f}",
        f"Recall      : {metrics['recall']:.4f}",
        f"F1          : {metrics['f1']:.4f}",
        f"Exact Match : {metrics['exact_match']:.4f}",
        f"TP/FP/FN    : {metrics['tp']} / {metrics['fp']} / {metrics['fn']}",
    ]
    (run_dir / "test_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(run_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return "\n".join(lines)
