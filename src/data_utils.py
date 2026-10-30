import json

import torch
from torch.utils.data import Dataset

DEFAULT_INSTRUCTION = (
    "请从下面的生物医学句子中提取所有基因（GENE）实体，"
    "若有多个实体请用逗号分隔；若句子中没有基因实体，请回答：无。"
)

ANSWER_MARK = "### 基因实体："

EMPTY_ANSWERS = {"", "无", "没有", "无基因", "未提及", "none", "null", "n/a", "na"}


def load_labels(labels_path):
    """备用函数，标签集合全局生成一次"""
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_answer(entities):
    """把entities列表拼成模型要学习的答案文本"""
    names = [e["name"].strip() for e in entities if e.get("name", "").strip()]
    return ", ".join(names) if names else "无"


def build_prompt(sentence, instruction=None):
    """拼出完整prompt"""
    instruction = instruction or DEFAULT_INSTRUCTION
    return (
        f"### 指令：\n{instruction}\n"
        f"### 句子：\n{sentence}\n"
        f"{ANSWER_MARK}\n"
    )


class Bc2gmNerDataset(Dataset):
    """把bc2gm JSON变成指令+基因实体的SFT样本"""

    def __init__(self, json_path, tokenizer, max_len=384, instruction=None, max_samples=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.instruction = instruction or DEFAULT_INSTRUCTION

        self.raw_items = load_raw(json_path)
        if max_samples:
            self.raw_items = self.raw_items[:max_samples]

        # 标准答案列表，顺序与样本一致，评估时和模型生成结果对齐
        self.gold_entities = [
            [e["name"] for e in item.get("entities", [])] for item in self.raw_items
        ]

    def __len__(self):
        return len(self.raw_items)

    def __getitem__(self, idx):
        item = self.raw_items[idx]
        sentence = item["sentence"]
        answer = build_answer(item.get("entities", []))

        # prompt与答案分开分词
        prompt = build_prompt(sentence, self.instruction)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        # 超长时从左侧截断prompt
        if len(prompt_ids) + len(answer_ids) > self.max_len:
            keep = max(self.max_len - len(answer_ids), 0)
            prompt_ids = prompt_ids[-keep:] if keep > 0 else []

        input_ids = prompt_ids + answer_ids

        # prompt部分label=-100，不计算loss
        labels = [-100] * len(prompt_ids) + answer_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def collate_fn(self, batch):
        """按本批次最长样本补齐"""
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_num = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_num)
            attention_mask.append(x["attention_mask"] + [0] * pad_num)
            labels.append(x["labels"] + [-100] * pad_num)  # padding位置不计算 loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
