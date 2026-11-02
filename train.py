import functools
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 保证能 import src.*

import torch
from transformers import GenerationConfig, Trainer, TrainingArguments

from src.common_utils import make_run_dir, save_json, set_seed, worker_init_fn
from src.config_utils import load_config_with_overrides
from src.data_utils import Bc2gmNerDataset
from src.evaluate_utils import NERGenerativeEvaluator, make_compute_metrics
from src.model_utils import attach_lora, build_base_model, build_tokenizer
from src.swanlab_utils import build_swanlab_callback


class NERTrainer(Trainer):
    """给数据加载器挂上worker_init_fn，固定多进程数据加载的随机种子"""

    def get_train_dataloader(self):
        loader = super().get_train_dataloader()
        loader.worker_init_fn = functools.partial(worker_init_fn, seed=self.args.seed)
        return loader


def _build_training_args(cfg, run_dir, has_swanlab):
    """组装TrainingArguments；对不同transformers版本的参数名做兼容。"""
    train_cfg = cfg["train"]
    sig_params = inspect.signature(TrainingArguments.__init__).parameters

    args_dict = dict(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval"]["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        learning_rate=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),   # warmup + 学习率调度，替代固定学习率
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),  # 梯度裁剪，防止梯度爆炸
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=10,
        save_total_limit=2,
        # 真正的 dev F1 改由训练结束后的 NERGenerativeEvaluator（逐批 generate）计算。
        load_best_model_at_end=False,
        bf16=train_cfg.get("bf16", True), 
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        seed=train_cfg["seed"],
        report_to="none",  # SwanLab 通过 callbacks 里的 SwanLabCallback 独立记录；4.44.0 的 report_to 不认识 "swanlab"
    )

    # 不同版本参数名兼容。训练中不评估（生成式评估要把整集logits攒进显存，必OOM）
    eval_key = "eval_strategy" if "eval_strategy" in sig_params else "evaluation_strategy"
    args_dict[eval_key] = "no"
    args_dict["save_strategy"] = "epoch"

    # 新版本支持gradient_checkpointing_kwargs，use_reentrant=False更稳
    if "gradient_checkpointing_kwargs" in sig_params:
        args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # 生成参数
    if "generation_config" in sig_params:
        args_dict["generation_config"] = GenerationConfig(
            max_new_tokens=cfg["eval"]["max_new_tokens"], # 答案长度上限
            do_sample=False, # 不随机采样，每生成一个词，都选概率最大的那个
            num_beams=1, # 束搜索（beam search）宽度
            pad_token_id=None,   # Trainer用tokenizer自动补齐
            eos_token_id=None,
        )

    return TrainingArguments(**args_dict)


def main(cfg, config_name):
    # 固定随机种子
    set_seed(cfg["train"]["seed"])

    # 实验目录。以配置文件名命名；已有权重则自动追加时间戳，不覆盖历史结果
    run_dir = make_run_dir(cfg["output"]["root"], config_name)
    save_json(cfg, run_dir / "config.json")  # 产物归档：配置随实验一起留存
    print(f"[run] 实验目录: {run_dir}")

    # 3) 数据预处理。bc2gm JSON -> SFT 样本
    tokenizer = build_tokenizer(cfg["model_name"])
    instruction = cfg.get("instruction")
    train_set = Bc2gmNerDataset(
        cfg["data"]["train_path"], tokenizer, max_len=cfg["max_len"], instruction=instruction
    )
    dev_set = Bc2gmNerDataset(
        cfg["data"]["dev_path"], tokenizer, max_len=cfg["max_len"], instruction=instruction
    )
    # 训练中途的验证可以只抽一部分样本提速，最终复评仍用完整dev
    eval_samples = cfg["eval"].get("train_eval_samples") or None
    dev_eval_set = (
        Bc2gmNerDataset(
            cfg["data"]["dev_path"], tokenizer, max_len=cfg["max_len"],
            instruction=instruction, max_samples=eval_samples,
        )
        if eval_samples
        else dev_set
    )
    print(f"[data] train={len(train_set)} 条 | dev={len(dev_set)} 条"
          f"（含无实体负样本，训练中途验证用 {len(dev_eval_set)} 条）")

    # 模型
    model = build_base_model(cfg)
    model = attach_lora(model, cfg)

    # SwanLab
    swanlab_cb = build_swanlab_callback(cfg, config_name)
    callbacks = [swanlab_cb] if swanlab_cb is not None else []

    # Trainer组装，数据集+collate_fn+评估指标
    training_args = _build_training_args(cfg, run_dir, has_swanlab=swanlab_cb is not None)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_eval_set,
        data_collator=train_set.collate_fn,  # 动态padding（Dataset成员函数）
        compute_metrics=make_compute_metrics(tokenizer, dev_eval_set.gold_entities),
    )
    try:  # 兼容不同 transformers 版本的参数名
        trainer = NERTrainer(processing_class=tokenizer, callbacks=callbacks, **trainer_kwargs)
    except TypeError:
        trainer = NERTrainer(tokenizer=tokenizer, callbacks=callbacks, **trainer_kwargs)

    # 训练
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # 记录本轮训练的显存峰值
    trainer.train()

    # 保存PEFT适配器+tokenizer（用训练结束后的最终权重）
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    peak_gb = (
        round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        if torch.cuda.is_available()
        else None
    )

    # 训练结束后在完整 dev 上做生成式评估（逐批 generate，内存友好），与 test.py 逻辑相同
    evaluator = NERGenerativeEvaluator(
        model, tokenizer,
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        batch_size=cfg["eval"]["batch_size"],
        max_length=cfg["max_len"],
    )
    dev_metrics, dev_predictions = evaluator.evaluate(dev_set, split_name="dev")
    save_json(dev_metrics, run_dir / "dev_metrics.json")
    save_json(dev_predictions[:200], run_dir / "dev_predictions_sample.json")

    summary = {
        "config_name": config_name,
        "mode": cfg["mode"],
        "r": cfg["lora"]["r"],
        "dev_f1": dev_metrics["f1"],
        "dev_precision": dev_metrics["precision"],
        "dev_recall": dev_metrics["recall"],
        "saved_model_dir": str(run_dir),
        "peak_memory_gb": peak_gb,
        "train_samples": len(train_set),
    }
    save_json(summary, run_dir / "train_summary.json")
    print(f"[train] dev F1={dev_metrics['f1']} | 显存峰值={peak_gb} GB")

    print(f"[done] 全部产物已归档到: {run_dir}")
    print("[done] 下一步: python test.py --config <同一个配置> 在 test 集上出最终 F1")
    return run_dir


if __name__ == "__main__":
    manager, _args = load_config_with_overrides()
    print(manager.summary())
    print("-" * 60)
    main(manager.to_dict(), manager.config_name)
