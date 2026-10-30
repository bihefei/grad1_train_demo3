import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_tokenizer(model_name):
    """加载Qwen分词器，pad_token 为空时补齐成eos"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_qlora_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 底座压成4-bit
        bnb_4bit_quant_type="nf4",            # NF4量化
        bnb_4bit_use_double_quant=True,       # 双重量化
        bnb_4bit_compute_dtype=torch.bfloat16,  # 存4-bit，算bf16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",        # 自动把模型放到GPU
        trust_remote_code=True,
    )

    # 把量化底座从"只读"状态改成"可训练但只训LoRA"的状态
    # 升fp32归一化层、冻结底座、开梯度检查点
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    except TypeError:  # 兼容旧版 peft 的参数名
        model = prepare_model_for_kbit_training(model, gradient_checkpointing=True)
    return model


def _build_lora_model(model_name, gradient_checkpointing):
    """LoRA，16-bit底座不量化"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False            # 梯度检查点与 KV cache 不共存
        model.enable_input_require_grads()        # 保证LoRA能收到梯度
    return model


def attach_lora(model, cfg):
    """把可训练的小矩阵A、B挂到底座上；QLoRA与LoRA共用"""
    lora_cfg = cfg["lora"]
    config = LoraConfig(
        r=lora_cfg["r"],                                # 小矩阵宽度
        lora_alpha=lora_cfg["lora_alpha"],              # 缩放因子
        target_modules=lora_cfg["target_modules"],      # Qwen的注意力投影层名
        lora_dropout=lora_cfg["lora_dropout"],          # 随机屏蔽防过拟合
        bias=lora_cfg.get("bias", "none"),              # 不训练偏置项
        task_type="CAUSAL_LM",                          # Qwen是因果生成式模型
    )
    model = get_peft_model(model, config) # 调用get_peft_model真正执行挂载
    model.print_trainable_parameters()  # 打印可训练参数占比    
    return model


def build_base_model(cfg):
    """按配置mode构建底座"""
    mode = cfg["mode"]
    model_name = cfg["model_name"]
    if mode == "qlora":
        print("[model] QLoRA 路线：4-bit(NF4) 量化底座 + LoRA（参考显存约 12G）")
        return _build_qlora_model(model_name)
    if mode == "lora":
        print("[model] LoRA 路线：16-bit 底座 + LoRA（参考显存约 27G）")
        return _build_lora_model(model_name, cfg["train"].get("gradient_checkpointing", True))
    raise ValueError(f"未知 mode: {mode}（只支持 lora / qlora）")


def build_model_and_tokenizer(cfg):
    tokenizer = build_tokenizer(cfg["model_name"])
    model = build_base_model(cfg) # 建底座
    model = attach_lora(model, cfg) # 挂 LoRA
    return model, tokenizer
