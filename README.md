# Demo3：大模型指令微调的实体识别（NER）

## 项目简介

基于 **Qwen2.5-7B**，在 **bc2gm**（BioCreative II Gene Mention）生物医学语料上做**基因实体识别**。
把 NER 任务改造成「指令 → 答案」的生成式任务，通过 **LoRA / QLoRA** 两种参数高效微调路线完成监督微调（SFT），并用 **SwanLab** 记录训练过程。

- 输入：一句生物医学英文，如 `Comparison with alkaline phosphatases and 5 - nucleotidase`
- 输出：这句话里的所有基因实体，如 `alkaline phosphatases`
- 数据中的负样本（无基因）教模型「没有就回答：无」

## 项目文件说明

| 文件名 | 作用说明 |
| :--- | :--- |
| `train.py` | 训练入口，加载数据/模型，用 Trainer 完成 SFT 并保存 LoRA 权重与最佳 F1 |
| `test.py` | 测试入口，在 test 集上计算最终 P/R/F1，产出 `test_result.txt` |
| `predict.py` | 预测入口，对单句或批量句子做基因实体识别 |
| `sweep_r.py` | 参数调优实验，批量训练不同 rank（r）并汇总趋势 |
| `configs/qwen25-7b-lora.json` | LoRA 路线配置（16-bit 底座，参考显存约 27G） |
| `configs/qwen25-7b-qlora.json` | QLoRA 路线配置（4-bit 底座，参考显存约 12G） |
| `src/config_utils.py` | 配置管理：读取 JSON 并支持命令行覆盖 |
| `src/common_utils.py` | 通用工具：随机种子、实验目录、结果落盘 |
| `src/data_utils.py` | 数据预处理：bc2gm JSON → SFT 样本（Dataset + 动态 padding） |
| `src/model_utils.py` | 模型加载与 LoRA/QLoRA 组装 |
| `src/evaluate_utils.py` | 实体级 F1 评估 |
| `src/swanlab_utils.py` | SwanLab 可视化回调 |
| `requirements.txt` | 项目依赖清单 |
| `data/bc2gm/` | 数据集文件（train/dev/test/labels 四个 JSON） |
| `outputs/` | 实验产物目录（按配置名归档） |

## 目录结构

```
demo3/
├── README.md                项目说明（本文档）
├── requirements.txt         依赖清单
├── train.py                 训练入口
├── test.py                  测试入口
├── predict.py               预测入口
├── sweep_r.py               r 调参实验
├── configs/
│   ├── qwen25-7b-lora.json     LoRA 路线配置
│   └── qwen25-7b-qlora.json    QLoRA 路线配置
├── src/                     需要被导入的类和工具
│   ├── config_utils.py
│   ├── common_utils.py
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── evaluate_utils.py
│   └── swanlab_utils.py
├── data/bc2gm/              bc2gm 数据集
└── outputs/                 实验产物
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 安装依赖：`pip install -r requirements.txt`
2. QLoRA 路线训练：`python train.py --config configs/qwen25-7b-qlora.json`
3. LoRA 路线训练：`python train.py --config configs/qwen25-7b-lora.json`
4. 在 test 集出最终指标：`python test.py --config configs/qwen25-7b-qlora.json`
5. 单句预测：`python predict.py --text "Comparison with alkaline phosphatases and 5 - nucleotidase"`
6. 批量预测：`python predict.py --file sentences.txt`
7. 参数调优（r 趋势）：`python sweep_r.py --config configs/qwen25-7b-qlora.json --r_list 4 8 16 32`

常用参数可直接在命令行覆盖，无需改配置文件，例如：

```bash
python train.py --config configs/qwen25-7b-qlora.json --r 16 --lr 1e-4 --epochs 2 --seed 101
```

## 方法说明（LoRA 与 QLoRA）

两者都只在冻结的 7B 底座旁训练一小对可学习矩阵（LoRA），区别在底座精度：

| | LoRA 路线 | QLoRA 路线 |
| :--- | :--- | :--- |
| 底座精度 | 16-bit | 4-bit（NF4 + 双重量化） |
| 参考显存 | 约 27G | 约 12G |
| 参考 F1 | 约 84% | 约 83% |
| 适用场景 | 显存充足，多拿约 1 个点 | 显存有限也能微调 7B |

## 配置说明

配置集中放在 `configs/*.json`，主要字段：

| 配置键 | 含义 | 默认值 |
| :--- | :--- | :--- |
| `mode` | 微调路线 `lora` / `qlora` | qlora |
| `model_name` | 底座模型 | Qwen/Qwen2.5-7B |
| `max_len` | 最大序列长度 | 384 |
| `lora.r` | rank，小矩阵容量，调参主角 | 8 |
| `lora.lora_alpha` | 缩放因子，经验取 alpha = 2r | 16 |
| `train.lr` | 学习率 | 1e-4 |
| `train.batch_size` | 单卡批大小 | 4 |
| `train.epochs` | 训练轮数 | 3 |
| `train.seed` | 随机种子 | 101 |

## 产物说明

每次实验的产物归档在 `outputs/<配置名>/`：

- `config.json`：本实验完整配置
- `adapter_model*`：LoRA 权重（仅几 MB）
- `train_summary.json`：最佳验证 F1 与显存峰值
- `dev_metrics.json` / `test_result.txt` / `predictions.json`：评估与预测明细

目录已存在权重时自动追加时间戳，**不覆盖历史结果**；test / predict 不指定 `--run_dir` 时自动选用最新目录。
