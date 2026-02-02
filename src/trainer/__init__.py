"""训练模块"""

import logging
import os
import platform
from pathlib import Path
from typing import Dict, Optional

from ..config import get_config

logger = logging.getLogger(__name__)

# ============ 平台检测 ============
IS_WINDOWS = platform.system() == "Windows"


# ============ GPU 检测 ============
def check_gpu_available() -> Dict[str, bool]:
    """检查 GPU 可用性

    Returns:
        Dict containing:
        - cuda: CUDA 是否可用
        - mps: Apple Silicon 是否可用
        - gpu_name: GPU 名称 (如果检测到)
    """
    result = {"cuda": False, "mps": False, "gpu_name": None}

    try:
        import torch

        if torch.cuda.is_available():
            result["cuda"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0)
            logger.info(f"检测到 CUDA GPU: {result['gpu_name']}")
            logger.info(
                f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB"
            )

        # 检查 Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["mps"] = True
            result["gpu_name"] = "Apple Silicon"
            logger.info("检测到 Apple Silicon MPS")

    except ImportError:
        logger.warning("PyTorch 未安装，无法检测 GPU")
    except Exception as e:
        logger.warning(f"GPU 检测失败: {e}")

    return result


def get_device_map() -> str:
    """获取最佳设备映射

    Returns:
        设备字符串: "cuda", "mps", 或 "auto"
    """
    gpu_info = check_gpu_available()

    if gpu_info["cuda"]:
        return "cuda"
    elif gpu_info["mps"]:
        return "mps"
    else:
        logger.info("未检测到 GPU，将使用 CPU 训练（可能较慢）")
        return "auto"


def prepare_training_data(
    dataset_path: str, output_path: str, chat_template: bool = True
) -> str:
    """准备训练数据"""
    import json

    config = get_config()

    # 读取数据
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 格式化
    formatted_data = []
    for item in data:
        if chat_template:
            formatted = {
                "messages": [
                    {"role": "user", "content": item.get("instruction", "")},
                    {"role": "assistant", "content": item.get("output", "")},
                ]
            }
        else:
            formatted = {
                "text": f"### Instruction: {item.get('instruction', '')}\n### Response: {item.get('output', '')}"
            }
        formatted_data.append(formatted)

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_path


def train_lora(
    model_name: str,
    data_path: str,
    output_dir: str,
    lora_config: Optional[Dict] = None,
    batch_size: int = 4,
    learning_rate: float = 0.0002,
    epochs: int = 3,
    max_length: int = 2048,
    resume_from: Optional[str] = None,
):
    """训练LoRA模型

    Args:
        model_name: 模型名称
        data_path: 训练数据路径
        output_dir: 输出目录
        lora_config: LoRA 配置
        batch_size: 批次大小
        learning_rate: 学习率
        epochs: 训练轮数
        max_length: 最大序列长度
        resume_from: 从检查点恢复的路径
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model

    config = get_config()
    lora_config = lora_config or config.training.lora.model_dump()

    # 检测 GPU
    gpu_info = check_gpu_available()
    device_map = get_device_map()

    # MPS 内存优化：减少序列长度
    if gpu_info["mps"] and max_length > 1024:
        logger.warning(
            "MPS 模式下自动将 max_length 从 %d 降至 1024 以避免 OOM", max_length
        )
        max_length = 1024

    # 加载模型
    logger.info(f"加载模型: {model_name}")
    logger.info(f"使用设备: {device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map=device_map
    )

    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("alpha", 16),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 加载数据
    dataset = load_dataset("json", data_files=data_path, split="train")

    # 数据预处理
    def tokenize_function(examples):
        texts = []
        for msg in examples.get("messages", []):
            text = tokenizer.apply_chat_template(msg, tokenize=False)
            texts.append(text)

        return tokenizer(
            texts, truncation=True, max_length=max_length, padding="max_length"
        )

    dataset = dataset.map(tokenize_function, batched=True)

    # 训练参数
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查点目录
    checkpoint_dir = output_dir / "checkpoints"

    # MPS 内存优化
    mps_memory_ratio = 0.5  # MPS 使用 50% 内存上限
    gradient_accumulation_steps = 4 if not gpu_info["mps"] else 2

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        fp16=gpu_info["cuda"],  # 仅在 CUDA 上启用 fp16
        bf16=gpu_info["mps"],  # MPS 使用 bf16
        optim="paged_adamw_8bit",
        load_best_model_at_end=False,
        resume_from_checkpoint=resume_from,
        max_grad_norm=1.0,
        warmup_steps=10,
    )

    # 训练
    logger.info("开始训练...")
    if gpu_info["mps"]:
        logger.warning(
            "MPS 模式下：如遇 OOM，可使用环境变量: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
        )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )

    trainer.train()

    # 保存模型
    model.save_pretrained(str(output_dir / "lora_model"))
    tokenizer.save_pretrained(str(output_dir / "lora_model"))

    logger.info(f"模型已保存到: {output_dir / 'lora_model'}")

    return str(output_dir / "lora_model")


def merge_model(base_model_path: str, lora_model_path: str, output_path: str):
    """合并模型"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", device_map="auto"
    )

    logger.info(f"加载LoRA: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    logger.info("合并模型...")
    merged_model = model.merge_and_unload()

    logger.info(f"保存合并模型: {output_path}")
    merged_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    logger.info("模型合并完成！")

    return output_path


"""
训练模块

本模块提供 LoRA 微调和模型合并功能。

## 主要功能

### 训练 (train_lora)

使用 LoRA (Low-Rank Adaptation) 方法微调大语言模型。

特点：
- 自动 GPU 检测和选择
- 支持从检查点恢复训练
- 混合精度训练 (FP16)
- 自动保存检查点

### 模型合并 (merge_model)

将微调后的 LoRA 权重合并回基础模型。

### 数据准备 (prepare_training_data)

将数据集转换为训练格式。

## GPU 支持

自动检测以下 GPU：
- **CUDA** - NVIDIA GPU
- **MPS** - Apple Silicon

```python
from src.trainer import check_gpu_available, get_device_map

# 检查 GPU
gpu_info = check_gpu_available()
print(gpu_info)
# {'cuda': True, 'mps': False, 'gpu_name': 'NVIDIA A100'}

# 获取最佳设备
device = get_device_map()  # 'cuda', 'mps', 或 'auto'
```

## 使用示例

```python
from src.trainer import train_lora, merge_model, prepare_training_data

# 准备训练数据
prepare_training_data(
    dataset_path="train.jsonl",
    output_path="prepared_train.jsonl",
    chat_template=True  # 使用聊天模板格式
)

# 训练 LoRA
train_lora(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    data_path="prepared_train.jsonl",
    output_dir="./output/my_model",
    batch_size=4,
    learning_rate=0.0002,
    epochs=3,
    resume_from=None  # 从检查点恢复
)

# 合并模型
merge_model(
    base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
    lora_model_path="./output/my_model/lora_model",
    output_path="./output/my_model/merged"
)
```

## 输出结构

```
output/
└── my_model/
    ├── checkpoints/          # 检查点目录
    │   ├── checkpoint-100/
    │   ├── checkpoint-200/
    │   └── ...
    ├── lora_model/           # LoRA 权重
    │   ├── adapter_config.json
    │   ├── adapter_model.bin
    │   └── ...
    └── merged/               # 合并后的模型
        ├── config.json
        ├── pytorch_model.bin
        └── ...
```

## LoRA 配置

推荐配置：

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| r | 8 | 8-64 | LoRA rank |
| alpha | 16 | 16-32 | 缩放因子 |
| dropout | 0.1 | 0.05-0.2 | Dropout 比例 |
| target_modules | q,k,v,o | - | 目标模块 |

## 性能优化

1. **批次大小**
   - 根据 GPU 显存调整
   - 建议从 4 开始尝试

2. **混合精度**
   - 自动在 CUDA 上启用
   - 可节省显存约 50%

3. **梯度累积**
   - 默认 4 步累积
   - 有效批次 = batch_size × 4

## 注意事项

- 需要安装 transformers, datasets, peft
- 首次运行会下载模型（可能较大）
- 建议使用 GPU 以获得合理训练速度
- 训练可能需要数小时（取决于数据量）
"""

__all__ = [
    "train_lora",
    "merge_model",
    "prepare_training_data",
    "check_gpu_available",
    "get_device_map",
]
