"""训练模块"""
import os
from pathlib import Path
from typing import Dict, Optional

from ..config import get_config


def prepare_training_data(
    dataset_path: str,
    output_path: str,
    chat_template: bool = True
) -> str:
    """准备训练数据"""
    import json
    
    config = get_config()
    
    # 读取数据
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 格式化
    formatted_data = []
    for item in data:
        if chat_template:
            formatted = {
                "messages": [
                    {"role": "user", "content": item.get("instruction", "")},
                    {"role": "assistant", "content": item.get("output", "")}
                ]
            }
        else:
            formatted = {
                "text": f"### Instruction: {item.get('instruction', '')}\n### Response: {item.get('output', '')}"
            }
        formatted_data.append(formatted)
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return output_path


def train_lora(
    model_name: str,
    data_path: str,
    output_dir: str,
    lora_config: Optional[Dict] = None,
    batch_size: int = 4,
    learning_rate: float = 0.0002,
    epochs: int = 3,
    max_length: int = 2048
):
    """训练LoRA模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    
    config = get_config()
    lora_config = lora_config or config.training.lora.model_dump()
    
    # 加载模型
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("alpha", 16),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
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
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    # 训练参数
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=4,
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        fp16=True,
        optim="paged_adamw_8bit"
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    model.save_pretrained(str(output_dir / "lora_model"))
    tokenizer.save_pretrained(str(output_dir / "lora_model"))
    
    print(f"模型已保存到: {output_dir / 'lora_model'}")
    
    return str(output_dir / "lora_model")


def merge_model(
    base_model_path: str,
    lora_model_path: str,
    output_path: str
):
    """合并模型"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    
    print(f"加载LoRA: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    print("合并模型...")
    merged_model = model.merge_and_unload()
    
    print(f"保存合并模型: {output_path}")
    merged_model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path
