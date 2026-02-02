"""model-finetune-tool - 方便的大模型微调工具

方便的大模型微调工具 🌹

主要功能：
- 📄 文档解析 (Word/PDF/Markdown)
- 🤖 LLM 训练数据生成
- 💾 数据集管理
- ⚡ LoRA 模型训练
"""

__version__ = "0.1.0"

__all__ = [
    # 配置模块
    "load_config",
    "get_config",
    "reload_config",
    "Config",
    "ConfigManager",
    # 数据集模块
    "DatasetManager",
    # LLM 模块
    "LLMClient",
    "CacheManager",
    # 训练模块
    "train_lora",
    "merge_model",
    "prepare_training_data",
    "check_gpu_available",
]
