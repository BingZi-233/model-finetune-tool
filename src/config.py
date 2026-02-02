"""配置文件加载模块"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2000


class DatabaseConfig(BaseModel):
    type: str = "sqlite"
    path: str = "./data/datasets.db"
    host: Optional[str] = None
    port: int = 3306
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


class DatasetConfig(BaseModel):
    input_dir: str = "./documents"
    output_dir: str = "./data"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class LoRAConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]


class TrainingConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora: LoRAConfig = LoRAConfig()
    batch_size: int = 4
    learning_rate: float = 0.0002
    epochs: int = 3
    max_length: int = 2048


class OutputConfig(BaseModel):
    model_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"


class GitConfig(BaseModel):
    auto_commit: bool = True
    commit_message: str = "Update dataset: {dataset_name}"


class Config(BaseModel):
    llm: LLMConfig
    database: DatabaseConfig = DatabaseConfig()
    datasets: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    output: OutputConfig = OutputConfig()
    git: GitConfig = GitConfig()


def load_yaml_with_env(yaml_str: str) -> Dict[str, Any]:
    """加载YAML并替换环境变量"""
    import re
    
    def replace_env(match):
        env_var = match.group(1)
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise ValueError(
                f"环境变量 '{env_var}' 未设置。\n"
                f"请在运行前设置该环境变量，例如:\n"
                f"  export {env_var}=your_value"
            )
        return env_value
    
    # 替换 ${VAR} 格式的环境变量
    yaml_content = re.sub(r'\$\{([^}]+)\}', replace_env, yaml_str)
    return yaml.safe_load(yaml_content)


def load_config(config_path: str = "config.yaml") -> Config:
    """加载配置文件"""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        yaml_content = f.read()
    
    config_dict = load_yaml_with_env(yaml_content)
    return Config(**config_dict)


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
