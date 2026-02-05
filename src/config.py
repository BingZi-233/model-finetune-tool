"""配置文件加载模块"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


# ============ 配置验证 ============
class LLMConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2000

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0 or v > 2:
            logger.warning(f"temperature 值 {v} 超出范围 [0, 2]，使用默认值 0.7")
            return 0.7
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v < 1:
            logger.warning(f"max_tokens 值 {v} 无效，使用默认值 2000")
            return 2000
        return v


class DatabaseConfig(BaseModel):
    type: str = "sqlite"
    path: str = "./data/datasets.db"
    host: Optional[str] = None
    port: int = 3306
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = ["sqlite", "mysql", "postgresql"]
        if v.lower() not in valid_types:
            logger.warning(f"不支持的数据库类型: {v}，使用 sqlite")
            return "sqlite"
        return v.lower()

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if v < 1 or v > 65535:
            logger.warning(f"端口号 {v} 无效，使用默认值 3306")
            return 3306
        return v


class DatasetConfig(BaseModel):
    input_dir: str = "./documents"
    output_dir: str = "./data"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 100:
            logger.warning(f"chunk_size 值 {v} 太小，使用默认值 1000")
            return 1000
        if v > 10000:
            logger.warning(f"chunk_size 值 {v} 太大，使用默认值 1000")
            return 1000
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v < 0:
            logger.warning(f"chunk_overlap 值 {v} 无效，使用默认值 200")
            return 200
        if v >= chunk_size:
            logger.warning(
                f"chunk_overlap ({v}) >= chunk_size ({chunk_size})，使用 chunk_size//2 = {chunk_size // 2}"
            )
            return chunk_size // 2
        return v


class LoRAConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]

    @field_validator("r")
    @classmethod
    def validate_r(cls, v: int) -> int:
        if v < 1:
            logger.warning(f"LoRA r 值 {v} 无效，使用默认值 8")
            return 8
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: int) -> int:
        if v < 1:
            logger.warning(f"LoRA alpha 值 {v} 无效，使用默认值 16")
            return 16
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if v < 0 or v > 1:
            logger.warning(f"LoRA dropout 值 {v} 超出范围 [0, 1]，使用默认值 0.1")
            return 0.1
        return v


class TrainingConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora: LoRAConfig = LoRAConfig()
    batch_size: int = 4
    learning_rate: float = 0.0002
    epochs: int = 3
    max_length: int = 2048

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1:
            logger.warning(f"batch_size 值 {v} 无效，使用默认值 4")
            return 4
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        if v <= 0:
            logger.warning(f"learning_rate 值 {v} 无效，使用默认值 0.0002")
            return 0.0002
        return v

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, v: int) -> int:
        if v < 1:
            logger.warning(f"epochs 值 {v} 无效，使用默认值 3")
            return 3
        return v

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        if v < 1:
            logger.warning(f"max_length 值 {v} 无效，使用默认值 2048")
            return 2048
        return v


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

    @model_validator(mode="after")
    def validate_output_dirs(self):
        """验证输出目录配置"""
        try:
            Path(self.output.model_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"无法创建 model_dir: {e}")
        return self


def load_yaml_with_env(yaml_str: str) -> Dict[str, Any]:
    """加载YAML并替换环境变量"""

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
    yaml_content = re.sub(r"\$\{([^}]+)\}", replace_env, yaml_str)
    return yaml.safe_load(yaml_content)


def load_config(config_path: str = "config.yaml") -> Config:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        Config 对象

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置无效
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    config_dict = load_yaml_with_env(yaml_content)

    # 验证配置
    try:
        config = Config(**config_dict)
        logger.info(f"成功加载配置: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        raise ValueError(f"配置无效: {e}") from e


# ============ 配置管理器（支持热重载） ============
class ConfigManager:
    """配置管理器，支持热重载和配置切换"""

    def __init__(self):
        self._config: Optional[Config] = None
        self._config_path: Optional[str] = None
        self._last_modified: Optional[float] = None

    def load_config(
        self, config_path: str = "config.yaml", force: bool = False
    ) -> Config:
        """加载配置文件，支持热重载

        Args:
            config_path: 配置文件路径
            force: 强制重新加载

        Returns:
            Config 对象
        """
        config_path = os.path.normpath(config_path)

        # 检查是否需要重新加载
        if not force and self._config is not None and self._config_path == config_path:
            try:
                current_mtime = os.path.getmtime(config_path)
                if self._last_modified == current_mtime:
                    return self._config
            except OSError:
                pass

        # 重新加载配置
        self._config_path = config_path
        self._config = load_config(config_path)

        try:
            self._last_modified = os.path.getmtime(config_path)
        except OSError:
            self._last_modified = None

        logger.info(f"配置已加载: {config_path}")
        return self._config

    def get_config(self) -> Config:
        """获取当前配置"""
        if self._config is None:
            return load_config()
        return self._config

    def clear_cache(self):
        """清除配置缓存"""
        self._config = None
        self._config_path = None
        self._last_modified = None


# 全局配置管理器实例
_config_manager = ConfigManager()


def get_config() -> Config:
    """获取全局配置"""
    return _config_manager.get_config()


def reload_config(config_path: str = "config.yaml") -> Config:
    """重新加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config 对象
    """
    return _config_manager.load_config(config_path, force=True)
