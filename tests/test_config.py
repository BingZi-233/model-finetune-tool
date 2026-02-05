"""
配置文件测试模块

本模块测试配置加载功能，确保配置正确解析和环境变量替换。
"""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    load_yaml_with_env,
    load_config,
    Config,
    LLMConfig,
    DatabaseConfig,
    TrainingConfig,
    get_config,
)


class TestLoadYamlWithEnv:
    """测试YAML加载和环境变量替换功能"""
    
    def test_replace_env_var(self):
        """测试环境变量替换功能"""
        os.environ["TEST_API_KEY"] = "test-key-123"
        
        yaml_content = """
llm:
  api_key: "${TEST_API_KEY}"
  model: "gpt-3.5-turbo"
"""
        result = load_yaml_with_env(yaml_content)
        
        assert result["llm"]["api_key"] == "test-key-123"
        assert result["llm"]["model"] == "gpt-3.5-turbo"
        
        # 清理
        del os.environ["TEST_API_KEY"]
    
    def test_missing_env_var(self):
        """测试缺失的环境变量会抛出错误"""
        yaml_content = """
llm:
  api_key: "${NONEXISTENT_VAR}"
"""
        # 现在会抛出ValueError
        with pytest.raises(ValueError) as exc_info:
            load_yaml_with_env(yaml_content)
        
        assert "NONEXISTENT_VAR" in str(exc_info.value)
    
    def test_multiple_env_vars(self):
        """测试多个环境变量替换"""
        os.environ["MODEL_NAME"] = "test-model"
        os.environ["API_BASE"] = "https://api.test.com"
        
        yaml_content = """
model: "${MODEL_NAME}"
base_url: "${API_BASE}"
"""
        result = load_yaml_with_env(yaml_content)
        
        assert result["model"] == "test-model"
        assert result["base_url"] == "https://api.test.com"
        
        # 清理
        del os.environ["MODEL_NAME"]
        del os.environ["API_BASE"]


class TestConfigModel:
    """测试配置模型类"""
    
    def test_llm_config_defaults(self):
        """测试LLM配置默认值"""
        config = LLMConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
    
    def test_database_config_defaults(self):
        """测试数据库配置默认值"""
        config = DatabaseConfig()
        
        assert config.type == "sqlite"
        assert config.path == "./data/datasets.db"
        assert config.host is None
        assert config.port == 3306
    
    def test_training_config_defaults(self):
        """测试训练配置默认值"""
        config = TrainingConfig()
        
        assert config.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.batch_size == 4
        assert config.learning_rate == 0.0002
        assert config.epochs == 3
        assert config.max_length == 2048
    
    def test_lora_config_defaults(self):
        """测试LoRA配置默认值"""
        from src.config import LoRAConfig
        
        config = LoRAConfig()
        
        assert config.r == 8
        assert config.alpha == 16
        assert config.dropout == 0.1
        assert "q_proj" in config.target_modules


class TestLoadConfig:
    """测试配置文件加载"""
    
    def test_load_valid_config(self):
        """测试加载有效配置文件"""
        config_content = """
llm:
  api_key: "test-key"
  model: "test-model"
  temperature: 0.5

database:
  type: "sqlite"
  path: ":memory:"

datasets:
  chunk_size: 500

training:
  epochs: 5
  batch_size: 8
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            try:
                config = load_config(f.name)
                
                assert config.llm.api_key == "test-key"
                assert config.llm.model == "test-model"
                assert config.llm.temperature == 0.5
                assert config.database.type == "sqlite"
                assert config.datasets.chunk_size == 500
                assert config.training.epochs == 5
                assert config.training.batch_size == 8
            finally:
                os.unlink(f.name)
    
    def test_config_file_not_found(self):
        """测试加载不存在的配置文件"""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")
    
    def test_nested_config_structure(self):
        """测试嵌套配置结构"""
        config_content = """
llm:
  api_key: "key"
  base_url: "https://api.test.com"
  model: "model"
  temperature: 0.8
  max_tokens: 1500

database:
  type: "mysql"
  host: "localhost"
  port: 3306
  username: "root"
  password: "password"
  database: "testdb"

datasets:
  input_dir: "./docs"
  output_dir: "./data"
  chunk_size: 800
  chunk_overlap: 100

training:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  lora:
    r: 16
    alpha: 32
    dropout: 0.2
    target_modules: ["q_proj", "k_proj"]
  batch_size: 2
  learning_rate: 0.0001
  epochs: 5
  max_length: 1024

output:
  model_dir: "./models"
  checkpoint_dir: "./checkpoints"

git:
  auto_commit: false
  commit_message: "Custom message"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            try:
                config = load_config(f.name)
                
                # 验证嵌套配置
                assert config.database.type == "mysql"
                assert config.database.host == "localhost"
                assert config.training.lora.r == 16
                assert config.training.lora.alpha == 32
                assert config.output.model_dir == "./models"
                assert config.git.auto_commit is False
            finally:
                os.unlink(f.name)


class TestLoadYamlWithEnvEdgeCases:
    """测试 YAML 加载边界情况"""

    def test_empty_yaml(self):
        """测试空 YAML 内容"""
        yaml_content = ""
        result = load_yaml_with_env(yaml_content)
        assert result is None or result == {}

    def test_yaml_with_special_chars(self):
        """测试包含特殊字符的环境变量"""
        os.environ["SPECIAL_CHARS"] = "value with spaces & symbols!@#"
        yaml_content = """
key: "${SPECIAL_CHARS}"
"""
        result = load_yaml_with_env(yaml_content)
        assert result["key"] == "value with spaces & symbols!@#"
        del os.environ["SPECIAL_CHARS"]

    def test_yaml_without_env_vars(self):
        """测试不包含环境变量的 YAML"""
        yaml_content = """
key1: value1
key2: 123
key3: true
"""
        result = load_yaml_with_env(yaml_content)
        assert result["key1"] == "value1"
        assert result["key2"] == 123
        assert result["key3"] is True


class TestConfigEdgeCases:
    """测试配置边界情况"""

    def test_llm_config_temperature_edge(self):
        """测试温度边界值"""
        import pytest
        from src.config import LLMConfig
        
        # 测试最小值
        config = LLMConfig(api_key="test", temperature=0.0)
        assert config.temperature == 0.0
        
        # 测试最大值
        config = LLMConfig(api_key="test", temperature=2.0)
        assert config.temperature == 2.0
        
        # 测试超出范围（应该使用默认值）
        config = LLMConfig(api_key="test", temperature=-1.0)
        assert config.temperature == 0.7
        
        config = LLMConfig(api_key="test", temperature=3.0)
        assert config.temperature == 0.7

    def test_database_config_port_edge(self):
        """测试端口边界值"""
        from src.config import DatabaseConfig
        
        # 测试最小值
        config = DatabaseConfig(port=1)
        assert config.port == 1
        
        # 测试最大值
        config = DatabaseConfig(port=65535)
        assert config.port == 65535
        
        # 测试超出范围
        config = DatabaseConfig(port=0)
        assert config.port == 3306
        
        config = DatabaseConfig(port=65536)
        assert config.port == 3306
