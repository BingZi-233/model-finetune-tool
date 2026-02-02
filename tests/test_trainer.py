"""
训练模块测试模块

本模块测试模型训练功能，包括数据准备、LoRA训练、模型合并等。
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestPrepareTrainingData:
    """测试训练数据准备功能"""
    
    def test_prepare_with_chat_template(self):
        """测试带聊天模板的数据准备"""
        from src.trainer import prepare_training_data
        
        # 创建测试数据
        test_data = [
            {"instruction": "你好", "input": "", "output": "你好！"},
            {"instruction": "你叫什么", "input": "", "output": "我是AI助手"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                output_path = f.name
            
            prepare_training_data(
                input_path,
                output_path,
                chat_template=True
            )
            
            # 验证输出
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                assert len(lines) == 2
                
                # 验证格式
                item = json.loads(lines[0])
                assert "messages" in item
                assert len(item["messages"]) == 2
                assert item["messages"][0]["role"] == "user"
                assert item["messages"][1]["role"] == "assistant"
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_prepare_without_chat_template(self):
        """测试不带聊天模板的数据准备"""
        from src.trainer import prepare_training_data
        
        test_data = [
            {"instruction": "问题", "input": "", "output": "答案"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                output_path = f.name
            
            prepare_training_data(
                input_path,
                output_path,
                chat_template=False
            )
            
            # 验证输出
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                item = json.loads(lines[0])
                assert "text" in item
                assert "### Instruction:" in item["text"]
                assert "### Response:" in item["text"]
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_prepare_creates_parent_dir(self):
        """测试自动创建输出目录"""
        from src.trainer import prepare_training_data
        
        test_data = [{"instruction": "问题", "output": "答案"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        try:
            output_path = "/tmp/nonexistent_dir/output.jsonl"
            
            prepare_training_data(input_path, output_path)
            
            assert os.path.exists(output_path)
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            if os.path.exists("/tmp/nonexistent_dir"):
                os.rmdir("/tmp/nonexistent_dir")


class TestMergeModel:
    """测试模型合并功能"""
    
    def test_merge_model_signature(self):
        """测试合并函数签名"""
        from src.trainer import merge_model
        import inspect
        
        sig = inspect.signature(merge_model)
        params = list(sig.parameters.keys())
        
        assert "base_model_path" in params
        assert "lora_model_path" in params
        assert "output_path" in params


class TestTrainLora:
    """测试LoRA训练功能"""
    
    def test_train_lora_signature(self):
        """测试训练函数签名"""
        from src.trainer import train_lora
        import inspect
        
        sig = inspect.signature(train_lora)
        params = list(sig.parameters.keys())
        
        assert "model_name" in params
        assert "data_path" in params
        assert "output_dir" in params
        assert "lora_config" in params
        assert "batch_size" in params
        assert "learning_rate" in params
        assert "epochs" in params
        assert "max_length" in params
