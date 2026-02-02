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


class TestTrainLora:
    """测试LoRA训练功能"""
    
    @patch('src.trainer.AutoModelForCausalLM')
    @patch('src.trainer.AutoTokenizer')
    @patch('src.trainer.load_dataset')
    @patch('src.trainer.get_peft_model')
    @patch('src.trainer.Trainer')
    def test_train_lora_basic(
        self,
        mock_trainer_class,
        mock_get_peft,
        mock_load_dataset,
        mock_tokenizer,
        mock_model
    ):
        """测试基本LoRA训练流程"""
        # 模拟模型加载
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # 模拟PeftModel
        mock_peft_model = Mock()
        mock_get_peft.return_value = mock_peft_model
        mock_peft_model.print_trainable_parameters = Mock()
        
        # 模拟数据集
        mock_dataset = Mock()
        mock_dataset.map = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        # 模拟Trainer
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        from src.trainer import train_lora
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"instruction": "问题", "output": "答案"}\n')
            data_path = f.name
        
        try:
            train_lora(
                model_name="test-model",
                data_path=data_path,
                output_dir="/tmp/test_output",
                batch_size=2,
                epochs=1
            )
            
            # 验证模型加载
            mock_model.from_pretrained.assert_called_once()
            mock_tokenizer.from_pretrained.assert_called_once()
            
            # 验证LoRA配置
            mock_get_peft.assert_called_once()
            
            # 验证数据集加载
            mock_load_dataset.assert_called_once()
            
            # 验证Trainer创建
            mock_trainer_class.assert_called_once()
            
            # 验证训练执行
            mock_trainer_instance.train.assert_called_once()
            
            # 验证模型保存
            mock_peft_model.save_pretrained.assert_called()
            mock_tokenizer_instance.save_pretrained.assert_called()
        finally:
            os.unlink(data_path)
    
    @patch('src.trainer.AutoModelForCausalLM')
    @patch('src.trainer.AutoTokenizer')
    @patch('src.trainer.load_dataset')
    @patch('src.trainer.get_peft_model')
    @patch('src.trainer.Trainer')
    def test_train_lora_with_custom_config(
        self,
        mock_trainer_class,
        mock_get_peft,
        mock_load_dataset,
        mock_tokenizer,
        mock_model
    ):
        """测试带自定义配置的LoRA训练"""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_load_dataset.return_value = Mock()
        mock_trainer_class.return_value = Mock()
        
        from src.trainer import train_lora
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"instruction": "问题", "output": "答案"}\n')
            data_path = f.name
        
        try:
            custom_lora_config = {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj"]
            }
            
            train_lora(
                model_name="test-model",
                data_path=data_path,
                output_dir="/tmp/test_output",
                lora_config=custom_lora_config,
                batch_size=4,
                learning_rate=0.001,
                epochs=5
            )
            
            # 验证LoRA配置被应用
            call_args = mock_get_peft.call_args
            lora_config = call_args[0][1]
            
            assert lora_config.r == 16
            assert lora_config.lora_alpha == 32
            assert lora_config.target_modules == ["q_proj", "k_proj"]
        finally:
            os.unlink(data_path)


class TestMergeModel:
    """测试模型合并功能"""
    
    @patch('src.trainer.AutoModelForCausalLM')
    @patch('src.trainer.AutoTokenizer')
    @patch('src.trainer.PeftModel')
    def test_merge_model_basic(self, mock_peft_model, mock_tokenizer, mock_model):
        """测试基本模型合并"""
        # 模拟基础模型
        mock_base_model = Mock()
        mock_model.from_pretrained.return_value = mock_base_model
        
        # 模拟PeftModel
        mock_peft_instance = Mock()
        mock_peft_model.from_pretrained.return_value = mock_peft_instance
        mock_peft_instance.merge_and_unload.return_value = mock_base_model
        
        from src.trainer import merge_model
        
        merge_model(
            base_model_path="/tmp/base_model",
            lora_model_path="/tmp/lora_model",
            output_path="/tmp/merged_model"
        )
        
        # 验证基础模型加载
        mock_model.from_pretrained.assert_called_with(
            "/tmp/base_model",
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 验证LoRA加载
        mock_peft_model.from_pretrained.assert_called_with(
            mock_base_model,
            "/tmp/lora_model"
        )
        
        # 验证合并
        mock_peft_instance.merge_and_unload.assert_called_once()
        
        # 验证保存
        mock_base_model.save_pretrained.assert_called_with("/tmp/merged_model")
        mock_tokenizer.from_pretrained.assert_called_with("/tmp/base_model")
        mock_tokenizer_instance.save_pretrained.assert_called_with("/tmp/merged_model")
