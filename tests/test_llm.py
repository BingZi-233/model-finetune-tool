"""
LLM调用测试模块

本模块测试LLM客户端功能，包括API调用、JSON解析、QA生成等。
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.llm import LLMClient, CacheManager


class TestLLMClient:
    """测试LLM客户端"""
    
    @patch('src.llm.OpenAI')
    def test_chat_basic(self, mock_openai_class):
        """测试基本对话功能"""
        # 模拟API响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "这是回复"
        mock_openai_class.return_value.chat.completions.create.return_value = mock_response
        
        client = LLMClient(api_key="test-key")
        result = client.chat([
            {"role": "user", "content": "你好"}
        ])
        
        assert result == "这是回复"
        
        # 验证API调用
        mock_openai_class.return_value.chat.completions.create.assert_called_once()
        call_args = mock_openai_class.return_value.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o"  # 更新为高质量配置
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "你好"}]
    
    @patch('src.llm.OpenAI')
    def test_chat_with_custom_params(self, mock_openai_class):
        """测试带自定义参数的对话"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "回复"
        mock_openai_class.return_value.chat.completions.create.return_value = mock_response
        
        client = LLMClient(api_key="test-key")
        client.chat(
            [{"role": "user", "content": "你好"}],
            temperature=0.5,
            max_tokens=100
        )
        
        call_args = mock_openai_class.return_value.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 100
    
    @patch('src.llm.OpenAI')
    def test_generate_qa_pairs(self, mock_openai_class):
        """测试生成QA对"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps([
            {"instruction": "问题1", "input": "", "output": "答案1"},
            {"instruction": "问题2", "input": "", "output": "答案2"},
            {"instruction": "问题3", "input": "", "output": "答案3"}
        ])
        mock_openai_class.return_value.chat.completions.create.return_value = mock_response
        
        client = LLMClient(api_key="test-key")
        result = client.generate_qa_pairs("这是一段测试文本", num_pairs=3)
        
        assert len(result) == 3
        assert result[0]["instruction"] == "问题1"
        assert result[0]["output"] == "答案1"
    
    @patch('src.llm.OpenAI')
    def test_generate_qa_pairs_from_code_block(self, mock_openai_class):
        """测试从代码块中提取JSON"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
这里有一些文本。

```json
[
  {"instruction": "问题1", "output": "答案1"},
  {"instruction": "问题2", "output": "答案2"}
]
```

更多文本。
"""
        mock_openai_class.return_value.chat.completions.create.return_value = mock_response
        
        client = LLMClient(api_key="test-key")
        result = client.generate_qa_pairs("测试文本")
        
        assert len(result) == 2
        assert result[0]["instruction"] == "问题1"
    
    @patch('src.llm.OpenAI')
    def test_generate_qa_pairs_with_fallback(self, mock_openai_class):
        """测试无效JSON时使用fallback生成"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "这不是JSON格式的响应内容，所以会触发fallback机制。"
        mock_openai_class.return_value.chat.completions.create.return_value = mock_response
        
        client = LLMClient(api_key="test-key")
        # 应该不会抛出异常，而是使用fallback
        result = client.generate_qa_pairs("这是一段足够长的测试文本，包含足够的字符来通过质量检查。")
        
        # 验证fallback生成了结果
        assert len(result) > 0
        assert "instruction" in result[0]
        assert "output" in result[0]
    
    @patch('src.llm.OpenAI')
    def test_generate_summarization(self, mock_openai_class):
        """测试生成摘要"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "这是摘要内容。"
        mock_openai_class.return_value.chat.completions.create.return_value = mock_response
        
        client = LLMClient(api_key="test-key")
        result = client.generate_summarization("这是一段很长的文本...")
        
        assert result == "这是摘要内容。"
    
    @patch('src.llm.OpenAI')
    def test_batch_generate_qa(self, mock_openai_class):
        """测试批量生成QA"""
        # 模拟每次调用返回2个QA
        def create_response(*args, **kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps([
                {"instruction": f"问题{i}", "output": f"答案{i}"}
                for i in range(2)
            ])
            return mock_response
        
        mock_openai_class.return_value.chat.completions.create.side_effect = create_response
        
        client = LLMClient(api_key="test-key")
        texts = ["文本1", "文本2", "文本3"]
        
        result = client.batch_generate_qa(texts, progress=False)
        
        assert len(result) == 6  # 3个文本 * 2个QA
        assert mock_openai_class.return_value.chat.completions.create.call_count == 3
    
    @patch('src.llm.OpenAI')
    def test_batch_generate_qa_skips_empty(self, mock_openai_class):
        """测试批量生成时跳过空文本"""
        def create_response(*args, **kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            # 返回2个QA对
            mock_response.choices[0].message.content = json.dumps([
                {"instruction": "问题", "output": "答案"}
            ])
            return mock_response
        
        mock_openai_class.return_value.chat.completions.create.side_effect = create_response
        
        client = LLMClient(api_key="test-key")
        # 使用更长的文本以确保生成结果
        texts = ["这是一段足够长的测试文本1，包含足够的字符来通过质量检查。", "", "  ", "这是一段足够长的测试文本2，包含足够的字符来通过质量检查。"]
        
        result = client.batch_generate_qa(texts, progress=False)
        
        # 只有2个非空文本被处理
        assert len(result) >= 2  # 至少2个QA对
        # 每个文本会尝试生成3次，所以可能是6次调用
        assert mock_openai_class.return_value.chat.completions.create.call_count == 6


class TestCacheManager:
    """测试缓存管理器"""
    
    def test_set_and_get(self):
        """测试设置和获取缓存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(cache_dir=temp_dir)
            
            # 设置缓存
            cache.set("test text", '{"result": "success"}')
            
            # 获取缓存
            result = cache.get("test text")
            
            assert result == '{"result": "success"}'
    
    def test_get_nonexistent(self):
        """测试获取不存在的缓存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(cache_dir=temp_dir)
            
            result = cache.get("nonexistent")
            
            assert result is None
    
    def test_different_texts_different_cache(self):
        """测试不同文本使用不同缓存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(cache_dir=temp_dir)
            
            cache.set("text1", "result1")
            cache.set("text2", "result2")
            
            assert cache.get("text1") == "result1"
            assert cache.get("text2") == "result2"
    
    def test_cache_key_consistency(self):
        """测试相同文本生成相同缓存key"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(cache_dir=temp_dir)
            
            # 设置缓存
            cache.set("same text", "result")
            
            # 使用相同文本获取
            result = cache.get("same text")
            
            assert result == "result"
    
    def test_cache_with_kwargs(self):
        """测试带参数的缓存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(cache_dir=temp_dir)
            
            # 不同参数应该生成不同缓存
            cache.set("text", "result1", param="a")
            cache.set("text", "result2", param="b")
            
            assert cache.get("text", param="a") == "result1"
            assert cache.get("text", param="b") == "result2"
