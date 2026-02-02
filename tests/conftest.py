"""
Pytest配置文件和共享夹具

本模块定义测试中使用的共享夹具和配置。
"""
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def temp_dir():
    """创建临时目录"""
    temp = tempfile.mkdtemp()
    yield temp
    # 清理
    import shutil
    if os.path.exists(temp):
        shutil.rmtree(temp)


@pytest.fixture
def sample_documents_dir(temp_dir):
    """创建测试文档目录"""
    doc_dir = os.path.join(temp_dir, "documents")
    os.makedirs(doc_dir)
    
    # 创建测试Markdown文件
    md_content = """# 测试文档

这是一个用于测试的Markdown文档。

## 第一章

这是第一章的内容，包含了一些重要的信息。

```python
def hello():
    print("Hello, World!")
```

## 第二章

这是第二章的内容，包含了更多的信息。

- 第一点
- 第二点
- 第三点
"""
    with open(os.path.join(doc_dir, "test.md"), 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 创建另一个测试文件
    md_content2 = """# 另一个测试文档

这是第二个测试文档。
"""
    with open(os.path.join(doc_dir, "test2.md"), 'w', encoding='utf-8') as f:
        f.write(md_content2)
    
    yield doc_dir
    
    # 清理
    import shutil
    if os.path.exists(doc_dir):
        shutil.rmtree(doc_dir)


@pytest.fixture
def sample_dataset_jsonl(temp_dir):
    """创建测试数据集JSONL文件"""
    data = [
        {"instruction": "什么是机器学习？", "input": "", "output": "机器学习是人工智能的一个分支..."},
        {"instruction": "请解释深度学习", "input": "", "output": "深度学习是机器学习的一种方法..."},
        {"instruction": "神经网络是什么？", "input": "", "output": "神经网络是一种模拟人脑的结构..."},
    ]
    
    file_path = os.path.join(temp_dir, "test_dataset.jsonl")
    
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    yield file_path
    
    # 清理
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def mock_llm_response():
    """模拟LLM响应"""
    import json
    return json.dumps([
        {"instruction": "问题1", "input": "", "output": "答案1"},
        {"instruction": "问题2", "input": "", "output": "答案2"},
        {"instruction": "问题3", "input": "", "output": "答案3"}
    ])


@pytest.fixture
def clean_config():
    """提供清洁的配置环境"""
    # 确保没有预加载的配置
    import sys
    if 'src.config' in sys.modules:
        del sys.modules['src.config']
    
    yield
    
    # 清理
    if 'src.config' in sys.modules:
        del sys.modules['src.config']


@pytest.fixture
def env_vars():
    """设置测试环境变量"""
    os.environ["TEST_API_KEY"] = "test-api-key-for-testing"
    os.environ["TEST_MODEL"] = "gpt-3.5-turbo-test"
    
    yield
    
    # 清理
    del os.environ["TEST_API_KEY"]
    del os.environ["TEST_MODEL"]
