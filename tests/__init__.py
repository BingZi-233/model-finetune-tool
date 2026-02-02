"""
测试模块

本模块包含项目的单元测试和集成测试。

## 测试结构

```
tests/
├── __init__.py           # 测试模块初始化
├── conftest.py           # pytest 夹具和配置
├── test_config.py        # 配置模块测试
├── test_parser.py        # 解析器测试
├── test_dataset.py       # 数据集模块测试
├── test_llm.py           # LLM 模块测试
└── test_trainer.py       # 训练模块测试
```

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_config.py -v

# 运行特定测试类
pytest tests/test_config.py::TestConfigModel -v

# 运行并生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

## 测试夹具

conftest.py 提供以下夹具：

| 夹具 | 描述 |
|------|------|
| temp_dir | 临时目录 |
| sample_documents_dir | 测试文档目录 |
| sample_dataset_jsonl | 测试数据集文件 |
| mock_llm_response | 模拟 LLM 响应 |
| clean_config | 清洁配置环境 |
| env_vars | 测试环境变量 |

## 编写测试

参考 [CONTRIBUTING.md](../CONTRIBUTING.md) 中的测试规范。

## CI/CD

测试在 GitHub Actions 中自动运行：
- Python 3.10, 3.11, 3.12
- 代码覆盖率报告
- 安全检查
"""
