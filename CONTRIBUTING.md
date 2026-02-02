# 贡献指南

> 感谢您考虑为 model-finetune-tool 贡献代码！

本指南将帮助您了解如何参与项目贡献。

## 目录

- [行为准则](#行为准则)
- [开始贡献](#开始贡献)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [测试](#测试)
- [文档](#文档)

---

## 行为准则

请尊重并友善地对待所有贡献者。我们期望：

- 使用包容和欢迎的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注社区共同利益

不可接受的行为包括：
- 使用性别歧视、种族歧视或其他歧视性语言
- 人身攻击或侮辱
- 公开或私下骚扰

## 开始贡献

### 通过 Issues 贡献

- 报告 bug
- 提出新功能建议
- 讨论架构方向
- 贡献文档改进

### 通过 Pull Requests 贡献

1. Fork 本仓库
2. 克隆到本地：
   ```bash
   git clone https://github.com/YOUR_USERNAME/model-finetune-tool.git
   cd model-finetune-tool
   ```

3. 创建特性分支：
   ```bash
   git checkout -b feature/amazing-feature
   # 或修复 bug
   git checkout -b fix/annoying-bug
   ```

4. 进行更改并测试

5. 提交更改：
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

6. 推送到您的 Fork：
   ```bash
   git push origin feature/amazing-feature
   ```

7. 创建 Pull Request

## 开发环境设置

### 前置要求

- Python 3.10+
- Git
- 推荐: GitHub CLI (`gh`)

### 安装开发依赖

```bash
# 克隆项目
git clone https://github.com/yourname/model-finetune-tool.git
cd model-finetune-tool

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -e .

# 安装开发依赖
pip install pytest pytest-cov
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行并生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

## 代码规范

### Python 风格

遵循 PEP 8 规范，使用以下工具：

```bash
# 语法检查
python -m py_compile src/*.py src/*/*.py

# 代码格式化 (需要安装 black)
pip install black
black src/ tests/
```

### 类型提示

所有新代码应包含类型提示：

```python
# ✅ 好的示例
def process_data(input_path: str, output_path: str) -> bool:
    ...

# ❌ 不好的示例
def process_data(input_path, output_path):
    ...
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def example_function(arg1: str, arg2: int = 10) -> bool:
    """函数简短描述。
    
    详细描述（如果需要）。
    
    Args:
        arg1: 参数1的描述
        arg2: 参数2的描述 (默认: 10)
        
    Returns:
        返回值的描述
        
    Raises:
        ValueError: 异常条件的描述
    """
    ...
```

### 异常处理

- 使用自定义异常类
- 提供有意义的错误消息
- 记录日志而非直接打印

```python
class CustomError(Exception):
    """自定义异常"""
    pass

try:
    risky_operation()
except SpecificError as e:
    logger.error(f"操作失败: {e}")
    raise CustomError("操作无法完成") from e
```

## 提交规范

### 提交信息格式

```
<类型>(<范围>): <描述>

[可选的正文]

[可选的脚注]
```

### 类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具更新

### 示例

```
feat(llm): 添加批量生成 QA 对功能

- 支持批量处理多个文本
- 添加进度条显示
- 优化缓存机制

Closes #123
```

## 测试

### 编写测试

- 所有新功能应有测试覆盖
- 使用 pytest 框架
- 测试文件放在 `tests/` 目录

```python
# tests/test_example.py
import pytest

class TestExample:
    def test_feature_works(self):
        """测试功能正常工作"""
        result = some_function("input")
        assert result == expected_output
    
    def test_edge_case(self):
        """测试边界情况"""
        with pytest.raises(ValueError):
            invalid_function()
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_config.py -v

# 运行特定测试类
pytest tests/test_config.py::TestConfigModel -v

# 运行并显示覆盖率
pytest tests/ --cov=src
```

## 文档

### 更新文档

- 更新 `README.md` 了解使用方法
- 更新 `docs/` 目录下的文档
- 为新功能添加使用示例

### 文档风格

- 使用中文（因为项目面向中文用户）
- 保持简洁明了
- 提供代码示例

## 审核流程

1. **自动检查** - CI 会运行测试和代码检查
2. **人工审核** - 维护者会审核您的 PR
3. **反馈** - 可能需要修改才能合并
4. **合并** - 审核通过后合并到主分支

## 建议

- 小步提交 - 每次提交做少量更改
- 清晰描述 - 说明为什么需要这个更改
- 添加测试 - 证明更改有效
- 保持简洁 - 避免不必要的复杂性

---

## 联系方式

- Issue: https://github.com/yourname/model-finetune-tool/issues
- 邮箱: maintainer@example.com

感谢您的贡献！🌹
