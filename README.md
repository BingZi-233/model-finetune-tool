# model-finetune-tool

方便的大模型微调工具 🌹

## 特性

- 📄 支持解析 Word(docx)、PDF、Markdown 文档
- 🤖 使用 OpenAI API 生成高质量训练数据
- 💾 SQLite 数据库缓存（支持 MySQL/PostgreSQL）
- ⚡ 基于 LoRA 的高效微调
- 📋 YAML 配置文件
- 📊 数据集版本管理
- 🪟 **Windows 兼容** - 完整的 Windows 支持
- 🔧 **开发友好** - 完善的测试和文档

## 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd model-finetune-tool

# 安装依赖
pip install -e .
```

**Windows 用户**：请参阅 [Windows 安装指南](docs/windows-guide.md) 获取详细的 Windows 安装说明。

## 快速开始（推荐）

### Linux / macOS

```bash
# 1. 配置 API Key
export OPENAI_API_KEY="your-api-key"

# 2. 初始化项目（自动创建配置和目录）
./finetune.sh init

# 3. 准备文档
cp your-documents/*.md documents/

# 4. 解析文档生成数据
./finetune.sh parse ./documents my_dataset

# 5. 训练模型
./finetune.sh train my_dataset

# 6. 合并模型
./finetune.sh merge my_dataset Qwen/Qwen2.5-0.5B-Instruct
```

### Windows

```cmd
:: 1. 配置 API Key
set OPENAI_API_KEY=your-api-key

:: 2. 初始化项目
finetune.bat init

:: 3. 准备文档
copy your-docs\*.md documents\

:: 4. 解析文档生成数据
finetune.bat parse .\documents my_dataset

:: 5. 训练模型
finetune.bat train my_dataset

:: 6. 合并模型
finetune.bat merge my_dataset Qwen/Qwen2.5-0.5B-Instruct
```

## 快速启动脚本功能

| 命令 | 说明 | 示例 |
|------|------|------|
| `./finetune.sh help` | 显示帮助 | - |
| `./finetune.sh init` | 初始化项目 | - |
| `./finetune.sh check` | 检查环境 | - |
| `./finetune.sh gpu` | 检查 GPU | - |
| `./finetune.sh parse <dir> <name>` | 解析文档 | `./finetune.sh parse ./documents my_dataset` |
| `./finetune.sh export <name>` | 导出数据 | `./finetune.sh export my_dataset -o train.jsonl` |
| `./finetune.sh train <name>` | 训练模型 | `./finetune.sh train my_dataset -e 3` |
| `./finetune.sh merge <name> <model>` | 合并模型 | `./finetune.sh merge my_dataset Qwen/Qwen2.5-0.5B-Instruct` |
| `./finetune.sh stats <name>` | 查看统计 | `./finetune.sh stats my_dataset` |
| `./finetune.sh clear <name>` | 清空数据 | `./finetune.sh clear my_dataset` |

## 高级用法

### 全局选项

```bash
# 详细输出
finetune parse ./documents my_dataset -v

# 安静模式
finetune parse ./documents my_dataset -q
```

### 验证 GPU

```bash
# 检查 GPU 可用性
finetune gpu-check
```

## 开发

### 开发环境

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码检查
make lint

# 格式化代码
make format
```

### 项目结构

```
model-finetune-tool/
├── finetune.sh           # Linux/Mac 快速启动脚本
├── finetune.bat          # Windows 快速启动脚本
├── src/
│   ├── __init__.py       # 包初始化，导出公共 API
│   ├── main.py           # CLI 入口
│   ├── config.py         # 配置加载和管理
│   ├── parser/           # 文档解析器
│   ├── dataset/          # 数据集管理
│   ├── llm/              # LLM 调用
│   └── trainer/          # 训练模块
├── tests/                # 测试文件
├── docs/                 # 文档
│   ├── quick-start.md    # 快速使用指南
│   ├── design.md         # 设计文档
│   ├── user-manual.md    # 详细用户手册
│   └── windows-guide.md  # Windows 安装指南
├── Makefile              # 开发命令
├── CHANGELOG.md          # 更新日志
└── CONTRIBUTING.md       # 贡献指南
```

## 文档

- [快速使用指南](docs/quick-start.md) - 🚀 快速上手（推荐）
- [用户手册](docs/user-manual.md) - 详细使用说明
- [设计文档](docs/design.md) - 架构设计
- [Windows 指南](docs/windows-guide.md) - Windows 安装说明
- [API 参考](docs/api/reference.md) - 编程接口
- [贡献指南](CONTRIBUTING.md) - 如何贡献代码

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新历史。

## 贡献

欢迎贡献代码！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## License

MIT
