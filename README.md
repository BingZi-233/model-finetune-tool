# model-finetune-tool

方便的大模型微调工具 🌹

## 特性

- 📄 支持解析 Word(docx)、PDF、Markdown 文档
- 🤖 使用 OpenAI API 生成高质量训练数据
- 💾 SQLite 数据库缓存（支持 MySQL/PostgreSQL）
- ⚡ 基于 LoRA 的高效微调
- 📋 YAML 配置文件
- 📊 数据集版本管理

## 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd model-finetune-tool

# 安装依赖
pip install -e .
```

## 配置

编辑 `config.yaml`：

```yaml
# OpenAI API配置
llm:
  api_key: "${OPENAI_API_KEY}"  # 使用环境变量
  model: "gpt-3.5-turbo"

# 数据库配置
database:
  type: "sqlite"
  path: "./data/datasets.db"

# 文档目录
datasets:
  input_dir: "./documents"
```

## 使用方法

### 1. 准备文档

将你的文档放入 `./documents` 目录，支持：
- `.docx` - Word文档
- `.pdf` - PDF文档
- `.md` - Markdown文档

### 2. 解析文档生成数据集

```bash
# 解析文档
finetune parse ./documents my_dataset

# 指定QA对数量
finetune parse ./documents my_dataset -n 5

# 查看统计
finetune stats my_dataset
```

### 3. 导出数据

```bash
# 导出为JSONL
finetune export my_dataset -o train.jsonl

# 导出为JSON
finetune export my_dataset -o train.json --format json
```

### 4. 训练模型

```bash
# 训练LoRA
finetune train my_dataset

# 指定参数
finetune train my_dataset -m Qwen/Qwen2.5-0.5B-Instruct -e 3 -b 4
```

### 5. 合并模型

```bash
# 合并基础模型和LoRA
finetune merge my_dataset Qwen/Qwen2.5-0.5B-Instruct
```

## 项目结构

```
model-finetune-tool/
├── config.yaml          # 配置文件
├── pyproject.toml       # 项目配置
├── src/
│   ├── main.py          # CLI入口
│   ├── config.py        # 配置加载
│   ├── parser/          # 文档解析器
│   ├── dataset/         # 数据集管理
│   ├── llm/             # LLM调用
│   └── trainer/         # 训练模块
├── data/                # 数据目录（git忽略）
│   └── datasets.db      # SQLite数据库
└── documents/           # 文档目录
```

## 数据库结构

每个数据集存储在SQLite中，包含：

- `documents` - 文档记录
- `dataset_items` - 训练数据条目

## 扩展

### 支持更多LLM

修改 `config.yaml`：

```yaml
llm:
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
```

### 自定义解析器

在 `src/parser/` 添加新的解析器类。

## License

MIT
