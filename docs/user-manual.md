# 📖 使用手册

> model-finetune-tool v0.1.0 使用指南

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 环境准备](#2-环境准备)
- [3. 安装部署](#3-安装部署)
- [4. 配置说明](#4-配置说明)
- [5. 使用教程](#5-使用教程)
- [6. 常见问题](#6-常见问题)
- [7. 最佳实践](#7-最佳实践)

---

## 1. 快速开始

### 1.1 5分钟上手

```bash
# 1. 克隆并安装
git clone https://github.com/yourname/model-finetune-tool.git
cd model-finetune-tool
pip install -e .

# 2. 配置API密钥
export OPENAI_API_KEY="sk-xxx"

# 3. 准备文档
mkdir -p documents
# 把你的docx/pdf/md文件放入documents目录

# 4. 解析文档生成数据
finetune parse ./documents my_dataset

# 5. 查看数据集
finetune stats my_dataset

# 6. 训练模型
finetune train my_dataset
```

### 1.2 预期输出

```
✅ 解析完成！共生成 150 条数据
✅ 训练完成！模型保存到: ./output/my_dataset/lora_model
```

---

## 2. 环境准备

### 2.1 系统要求

| 要求 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Python | 3.10 | 3.11+ |
| 内存 | 4GB | 16GB+ |
| 磁盘 | 10GB | 50GB+ |
| GPU | 可选 | NVIDIA 8GB+ |

### 2.2 硬件配置建议

| 使用场景 | 配置 | 说明 |
|----------|------|------|
| 学习/测试 | CPU即可 | 解析文档、生成数据 |
| 微调训练 | GPU 8GB+ | Qwen2.5-0.5B 可在消费级GPU运行 |
| 生产部署 | GPU 16GB+ | 可运行更大模型 |

### 2.3 依赖环境

```bash
# Python 3.10+
python --version

# Git
git --version

# (可选) CUDA (用于GPU训练)
nvidia-smi
```

---

## 3. 安装部署

### 3.1 克隆项目

```bash
# 克隆仓库
git clone https://github.com/yourname/model-finetune-tool.git
cd model-finetune-tool
```

### 3.2 安装依赖

#### 方式一：pip (推荐)

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate   # Windows

# 安装项目
pip install -e .
```

#### 方式二：Poetry

```bash
# 安装Poetry (如果未安装)
pip install poetry

# 安装依赖
poetry install
```

### 3.3 验证安装

```bash
# 查看版本
finetune --version

# 查看帮助
finetune --help

# 预期输出：
# Usage: finetune [OPTIONS] COMMAND [ARGS]...
# 
# Commands:
#   clear     清空数据集
#   export    导出数据集
#   init      初始化项目
#   merge     合并模型
#   parse     解析文档
#   stats     查看统计
#   train     训练模型
```

---

## 4. 配置说明

### 4.1 配置文件位置

项目根目录下的 `config.yaml`：

```bash
model-finetune-tool/
├── config.yaml          # 主配置文件
├── pyproject.toml       # 项目配置
└── ...
```

### 4.2 最小配置

只需要配置LLM API密钥即可开始：

```yaml
llm:
  api_key: "${OPENAI_API_KEY}"  # 设置环境变量OPENAI_API_KEY
```

### 4.3 完整配置

```yaml
# =====================================
# LLM配置 (必需)
# =====================================
llm:
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000

# =====================================
# 数据库配置 (可选)
# =====================================
database:
  type: "sqlite"
  path: "./data/datasets.db"

# =====================================
# 数据集配置 (可选)
# =====================================
datasets:
  input_dir: "./documents"
  chunk_size: 1000
  chunk_overlap: 200

# =====================================
# 训练配置 (可选)
# =====================================
training:
  model_name: "Qwen/Qwen2.5-0.5B-Instruct"
  lora:
    r: 8
    alpha: 16
    dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  batch_size: 4
  learning_rate: 0.0002
  epochs: 3
  max_length: 2048

# =====================================
# 输出配置 (可选)
# =====================================
output:
  model_dir: "./output"
  checkpoint_dir: "./checkpoints"

# =====================================
# Git配置 (可选)
# =====================================
git:
  auto_commit: true
  commit_message: "Update dataset: {dataset_name}"
```

### 4.4 配置项详解

#### LLM配置

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | 是 | - | API密钥，支持环境变量 |
| `base_url` | 否 | OpenAI官方 | API基础URL |
| `model` | 否 | gpt-3.5-turbo | 模型名称 |
| `temperature` | 否 | 0.7 | 生成温度 (0-2) |
| `max_tokens` | 否 | 2000 | 最大生成长度 |

#### 数据库配置

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | 否 | sqlite | 数据库类型: sqlite/mysql/postgresql |
| `path` | 否 | ./data/datasets.db | SQLite文件路径 |
| `host` | 否 | localhost | MySQL/PostgreSQL主机 |
| `port` | 否 | 3306/5432 | 端口 |
| `username` | 否 | - | 用户名 |
| `password` | 否 | - | 密码 |
| `database` | 否 | - | 数据库名 |

#### 训练配置

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | 否 | Qwen2.5-0.5B | HuggingFace模型名 |
| `lora.r` | 否 | 8 | LoRA rank |
| `lora.alpha` | 否 | 16 | LoRA alpha |
| `lora.dropout` | 否 | 0.1 | Dropout比例 |
| `batch_size` | 否 | 4 | 批次大小 |
| `learning_rate` | 否 | 0.0002 | 学习率 |
| `epochs` | 否 | 3 | 训练轮数 |

---

## 5. 使用教程

### 5.1 教程一：基础使用流程

#### 步骤1：准备文档

创建 `documents` 目录，放入你的文档：

```bash
mkdir -p documents

# 放入文档
cp /path/to/your/*.md documents/
cp /path/to/your/*.docx documents/
cp /path/to/your/*.pdf documents/
```

支持的格式：
- `.docx` - Word文档
- `.pdf` - PDF文档  
- `.md` - Markdown文档

#### 步骤2：配置API密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="sk-your-api-key"

# 或在config.yaml中直接配置
# llm:
#   api_key: "sk-your-api-key"
```

#### 步骤3：解析文档生成数据

```bash
# 基本用法
finetune parse ./documents my_dataset

# 高级用法
finetune parse ./documents my_dataset \
    --recursive \
    --chunk-size 1500 \
    --qa-pairs 5

# 参数说明：
# --recursive    递归处理子目录
# --chunk-size   文本块大小 (默认1000)
# --qa-pairs     每块生成的QA对数 (默认3)
```

预期输出：
```
解析文档: ./documents
找到 10 个文档
处理文档: ./documents/test.md: 100%|████████████████| 10/10
✅ 完成！共生成 150 条数据
```

#### 步骤4：查看数据集统计

```bash
finetune stats my_dataset
```

输出：
```
数据集: my_dataset
总条目: 150
```

#### 步骤5：导出数据（可选）

```bash
# 导出为JSONL
finetune export my_dataset -o train.jsonl

# 导出为JSON
finetune export my_dataset -o train.json --format json

# 查看导出的数据
head -n 5 train.jsonl
```

#### 步骤6：训练模型

```bash
# 基本用法
finetune train my_dataset

# 高级用法
finetune train my_dataset \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 5 \
    --batch-size 4

# 参数说明：
# --model     模型名称 (HuggingFace)
# --epochs    训练轮数
# --batch-size 批次大小
```

预期输出：
```
开始训练模型: Qwen/Qwen2.5-0.5B-Instruct
加载模型: Qwen/Qwen2.5-0.5B-Instruct
trainable params: 1,048,576 || all params: 487,616,000 || trainable%: 0.2150
开始训练...
Epoch 1/5: 100%|████| 10/10
✅ 训练完成！模型保存到: ./output/my_dataset/lora_model
```

#### 步骤7：合并模型（可选）

```bash
# 合并基础模型和LoRA
finetune merge my_dataset Qwen/Qwen2.5-0.5B-Instruct

# 输出目录: ./output/my_dataset/merged
```

### 5.2 教程二：使用MySQL/PostgreSQL

#### 使用MySQL

```bash
# 安装MySQL驱动
pip install pymysql

# 配置config.yaml
database:
  type: "mysql"
  host: "localhost"
  port: 3306
  username: "root"
  password: "your_password"
  database: "model_finetune"
```

```bash
# 创建数据库
mysql -u root -p
CREATE DATABASE model_finetune;
```

#### 使用PostgreSQL

```bash
# 安装PostgreSQL驱动
pip install psycopg2-binary

# 配置config.yaml
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  username: "postgres"
  password: "your_password"
  database: "model_finetune"
```

```bash
# 创建数据库
psql -U postgres
CREATE DATABASE model_finetune;
```

### 5.3 教程三：使用其他LLM

#### DeepSeek

```yaml
llm:
  api_key: "${DEEPSEEK_API_KEY}"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
```

#### 阿里通义千问

```yaml
llm:
  api_key: "${DASHSCOPE_API_KEY}"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-turbo"
```

#### Ollama (本地)

```yaml
llm:
  api_key: "ollama"  # 任意非空值
  base_url: "http://localhost:11434/v1"
  model: "llama3"
```

### 5.4 教程四：模型推理测试

训练完成后，测试模型效果：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载合并后的模型
model_path = "./output/my_dataset/merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 推理测试
prompt = "请介绍一下你自己"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 6. 常见问题

### Q1: 解析PDF报错？

**问题**：`fitz` 模块未安装  
**解决**：
```bash
pip install pymupdf
```

### Q2: LLM调用超时？

**问题**：网络问题或API限流  
**解决**：
```yaml
# 增加超时时间
llm:
  timeout: 60  # 60秒
```

### Q3: 内存不足？

**问题**：模型太大  
**解决**：
```yaml
# 使用更小的模型
training:
  model_name: "Qwen/Qwen2.5-0.5B-Instruct"

# 减小批次大小
training:
  batch_size: 1
```

### Q4: 训练loss不下降？

**问题**：数据质量或超参数问题  
**解决**：
1. 检查数据质量
2. 调整学习率：`0.0001` ~ `0.0003`
3. 增加训练轮数

### Q5: 生成的JSON格式错误？

**问题**：LLM响应不稳定  
**解决**：
```yaml
# 使用更稳定的模型
llm:
  model: "gpt-4o-mini"

# 或在prompt中强调JSON格式
```

### Q6: 如何增量更新数据集？

```bash
# 添加新文档后重新解析
finetune parse ./documents my_dataset

# 已处理的文档会自动跳过
```

### Q7: 如何删除数据集？

```bash
# 清空数据集（保留配置）
finetune clear my_dataset

# 删除整个数据库文件
rm ./data/datasets.db
```

### Q8: 训练速度慢？

**建议**：
1. 使用GPU加速
2. 减小 `max_length`
3. 增加 `batch_size`
4. 使用 `fp16` 混合精度

### Q9: 如何查看训练日志？

```bash
# 训练日志保存在 checkpoints 目录
cat ./output/my_dataset/checkpoints/trainer_state.json
```

### Q10: 怎么切换不同的数据集？

```bash
# 解析为新数据集
finetune parse ./documents new_dataset

# 训练新数据集
finetune train new_dataset
```

---

## 7. 最佳实践

### 7.1 数据准备

| 建议 | 说明 |
|------|------|
| ✅ 清理无关内容 | 删除广告、导航栏等 |
| ✅ 统一格式 | 建议使用Markdown |
| ✅ 控制长度 | 单个文件不宜过大 |
| ✅ 丰富内容 | 包含多种主题和问答类型 |

### 7.2 数据量建议

| 模型规模 | 建议数据量 | 说明 |
|----------|------------|------|
| 0.5B | 1K-5K | 小模型数据量不宜过多 |
| 1B | 5K-20K | 中等规模 |
| 3B+ | 20K+ | 大模型可吸收更多数据 |

### 7.3 超参数选择

| 参数 | 推荐值 | 调整建议 |
|------|--------|----------|
| learning_rate | 0.0002 | 数据量大时适当减小 |
| batch_size | 4-8 | 显存允许时增大 |
| epochs | 3-5 | 根据loss收敛情况调整 |
| lora.r | 8-16 | 复杂任务用大值 |

### 7.4 故障排查

```bash
# 1. 启用详细日志
finetune parse ./documents my_dataset -v

# 2. 检查数据库
sqlite3 ./data/datasets.db
sqlite> SELECT COUNT(*) FROM dataset_items;

# 3. 验证配置文件
python -c "from src.config import load_config; load_config('config.yaml')"
```

### 7.5 资源清理

```bash
# 清理临时文件
rm -rf ./tmp/*
rm -rf ./data/cache/*

# 清理旧的检查点
rm -rf ./output/*/checkpoints/*
```

---

## 附录

### A. 命令参考

| 命令 | 描述 |
|------|------|
| `finetune init` | 初始化项目 |
| `finetune parse <dir> <name>` | 解析文档 |
| `finetune export <name>` | 导出数据 |
| `finetune stats <name>` | 查看统计 |
| `finetune train <name>` | 训练模型 |
| `finetune merge <name> <base>` | 合并模型 |
| `finetune clear <name>` | 清空数据 |

### B. 配置文件模板

见 `config.yaml`

### C. 相关资源

- [项目仓库](https://github.com/yourname/model-finetune-tool)
- [HuggingFace Hub](https://huggingface.co/models)
- [OpenAI API文档](https://platform.openai.com/docs)
- [LoRA论文](https://arxiv.org/abs/2106.09685)

---

**祝使用愉快！** 🎉
