# 快速使用指南

> model-finetune-tool 快速上手指南

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 快速启动](#2-快速启动)
- [3. 完整流程](#3-完整流程)
- [4. 常见问题](#4-常见问题)

---

## 1. 环境准备

### 前置要求

| 要求 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Python | 3.10 | 3.11+ |
| Git | 任意版本 | 最新版 |
| 内存 | 4GB | 8GB+ |
| 磁盘 | 10GB | 50GB+ |

### 1.1 克隆项目

```bash
# 克隆项目
git clone https://github.com/BingZi-233/model-finetune-tool.git
cd model-finetune-tool
```

### 1.2 设置 API Key

**Linux / macOS:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

> 💡 提示：将上述命令添加到 `~/.bashrc`、`~/.zshrc` 或 PowerShell Profile 中，永久生效。

---

## 2. 快速启动

### 2.1 使用启动脚本（推荐）

#### Linux / macOS

```bash
# 查看帮助
./finetune.sh help

# 初始化项目（自动创建配置和目录）
./finetune.sh init

# 检查环境
./finetune.sh check
```

#### Windows

```cmd
:: 查看帮助
finetune.bat help

:: 初始化项目
finetune.bat init

:: 检查环境
finetune.bat check
```

### 2.2 手动启动

#### Linux / macOS

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e .

# 配置 API Key
export OPENAI_API_KEY="your-api-key"

# 运行
finetune --help
```

#### Windows

```cmd
:: 创建虚拟环境
python -m venv venv

:: 激活虚拟环境
venv\Scripts\activate

:: 安装依赖
pip install -e .

:: 配置 API Key
set OPENAI_API_KEY=your-api-key

:: 运行
finetune --help
```

---

## 3. 完整流程

### 步骤 1: 准备文档

创建 `documents` 目录，放入你的文档：

```bash
mkdir -p documents

# 放入文档
cp /path/to/your/*.md documents/
cp /path/to/your/*.docx documents/
cp /path/to/your/*.pdf documents/
```

**支持的格式：**
- `.md` - Markdown
- `.docx` - Word
- `.pdf` - PDF

### 步骤 2: 解析文档

```bash
# 基本用法
./finetune.sh parse ./documents my_dataset

# 高级用法
./finetune.sh parse ./documents my_dataset \
    --chunk-size 1500 \
    --qa-pairs 5
```

**参数说明：**
- `--chunk-size` - 文本块大小（默认 1000）
- `--qa-pairs` - 每块生成的 QA 对数（默认 3）

**预期输出：**
```
🔄 处理文档: 100%|████████████████| 10/10
✅ 完成！共生成 150 条数据
```

### 步骤 3: 查看数据

```bash
./finetune.sh stats my_dataset
```

**输出示例：**
```
数据集: my_dataset
总条目: 150
```

### 步骤 4: 导出数据（可选）

```bash
# 导出为 JSONL
./finetune.sh export my_dataset -o train.jsonl

# 导出为 JSON
./finetune.sh export my_dataset -o train.json --format json
```

### 步骤 5: 训练模型

```bash
# 基本用法
./finetune.sh train my_dataset

# 高级用法
./finetune.sh train my_dataset \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    -e 3 \
    -b 4
```

**参数说明：**
- `-m, --model` - 模型名称（默认 Qwen/Qwen2.5-0.5B-Instruct）
- `-e, --epochs` - 训练轮数（默认 3）
- `-b, --batch-size` - 批次大小（默认 4）

**预期输出：**
```
✅ 加载模型: Qwen/Qwen2.5-0.5B-Instruct
✅ trainable params: 1,048,576
🔄 开始训练...
✅ 训练完成！模型保存到: ./output/my_dataset/lora_model
```

### 步骤 6: 合并模型

```bash
./finetune.sh merge my_dataset Qwen/Qwen2.5-0.5B-Instruct
```

**输出目录：**
```
output/
└── my_dataset/
    ├── checkpoints/
    ├── lora_model/
    └── merged/
```

---

## 4. 常见问题

### Q1: 找不到 Python？

**错误信息：** `未找到 Python，请先安装 Python 3.10+`

**解决：**
1. 访问 [Python 官网](https://python.org/downloads)
2. 下载并安装 Python 3.11
3. 勾选 ✅ `Add Python to PATH`

### Q2: API Key 无效？

**错误信息：** `Environment variable not found: OPENAI_API_KEY`

**解决：**
1. 检查 API Key 是否正确设置
2. 确认 API Key 有足够的余额
3. 检查 API Key 是否过期

### Q3: 训练太慢？

**解决：**
1. 确保使用 GPU 训练
2. 减小 `batch_size`
3. 减小 `max_length`

### Q4: 内存不足？

**错误信息：** `CUDA out of memory`

**解决：**
1. 减小 `batch_size`（如改为 1 或 2）
2. 使用更小的模型（如 Qwen/Qwen2.5-0.5B-Instruct）
3. 使用 CPU 训练（仅用于测试）

### Q5: 文档解析失败？

**解决：**
1. 检查文档格式是否支持
2. 检查文件权限
3. 使用 `--verbose` 查看详细错误

### Q6: Windows 上无法运行脚本？

**解决：**
1. 使用 `finetune.bat` 代替 `./finetune.sh`
2. 或在 PowerShell 中运行：
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\finetune.sh help
   ```

---

## 脚本命令参考

| 命令 | 说明 |
|------|------|
| `help` | 显示帮助 |
| `init` | 初始化项目 |
| `check` | 检查环境 |
| `gpu` | 检查 GPU |
| `parse <dir> <name>` | 解析文档 |
| `export <name>` | 导出数据 |
| `train <name>` | 训练模型 |
| `merge <name> <model>` | 合并模型 |
| `stats <name>` | 查看统计 |
| `clear <name>` | 清空数据 |

---

## 下一步

- 📖 阅读 [用户手册](docs/user-manual.md) 了解详细用法
- 🏗️ 阅读 [设计文档](docs/design.md) 了解架构
- 💻 阅读 [API 文档](docs/api/reference.md) 了解编程接口

---

**祝使用愉快！** 🎉
