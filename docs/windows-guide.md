# Windows 安装指南

> model-finetune-tool v0.1.0 Windows 平台安装说明

## 目录

- [1. 系统要求](#1-系统要求)
- [2. 安装步骤](#2-安装步骤)
- [3. 常见问题](#3-常见问题)
- [4. 故障排查](#4-故障排查)

---

## 1. 系统要求

| 要求 | 详情 |
|------|------|
| 操作系统 | Windows 10 / Windows 11 |
| Python | 3.10 或更高版本 |
| 内存 | 4GB+ (推荐 8GB+) |
| 磁盘 | 10GB+ 可用空间 |
| GPU | 可选 (用于训练) |

## 2. 安装步骤

### 2.1 安装 Python

1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载 Python 3.10 或更高版本
3. **重要**: 安装时勾选 ✅ `Add Python to PATH`
4. 点击 `Install Now`

### 2.2 验证 Python 安装

打开 **命令提示符** (CMD) 或 **PowerShell**，运行：

```cmd
python --version
```

输出示例：
```
Python 3.11.5
```

### 2.3 安装 Git (可选)

如需从 Git 克隆项目，需要安装 Git：

1. 访问 [Git for Windows](https://git-scm.com/download/win)
2. 下载并安装
3. 验证安装：
```cmd
git --version
```

### 2.4 克隆并安装项目

#### 方式一：从 Git 克隆

```cmd
git clone https://github.com/BingZi-233/model-finetune-tool.git
cd model-finetune-tool
```

#### 方式二：下载 ZIP

1. 下载项目 ZIP 文件
2. 解压到 `C:\Users\你的用户名\model-finetune-tool`
3. 打开 CMD，进入目录：
```cmd
cd C:\Users\你的用户名\model-finetune-tool
```

### 2.5 创建虚拟环境

```cmd
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate
```

激活成功后，命令行会显示 `(venv)` 前缀。

### 2.6 安装依赖

```cmd
pip install -e .
```

### 2.7 配置环境变量

创建 `.env` 文件：

```cmd
notepad .env
```

添加以下内容：

```
OPENAI_API_KEY=your_api_key_here
```

> 💡 提示：在 PowerShell 中使用 `notepad .env` 创建文件

### 2.8 验证安装

```cmd
finetune --help
```

预期输出：
```
Usage: finetune [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  clear     清空数据集
  export    导出数据集
  init      初始化项目
  merge     合并模型
  parse     解析文档
  stats     查看统计
  train     训练模型
```

---

## 3. 常见问题

### Q1: pip 安装失败？

**问题**：`pip` 找不到包或版本冲突

**解决**：
```cmd
# 升级 pip
python -m pip install --upgrade pip

# 清除缓存后重试
pip cache purge
pip install -e .
```

### Q2: Python 不在 PATH 中？

**解决**：
1. 搜索 "环境变量"
2. 点击 "编辑系统环境变量"
3. 点击 "环境变量"
4. 在 "系统变量" 中找到 "Path"
5. 添加 Python 路径，例如：
   - `C:\Users\用户名\AppData\Local\Programs\Python\Python311\`
   - `C:\Users\用户名\AppData\Local\Programs\Python\Python311\Scripts\`

### Q3: 虚拟环境无法激活？

**问题**：`venv\Scripts\activate` 报错

**解决**：
```cmd
# 使用完整路径
C:\项目路径\venv\Scripts\activate.bat

# 或在 PowerShell 中
C:\项目路径\venv\Scripts\Activate.ps1
```

如果 PowerShell 报错，执行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q4: 模块导入错误？

**问题**：`ModuleNotFoundError: No module named 'xxx'`

**解决**：
```cmd
# 重新安装依赖
pip install -e .

# 或单独安装缺失的模块
pip install 模块名
```

---

## 4. 故障排查

### 4.1 文件路径问题

Windows 路径使用反斜杠 `\`，可能与代码中的 `/` 冲突。

**解决**：项目代码已内置路径规范化处理，无需手动转换。

### 4.2 编码问题

Windows CMD 默认使用 GBK 编码，可能导致中文显示乱码。

**解决**：
```cmd
# 设置 UTF-8 编码
chcp 65001
```

或在 PowerShell 中：
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### 4.3 权限问题

如果遇到权限错误，尝试：

1. **以管理员身份运行** CMD
2. 或修改目录权限：
```cmd
# 右键项目文件夹 > 属性 > 安全
# 添加当前用户并赋予完全控制权
```

### 4.4 杀毒软件拦截

某些杀毒软件可能拦截 Python 脚本。

**解决**：
1. 将 Python 添加到白名单
2. 将项目目录添加到白名单

### 4.5 日志查看

如果遇到错误，查看日志：

```cmd
# 启用详细输出
finetune parse ./documents my_dataset -v
```

---

## Windows 特定注意事项

### 1. 路径格式

项目代码会自动处理路径分隔符，但手动输入路径时：
- ✅ 支持：`./documents` 或 `.\documents`
- ✅ 支持：`C:\Users\文档`
- ❌ 避免：混合使用 `/` 和 `\`

### 2. 文件大小限制

Windows 文件系统 (FAT32) 有单文件 4GB 限制。

**解决**：使用 NTFS 格式的磁盘。

### 3. 临时文件

Windows 临时文件目录：
```cmd
# 查看临时目录
echo %TEMP%
```

---

## 性能优化建议

### 1. 禁用杀毒扫描

将以下目录添加到杀毒软件白名单：
- 项目目录
- Python 安装目录
- `C:\Users\用户名\AppData\Local\Temp\`

### 2. 使用 SSD

如果使用机械硬盘，训练速度会明显较慢。

### 3. 分配更多内存

如需训练大模型，可在 `config.yaml` 中减小 `batch_size`。

---

## 相关资源

- [Python 官方文档](https://docs.python.org/3/)
- [Windows 命令提示符教程](https://learn.microsoft.com/windows-server/administration/windows-commands/windows-commands)
- [pip 用户指南](https://pip.pypa.io/en/stable/user_guide/)

---

**祝使用愉快！** 🎉
