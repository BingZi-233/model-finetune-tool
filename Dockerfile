# ================================================
# model-finetune-tool Dockerfile
# ================================================
# 构建镜像: docker build -t model-finetune-tool .
# 运行容器: docker run -it --rm -v $(pwd):/app model-finetune-tool finetune --help
# ================================================

# 基础镜像 - 使用 Python 3.11 slim
FROM python:3.11-slim

# 维护者信息
LABEL maintainer="model-finetune-tool"
LABEL description="方便的大模型微调工具"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（用于 PDF 处理等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data /app/output /app/documents

# 设置默认命令
CMD ["finetune", "--help"]

# ================================================
# 构建说明
# ================================================
# 1. 构建镜像:
#    docker build -t model-finetune-tool .
#
# 2. 运行容器:
#    docker run -it --rm \
#        -v $(pwd)/documents:/app/documents \
#        -v $(pwd)/data:/app/data \
#        -v $(pwd)/output:/app/output \
#        -e OPENAI_API_KEY=your-api-key \
#        model-finetune-tool \
#        finetune parse ./documents my_dataset
#
# 3. 使用 Docker Compose (推荐):
#    docker-compose up -d
# ================================================
