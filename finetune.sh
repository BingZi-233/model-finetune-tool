#!/bin/bash
# ================================================
# model-finetune-tool 快速启动脚本
# ================================================
# 使用方法:
#   ./finetune.sh --help
#   ./finetune.sh init
#   ./finetune.sh parse ./documents my_dataset
# ================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# 打印函数
print_info() {
    echo "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo "${RED}[ERROR]${NC} $1"
}

# 检查 Python 是否安装
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        print_info "使用 Python 3"
        return 0
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        print_info "使用 Python"
        return 0
    else
        print_error "未找到 Python，请先安装 Python 3.10+"
        echo "安装方法:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "  macOS: brew install python"
        echo "  Windows: https://python.org/downloads"
        return 1
    fi
}

# 检查并创建虚拟环境
check_venv() {
    if [ -d "${VENV_DIR}" ]; then
        print_info "找到虚拟环境"
        return 0
    fi
    
    print_info "创建虚拟环境..."
    $PYTHON_CMD -m venv "${VENV_DIR}"
    
    if [ $? -eq 0 ]; then
        print_success "虚拟环境创建成功"
    else
        print_error "虚拟环境创建失败"
        return 1
    fi
}

# 激活虚拟环境并安装依赖
setup_dependencies() {
    # 激活虚拟环境
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
    else
        print_warning "无法激活虚拟环境，直接使用系统 Python"
    fi
    
    # 检查依赖是否安装
    if ! python -c "import click" 2>/dev/null; then
        print_info "安装项目依赖..."
        pip install . --quiet
        
        if [ $? -eq 0 ]; then
            print_success "依赖安装成功"
        else
            print_error "依赖安装失败"
            return 1
        fi
    else
        print_info "依赖已安装"
    fi
}

# 检查配置文件
check_config() {
    if [ -f "${CONFIG_FILE}" ]; then
        print_info "找到配置文件: ${CONFIG_FILE}"
        return 0
    fi
    
    print_warning "未找到配置文件，正在创建..."
    
    # 检查 API Key
    if [ -z "${OPENAI_API_KEY}" ]; then
        print_warning "环境变量 OPENAI_API_KEY 未设置"
        echo "请设置环境变量或编辑 config.yaml"
        echo ""
        echo "设置方法:"
        echo "  export OPENAI_API_KEY='your-api-key'"
    fi
    
    # 创建默认配置
    cat > "${CONFIG_FILE}" << 'EOF'
# model-finetune-tool 配置文件
# 请根据需要修改配置

# LLM配置
llm:
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000

# 数据库配置
database:
  type: "sqlite"
  path: "./data/datasets.db"

# 数据集配置
datasets:
  input_dir: "./documents"
  output_dir: "./data"
  chunk_size: 1000
  chunk_overlap: 200

# 训练配置
training:
  model_name: "Qwen/Qwen2.5-0.5B-Instruct"
  batch_size: 4
  learning_rate: 0.0002
  epochs: 3
  max_length: 2048

# 输出配置
output:
  model_dir: "./output"
  checkpoint_dir: "./checkpoints"

# Git配置
git:
  auto_commit: false
  commit_message: "Update dataset: {dataset_name}"
EOF
    
    print_success "配置文件创建成功: ${CONFIG_FILE}"
    print_warning "请编辑配置文件设置 API Key"
}

# 检查 GPU
check_gpu() {
    print_info "检查 GPU 可用性..."

    # 尝试导入 torch
    if python -c "import torch" 2>/dev/null; then
        # 检查 CUDA (NVIDIA GPU)
        if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            print_success "检测到 GPU: ${GPU_NAME}"
            return 0
        fi

        # 检查 MPS (Apple Silicon)
        if python -c "import torch; print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "检测到 GPU: Apple Silicon (MPS)"
            return 0
        fi
    fi

    print_info "未检测到 GPU，将使用 CPU 训练（可能较慢）"
    return 0
}

# 创建目录结构
create_dirs() {
    print_info "创建目录结构..."
    
    mkdir -p "${SCRIPT_DIR}/data"
    mkdir -p "${SCRIPT_DIR}/documents"
    mkdir -p "${SCRIPT_DIR}/output"
    mkdir -p "${SCRIPT_DIR}/checkpoints"
    
    # 创建 .gitkeep 以保留目录（注意：documents目录不创建，避免解析时出现问题）
    touch "${SCRIPT_DIR}/data/.gitkeep"
    touch "${SCRIPT_DIR}/output/.gitkeep"
    touch "${SCRIPT_DIR}/checkpoints/.gitkeep"
    
    print_success "目录创建完成"
}

# 显示帮助
show_help() {
    cat << EOF
==========================================
model-finetune-tool 快速启动脚本
==========================================

用法: ./finetune.sh [命令] [选项]

命令:
    help          显示此帮助信息
    init          初始化项目（创建配置和目录）
    check         检查环境和依赖
    gpu           检查 GPU 可用性
    parse <dir> <name>
                  解析文档生成数据集
                  示例: ./finetune.sh parse ./documents my_dataset
    
    export <name> [选项]
                  导出数据集
                  示例: ./finetune.sh export my_dataset -o train.jsonl
    
    train <name> [选项]
                  训练模型
                  示例: ./finetune.sh train my_dataset -e 3 -b 4
    
    merge <name> <base_model>
                  合并模型
                  示例: ./finetune.sh merge my_dataset Qwen/Qwen2.5-0.5B-Instruct
    
    stats <name>  查看数据集统计
                  示例: ./finetune.sh stats my_dataset
    
    clear <name>  清空数据集
                  示例: ./finetune.sh clear my_dataset

选项:
    -v, --verbose  详细输出
    -q, --quiet    安静模式
    -h, --help     显示帮助

环境变量:
    OPENAI_API_KEY    OpenAI API Key（必需）
    OPENAI_BASE_URL   API 地址（可选）
    LOG_LEVEL         日志级别（可选）

示例:
    # 设置 API Key
    export OPENAI_API_KEY='your-api-key'
    
    # 初始化项目
    ./finetune.sh init
    
    # 准备文档
    cp your-docs/*.md documents/
    
    # 解析文档生成数据
    ./finetune.sh parse ./documents my_dataset
    
    # 训练模型
    ./finetune.sh train my_dataset
    
    # 合并模型
    ./finetune.sh merge my_dataset Qwen/Qwen2.5-0.5B-Instruct

==========================================
EOF
}

# 主函数
main() {
    echo ""
    echo "${BLUE}==========================================${NC}"
    echo "${BLUE}  model-finetune-tool 快速启动${NC}"
    echo "${BLUE}==========================================${NC}"
    echo ""
    
    # 解析参数
    case "${1:-help}" in
        help|--help|-h)
            show_help
            exit 0
            ;;
        init)
            check_python || exit 1
            check_config
            create_dirs
            check_venv
            setup_dependencies
            ;;
        check)
            check_python || exit 1
            check_venv
            setup_dependencies
            check_config
            check_gpu
            ;;
        gpu)
            check_python || exit 1
            check_gpu
            ;;
        parse)
            shift
            check_python || exit 1
            check_venv
            setup_dependencies
            source "${VENV_DIR}/bin/activate" 2>/dev/null
            finetune parse "$@"
            ;;
        export)
            shift
            check_python || exit 1
            check_venv
            setup_dependencies
            source "${VENV_DIR}/bin/activate" 2>/dev/null
            finetune export "$@"
            ;;
        train)
            shift
            check_python || exit 1
            check_venv
            setup_dependencies
            check_gpu
            source "${VENV_DIR}/bin/activate" 2>/dev/null
            finetune train "$@"
            ;;
        merge)
            shift
            check_python || exit 1
            check_venv
            setup_dependencies
            source "${VENV_DIR}/bin/activate" 2>/dev/null
            finetune merge "$@"
            ;;
        stats)
            shift
            check_python || exit 1
            check_venv
            setup_dependencies
            source "${VENV_DIR}/bin/activate" 2>/dev/null
            finetune stats "$@"
            ;;
        clear)
            shift
            check_python || exit 1
            check_venv
            setup_dependencies
            source "${VENV_DIR}/bin/activate" 2>/dev/null
            finetune clear "$@"
            ;;
        *)
            print_error "未知命令: $1"
            echo ""
            echo "使用 ./finetune.sh help 查看帮助"
            exit 1
            ;;
    esac
    
    echo ""
    print_success "操作完成！"
}

# 执行主函数
main "$@"
