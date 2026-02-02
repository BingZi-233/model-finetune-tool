@echo off
REM ================================================
REM model-finetune-tool 快速启动脚本 (Windows)
REM ================================================
REM 使用方法:
REM   finetune.bat help
REM   finetune.bat init
REM   finetune.bat parse .\documents my_dataset
REM ================================================

setlocal enabledelayedexpansion

REM 颜色定义
set "BLUE=[0;34m"
set "GREEN=[0;32m"
set "YELLOW=[1;33m"
set "RED=[0;31m"
set "NC=[0m"

REM 配置
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "CONFIG_FILE=%SCRIPT_DIR%\config.yaml"

REM 打印函数
:print_info
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM 检查 Python 是否安装
:check_python
set "PYTHON_CMD="
where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    call :print_info "找到 Python"
    goto :eof
)
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python3"
    call :print_info "找到 Python 3"
    goto :eof
)
call :print_error "未找到 Python，请先安装 Python 3.10+"
echo.
echo 安装方法:
echo   - 下载地址: https://python.org/downloads
echo   - 安装时务必勾选 "Add Python to PATH"
goto :eof

REM 检查并创建虚拟环境
:check_venv
if exist "%VENV_DIR%" (
    call :print_info "找到虚拟环境"
    goto :eof
)
call :print_info "创建虚拟环境..."
%PYTHON_CMD% -m venv "%VENV_DIR%"
if %errorlevel% equ 0 (
    call :print_success "虚拟环境创建成功"
) else (
    call :print_error "虚拟环境创建失败"
    exit /b 1
)
goto :eof

REM 安装依赖
:setup_dependencies
REM 激活虚拟环境
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    call :print_warning "无法激活虚拟环境，使用系统 Python"
)

REM 检查依赖
%PYTHON_CMD% -c "import click" 2>nul
if not %errorlevel% equ 0 (
    call :print_info "安装项目依赖..."
    pip install -e . --quiet
    if %errorlevel% equ 0 (
        call :print_success "依赖安装成功"
    ) else (
        call :print_error "依赖安装失败"
        exit /b 1
    )
) else (
    call :print_info "依赖已安装"
)
goto :eof

REM 检查配置文件
:check_config
if exist "%CONFIG_FILE%" (
    call :print_info "找到配置文件: %CONFIG_FILE%"
    goto :eof
)

call :print_warning "未找到配置文件，正在创建..."

REM 检查 API Key
if not defined OPENAI_API_KEY (
    call :print_warning "环境变量 OPENAI_API_KEY 未设置"
    echo 请设置环境变量或编辑 config.yaml
    echo.
    echo 设置方法:
    echo   set OPENAI_API_KEY=your-api-key
)

REM 创建默认配置
(
echo # model-finetune-tool 配置文件
echo # 请根据需要修改配置
echo.
echo # LLM配置
echo llm:
echo   api_key: "${OPENAI_API_KEY}"
echo   base_url: "https://api.openai.com/v1"
echo   model: "gpt-3.5-turbo"
echo   temperature: 0.7
echo   max_tokens: 2000
echo.
echo # 数据库配置
echo database:
echo   type: "sqlite"
echo   path: "./data/datasets.db"
echo.
echo # 数据集配置
echo datasets:
echo   input_dir: "./documents"
echo   output_dir: "./data"
echo   chunk_size: 1000
echo   chunk_overlap: 200
echo.
echo # 训练配置
echo training:
echo   model_name: "Qwen/Qwen2.5-0.5B-Instruct"
echo   batch_size: 4
echo   learning_rate: 0.0002
echo   epochs: 3
echo   max_length: 2048
echo.
echo # 输出配置
echo output:
echo   model_dir: "./output"
echo   checkpoint_dir: "./checkpoints"
echo.
echo # Git配置
echo git:
echo   auto_commit: false
echo   commit_message: "Update dataset: {dataset_name}"
) > "%CONFIG_FILE%"

call :print_success "配置文件创建成功: %CONFIG_FILE%"
call :print_warning "请编辑配置文件设置 API Key"
goto :eof

REM 创建目录结构
:create_dirs
call :print_info "创建目录结构..."
if not exist "%SCRIPT_DIR%\data" mkdir "%SCRIPT_DIR%\data"
if not exist "%SCRIPT_DIR%\documents" mkdir "%SCRIPT_DIR%\documents"
if not exist "%SCRIPT_DIR%\output" mkdir "%SCRIPT_DIR%\output"
if not exist "%SCRIPT_DIR%\checkpoints" mkdir "%SCRIPT_DIR%\checkpoints"
call :print_success "目录创建完成"
goto :eof

REM 显示帮助
:show_help
echo ==========================================
echo   model-finetune-tool 快速启动 (Windows)
echo ==========================================
echo.
echo 用法: finetune.bat [命令] [选项]
echo.
echo 命令:
echo   help          显示此帮助信息
echo   init          初始化项目（创建配置和目录）
echo   check         检查环境和依赖
echo   gpu           检查 GPU 可用性
echo   parse ^<dir^> ^<name^>
echo                 解析文档生成数据集
echo                 示例: finetune.bat parse .\documents my_dataset
echo.
echo   export ^<name^> [选项]
echo                 导出数据集
echo                 示例: finetune.bat export my_dataset -o train.jsonl
echo.
echo   train ^<name^> [选项]
echo                 训练模型
echo                 示例: finetune.bat train my_dataset -e 3 -b 4
echo.
echo   merge ^<name^> ^<base_model^>
echo                 合并模型
echo                 示例: finetune.bat merge my_dataset Qwen/Qwen2.5-0.5B-Instruct
echo.
echo   stats ^<name^  查看数据集统计
echo                 示例: finetune.bat stats my_dataset
echo.
echo   clear ^<name^  清空数据集
echo                 示例: finetune.bat clear my_dataset
echo.
echo 选项:
echo   -v, --verbose  详细输出
echo   -q, --quiet    安静模式
echo   -h, --help     显示帮助
echo.
echo 环境变量:
echo   OPENAI_API_KEY    OpenAI API Key（必需）
echo   OPENAI_BASE_URL   API 地址（可选）
echo   LOG_LEVEL         日志级别（可选）
echo.
echo 示例:
echo   :: 设置 API Key
echo   set OPENAI_API_KEY=your-api-key
echo.
echo   :: 初始化项目
echo   finetune.bat init
echo.
echo   :: 准备文档
echo   copy your-docs\*.md documents\
echo.
echo   :: 解析文档生成数据
echo   finetune.bat parse .\documents my_dataset
echo.
echo   :: 训练模型
echo   finetune.bat train my_dataset
echo.
echo   :: 合并模型
echo   finetune.bat merge my_dataset Qwen/Qwen2.5-0.5B-Instruct
echo.
echo ==========================================
goto :eof

REM 主程序
:main
echo.
echo ==========================================
echo   model-finetune-tool 快速启动
echo ==========================================
echo.

if "%~1"=="" (
    call :show_help
    exit /b 0
)

if "%~1"=="help" (
    call :show_help
    exit /b 0
)
if "%~1"=="-h" (
    call :show_help
    exit /b 0
)
if "%~1"=="--help" (
    call :show_help
    exit /b 0
)

if "%~1"=="init" (
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_config
    call :create_dirs
    call :check_venv
    call :setup_dependencies
    goto :main_end
)

if "%~1"=="check" (
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_venv
    call :setup_dependencies
    call :check_config
    call :check_gpu
    goto :main_end
)

if "%~1"=="gpu" (
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_gpu
    goto :main_end
)

if "%~1"=="parse" (
    shift
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_venv
    call :setup_dependencies
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    finetune parse %*
    goto :main_end
)

if "%~1"=="export" (
    shift
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_venv
    call :setup_dependencies
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    finetune export %*
    goto :main_end
)

if "%~1"=="train" (
    shift
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_venv
    call :setup_dependencies
    call :check_gpu
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    finetune train %*
    goto :main_end
)

if "%~1"=="merge" (
    shift
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_venv
    call :setup_dependencies
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    finetune merge %*
    goto :main_end
)

if "%~1"=="stats" (
    shift
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_venv
    call :setup_dependencies
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    finetune stats %*
    goto :main_end
)

if "%~1"=="clear" (
    shift
    call :check_python
    if !errorlevel! equ 1 exit /b 1
    call :check_venv
    call :setup_dependencies
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        call "%VENV_DIR%\Scripts\activate.bat"
    )
    finetune clear %*
    goto :main_end
)

call :print_error "未知命令: %~1"
echo.
echo 使用 finetune.bat help 查看帮助

:main_end
echo.
call :print_success "操作完成！"
endlocal
goto :eof

:check_gpu
call :print_info "检查 GPU 可用性..."
%PYTHON_CMD% -c "import torch; print(torch.cuda.is_available())" 2>nul | findstr /c:"True" >nul
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('%PYTHON_CMD% -c "import torch; print(torch.cuda.get_device_name(0))" 2^>nul') do set "GPU_NAME=%%i"
    call :print_success "检测到 GPU: !GPU_NAME!"
) else (
    call :print_info "未检测到 GPU，将使用 CPU 训练（可能较慢）"
)
goto :eof

REM 执行主程序
call :main %*
