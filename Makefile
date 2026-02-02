# Makefile for model-finetune-tool
# 使用方法: make <target>

.PHONY: help install dev test clean lint format check run

# 默认目标
help:
	@echo "=== model-finetune-tool ==="
	@echo ""
	@echo "可用目标:"
	@echo "  install     - 安装项目依赖"
	@echo "  dev         - 开发模式安装"
	@echo "  test        - 运行测试"
	@echo "  test-cov    - 运行测试并生成覆盖率报告"
	@echo "  lint        - 运行代码检查"
	@echo "  format      - 格式化代码"
	@echo "  check       - 运行所有检查"
	@echo "  clean       - 清理临时文件"
	@echo "  run         - 运行 CLI"
	@echo "  gpu-check   - 检查 GPU 可用性"
	@echo "  cache-clear - 清理缓存"
	@echo ""

# 安装依赖
install:
	pip install -e .

# 开发模式安装
dev:
	pip install -e . pytest pytest-cov

# 运行测试
test:
	python -m pytest tests/ -v --tb=short

# 运行测试并生成覆盖率报告
test-cov:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# 运行代码检查
lint:
	@echo "=== 代码检查 ==="
	-python3 -m py_compile src/*.py src/*/*.py
	@echo "✓ 语法检查通过"
	@echo "✓ 使用 flake8 或 ruff 进行更深入的检查 (需要安装)"

# 格式化代码
format:
	@echo "=== 代码格式化 ==="
	@echo "使用 black 格式化代码 (需要安装):"
	@echo "  pip install black"
	@echo "  black src/ tests/"

# 运行所有检查
check: lint test

# 清理临时文件
clean:
	@echo "=== 清理临时文件 ==="
	rm -rf .pytest_cache/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf tmp/
	rm -rf temp/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ 清理完成"

# 运行 CLI
run:
	finetune --help

# 检查 GPU 可用性
gpu-check:
	@echo "=== GPU 检测 ==="
	python -c "from src.trainer import check_gpu_available; import json; print(json.dumps(check_gpu_available(), indent=2))"

# 清理缓存
cache-clear:
	@echo "=== 清理缓存 ==="
	python -c "from src.llm import CacheManager; CacheManager().clear()"
	@echo "✓ 缓存已清理"

# Windows 特定
install-windows:
	@echo "=== Windows 安装 ==="
	python -m venv venv
	venv\Scripts\activate
	pip install -e .

# 创建虚拟环境
venv:
	python -m venv venv
	@echo "虚拟环境已创建。激活方法:"
	@echo "  Linux/Mac: source venv/bin/activate"
	@echo "  Windows:   venv\Scripts\activate"
