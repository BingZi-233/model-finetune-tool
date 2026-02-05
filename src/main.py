"""主程序入口"""

import hashlib
import json
import logging
import os
import platform
import re
import signal
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import click
from tqdm import tqdm

from .config import get_config, load_config, reload_config
from .parser import ParserManager
from .dataset import DatasetManager
from .llm import LLMClient, CacheManager
from .trainer import train_lora, merge_model, prepare_training_data

logger = logging.getLogger(__name__)

# 配置日志输出到 stderr
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

# ============ 平台检测 ============
IS_WINDOWS = platform.system() == "Windows"
# IS_MACOS = platform.system() == "Darwin"  # 已定义但未使用

# ============ 全局状态 ============
_cli_verbose = False
_cli_quiet = False


# ============ 信号处理 ============
def setup_signal_handlers():
    """设置信号处理器"""

    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在安全退出...")
        logger.info("\n[WARN] 检测到中断信号，正在安全退出...")
        sys.exit(0)

    # 注册信号处理器
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    else:
        # Windows 不支持 SIGTERM
        signal.signal(signal.SIGINT, signal_handler)


def enable_verbose():
    """启用详细输出"""
    global _cli_verbose
    _cli_verbose = True
    logging.getLogger().setLevel(logging.DEBUG)


def enable_quiet():
    """启用安静模式"""
    global _cli_quiet
    _cli_quiet = True
    logging.getLogger().setLevel(logging.WARNING)


# ============ 常量定义 ============
MIN_CHUNK_LENGTH = 100  # 最小文本块长度
MAX_TEXT_LENGTH = 100000  # 最大输入文本长度 (100KB)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 最大文件大小 (50MB)


# ============ 跨平台工具函数 ============
def normalize_path(path: str) -> str:
    """规范化路径，处理不同操作系统的路径分隔符

    Args:
        path: 原始路径

    Returns:
        规范化后的路径
    """
    # 将正斜杠转换为当前系统的路径分隔符
    return path.replace("/", os.sep).replace("\\", os.sep)


def validate_path(path: str, base_dir: Optional[str] = None) -> str:
    """验证路径安全性，防止路径遍历攻击（跨平台版本）

    Args:
        path: 要验证的路径
        base_dir: 基础目录，限制路径在此目录内

    Returns:
        验证后的绝对路径

    Raises:
        ValueError: 路径不合法
    """
    # 规范化路径分隔符
    path = normalize_path(path)

    # 使用 Path 对象处理路径（跨平台）
    try:
        path_obj = Path(path)

        # 检查路径是否存在
        if not path_obj.exists():
            raise ValueError(f"路径不存在: {path}")

        # 获取绝对路径
        clean_path = str(path_obj.resolve())

        # 检查基础目录限制
        if base_dir:
            base_dir = normalize_path(base_dir)
            base_path = Path(base_dir).resolve()

            # 尝试多种方式检查路径是否在基础目录内
            try:
                clean_path_obj = Path(clean_path)
                # 检查路径是否以基础目录开头
                if IS_WINDOWS:
                    # Windows 不区分大小写
                    if (
                        not clean_path_obj.resolve().parts[: len(base_path.parts)]
                        == base_path.parts
                    ):
                        raise ValueError(f"路径访问被拒绝: {path}")
                else:
                    # Linux/Mac
                    if (
                        not clean_path_obj.resolve().parts[: len(base_path.parts)]
                        == base_path.parts
                    ):
                        raise ValueError(f"路径访问被拒绝: {path}")
            except ValueError:
                raise
            except Exception:
                # 如果 resolve() 失败，使用字符串比较
                if not clean_path.startswith(str(base_path) + os.sep):
                    raise ValueError(f"路径访问被拒绝: {path}")

        return clean_path

    except OSError as e:
        raise ValueError(f"路径访问错误: {e}")


def validate_file_size(file_path: str, max_size: int = MAX_FILE_SIZE) -> bool:
    """验证文件大小

    Args:
        file_path: 文件路径
        max_size: 最大允许大小

    Returns:
        文件大小是否合法

    Raises:
        ValueError: 文件过大
    """
    # 规范化路径
    file_path = normalize_path(file_path)

    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            raise ValueError(
                f"文件过大: {file_path} ({file_size / 1024 / 1024:.1f}MB > "
                f"{max_size / 1024 / 1024:.1f}MB)"
            )
    return True


def validate_text_length(text: str, max_length: int = MAX_TEXT_LENGTH) -> None:
    """验证文本长度

    Args:
        text: 要验证的文本
        max_length: 最大允许长度

    Raises:
        ValueError: 文本过长
    """
    if len(text) > max_length:
        raise ValueError(f"输入文本过长 ({len(text)} > {max_length} 字符)")


def get_default_config_path() -> str:
    """获取默认配置文件路径（跨平台）

    Returns:
        默认配置文件路径
    """
    return "config.yaml"


def get_data_dir() -> Path:
    """获取数据目录路径（跨平台）

    Returns:
        数据目录 Path 对象
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir


# ============ CLI 全局选项 ============
def verbose_option(f):
    """添加 verbose 选项的装饰器"""

    def callback(ctx, param, value):
        if value:
            enable_verbose()
        return value

    return click.option(
        "--verbose",
        "-v",
        is_flag=True,
        help="启用详细输出",
        expose_value=False,
        callback=callback,
    )(f)


def quiet_option(f):
    """添加 quiet 选项的装饰器"""

    def callback(ctx, param, value):
        if value:
            enable_quiet()
        return value

    return click.option(
        "--quiet",
        "-q",
        is_flag=True,
        help="安静模式，减少输出",
        expose_value=False,
        callback=callback,
    )(f)


# ============ CLI 命令 ============
@click.group()
@click.option("--config", "-c", default="config.yaml", help="配置文件路径")
@click.option("--verbose", "-v", is_flag=True, help="启用详细输出")
@click.option("--quiet", "-q", is_flag=True, help="安静模式")
@click.pass_context
def cli(ctx, config, verbose, quiet):
    """模型微调工具"""
    # 设置信号处理器
    setup_signal_handlers()

    # 处理全局选项
    if verbose and quiet:
        logger.info("[WARN] 不能同时使用 --verbose 和 --quiet")

    if verbose:
        enable_verbose()

    if quiet:
        enable_quiet()

    # 保存配置路径到上下文
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    logger.debug(f"配置路径: {config}")
    logger.debug(f"平台: {platform.system()}")


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="配置文件路径")
def init(config: str):
    """初始化项目"""
    config_path = Path(config)
    if config_path.exists():
        logger.info(f"配置文件已存在: {config}")
    else:
        logger.info(f"创建配置: {config}")


@cli.command()
@click.argument("input_dir")
@click.argument("dataset_name")
@click.option("--recursive/--no-recursive", default=True, help="递归解析子目录")
@click.option("--chunk-size", "-s", default=None, help="文本块大小 (100-10000)")
@click.option("--qa-pairs", "-n", default=3, help="每段文本生成的QA对数量 (1-20)")
@click.pass_context
def parse(
    ctx,
    input_dir: str,
    dataset_name: str,
    recursive: bool,
    chunk_size: Optional[int],
    qa_pairs: int,
):
    """解析文档并生成数据集"""
    # 获取全局配置路径
    config_path = ctx.obj.get("config", "config.yaml")

    logger.info("=" * 60)
    logger.info("🚀 开始解析文档并生成数据集")
    logger.info("=" * 60)
    logger.info(f"📁 输入目录: {input_dir}")
    logger.info(f"📊 数据集名称: {dataset_name}")
    logger.info(f"🔄 递归扫描: {'是' if recursive else '否'}")

    # 验证参数
    if chunk_size is not None:
        if chunk_size < 100 or chunk_size > 10000:
            raise click.BadParameter(
                f"chunk_size 必须在 100-10000 之间", param_hint="--chunk-size"
            )

    if qa_pairs < 1 or qa_pairs > 20:
        raise click.BadParameter(f"qa-pairs 必须在 1-20 之间", param_hint="--qa-pairs")

    # 验证 dataset_name
    if not dataset_name or not dataset_name.strip():
        raise click.BadParameter("dataset_name 不能为空", param_hint="DATASET_NAME")

    # 验证路径安全性
    try:
        input_dir = validate_path(input_dir)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        return

    try:
        cfg = load_config(config_path)
    except Exception as e:
        print(f"[ERROR] 加载配置失败: {e}", file=sys.stderr, flush=True)
        return

    if chunk_size:
        cfg.datasets.chunk_size = chunk_size

    logger.info(f"📏 文本块大小: {cfg.datasets.chunk_size}")
    logger.info(f"❓ 每个文本块生成QA对数量: {qa_pairs}")
    logger.info(f"🤖 LLM模型: {cfg.llm.model}")
    logger.info("-" * 60)

    # 初始化管理器
    parser = ParserManager()
    db_manager = DatasetManager()
    llm_client = LLMClient()

    # 解析文档
    logger.info(f"📂 开始扫描文档目录...")

    try:
        documents = parser.parse_directory(input_dir, recursive)
    except (OSError, IOError) as e:
        logger.error(f"[ERROR] 读取文档目录失败: {e}")
        return
    except ValueError as e:
        logger.error(f"[ERROR] 文档格式错误: {e}")
        return
    except Exception as e:
        logger.error(f"[ERROR] 解析文档失败: {e}")
        logger.debug(f"详细错误信息:", exc_info=True)
        return

    if not documents:
        logger.warning("[WARN] 没有找到可解析的文档")
        return

    logger.info("-" * 60)
    logger.info(f"[OK] 扫描完成! 发现 {len(documents)} 个有效文档")

    # 统计总段落数
    total_paragraphs = sum(len(paras) for paras in documents.values())
    logger.info(f"📝 总段落数: {total_paragraphs}")

    # 处理每个文档
    total_items = 0
    skipped_files = 0
    error_files = []
    total_chunks = 0

    logger.info("-" * 60)
    logger.info("🔄 开始生成QA对...")
    logger.info("-" * 60)

    for file_path, paragraphs in tqdm(documents.items(), desc="🔄 处理文档"):
        # 验证文件大小
        try:
            validate_file_size(file_path)
        except ValueError as e:
            logger.warning(f"[WARN] 跳过大文件: {e}")
            continue

        # 计算内容hash
        content_hash = hashlib.md5("".join(paragraphs).encode()).hexdigest()

        # 检查是否已处理
        if db_manager.document_exists(file_path, content_hash):
            skipped_files += 1
            continue

        # 添加文档记录
        doc_id = db_manager.add_document(file_path, content_hash)

        # 切分文本
        chunks = []
        for i, para in enumerate(paragraphs):
            if len(para) > cfg.datasets.chunk_size:
                # 长文本切分成小块
                for j in range(
                    0, len(para), cfg.datasets.chunk_size - cfg.datasets.chunk_overlap
                ):
                    chunk = para[j : j + cfg.datasets.chunk_size]
                    if len(chunk) > MIN_CHUNK_LENGTH:  # 使用常量
                        chunks.append(chunk)
            else:
                if len(para) > MIN_CHUNK_LENGTH:  # 使用常量
                    chunks.append(para)

        total_chunks += len(chunks)
        logger.info(
            f"📄 [{Path(file_path).name}] {len(paragraphs)} 段落 → {len(chunks)} 文本块"
        )

        # 生成QA对
        for chunk_idx, chunk in enumerate(chunks):
            # 验证文本长度
            try:
                validate_text_length(chunk)
            except ValueError as e:
                logger.warning(f"[WARN] 跳过过长文本块: {e}")
                continue

            # 输出当前处理进度到 stderr
            file_name = Path(file_path).name
            total_chunks_processed = sum(
                1 for f, p in documents.items() for _ in range(min(len(p), 100))
            )  # 估算
            logger.info(
                f"🔄 处理中: [{file_name}] {chunk_idx + 1}/{len(chunks)} 文本块..."
            )

            try:
                # 生成QA对（会显示LLM响应）
                qa = llm_client.generate_qa_pairs(chunk, qa_pairs)

                # 输出生成结果
                if qa:
                    logger.info(
                        f"   [OK] 生成 {len(qa)} 个QA对 (总计: {total_items + len(qa)})"
                    )
                else:
                    logger.warning(f"   [WARN] 未生成任何QA对")

                for qa_item in qa:
                    db_manager.add_dataset_item(
                        dataset_name=dataset_name,
                        instruction=qa_item.get("instruction", ""),
                        input_=qa_item.get("input", ""),
                        output=qa_item.get("output", ""),
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        source_file=file_path,
                    )
                    total_items += 1
            except Exception as e:
                error_files.append((file_path, str(e)))
                logger.error(f"   [ERROR] 生成失败: {e}")
                logger.error(f"生成QA失败: {e}")
                continue

        # 每个文件处理完成后输出总结
        logger.info(
            f"\n[OK] [{file_name}] 处理完成! 本文件生成 {sum(1 for _ in chunks)} 个文本块"
        )

    logger.info("-" * 60)
    logger.info("📊 处理完成! 统计信息:")
    logger.info("=" * 60)
    logger.info(f"[OK] 成功处理文档: {len(documents) - skipped_files - len(error_files)}")
    logger.info(f"📌 跳过已处理文档: {skipped_files}")
    if error_files:
        logger.error(f"[ERROR] 处理失败文档: {len(error_files)}")
    logger.info(f"📦 总文本块数: {total_chunks}")
    logger.info(f"🎯 生成QA对总数: {total_items}")
    logger.info(f"📁 数据集: {dataset_name}")
    logger.info("=" * 60)


@cli.command()
@click.argument("dataset_name")
@click.option(
    "--format", "output_format", type=click.Choice(["jsonl", "json"]), default="jsonl"
)
@click.option("--output", "-o", help="输出文件路径")
def export(dataset_name: str, output_format: str, output: Optional[str]):
    """导出数据集"""
    db_manager = DatasetManager()

    if output is None:
        output = f"{dataset_name}.{output_format}"

    if output_format == "jsonl":
        count = db_manager.save_to_jsonl(dataset_name, output)
        logger.info(f"[OK] 导出 {count} 条数据到 {output}")
    else:
        data = db_manager.export_dataset(dataset_name)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[OK] 导出 {len(data)} 条数据到 {output}")


@cli.command()
@click.argument("dataset_name")
def stats(dataset_name: str):
    """查看数据集统计"""
    db_manager = DatasetManager()
    stats = db_manager.get_dataset_stats(dataset_name)

    logger.info(f"数据集: {stats['dataset_name']}")
    logger.info(f"总条目: {stats['total_items']}")


@cli.command()
@click.argument("dataset_name")
@click.option("--model", "-m", help="模型名称")
@click.option("--epochs", "-e", default=None, help="训练轮数")
@click.option("--batch-size", "-b", default=None, help="批次大小")
@click.option("--max-length", "-l", default=None, help="最大序列长度")
def train(
    dataset_name: str,
    model: Optional[str],
    epochs: Optional[int],
    batch_size: Optional[int],
    max_length: Optional[int],
):
    """训练模型"""
    # 验证参数
    if epochs is not None and (epochs < 1 or epochs > 100):
        raise click.BadParameter(f"epochs 必须在 1-100 之间", param_hint="--epochs")

    if batch_size is not None and (batch_size < 1 or batch_size > 64):
        raise click.BadParameter(f"batch_size 必须在 1-64 之间", param_hint="--batch-size")

    if max_length is not None and (max_length < 128 or max_length > 8192):
        raise click.BadParameter(f"max_length 必须在 128-8192 之间", param_hint="--max-length")

    cfg = get_config()

    model_name = model or cfg.training.model_name
    epochs = epochs or cfg.training.epochs
    batch_size = batch_size or cfg.training.batch_size
    max_length = max_length or cfg.training.max_length

    # 导出数据到临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        data_path = tmp.name
    db_manager = DatasetManager()
    db_manager.save_to_jsonl(dataset_name, data_path)

    # 准备数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        prepared_path = tmp.name
    prepare_training_data(data_path, prepared_path)

    output_dir = f"./output/{dataset_name}"

    logger.info(f"开始训练模型: {model_name}")

    train_lora(
        model_name=model_name,
        data_path=prepared_path,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        max_length=max_length,
    )

    logger.info(f"[OK] 训练完成！模型保存到: {output_dir}")


@cli.command()
@click.argument("dataset_name")
@click.argument("base_model")
@click.option("--output", "-o", help="输出路径")
def merge(dataset_name: str, base_model: str, output: Optional[str]):
    """合并模型"""
    lora_path = f"./output/{dataset_name}/lora_model"

    if not Path(lora_path).exists():
        logger.error(f"[ERROR] LoRA模型不存在: {lora_path}")
        return

    output_path = output or f"./output/{dataset_name}/merged"

    merge_model(base_model, lora_path, output_path)
    logger.info(f"[OK] 模型已合并到: {output_path}")


@cli.command()
@click.argument("dataset_name")
def clear(dataset_name: str):
    """清空数据集"""
    db_manager = DatasetManager()
    db_manager.clear_dataset(dataset_name)
    logger.info(f"[OK] 已清空数据集: {dataset_name}")


def main():
    """主入口"""
    cli()


if __name__ == "__main__":
    main()
