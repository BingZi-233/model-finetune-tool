"""主程序入口"""
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

import click
from tqdm import tqdm

from .config import get_config, load_config
from .parser import ParserManager
from .dataset import DatasetManager
from .llm import LLMClient, CacheManager
from .trainer import train_lora, merge_model, prepare_training_data


@click.group()
def cli():
    """模型微调工具"""
    pass


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='配置文件路径')
def init(config: str):
    """初始化项目"""
    config_path = Path(config)
    if config_path.exists():
        click.echo(f"配置文件已存在: {config}")
    else:
        click.echo(f"创建配置: {config}")


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='配置文件路径')
@click.argument('input_dir')
@click.argument('dataset_name')
@click.option('--recursive/--no-recursive', default=True, help='递归解析子目录')
@click.option('--chunk-size', '-s', default=None, help='文本块大小')
@click.option('--qa-pairs', '-n', default=3, help='每段文本生成的QA对数量')
def parse(
    config: str,
    input_dir: str,
    dataset_name: str,
    recursive: bool,
    chunk_size: Optional[int],
    qa_pairs: int
):
    """解析文档并生成数据集"""
    cfg = load_config(config)
    
    if chunk_size:
        cfg.datasets.chunk_size = chunk_size
    
    # 初始化管理器
    parser = ParserManager()
    db_manager = DatasetManager()
    llm_client = LLMClient()
    
    # 解析文档
    click.echo(f"解析文档: {input_dir}")
    documents = parser.parse_directory(input_dir, recursive)
    
    click.echo(f"找到 {len(documents)} 个文档")
    
    # 处理每个文档
    total_items = 0
    
    for file_path, paragraphs in tqdm(documents.items(), desc="处理文档"):
        # 计算内容hash
        content_hash = hashlib.md5(''.join(paragraphs).encode()).hexdigest()
        
        # 检查是否已处理
        if db_manager.document_exists(file_path, content_hash):
            click.echo(f"跳过已处理: {Path(file_path).name}")
            continue
        
        # 添加文档记录
        doc_id = db_manager.add_document(file_path, content_hash)
        
        # 切分文本
        chunks = []
        for i, para in enumerate(paragraphs):
            if len(para) > cfg.datasets.chunk_size:
                # 长文本切分成小块
                for j in range(0, len(para), cfg.datasets.chunk_size - cfg.datasets.chunk_overlap):
                    chunk = para[j:j + cfg.datasets.chunk_size]
                    if len(chunk) > 100:  # 过滤太短的块
                        chunks.append(chunk)
            else:
                chunks.append(para)
        
        # 生成QA对
        for chunk_idx, chunk in enumerate(chunks):
            try:
                qa = llm_client.generate_qa_pairs(chunk, qa_pairs)
                
                for qa_item in qa:
                    db_manager.add_dataset_item(
                        dataset_name=dataset_name,
                        instruction=qa_item.get("instruction", ""),
                        input_=qa_item.get("input", ""),
                        output=qa_item.get("output", ""),
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        source_file=file_path
                    )
                    total_items += 1
            except Exception as e:
                click.echo(f"生成QA失败: {e}")
                continue
    
    click.echo(f"✅ 完成！共生成 {total_items} 条数据")


@cli.command()
@click.argument('dataset_name')
@click.option('--format', 'output_format', type=click.Choice(['jsonl', 'json']), default='jsonl')
@click.option('--output', '-o', help='输出文件路径')
def export(dataset_name: str, output_format: str, output: Optional[str]):
    """导出数据集"""
    db_manager = DatasetManager()
    
    if output is None:
        output = f"{dataset_name}.{output_format}"
    
    if output_format == 'jsonl':
        count = db_manager.save_to_jsonl(dataset_name, output)
        click.echo(f"✅ 导出 {count} 条数据到 {output}")
    else:
        data = db_manager.export_dataset(dataset_name)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        click.echo(f"✅ 导出 {len(data)} 条数据到 {output}")


@cli.command()
@click.argument('dataset_name')
def stats(dataset_name: str):
    """查看数据集统计"""
    db_manager = DatasetManager()
    stats = db_manager.get_dataset_stats(dataset_name)
    
    click.echo(f"数据集: {stats['dataset_name']}")
    click.echo(f"总条目: {stats['total_items']}")


@cli.command()
@click.argument('dataset_name')
@click.option('--model', '-m', help='模型名称')
@click.option('--epochs', '-e', default=None, help='训练轮数')
@click.option('--batch-size', '-b', default=None, help='批次大小')
def train(dataset_name: str, model: Optional[str], epochs: Optional[int], batch_size: Optional[int]):
    """训练模型"""
    cfg = get_config()
    
    model_name = model or cfg.training.model_name
    epochs = epochs or cfg.training.epochs
    batch_size = batch_size or cfg.training.batch_size
    
    # 导出数据
    data_path = f"/tmp/{dataset_name}_train.jsonl"
    db_manager = DatasetManager()
    db_manager.save_to_jsonl(dataset_name, data_path)
    
    # 准备数据
    prepared_path = f"/tmp/{dataset_name}_prepared.jsonl"
    prepare_training_data(data_path, prepared_path)
    
    output_dir = f"./output/{dataset_name}"
    
    click.echo(f"开始训练模型: {model_name}")
    
    train_lora(
        model_name=model_name,
        data_path=prepared_path,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs
    )
    
    click.echo(f"✅ 训练完成！模型保存到: {output_dir}")


@cli.command()
@click.argument('dataset_name')
@click.argument('base_model')
@click.option('--output', '-o', help='输出路径')
def merge(dataset_name: str, base_model: str, output: Optional[str]):
    """合并模型"""
    lora_path = f"./output/{dataset_name}/lora_model"
    
    if not Path(lora_path).exists():
        click.echo(f"❌ LoRA模型不存在: {lora_path}")
        return
    
    output_path = output or f"./output/{dataset_name}/merged"
    
    merge_model(base_model, lora_path, output_path)
    click.echo(f"✅ 模型已合并到: {output_path}")


@cli.command()
@click.argument('dataset_name')
def clear(dataset_name: str):
    """清空数据集"""
    db_manager = DatasetManager()
    db_manager.clear_dataset(dataset_name)
    click.echo(f"✅ 已清空数据集: {dataset_name}")


def main():
    """主入口"""
    cli()


if __name__ == "__main__":
    main()
