"""数据集管理模块"""
import json
import logging
import os
import platform
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Index
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from sqlalchemy.pool import StaticPool

from ..config import get_config

logger = logging.getLogger(__name__)

# ============ 平台检测 ============
IS_WINDOWS = platform.system() == "Windows"


class Base(DeclarativeBase):
    """SQLAlchemy基类"""
    pass


class Document(Base):
    """文档表"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(500), unique=True, nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    content_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 额外数据
    extra_data = Column(JSON, nullable=True)
    
    # 索引
    __table_args__ = (
        Index('idx_document_file_type', 'file_type'),
        Index('idx_document_created_at', 'created_at'),
    )


class DatasetItem(Base):
    """数据集条目表"""
    __tablename__ = "dataset_items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    document_id = Column(Integer, nullable=True, index=True)
    
    instruction = Column(Text, nullable=False)
    input_ = Column(Text, nullable=True)
    output = Column(Text, nullable=True)
    
    chunk_index = Column(Integer, default=0)
    source_file = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(JSON, nullable=True)
    
    # 复合索引
    __table_args__ = (
        Index('idx_item_dataset_created', 'dataset_name', 'created_at'),
        Index('idx_item_document_source', 'document_id', 'source_file'),
    )


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, db_path: Optional[str] = None):
        config = get_config()
        
        if db_path is None:
            db_path = config.database.path
        
        # 确保目录存在
        db_path = os.path.normpath(db_path)  # 跨平台路径规范化
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 创建引擎
        if db_path == ":memory:":
            # 内存数据库
            self.engine = create_engine(
                "sqlite:///:memory:",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False}
            )
        elif db_path.endswith(".db") or "sqlite" in db_path:
            # SQLite 数据库 - 使用 4 斜杠格式确保跨平台兼容性
            # Windows: sqlite:///C:/path/to/db.db
            # Linux/Mac: sqlite:///path/to/db.db
            if IS_WINDOWS and os.path.isabs(db_path):
                # Windows 绝对路径需要使用 4 斜杠
                db_uri = f"sqlite:///{db_path}"
            else:
                db_uri = f"sqlite:///{db_path}"
            
            self.engine = create_engine(
                db_uri,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                pool_pre_ping=True
            )
        elif "mysql" in db_path:
            self.engine = create_engine(db_path)
        elif "postgresql" in db_path:
            self.engine = create_engine(db_path)
        else:
            raise ValueError(f"不支持的数据库路径: {db_path}")
        
        # 创建表
        Base.metadata.create_all(self.engine)
        
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """获取数据库会话"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def add_document(self, file_path: str, content_hash: str, extra_data: Optional[Dict] = None):
        """添加文档记录"""
        logger.info(f"添加文档: {file_path}")
        with self.session() as session:
            # 检查是否已存在
            existing = session.query(Document).filter(
                Document.file_path == file_path
            ).first()
            
            if existing:
                # 更新现有记录
                logger.debug(f"更新文档: {file_path}")
                existing.content_hash = content_hash
                existing.extra_data = extra_data
                session.flush()
                return existing.id
            else:
                # 创建新记录
                doc = Document(
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    file_type=Path(file_path).suffix.lower(),
                    content_hash=content_hash,
                    extra_data=extra_data
                )
                session.add(doc)
                session.flush()
                logger.debug(f"新建文档: {file_path}, id={doc.id}")
                return doc.id
    
    def get_document(self, file_path: str) -> Optional[Document]:
        """获取文档记录"""
        with self.session() as session:
            return session.query(Document).filter(
                Document.file_path == file_path
            ).first()
    
    def document_exists(self, file_path: str, content_hash: str) -> bool:
        """检查文档是否已存在"""
        with self.session() as session:
            return session.query(Document).filter(
                Document.file_path == file_path,
                Document.content_hash == content_hash
            ).first() is not None
    
    def add_dataset_item(
        self,
        dataset_name: str,
        instruction: str,
        input_: Optional[str] = None,
        output: Optional[str] = None,
        document_id: Optional[int] = None,
        chunk_index: int = 0,
        source_file: Optional[str] = None,
        extra_data: Optional[Dict] = None
    ):
        """添加数据集条目"""
        logger.debug(f"添加数据集条目: dataset={dataset_name}")
        with self.session() as session:
            item = DatasetItem(
                dataset_name=dataset_name,
                instruction=instruction,
                input_=input_,
                output=output,
                document_id=document_id,
                chunk_index=chunk_index,
                source_file=source_file,
                extra_data=extra_data
            )
            session.add(item)
            session.flush()
            return item.id
    
    def get_dataset_items(
        self,
        dataset_name: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[DatasetItem]:
        """获取数据集条目"""
        with self.session() as session:
            query = session.query(DatasetItem).filter(
                DatasetItem.dataset_name == dataset_name
            )
            
            if offset > 0:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
    
    def export_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """导出数据集为JSON格式"""
        logger.info(f"导出数据集: {dataset_name}")
        with self.session() as session:
            items = session.query(DatasetItem).filter(
                DatasetItem.dataset_name == dataset_name
            ).all()
            
            result = [
                {
                    "instruction": item.instruction,
                    "input": item.input_,
                    "output": item.output,
                    "extra_data": item.extra_data
                }
                for item in items
            ]
            
            logger.debug(f"导出 {len(result)} 条数据")
            return result
    
    def save_to_jsonl(self, dataset_name: str, output_path: str):
        """保存数据集为JSONL格式"""
        items = self.export_dataset(dataset_name)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return len(items)
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, int]:
        """获取数据集统计信息"""
        logger.debug(f"获取数据集统计: {dataset_name}")
        with self.session() as session:
            total = session.query(DatasetItem).filter(
                DatasetItem.dataset_name == dataset_name
            ).count()
            
            return {
                "total_items": total,
                "dataset_name": dataset_name
            }
    
    def clear_dataset(self, dataset_name: str):
        """清空数据集"""
        logger.warning(f"清空数据集: {dataset_name}")
        with self.session() as session:
            deleted_count = session.query(DatasetItem).filter(
                DatasetItem.dataset_name == dataset_name
            ).delete()
            logger.info(f"删除了 {deleted_count} 条数据")


"""
数据集管理模块

本模块提供数据集的创建、存储、查询和导出功能。

## 数据库结构

### documents 表

存储已解析的文档信息：

| 字段 | 类型 | 描述 |
|------|------|------|
| id | Integer | 主键，自增 |
| file_path | String(500) | 文件路径，唯一 |
| file_name | String(255) | 文件名 |
| file_type | String(50) | 文件类型 (.docx/.pdf/.md) |
| content_hash | String(64) | 内容哈希，用于去重 |
| created_at | DateTime | 创建时间 |
| extra_data | JSON | 额外元数据 |

### dataset_items 表

存储训练数据条目：

| 字段 | 类型 | 描述 |
|------|------|------|
| id | Integer | 主键，自增 |
| dataset_name | String(100) | 数据集名称 |
| document_id | Integer | 关联文档 ID |
| instruction | Text | 问题/指令 |
| input | Text | 输入内容 |
| output | Text | 输出答案 |
| chunk_index | Integer | 文本块索引 |
| source_file | String(500) | 源文件路径 |
| created_at | DateTime | 创建时间 |
| extra_data | JSON | 额外元数据 |

## 支持的数据库

- **SQLite** - 默认，轻量级
- **MySQL** - 需要安装 pymysql
- **PostgreSQL** - 需要安装 psycopg2-binary

## 使用示例

```python
from src.dataset import DatasetManager

# 创建管理器
manager = DatasetManager()

# 添加文档
doc_id = manager.add_document(
    file_path="/path/to/doc.md",
    content_hash="abc123"
)

# 添加数据集条目
item_id = manager.add_dataset_item(
    dataset_name="my_dataset",
    instruction="问题",
    input="",
    output="答案",
    document_id=doc_id
)

# 获取数据集条目
items = manager.get_dataset_items(
    dataset_name="my_dataset",
    limit=10,
    offset=0
)

# 导出数据集
data = manager.export_dataset("my_dataset")

# 导出为 JSONL
manager.save_to_jsonl("my_dataset", "train.jsonl")

# 获取统计信息
stats = manager.get_dataset_stats("my_dataset")

# 清空数据集
manager.clear_dataset("my_dataset")
```

## 事务管理

使用上下文管理器确保数据一致性：

```python
with manager.session() as session:
    # 在这里执行数据库操作
    # 成功自动提交，失败自动回滚
    session.add(new_item)
```

## 索引优化

以下查询字段已添加索引：
- documents.file_type
- documents.created_at
- dataset_items.dataset_name
- dataset_items.document_id
- dataset_items.dataset_name + created_at (复合索引)

## 注意事项

- 文档通过 content_hash 去重，相同内容不会重复添加
- 数据库连接自动管理，无需手动关闭
- 建议定期备份数据库文件
"""

__all__ = [
    'DatasetManager',
    'Document',
    'DatasetItem',
]
