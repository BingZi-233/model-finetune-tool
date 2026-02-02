"""数据集管理模块"""
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from sqlalchemy.pool import StaticPool

from .config import get_config


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
    
    # 元数据
    metadata = Column(JSON, nullable=True)


class DatasetItem(Base):
    """数据集条目表"""
    __tablename__ = "dataset_items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    document_id = Column(Integer, nullable=True)
    
    instruction = Column(Text, nullable=False)
    input = Column(Text, nullable=True)
    output = Column(Text, nullable=True)
    
    chunk_index = Column(Integer, default=0)
    source_file = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, db_path: Optional[str] = None):
        config = get_config()
        
        if db_path is None:
            db_path = config.database.path
        
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 创建引擎
        if db_path.endswith(".db") or "sqlite" in db_path:
            self.engine = create_engine(
                f"sqlite:///{db_path}",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False}
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
    
    def add_document(self, file_path: str, content_hash: str, metadata: Optional[Dict] = None):
        """添加文档记录"""
        with self.session() as session:
            doc = Document(
                file_path=file_path,
                file_name=Path(file_path).name,
                file_type=Path(file_path).suffix.lower(),
                content_hash=content_hash,
                metadata=metadata
            )
            session.merge(doc)  # 如果存在则更新
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
        metadata: Optional[Dict] = None
    ):
        """添加数据集条目"""
        with self.session() as session:
            item = DatasetItem(
                dataset_name=dataset_name,
                instruction=instruction,
                input=input_,
                output=output,
                document_id=document_id,
                chunk_index=chunk_index,
                source_file=source_file,
                metadata=metadata
            )
            session.add(item)
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
        items = self.get_dataset_items(dataset_name)
        return [
            {
                "instruction": item.instruction,
                "input": item.input,
                "output": item.output,
                "metadata": item.metadata
            }
            for item in items
        ]
    
    def save_to_jsonl(self, dataset_name: str, output_path: str):
        """保存数据集为JSONL格式"""
        items = self.export_dataset(dataset_name)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return len(items)
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, int]:
        """获取数据集统计信息"""
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
        with self.session() as session:
            session.query(DatasetItem).filter(
                DatasetItem.dataset_name == dataset_name
            ).delete()
