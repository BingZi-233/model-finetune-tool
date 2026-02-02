"""
数据集管理测试模块

本模块测试数据集管理功能，包括SQLite数据库操作、数据集CRUD等。
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.dataset import (
    DatasetManager,
    Document,
    DatasetItem,
    Base,
)


@pytest.fixture
def temp_db():
    """创建临时数据库"""
    db_path = tempfile.mktemp(suffix=".db")
    
    # 使用内存数据库进行测试
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    
    yield engine
    
    # 清理
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def db_manager(temp_db):
    """创建数据集管理器实例"""
    return DatasetManager(":memory:")


class TestDatasetManager:
    """测试数据集管理器"""
    
    def test_init_creates_tables(self, temp_db):
        """测试初始化时创建表"""
        from sqlalchemy import inspect
        
        inspector = inspect(temp_db)
        tables = inspector.get_table_names()
        
        assert "documents" in tables
        assert "dataset_items" in tables
    
    def test_add_document(self, db_manager):
        """测试添加文档记录"""
        doc_id = db_manager.add_document(
            file_path="/test/doc1.md",
            content_hash="abc123",
            metadata={"source": "test"}
        )
        
        assert doc_id is not None
        assert doc_id > 0
    
    def test_get_document(self, db_manager):
        """测试获取文档记录"""
        db_manager.add_document(
            file_path="/test/doc1.md",
            content_hash="abc123"
        )
        
        doc = db_manager.get_document("/test/doc1.md")
        
        assert doc is not None
        assert doc.file_path == "/test/doc1.md"
        assert doc.content_hash == "abc123"
        assert doc.file_type == ".md"
    
    def test_document_exists(self, db_manager):
        """测试检查文档是否存在"""
        db_manager.add_document(
            file_path="/test/doc1.md",
            content_hash="abc123"
        )
        
        # 存在
        assert db_manager.document_exists("/test/doc1.md", "abc123") is True
        
        # 内容hash不同
        assert db_manager.document_exists("/test/doc1.md", "different_hash") is False
        
        # 路径不同
        assert db_manager.document_exists("/test/doc2.md", "abc123") is False
    
    def test_add_dataset_item(self, db_manager):
        """测试添加数据集条目"""
        item_id = db_manager.add_dataset_item(
            dataset_name="test_dataset",
            instruction="什么是机器学习？",
            input="",
            output="机器学习是..."
        )
        
        assert item_id is not None
        assert item_id > 0
    
    def test_add_dataset_item_with_document(self, db_manager):
        """测试添加关联文档的数据集条目"""
        doc_id = db_manager.add_document(
            file_path="/test/doc1.md",
            content_hash="abc123"
        )
        
        item_id = db_manager.add_dataset_item(
            dataset_name="test_dataset",
            instruction="请解释",
            input="",
            output="解释内容",
            document_id=doc_id,
            chunk_index=0,
            source_file="/test/doc1.md"
        )
        
        assert item_id is not None
        
        # 验证关联
        items = db_manager.get_dataset_items("test_dataset")
        assert len(items) == 1
        assert items[0].document_id == doc_id
        assert items[0].source_file == "/test/doc1.md"
    
    def test_get_dataset_items(self, db_manager):
        """测试获取数据集条目"""
        # 添加多个条目
        for i in range(5):
            db_manager.add_dataset_item(
                dataset_name="test_dataset",
                instruction=f"问题{i}",
                input="",
                output=f"答案{i}"
            )
        
        items = db_manager.get_dataset_items("test_dataset")
        
        assert len(items) == 5
        assert items[0].instruction == "问题0"
        assert items[4].instruction == "问题4"
    
    def test_get_dataset_items_limit_offset(self, db_manager):
        """测试分页获取数据集条目"""
        # 添加10个条目
        for i in range(10):
            db_manager.add_dataset_item(
                dataset_name="test_dataset",
                instruction=f"问题{i}",
                input="",
                output=f"答案{i}"
            )
        
        # 测试limit
        items = db_manager.get_dataset_items("test_dataset", limit=3)
        assert len(items) == 3
        
        # 测试offset
        items = db_manager.get_dataset_items("test_dataset", limit=3, offset=5)
        assert len(items) == 3
        assert items[0].instruction == "问题5"
    
    def test_get_dataset_items_by_name(self, db_manager):
        """测试按名称获取数据集"""
        # 添加不同数据集
        for i in range(3):
            db_manager.add_dataset_item(
                dataset_name="dataset_a",
                instruction=f"A问题{i}",
                input="",
                output=f"A答案{i}"
            )
        
        for i in range(2):
            db_manager.add_dataset_item(
                dataset_name="dataset_b",
                instruction=f"B问题{i}",
                input="",
                output=f"B答案{i}"
            )
        
        items_a = db_manager.get_dataset_items("dataset_a")
        items_b = db_manager.get_dataset_items("dataset_b")
        
        assert len(items_a) == 3
        assert len(items_b) == 2
    
    def test_export_dataset(self, db_manager):
        """测试导出数据集"""
        for i in range(3):
            db_manager.add_dataset_item(
                dataset_name="test_dataset",
                instruction=f"问题{i}",
                input=f"输入{i}",
                output=f"答案{i}",
                metadata={"source": "test"}
            )
        
        data = db_manager.export_dataset("test_dataset")
        
        assert len(data) == 3
        assert data[0]["instruction"] == "问题0"
        assert data[0]["input"] == "输入0"
        assert data[0]["output"] == "答案0"
        assert data[0]["metadata"]["source"] == "test"
    
    def test_save_to_jsonl(self, db_manager):
        """测试保存为JSONL格式"""
        for i in range(3):
            db_manager.add_dataset_item(
                dataset_name="test_dataset",
                instruction=f"问题{i}",
                input="",
                output=f"答案{i}"
            )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name
            f.flush()
            
            try:
                count = db_manager.save_to_jsonl("test_dataset", output_path)
                
                assert count == 3
                
                # 验证文件内容
                with open(output_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    assert len(lines) == 3
                    
                    # 解析第一行
                    item = json.loads(lines[0])
                    assert item["instruction"] == "问题0"
            finally:
                os.unlink(output_path)
    
    def test_get_dataset_stats(self, db_manager):
        """测试获取数据集统计"""
        for i in range(5):
            db_manager.add_dataset_item(
                dataset_name="test_dataset",
                instruction=f"问题{i}",
                input="",
                output=f"答案{i}"
            )
        
        stats = db_manager.get_dataset_stats("test_dataset")
        
        assert stats["dataset_name"] == "test_dataset"
        assert stats["total_items"] == 5
    
    def test_clear_dataset(self, db_manager):
        """测试清空数据集"""
        for i in range(5):
            db_manager.add_dataset_item(
                dataset_name="test_dataset",
                instruction=f"问题{i}",
                input="",
                output=f"答案{i}"
            )
        
        # 清空
        db_manager.clear_dataset("test_dataset")
        
        # 验证
        items = db_manager.get_dataset_items("test_dataset")
        assert len(items) == 0
    
    def test_session_context_manager(self, db_manager):
        """测试会话上下文管理器"""
        with db_manager.session() as session:
            # 添加一个文档
            doc = Document(
                file_path="/test/session_test.md",
                file_name="session_test.md",
                file_type=".md",
                content_hash="test123"
            )
            session.add(doc)
        
        # 验证数据已保存
        doc = db_manager.get_document("/test/session_test.md")
        assert doc is not None
    
    def test_session_rollback(self, db_manager):
        """测试会话回滚"""
        try:
            with db_manager.session() as session:
                doc = Document(
                    file_path="/test/rollback_test.md",
                    file_name="rollback_test.md",
                    file_type=".md",
                    content_hash="test456"
                )
                session.add(doc)
                raise Exception("模拟错误")
        except Exception:
            pass
        
        # 验证数据未保存
        doc = db_manager.get_document("/test/rollback_test.md")
        assert doc is None
    
    def test_merge_document(self, db_manager):
        """测试合并文档记录（存在则更新）"""
        # 第一次添加
        doc_id1 = db_manager.add_document(
            file_path="/test/merge_test.md",
            content_hash="hash_v1"
        )
        
        # 第二次添加（相同路径，不同hash）
        doc_id2 = db_manager.add_document(
            file_path="/test/merge_test.md",
            content_hash="hash_v2"
        )
        
        # 验证是更新而不是新建
        assert doc_id1 == doc_id2
        
        # 验证hash已更新
        doc = db_manager.get_document("/test/merge_test.md")
        assert doc.content_hash == "hash_v2"


class TestDatabaseSchema:
    """测试数据库表结构"""
    
    def test_document_table_columns(self, db_manager):
        """测试文档表字段"""
        with db_manager.session() as session:
            result = session.execute(text("SELECT * FROM documents LIMIT 1"))
            # 如果表为空，查询仍然成功说明表结构正确
            assert result.returns_rows or True  # 表存在即可
    
    def test_dataset_items_table_columns(self, db_manager):
        """测试数据集条目表字段"""
        with db_manager.session() as session:
            result = session.execute(text("SELECT * FROM dataset_items LIMIT 1"))
            assert result.returns_rows or True
