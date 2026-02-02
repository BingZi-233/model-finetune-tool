"""
文档解析器测试模块

本模块测试各类文档解析器的功能，包括Word、PDF和Markdown文档的解析。
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.parser import (
    DocxParser,
    PdfParser,
    MarkdownParser,
    ParserManager,
    BaseParser,
)


class TestDocxParser:
    """测试Word文档解析器"""
    
    def test_supports_docx(self):
        """测试支持docx文件"""
        parser = DocxParser()
        assert parser.supports("test.docx") is True
        assert parser.supports("document.DOCX") is True
        assert parser.supports("test.txt") is False
        assert parser.supports("test.pdf") is False
    
    @patch('src.parser.Document')
    def test_parse_simple_paragraphs(self, mock_document_class):
        """测试解析简单段落"""
        # 模拟Document对象
        mock_doc = Mock()
        mock_doc.paragraphs = [
            Mock(text="这是第一段"),
            Mock(text="这是第二段"),
            Mock(text=""),
            Mock(text="这是第三段"),
        ]
        mock_doc.tables = []
        mock_document_class.return_value = mock_doc
        
        parser = DocxParser()
        result = parser.parse("test.docx")
        
        assert len(result) == 3
        assert result[0] == "这是第一段"
        assert result[1] == "这是第二段"
        assert result[2] == "这是第三段"
    
    @patch('src.parser.Document')
    def test_parse_with_tables(self, mock_document_class):
        """测试解析包含表格的文档"""
        mock_doc = Mock()
        mock_doc.paragraphs = [
            Mock(text="标题段落"),
        ]
        
        # 模拟表格
        mock_table = Mock()
        mock_row = Mock()
        mock_cell1 = Mock(text="单元格1")
        mock_cell2 = Mock(text="单元格2")
        mock_row.cells = [mock_cell1, mock_cell2]
        mock_table.rows = [mock_row]
        mock_doc.tables = [mock_table]
        
        mock_document_class.return_value = mock_doc
        
        parser = DocxParser()
        result = parser.parse("test.docx")
        
        assert len(result) == 3
        assert "标题段落" in result
        assert "单元格1" in result
        assert "单元格2" in result
    
    @patch('src.parser.Document')
    def test_filter_empty_paragraphs(self, mock_document_class):
        """测试过滤空段落"""
        mock_doc = Mock()
        mock_doc.paragraphs = [
            Mock(text=""),
            Mock(text="有效段落"),
            Mock(text="  "),
            Mock(text="另一个有效段落"),
            Mock(text="\t\n"),
        ]
        mock_doc.tables = []
        mock_document_class.return_value = mock_doc
        
        parser = DocxParser()
        result = parser.parse("test.docx")
        
        assert len(result) == 2
        assert result[0] == "有效段落"
        assert result[1] == "另一个有效段落"


class TestPdfParser:
    """测试PDF文档解析器"""
    
    def test_supports_pdf(self):
        """测试支持pdf文件"""
        parser = PdfParser()
        assert parser.supports("document.pdf") is True
        assert parser.supports("file.PDF") is True
        assert parser.supports("test.docx") is False
    
    @patch('src.parser.fitz')
    def test_parse_single_page(self, mock_fitz):
        """测试解析单页PDF"""
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "第一段\n\n第二段\n\n第三段"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_fitz.open.return_value = mock_doc
        
        parser = PdfParser()
        result = parser.parse("test.pdf")
        
        assert len(result) == 3
        assert result[0] == "第一段"
        assert result[1] == "第二段"
        assert result[2] == "第三段"
    
    @patch('src.parser.fitz')
    def test_parse_multi_page(self, mock_fitz):
        """测试解析多页PDF"""
        mock_doc = Mock()
        
        pages = []
        for i in range(3):
            mock_page = Mock()
            mock_page.get_text.return_value = f"第{i+1}页内容"
            pages.append(mock_page)
        
        mock_doc.__iter__ = Mock(return_value=iter(pages))
        mock_fitz.open.return_value = mock_doc
        
        parser = PdfParser()
        result = parser.parse("test.pdf")
        
        assert len(result) == 3
        assert result[0] == "第1页内容"
        assert result[1] == "第2页内容"
        assert result[2] == "第3页内容"
    
    @patch('src.parser.fitz')
    def test_filter_empty_paragraphs(self, mock_fitz):
        """测试过滤空段落"""
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "有效内容\n\n\n\n另一个有效内容"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_fitz.open.return_value = mock_doc
        
        parser = PdfParser()
        result = parser.parse("test.pdf")
        
        assert len(result) == 2


class TestMarkdownParser:
    """测试Markdown文档解析器"""
    
    def test_supports_md(self):
        """测试支持md文件"""
        parser = MarkdownParser()
        assert parser.supports("readme.md") is True
        assert parser.supports("DOCUMENT.MD") is True
        assert parser.supports("test.txt") is False
    
    def test_parse_simple_markdown(self):
        """测试解析简单Markdown"""
        content = """
# 标题

这是一个段落。

## 子标题

另一个段落。
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                parser = MarkdownParser()
                result = parser.parse(f.name)
                
                # 验证标题符号被移除
                assert "# 标题" not in result
                assert "标题" in result
                assert "## 子标题" not in result
                assert "子标题" in result
            finally:
                os.unlink(f.name)
    
    def test_parse_with_yaml_front_matter(self):
        """测试解析带YAML前置元数据的Markdown"""
        content = """---
title: "测试文档"
author: "测试作者"
---

# 正文标题

这是正文内容。
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                parser = MarkdownParser()
                result = parser.parse(f.name)
                
                # 验证元数据被移除
                assert len(result) >= 2
                assert "title" not in result[0].lower() or "正文标题" in result[0]
            finally:
                os.unlink(f.name)
    
    def test_parse_with_code_blocks(self):
        """测试解析带代码块的Markdown"""
        content = """
这是一个段落。

```python
def hello():
    print("Hello")
```

另一个段落。
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                parser = MarkdownParser()
                result = parser.parse(f.name)
                
                # 验证代码块被移除
                assert len(result) == 2
                assert "python" not in result[0]
                assert "print" not in result[0]
            finally:
                os.unlink(f.name)
    
    def test_parse_with_links(self):
        """测试解析带链接的Markdown"""
        content = """
这是一个[链接文本](https://example.com)。

另一个段落。
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                parser = MarkdownParser()
                result = parser.parse(f.name)
                
                # 验证链接被提取为纯文本
                assert len(result) == 2
                assert "链接文本" in result[0]
                assert "https://example.com" not in result[0]
            finally:
                os.unlink(f.name)
    
    def test_parse_with_lists(self):
        """测试解析带列表的Markdown"""
        content = """
- 第一项
- 第二项
- 第三项

1. 第一点
2. 第二点
3. 第三点
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                parser = MarkdownParser()
                result = parser.parse(f.name)
                
                # 验证列表符号被移除
                assert len(result) >= 2
                for item in result:
                    assert not item.startswith("-")
                    assert not item.startswith("1.")
            finally:
                os.unlink(f.name)
    
    def test_filter_short_paragraphs(self):
        """测试过滤太短的段落"""
        content = """
# 短

a

这是一个有效段落。
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                parser = MarkdownParser()
                result = parser.parse(f.name)
                
                # 验证过短的段落被过滤
                assert len(result) <= 2
            finally:
                os.unlink(f.name)


class TestParserManager:
    """测试解析器管理器"""
    
    def test_parse_single_file(self):
        """测试解析单个文件"""
        content = "# 测试文档\n\n这是一个测试段落。"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                manager = ParserManager()
                result = manager.parse_file(f.name)
                
                assert len(result) >= 1
                assert "测试文档" in result[0] or "测试段落" in result[0]
            finally:
                os.unlink(f.name)
    
    def test_parse_directory(self):
        """测试解析整个目录"""
        # 创建临时目录和文件
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 创建测试文件
            md_file = os.path.join(temp_dir, "test.md")
            with open(md_file, 'w') as f:
                f.write("# 测试\n\n测试内容")
            
            another_file = os.path.join(temp_dir, "test2.md")
            with open(another_file, 'w') as f:
                f.write("# 另一个测试\n\n另一个内容")
            
            manager = ParserManager()
            result = manager.parse_directory(temp_dir, recursive=False)
            
            assert len(result) == 2
            assert str(md_file) in result
            assert str(another_file) in result
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_parse_directory_recursive(self):
        """测试递归解析目录"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 创建子目录
            sub_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(sub_dir)
            
            # 创建文件
            file1 = os.path.join(temp_dir, "test1.md")
            with open(file1, 'w') as f:
                f.write("# 测试1\n\n内容1")
            
            file2 = os.path.join(sub_dir, "test2.md")
            with open(file2, 'w') as f:
                f.write("# 测试2\n\n内容2")
            
            manager = ParserManager()
            
            # 测试递归
            result = manager.parse_directory(temp_dir, recursive=True)
            assert len(result) == 2
            
            # 测试非递归
            result = manager.parse_directory(temp_dir, recursive=False)
            assert len(result) == 1
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_get_supported_extensions(self):
        """测试获取支持的文件扩展名"""
        manager = ParserManager()
        extensions = manager.get_supported_extensions()
        
        assert ".docx" in extensions
        assert ".pdf" in extensions
        assert ".md" in extensions
    
    def test_unsupported_file_raises_error(self):
        """测试不支持的文件格式抛出异常"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            f.flush()
            
            try:
                manager = ParserManager()
                with pytest.raises(ValueError) as exc_info:
                    manager.parse_file(f.name)
                
                assert "不支持" in str(exc_info.value)
            finally:
                os.unlink(f.name)
