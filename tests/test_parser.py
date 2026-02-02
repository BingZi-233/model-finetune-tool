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
    
    def test_supports_various_extensions(self):
        """测试各种扩展名"""
        parser = DocxParser()
        assert parser.supports("file.docx") is True
        assert parser.supports("file.DOCX") is True
        assert parser.supports("file.txt") is False
        assert parser.supports("file.pdf") is False


class TestPdfParser:
    """测试PDF文档解析器"""
    
    def test_supports_pdf(self):
        """测试支持pdf文件"""
        parser = PdfParser()
        assert parser.supports("document.pdf") is True
        assert parser.supports("file.PDF") is True
        assert parser.supports("test.docx") is False


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
                assert len(result) >= 2
                assert "标题" in result[0] or "段落" in result[0]
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

## 另一个部分

更多内容。
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                parser = MarkdownParser()
                result = parser.parse(f.name)
                
                # 验证有内容被解析
                assert len(result) >= 1
                # 验证不包含 front matter 中的字段
                for item in result:
                    assert "title" not in item.lower()
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
            finally:
                os.unlink(f.name)
    
    def test_parse_with_lists(self):
        """测试解析带列表的Markdown"""
        content = """
- 第一项
- 第二项
- 第三点
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            try:
                parser = MarkdownParser()
                result = parser.parse(f.name)
                
                # 验证列表符号被移除
                for item in result:
                    assert not item.startswith("-")
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
                f.write("# 测试\n\n这是测试内容，足够长以通过过滤。")
            
            another_file = os.path.join(temp_dir, "test2.md")
            with open(another_file, 'w') as f:
                f.write("# 另一个测试\n\n这是另一个测试内容，也足够长。")
            
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
                f.write("# 测试1\n\n这是测试内容1，足够长以通过过滤。")
            
            file2 = os.path.join(sub_dir, "test2.md")
            with open(file2, 'w') as f:
                f.write("# 测试2\n\n这是测试内容2，也足够长。")
            
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
        # 验证有解析器
        assert len(manager.parsers) == 3
        # 验证每个解析器的supports方法都能正常工作
        for parser in manager.parsers:
            # 测试每个解析器支持的格式
            if isinstance(parser, DocxParser):
                assert parser.supports("test.docx") is True
            elif isinstance(parser, PdfParser):
                assert parser.supports("test.pdf") is True
            elif isinstance(parser, MarkdownParser):
                assert parser.supports("test.md") is True
    
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
