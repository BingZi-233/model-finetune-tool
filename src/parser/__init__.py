"""文档解析器基类"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Type


class BaseParser(ABC):
    """解析器基类"""
    
    @abstractmethod
    def parse(self, file_path: str) -> List[str]:
        """解析文件，返回文本段落列表"""
        pass
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """检查是否支持该文件格式"""
        pass


class DocxParser(BaseParser):
    """Word文档解析器"""
    
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in ['.docx']
    
    def parse(self, file_path: str) -> List[str]:
        from docx import Document
        
        doc = Document(file_path)
        paragraphs = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        
        # 提取表格内容
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text = cell.text.strip()
                    if text:
                        paragraphs.append(text)
        
        return paragraphs


class PdfParser(BaseParser):
    """PDF文档解析器"""
    
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == '.pdf'
    
    def parse(self, file_path: str) -> List[str]:
        import fitz  # PyMuDF
        
        doc = fitz.open(file_path)
        paragraphs = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            # 按段落分割
            for para in text.split('\n\n'):
                para = para.strip()
                if para:
                    paragraphs.append(para)
        
        return paragraphs


class MarkdownParser(BaseParser):
    """Markdown文档解析器"""
    
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == '.md'
    
    def parse(self, file_path: str) -> List[str]:
        import re
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除YAML front matter
        # 移除YAML front matter
        content = re.sub(r'^---\n[\s\S]*?---\n', '', content)
        
        # 移除代码块
        content = re.sub(r'```[\s\S]*?```', '', content)
        
        # 移除行内代码
        content = re.sub(r'`[^`]+`', '', content)
        
        # 移除图片链接
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        
        # 移除链接
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        paragraphs = []
        for para in content.split('\n\n'):
            para = para.strip()
            # 移除标题符号
            para = re.sub(r'^#+\s+', '', para)
            # 移除列表符号
            para = re.sub(r'^[\s]*[-*+]\s+', '', para)
            para = re.sub(r'^[\s]*\d+\.\s+', '', para)
            
            if para and len(para) > 5:  # 过滤太短的段落
                paragraphs.append(para)
        
        return paragraphs


class ParserManager:
    """解析器管理器"""
    
    def __init__(self):
        self.parsers: List[BaseParser] = [
            DocxParser(),
            PdfParser(),
            MarkdownParser(),
        ]
    
    def parse_file(self, file_path: str) -> List[str]:
        """解析单个文件"""
        for parser in self.parsers:
            if parser.supports(file_path):
                return parser.parse(file_path)
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    def parse_directory(self, dir_path: str, recursive: bool = True) -> Dict[str, List[str]]:
        """解析整个目录"""
        from pathlib import Path
        
        result = {}
        path = Path(dir_path)
        
        if recursive:
            files = list(path.rglob("*"))
        else:
            files = list(path.glob("*"))
        
        for file_path in files:
            if file_path.is_file():
                try:
                    content = self.parse_file(str(file_path))
                    if content:
                        result[str(file_path)] = content
                except Exception as e:
                    print(f"解析失败 {file_path}: {e}")
        
        return result
    
    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名"""
        extensions = []
        for parser in self.parsers:
            if hasattr(parser, 'extensions'):
                extensions.extend(parser.extensions)
        return list(set(extensions))
