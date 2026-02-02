"""LLM调用模块"""
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import get_config


class LLMClient:
    """OpenAI兼容的LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        config = get_config()
        
        self.api_key = api_key or config.llm.api_key
        self.base_url = base_url or config.llm.base_url
        self.model = model or config.llm.model
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """发送对话请求"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def generate_qa_pairs(
        self,
        text: str,
        num_pairs: int = 3,
        language: str = "zh"
    ) -> List[Dict[str, str]]:
        """从文本生成QA对"""
        lang_prompt = "中文" if language == "zh" else "English"
        
        system_prompt = f"""你是一个专业的教学助手，负责从文档中生成高质量的问答对。
请根据提供的文本内容，生成{num_pairs}个问答对。
要求：
1. 问题要简洁明确，覆盖文本的核心内容
2. 答案要准确，基于文本内容
3. 保持{lang_prompt}输出
4. 输出格式为JSON数组
"""
        
        user_prompt = f"""请分析以下文本，生成{num_pairs}个问答对：

{text}

请按以下JSON格式输出：
[
  {{"instruction": "问题1", "input": "", "output": "答案1"}},
  {{"instruction": "问题2", "input": "", "output": "答案2"}}
]
"""
        
        response = self.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # 提取JSON
        return self._extract_json(response)
    
    def generate_summarization(
        self,
        text: str,
        language: str = "zh"
    ) -> str:
        """生成文本摘要"""
        lang_prompt = "中文" if language == "zh" else "English"
        
        response = self.chat([
            {
                "role": "system",
                "content": f"请用{lang_prompt}简洁地总结以下文本，保留关键信息。"
            },
            {
                "role": "user",
                "content": f"总结这段文本：\n\n{text}"
            }
        ])
        
        return response.strip()
    
    def generate_conversation(
        self,
        text: str,
        num_turns: int = 2,
        language: str = "zh"
    ) -> List[Dict[str, str]]:
        """生成对话形式的数据"""
        lang_prompt = "中文" if language == "zh" else "English"
        
        system_prompt = f"""你是一个乐于助人的助手。请根据提供的文档内容，生成一个{n_turns}轮对话示例。
要求：
1. 对话要自然流畅
2. 内容要基于文档
3. 保持{lang_prompt}
4. 输出格式为JSON数组
"""
        
        response = self.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"基于以下内容生成对话：\n\n{text}"}
        ])
        
        return self._extract_json(response)
    
    def _extract_json(self, response: str) -> List[Dict[str, str]]:
        """从响应中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试从代码块中提取
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试查找JSON数组
        array_match = re.search(r'(\[[\s\S]*?\])', response)
        if array_match:
            try:
                return json.loads(array_match.group(1))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"无法解析LLM响应中的JSON:\n{response}")
    
    def batch_generate_qa(
        self,
        texts: List[str],
        num_pairs_per_text: int = 3,
        progress: bool = True
    ) -> List[Dict[str, str]]:
        """批量生成QA对"""
        from tqdm import tqdm
        
        all_pairs = []
        texts = [t for t in texts if t.strip()]
        
        iterator = tqdm(texts, desc="生成QA对") if progress else texts
        
        for text in iterator:
            try:
                pairs = self.generate_qa_pairs(text, num_pairs_per_text)
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"生成QA对失败: {e}")
                continue
        
        return all_pairs


class CacheManager:
    """LLM响应缓存管理器"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, text: str, **kwargs) -> str:
        """生成缓存key"""
        import hashlib
        content = text + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, **kwargs) -> Optional[str]:
        """获取缓存"""
        key = self._get_cache_key(text, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def set(self, text: str, response: str, **kwargs):
        """设置缓存"""
        key = self._get_cache_key(text, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(response)
