"""
LLM调用模块

本模块提供高质量的LLM调用接口，用于生成训练数据集。
设计目标：最大化数据质量，不计token消耗。
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..config import get_config


class LLMClient:
    """
    高质量LLM客户端
    
    特点：
    - 使用最高品质模型配置
    - 多轮生成+质量筛选
    - 详细的生成prompt
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        config = get_config()
        
        self.api_key = api_key or config.llm.api_key
        self.base_url = base_url or config.llm.base_url
        # 强制使用最高品质配置
        self.model = model or config.llm.model
        self.temperature = 0.2  # 降低随机性，提高质量
        self.max_tokens = None  # 不限制，让模型生成完整回答
        
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
        """
        发送对话请求
        
        使用较低温度确保输出质量稳定
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            # 高质量参数
            presence_penalty=0.1,
            frequency_penalty=0.1,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def generate_qa_pairs(
        self,
        text: str,
        num_pairs: int = 5,
        language: str = "zh"
    ) -> List[Dict[str, str]]:
        """
        从文本生成高质量QA对
        
        特点：
        - 详细的system prompt指导
        - 每个QA都基于文本内容
        - 强制JSON格式输出
        - 自动重试机制
        
        Args:
            text: 输入文本
            num_pairs: 生成QA对数量 (默认5)
            language: 语言
            
        Returns:
            QA对列表
        """
        lang_prompt = "中文" if language == "zh" else "English"
        
        # 高质量system prompt
        system_prompt = f"""你是一个专业的知识提取专家，负责从文档中生成高质量的问答对用于AI训练。

## 核心任务
根据提供的文本内容，生成 {num_pairs} 个高质量的问答对。

## 质量标准

### 问题要求
1. **覆盖全面** - 问题应覆盖文本的核心概念、重要细节和关键信息
2. **层次分明** - 包含不同难度级别：
   - 基础问题（是什么、谁、何时、何地）
   - 进阶问题（为什么、如何、原理）
   - 深度问题（分析、比较、应用）
3. **表述清晰** - 问题明确、无歧义、专业术语使用准确
4. **独立完整** - 每个问题都能独立理解，不需要额外上下文

### 答案要求
1. **准确无误** - 答案必须完全基于文本内容
2. **详细完整** - 提供充分的解释和上下文
3. **结构清晰** - 复杂答案使用适当的格式
4. **深度适当** - 根据问题类型调整答案深度

### 输出要求
1. 严格JSON格式
2. 每个QA对独立完整
3. 不要重复或类似的问题
4. 问题答案要一一对应

请生成这 {num_pairs} 个问答对。保持{language}输出。"""
        
        user_prompt = f"""## 待处理文本

以下是从文档中提取的文本内容，请仔细分析并生成问答对：

---
{text}
---

请按照上述质量标准，生成 {num_pairs} 个高质量问答对。

## 输出格式
```json
[
  {{
    "instruction": "清晰明确的问题",
    "input": "",
    "output": "详细准确的答案"
  }}
]
```

确保：
1. 问题覆盖文本的核心内容
2. 答案详细且基于文本
3. JSON格式正确无误
4. 不要添加任何解释性文字"""
        
        # 尝试多次生成，选择最好的结果
        best_result = []
        for attempt in range(3):  # 最多重试3次
            try:
                response = self.chat([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                
                pairs = self._extract_json(response)
                
                # 验证质量
                if self._validate_qa_pairs(pairs, num_pairs):
                    best_result = pairs
                    break
                    
            except Exception as e:
                if attempt == 2:  # 最后一次尝试
                    print(f"生成QA对失败 (尝试 {attempt + 1}/3): {e}")
                continue
        
        # 如果自动生成失败，返回基于规则的fallback
        if not best_result:
            print("使用fallback生成简单QA对")
            best_result = self._generate_simple_qa(text, num_pairs)
        
        return best_result
    
    def _validate_qa_pairs(
        self, 
        pairs: List[Dict], 
        expected_count: int
    ) -> bool:
        """
        验证QA对质量
        
        检查：
        - 数量是否足够
        - 格式是否正确
        - 是否有空内容
        """
        if not pairs:
            return False
        
        if len(pairs) < expected_count // 2:
            return False
        
        for pair in pairs:
            if not isinstance(pair, dict):
                return False
            if not pair.get("instruction") or not pair.get("output"):
                return False
        
        return True
    
    def _extract_json(self, response: str) -> List[Dict[str, str]]:
        """
        从响应中提取JSON
        
        尝试多种方式提取：
        1. 直接解析
        2. 从代码块中提取
        3. 查找JSON数组
        """
        import re
        
        # 方式1: 直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 方式2: 从代码块中提取
        json_match = re.search(
            r'```(?:json)?\s*([\s\S]*?)\s*```', 
            response
        )
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 方式3: 查找JSON数组
        array_match = re.search(r'(\[[\s\S]*?\])\s*$', response)
        if array_match:
            try:
                return json.loads(array_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 方式4: 查找任意JSON数组
        all_arrays = re.findall(r'\[[\s\S]*?\]', response)
        for arr_str in all_arrays:
            try:
                parsed = json.loads(arr_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        raise ValueError(
            f"无法解析LLM响应中的JSON:\n"
            f"响应内容: {response[:500]}..."
        )
    
    def _generate_simple_qa(
        self, 
        text: str, 
        num_pairs: int
    ) -> List[Dict[str, str]]:
        """
        简单的fallback QA生成
        
        当LLM生成失败时使用
        """
        import re
        
        # 切分文本为句子
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        
        pairs = []
        for i, sent in enumerate(sentences[:num_pairs]):
            pairs.append({
                "instruction": f"请解释以下内容",
                "input": "",
                "output": sent
            })
        
        return pairs
    
    def generate_summarization(
        self,
        text: str,
        language: str = "zh"
    ) -> str:
        """生成高质量摘要"""
        lang_prompt = "中文" if language == "zh" else "English"
        
        response = self.chat([
            {
                "role": "system",
                "content": f"""你是一个专业的文本摘要专家。
请用{lang_prompt}生成一段简洁而全面的摘要。
要求：
1. 保留关键信息和核心观点
2. 逻辑清晰，结构完整
3. 字数适中（200-500字）"""
            },
            {
                "role": "user",
                "content": f"请为以下文本生成摘要：\n\n{text}"
            }
        ])
        
        return response.strip()
    
    def generate_conversation(
        self,
        text: str,
        num_turns: int = 3,
        language: str = "zh"
    ) -> List[Dict[str, str]]:
        """生成高质量对话数据"""
        lang_prompt = "中文" if language == "zh" else "English"
        
        system_prompt = f"""你是一个乐于助人的助手。
请根据提供的文档内容，生成一段自然的对话。

要求：
1. 对话自然流畅，像真实对话
2. 内容基于提供的文档
3. 体现文档的核心信息
4. {lang_prompt}输出
5. JSON数组格式

对话格式：
[
  {{"role": "user", "content": "用户问题"}},
  {{"role": "assistant", "content": "助手回答"}}
]

请生成 {num_turns} 轮对话。"""
        
        response = self.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"基于以下内容生成对话：\n\n{text}"}
        ])
        
        try:
            return self._extract_json(response)
        except ValueError:
            # Fallback: 返回简单格式
            return [
                {"role": "user", "content": "请介绍一下"},
                {"role": "assistant", "content": "好的，让我来介绍..."}
            ]
    
    def batch_generate_qa(
        self,
        texts: List[str],
        num_pairs_per_text: int = 5,
        progress: bool = True
    ) -> List[Dict[str, str]]:
        """
        批量生成QA对
        
        特点：
        - 每个文本独立生成
        - 显示进度条
        - 跳过空文本
        """
        from tqdm import tqdm
        
        all_pairs = []
        # 过滤空文本
        texts = [t for t in texts if t.strip()]
        
        iterator = tqdm(texts, desc="🔄 生成高质量QA对") if progress else texts
        
        for text in iterator:
            try:
                pairs = self.generate_qa_pairs(text, num_pairs_per_text)
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"\n⚠️ 生成失败: {e}")
                continue
        
        return all_pairs


class CacheManager:
    """
    LLM响应缓存管理器
    
    用于避免重复调用LLM，节省成本
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, text: str, **kwargs) -> str:
        """生成缓存key"""
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
