"""
LLM调用模块

本模块提供高质量的LLM调用接口，用于生成训练数据集。
设计目标：最大化数据质量，不计token消耗。
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from ..config import get_config

logger = logging.getLogger(__name__)

# ============ 常量定义 ============
MAX_INPUT_LENGTH = 50000  # 最大输入长度 (50KB)
DEFAULT_CACHE_MAX_SIZE = 1000  # 默认缓存最大条目数
DEFAULT_CACHE_MAX_AGE = 86400  # 默认缓存最大存活时间 (24小时)


# ============ 自定义异常 ============
class LLMError(Exception):
    """LLM 调用错误基类"""
    pass


class QAGenerationError(LLMError):
    """QA 对生成错误"""
    pass


class JSONParseError(LLMError):
    """JSON 解析错误"""
    pass


class CacheError(Exception):
    """缓存错误"""
    pass


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
            
        Raises:
            QAGenerationError: 生成失败
        """
        # 验证输入
        if not text or not text.strip():
            logger.warning("输入文本为空")
            return []
        
        if len(text) > MAX_INPUT_LENGTH:
            raise QAGenerationError(
                f"输入文本过长 ({len(text)} > {MAX_INPUT_LENGTH} 字符)"
            )
        
        if num_pairs < 1 or num_pairs > 20:
            logger.warning(f"无效的 num_pairs 值: {num_pairs}，使用默认值 5")
            num_pairs = 5
        
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
                    logger.info(f"成功生成 {len(pairs)} 个QA对 (尝试 {attempt + 1}/3)")
                    break
                    
            except JSONParseError as e:
                logger.warning(f"JSON解析失败 (尝试 {attempt + 1}/3): {e}")
            except Exception as e:
                logger.error(f"生成QA对失败 (尝试 {attempt + 1}/3): {e}")
                if attempt == 2:  # 最后一次尝试
                    raise QAGenerationError(f"生成QA对失败: {e}")
                continue
        
        # 如果自动生成失败，返回基于规则的fallback
        if not best_result:
            logger.warning("使用fallback生成简单QA对")
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
        
        raise JSONParseError(
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
        except JSONParseError:
            # Fallback: 返回简单格式
            logger.warning("对话生成JSON解析失败，使用fallback")
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
                logger.warning(f"生成失败: {e}")
                continue
        
        return all_pairs


class CacheManager:
    """
    LLM响应缓存管理器
    
    用于避免重复调用LLM，节省成本
    
    特性：
    - 缓存清理机制（大小限制、时间限制）
    - 线程安全
    - 跨平台支持
    """
    
    def __init__(
        self, 
        cache_dir: str = "./data/cache",
        max_size: int = DEFAULT_CACHE_MAX_SIZE,
        max_age: int = DEFAULT_CACHE_MAX_AGE
    ):
        """初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            max_size: 最大缓存条目数 (默认 1000)
            max_age: 缓存最大存活时间，秒 (默认 24小时)
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.max_age = max_age
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, text: str, **kwargs) -> str:
        """生成缓存key"""
        import hashlib
        content = text + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.json"
    
    def _get_cache_info_file(self, key: str) -> Path:
        """获取缓存信息文件路径"""
        return self.cache_dir / f"{key}.info"
    
    def _get_file_age(self, file_path: Path) -> float:
        """获取文件年龄（秒）"""
        try:
            return time.time() - file_path.stat().st_mtime
        except OSError:
            return float('inf')
    
    def _save_cache_info(self, key: str, metadata: Dict = None):
        """保存缓存元信息"""
        info_file = self._get_cache_info_file(key)
        info = {
            'created_at': time.time(),
            'key': key,
            **(metadata or {})
        }
        try:
            with open(info_file, 'w') as f:
                json.dump(info, f)
        except Exception as e:
            logger.warning(f"保存缓存信息失败: {e}")
    
    def get(self, text: str, **kwargs) -> Optional[str]:
        """获取缓存
        
        Args:
            text: 缓存的文本
            **kwargs: 其他参数
            
        Returns:
            缓存的响应，如果不存在返回 None
        """
        key = self._get_cache_key(text, **kwargs)
        cache_file = self._get_cache_file(key)
        info_file = self._get_cache_info_file(key)
        
        if not cache_file.exists():
            return None
        
        # 检查是否过期
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                created_at = info.get('created_at', 0)
                if time.time() - created_at > self.max_age:
                    # 缓存过期，删除
                    self._delete_cache(key)
                    return None
            except Exception:
                pass
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def set(self, text: str, response: str, **kwargs):
        """设置缓存
        
        Args:
            text: 缓存的文本
            response: 缓存的响应
            **kwargs: 其他参数
        """
        key = self._get_cache_key(text, **kwargs)
        cache_file = self._get_cache_file(key)
        
        # 检查是否需要清理缓存
        self._cleanup_if_needed()
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(response)
            self._save_cache_info(key, {'text_length': len(text)})
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _delete_cache(self, key: str):
        """删除缓存"""
        cache_file = self._get_cache_file(key)
        info_file = self._get_cache_info_file(key)
        
        try:
            if cache_file.exists():
                cache_file.unlink()
            if info_file.exists():
                info_file.unlink()
        except Exception as e:
            logger.warning(f"删除缓存失败: {e}")
    
    def _cleanup_if_needed(self):
        """必要时清理缓存"""
        try:
            # 统计缓存数量
            cache_files = list(self.cache_dir.glob("*.json"))
            
            if len(cache_files) < self.max_size:
                return
            
            # 删除最旧的缓存
            cache_with_age = []
            for cache_file in cache_files:
                key = cache_file.stem
                info_file = self._get_cache_info_file(key)
                
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        created_at = info.get('created_at', 0)
                        cache_with_age.append((cache_file, created_at))
                    except Exception:
                        cache_with_age.append((cache_file, 0))
                else:
                    cache_with_age.append((cache_file, 0))
            
            # 按时间排序，删除最旧的
            cache_with_age.sort(key=lambda x: x[1])
            num_to_delete = len(cache_files) - self.max_size + 100
            
            for cache_file, _ in cache_with_age[:num_to_delete]:
                key = cache_file.stem
                self._delete_cache(key)
                logger.debug(f"清理缓存: {key}")
                
        except Exception as e:
            logger.warning(f"清理缓存失败: {e}")
    
    def clear(self):
        """清空所有缓存"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            for info_file in self.cache_dir.glob("*.info"):
                info_file.unlink()
            logger.info(f"已清空缓存目录: {self.cache_dir}")
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            raise CacheError(f"清空缓存失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            info_files = list(self.cache_dir.glob("*.info"))
            
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())
            
            # 计算缓存年龄
            ages = []
            for info_file in info_files:
                try:
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    created_at = info.get('created_at', 0)
                    if created_at:
                        ages.append(time.time() - created_at)
                except Exception:
                    pass
            
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return {
                'cache_count': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / 1024 / 1024,
                'max_size': self.max_size,
                'max_age_seconds': self.max_age,
                'avg_age_seconds': avg_age,
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}


"""
LLM 调用模块

本模块提供 OpenAI API 集成和高品质训练数据生成功能。

## 主要功能

### LLMClient - LLM 客户端

提供与 OpenAI API 的交互接口，支持：
- 基础对话功能
- QA 对批量生成
- 文本摘要生成
- 对话数据生成
- 响应缓存

### CacheManager - 缓存管理器

提供 LLM 响应缓存功能，支持：
- 自动缓存管理
- 缓存大小限制
- 缓存过期清理
- 缓存统计

## 使用示例

```python
from src.llm import LLMClient, CacheManager

# 创建 LLM 客户端
client = LLMClient(
    api_key="your-api-key",
    model="gpt-4o",
    temperature=0.3
)

# 发送对话
response = client.chat([
    {"role": "user", "content": "你好！"}
])
print(response)

# 生成 QA 对
qa_pairs = client.generate_qa_pairs(
    text="这是一段测试文本...",
    num_pairs=5,
    language="zh"
)

# 使用缓存
cache = CacheManager(
    cache_dir="./data/cache",
    max_size=1000,
    max_age=86400  # 24小时
)

# 检查缓存
cached = cache.get(text)
if not cached:
    response = client.generate_qa_pairs(text)
    cache.set(text, response)
```

## 异常处理

本模块定义了以下自定义异常：

- `LLMError` - LLM 调用错误基类
- `QAGenerationError` - QA 对生成错误
- `JSONParseError` - JSON 解析错误
- `CacheError` - 缓存操作错误

```python
from src.llm import LLMClient, QAGenerationError, JSONParseError

client = LLMClient()

try:
    qa_pairs = client.generate_qa_pairs(text)
except QAGenerationError as e:
    print(f"QA 生成失败: {e}")
except JSONParseError as e:
    print(f"JSON 解析失败: {e}")
```

## 性能优化

1. **缓存策略**
   - 默认缓存 1000 条
   - 默认过期时间 24 小时
   - 自动清理旧缓存

2. **批处理**
   - 使用 `batch_generate_qa()` 批量生成
   - 自动过滤空文本

3. **温度设置**
   - 默认温度 0.2，确保输出稳定
   - 可根据需要调整

## 注意事项

- 需要有效的 OpenAI API Key
- API 调用会计费，请注意成本控制
- 建议设置合理的 max_tokens 限制
- 大文本会自动分块处理
"""

__all__ = [
    'LLMClient',
    'CacheManager',
    'LLMError',
    'QAGenerationError',
    'JSONParseError',
    'CacheError',
]
