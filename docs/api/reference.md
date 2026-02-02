# API 参考文档

> model-finetune-tool v0.1.0 编程接口

## 目录

- [配置模块](#配置模块)
- [数据集模块](#数据集模块)
- [LLM 模块](#llm-模块)
- [训练模块](#训练模块)
- [解析器模块](#解析器模块)

---

## 配置模块

### load_config

```python
from src.config import load_config

config = load_config(config_path: str = "config.yaml") -> Config
```

加载配置文件。

**参数:**
- `config_path` (str): 配置文件路径，默认 `"config.yaml"`

**返回:**
- `Config`: 配置对象

**示例:**
```python
from src.config import load_config

config = load_config("config.yaml")
print(config.llm.model)
```

### get_config

```python
from src.config import get_config

config = get_config() -> Config
```

获取全局配置实例（单例模式）。

**返回:**
- `Config`: 配置对象

**示例:**
```python
from src.config import get_config

config = get_config()
print(config.training.epochs)
```

### reload_config

```python
from src.config import reload_config

config = reload_config(config_path: str = "config.yaml", force: bool = True) -> Config
```

重新加载配置文件。

**参数:**
- `config_path` (str): 配置文件路径
- `force` (bool): 强制重新加载，默认 `True`

**返回:**
- `Config`: 配置对象

### ConfigManager

```python
from src.config import ConfigManager

manager = ConfigManager()

# 加载配置
config = manager.load_config("config.yaml")

# 获取配置
config = manager.get_config()

# 清除缓存
manager.clear_cache()
```

配置管理器，支持热重载。

---

## 数据集模块

### DatasetManager

```python
from src.dataset import DatasetManager

manager = DatasetManager(db_path: Optional[str] = None)
```

数据集管理器。

#### 方法

##### add_document

```python
doc_id = manager.add_document(
    file_path: str,
    content_hash: str,
    extra_data: Optional[Dict] = None
) -> int
```

添加文档记录。

**参数:**
- `file_path` (str): 文件路径
- `content_hash` (str): 内容哈希
- `extra_data` (Dict): 额外数据（可选）

**返回:**
- (int): 文档 ID

##### document_exists

```python
exists = manager.document_exists(file_path: str, content_hash: str) -> bool
```

检查文档是否已存在。

##### add_dataset_item

```python
item_id = manager.add_dataset_item(
    dataset_name: str,
    instruction: str,
    input_: Optional[str] = None,
    output: Optional[str] = None,
    document_id: Optional[int] = None,
    chunk_index: int = 0,
    source_file: Optional[str] = None,
    extra_data: Optional[Dict] = None
) -> int
```

添加数据集条目。

##### get_dataset_items

```python
items = manager.get_dataset_items(
    dataset_name: str,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[DatasetItem]
```

获取数据集条目。

##### export_dataset

```python
data = manager.export_dataset(dataset_name: str) -> List[Dict[str, Any]]
```

导出数据集为 JSON 格式。

##### save_to_jsonl

```python
count = manager.save_to_jsonl(dataset_name: str, output_path: str) -> int
```

保存数据集为 JSONL 格式。

##### get_dataset_stats

```python
stats = manager.get_dataset_stats(dataset_name: str) -> Dict[str, int]
```

获取数据集统计信息。

##### clear_dataset

```python
manager.clear_dataset(dataset_name: str)
```

清空数据集。

---

## LLM 模块

### LLMClient

```python
from src.llm import LLMClient

client = LLMClient(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None
)
```

LLM 客户端。

#### 方法

##### chat

```python
response = client.chat(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str
```

发送对话请求。

**参数:**
- `messages` (List): 消息列表
- `temperature` (float): 温度（可选）
- `max_tokens` (int): 最大 token 数（可选）

**返回:**
- (str): 模型响应

##### generate_qa_pairs

```python
qa_pairs = client.generate_qa_pairs(
    text: str,
    num_pairs: int = 5,
    language: str = "zh"
) -> List[Dict[str, str]]
```

生成 QA 对。

##### generate_summarization

```python
summary = client.generate_summarization(
    text: str,
    language: str = "zh"
) -> str
```

生成文本摘要。

##### generate_conversation

```python
conversation = client.generate_conversation(
    text: str,
    num_turns: int = 3,
    language: str = "zh"
) -> List[Dict[str, str]]
```

生成对话数据。

##### batch_generate_qa

```python
all_pairs = client.batch_generate_qa(
    texts: List[str],
    num_pairs_per_text: int = 5,
    progress: bool = True
) -> List[Dict[str, str]]
```

批量生成 QA 对。

### CacheManager

```python
from src.llm import CacheManager

cache = CacheManager(
    cache_dir: str = "./data/cache",
    max_size: int = 1000,
    max_age: int = 86400
)
```

LLM 响应缓存管理器。

#### 方法

##### get

```python
cached = cache.get(text: str, **kwargs) -> Optional[str]
```

获取缓存。

##### set

```python
cache.set(text: str, response: str, **kwargs)
```

设置缓存。

##### clear

```python
cache.clear()
```

清空所有缓存。

##### get_stats

```python
stats = cache.get_stats() -> Dict[str, Any]
```

获取缓存统计信息。

---

## 训练模块

### train_lora

```python
from src.trainer import train_lora

lora_path = train_lora(
    model_name: str,
    data_path: str,
    output_dir: str,
    lora_config: Optional[Dict] = None,
    batch_size: int = 4,
    learning_rate: float = 0.0002,
    epochs: int = 3,
    max_length: int = 2048,
    resume_from: Optional[str] = None
) -> str
```

训练 LoRA 模型。

### merge_model

```python
from src.trainer import merge_model

merge_model(
    base_model_path: str,
    lora_model_path: str,
    output_path: str
) -> str
```

合并模型。

### prepare_training_data

```python
from src.trainer import prepare_training_data

output_path = prepare_training_data(
    dataset_path: str,
    output_path: str,
    chat_template: bool = True
) -> str
```

准备训练数据。

### check_gpu_available

```python
from src.trainer import check_gpu_available

gpu_info = check_gpu_available() -> Dict[str, bool]
```

检查 GPU 可用性。

### get_device_map

```python
from src.trainer import get_device_map

device = get_device_map() -> str
```

获取最佳设备映射。

---

## 解析器模块

### ParserManager

```python
from src.parser import ParserManager

manager = ParserManager()
```

解析器管理器。

#### 方法

##### parse_file

```python
paragraphs = manager.parse_file(file_path: str) -> List[str]
```

解析单个文件。

##### parse_directory

```python
documents = manager.parse_directory(
    dir_path: str,
    recursive: bool = True
) -> Dict[str, List[str]]
```

解析整个目录。

##### get_supported_extensions

```python
extensions = manager.get_supported_extensions() -> List[str]
```

获取支持的文件扩展名。

---

## 异常类

### LLMError

```python
from src.llm import LLMError

raise LLMError("LLM 调用错误")
```

LLM 调用错误基类。

### QAGenerationError

```python
from src.llm import QAGenerationError

raise QAGenerationError("QA 生成失败")
```

QA 对生成错误。

### JSONParseError

```python
from src.llm import JSONParseError

raise JSONParseError("JSON 解析失败")
```

JSON 解析错误。

### CacheError

```python
from src.llm import CacheError

raise CacheError("缓存操作失败")
```

缓存操作错误。

---

## 配置模型

### LLMConfig

```python
from src.config import LLMConfig

config = LLMConfig(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 2000
)
```

### DatabaseConfig

```python
from src.config import DatabaseConfig

config = DatabaseConfig(
    type: str = "sqlite",
    path: str = "./data/datasets.db",
    host: Optional[str] = None,
    port: int = 3306,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None
)
```

### DatasetConfig

```python
from src.config import DatasetConfig

config = DatasetConfig(
    input_dir: str = "./documents",
    output_dir: str = "./data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
)
```

### TrainingConfig

```python
from src.config import TrainingConfig

config = TrainingConfig(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    batch_size: int = 4,
    learning_rate: float = 0.0002,
    epochs: int = 3,
    max_length: int = 2048
)
```

### LoRAConfig

```python
from src.config import LoRAConfig

config = LoRAConfig(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

---

*最后更新: 2026-02-02*
