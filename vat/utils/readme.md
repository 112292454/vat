# VAT 模块文档：Utils（通用工具）

> 提供项目全局共用的基础工具：GPU 管理、日志、缓存、文本处理、文件操作、输出验证等。

---

## 1. 模块组成

| 文件 | 职责 |
|------|------|
| `gpu.py` | GPU 信息获取、自动选择、device 管理 |
| `logger.py` | 统一日志系统（ContextVars video_id 注入） |
| `cache.py` | 基于 diskcache 的 API 结果缓存 |
| `cache_metadata.py` | 子步骤配置快照与缓存失效检测 |
| `output_validator.py` | ASR/LLM 输出崩溃模式检测 |
| `text_utils.py` | 多语言文本处理（CJK 判断、字数统计） |
| `file_ops.py` | 文件操作（处理产物删除，保护原始文件） |
| `resource_lock.py` | 跨进程资源锁（youtube_download / bilibili_upload 速率协调） |

---

## 2. GPU 管理（gpu.py）

遵循项目 GPU 原则：**默认必须使用 GPU，禁止静默回退 CPU**。

### 核心函数

| 函数 | 说明 |
|------|------|
| `get_available_gpus()` | 通过 nvidia-smi 获取所有 GPU 信息（显存、利用率） |
| `select_best_gpu()` | 自动选择显存最空闲的 GPU |
| `resolve_gpu_device(device)` | 解析 device 参数：`"auto"` → 自动选择，`"cpu"` → CPU，`"0"` → 指定 GPU |
| `is_cuda_available()` | 检查 CUDA 是否可用 |
| `set_cuda_visible_devices(gpus)` | 设置 `CUDA_VISIBLE_DEVICES` 环境变量 |
| `get_gpu_for_subprocess()` | 为子进程分配 GPU（多 GPU 调度使用） |
| `log_gpu_status()` | 打印所有 GPU 状态 |

### GPUInfo 数据结构

```python
@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int
```

---

## 3. 日志系统（logger.py）

### 设计

- 使用 Python `logging` + `ContextVars` 管理 video_id 上下文
- 每个线程有独立的 video_id（线程安全）
- 日志格式：`时间 | 级别 | [video_id] | 模块名 | 消息`

### 使用

```python
from vat.utils.logger import setup_logger, set_video_id

logger = setup_logger("my_module")
set_video_id("abc123")  # 后续该线程的日志自动带 [abc123] 前缀
logger.info("处理开始")
# 输出: 2026-03-02 10:00:00 | INFO     | [abc123] | my_module | 处理开始
```

---

## 4. 缓存系统（cache.py）

### 设计

- 基于 diskcache（SQLite 后端），通过 `config.storage.cache_enabled` 控制开关
- **默认关闭**：零 SQLite 依赖，零性能开销
- 启用时：持久化缓存 LLM/翻译结果，重跑同一视频可复用

### 缓存实例

| 实例 | 用途 |
|------|------|
| `_llm_cache` | LLM API 调用结果 |
| `_asr_cache` | ASR 结果 |
| `_translate_cache` | 翻译结果 |

### 核心函数

| 函数 | 说明 |
|------|------|
| `init_caches(cache_dir, enabled)` | 初始化缓存系统（幂等+线程安全） |
| `get_llm_cache()` | 获取 LLM 缓存实例 |
| `memoize(cache, expire, typed)` | 装饰器：自动缓存函数返回值 |
| `generate_cache_key(...)` | 生成缓存键 |

---

## 5. 缓存元数据（cache_metadata.py）

追踪每个处理子步骤的配置快照，用于判断缓存是否失效。

- 每个视频输出目录有 `.cache_metadata.json`
- 记录 whisper/split/optimize 各步骤的关键配置和完成时间
- 配置变更时自动失效，强制重新处理

### 关键配置定义

| 阶段 | 监控的配置项 |
|------|-------------|
| whisper | model, language, compute_type, vad_filter, enable_chunked, chunk_length_sec, backend |
| split | enable, mode, max_words_cjk, max_words_english, model |
| optimize | enable, custom_prompt, model, batch_size, thread_num |

---

## 6. 输出验证（output_validator.py）

检测 ASR/LLM 输出中的崩溃模式：

| 检测项 | 说明 | 阈值 |
|--------|------|------|
| 单字符重复 | 如 `あああああああああああ` | 警告 ≥10，灾难 ≥20 |
| 短模式重复 | 如 `チャカナカマカ` 循环 | 警告 ≥6，灾难 ≥12 |
| 独特字符比 | `unique_chars / len` 过低 | < 0.05 |
| 异常长片段 | 单条字幕 > 30 秒且文本异常 | 结合上下文判断 |

验证结果 `ValidationResult` 包含 `is_valid`、`is_catastrophic`（灾难性→应丢弃）、`warnings` 列表。

---

## 7. 文本工具（text_utils.py）

| 函数 | 说明 |
|------|------|
| `is_mainly_cjk(text, threshold)` | 判断文本是否主要为 CJK / 亚洲语言（阈值 50%） |
| `count_words(text)` | 混合语言字数统计：CJK 按字符计 + 拉丁系按空格分词 |
| `is_pure_punctuation(text)` | 判断文本是否仅含标点 |

用于 Split 阶段的断句长度控制（CJK 和英文使用不同的最大字数阈值）。

---

## 8. 文件操作（file_ops.py）

| 函数 | 说明 |
|------|------|
| `is_processed_file(path)` | 判断文件是否为处理产物（非原始下载） |
| `delete_processed_files(dir)` | 删除处理产物，保留原始下载文件 |

安全策略：只有明确识别为处理产物的才删除，未知文件一律保留。

已知处理产物：`original_raw.srt`、`original.srt`、`optimized.srt`、`translated.srt`、`final.mp4`、`.ass`、`.wav` 等。

---

## 9. 资源锁（resource_lock.py）

基于 SQLite 的跨进程资源锁，用于协调多个 VAT 进程（watch 自动提交 + 用户手动执行）间的下载和上传速率。当前真实接入点在 `YouTubeDownloader.download()` 与 `BilibiliUploader` 的真实上传方法中。详见 `docs/WATCH_MODE_SPEC.md` §3.4。

### 资源类型

| 资源 | 最大并发 | 最小间隔 | 锁超时 |
|------|----------|----------|--------|
| `youtube_download` | 1 | 配置项 `download_delay` | 30 分钟 |
| `bilibili_upload` | 1 | 配置项 `upload_interval` | 60 分钟 |

### 使用方式

```python
from vat.utils.resource_lock import resource_lock

with resource_lock(db_path, 'youtube_download', timeout_seconds=300):
    do_download()
# 自动释放 + 冷却间隔控制
```

### 健壮性

- **PID 存活检测**：处理持有者崩溃场景
- **心跳机制**：每 30s 更新 `last_activity_at`，检测僵死进程
- **过期自动释放**：防止死锁
- **冷却间隔**：释放后强制等待最小间隔，避免频繁请求
