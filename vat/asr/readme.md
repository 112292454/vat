# VAT 模块文档：ASR 阶段组（WHISPER + SPLIT）

> **重构说明**：ASR 现在是一个**阶段组**，包含两个独立可执行的细粒度阶段：
> - `WHISPER`：Whisper 模型推理，生成原始转录
> - `SPLIT`：LLM 智能断句，生成语义完整句子

---

## 1. 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ASR 阶段组 (asr)                                   │
│                                                                              │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │     WHISPER 阶段                │   │      SPLIT 阶段                 │  │
│  │     (TaskStep.WHISPER)          │   │      (TaskStep.SPLIT)           │  │
│  │                                 │   │                                 │  │
│  │  ┌─────────────────────────┐   │   │   ┌─────────────────────────┐   │  │
│  │  │ 1. 查找视频文件          │   │   │   │ 1. 加载 original_raw.srt │   │  │
│  │  │    _find_video_file()   │   │   │   │                         │   │  │
│  │  └───────────┬─────────────┘   │   │   └───────────┬─────────────┘   │  │
│  │              ▼                  │   │               ▼                  │  │
│  │  ┌─────────────────────────┐   │   │   ┌─────────────────────────┐   │  │
│  │  │ 2. 提取音频 (ffmpeg)    │   │   │   │ 2. 检查是否启用断句     │   │  │
│  │  │    → <video>.wav        │   │   │   │    split.enable?        │   │  │
│  │  └───────────┬─────────────┘   │   │   └───────────┬─────────────┘   │  │
│  │              ▼                  │   │               ▼                  │  │
│  │  ┌─────────────────────────┐   │   │   ┌─────────────────────────┐   │  │
│  │  │ 3. 检查缓存/配置快照    │   │   │   │ 3. 检查缓存/配置快照    │   │  │
│  │  │    _should_use_cache()  │   │   │   │    _should_use_cache()  │   │  │
│  │  └───────────┬─────────────┘   │   │   └───────────┬─────────────┘   │  │
│  │              ▼                  │   │               ▼                  │  │
│  │  ┌─────────────────────────┐   │   │   ┌─────────────────────────┐   │  │
│  │  │ 4. Whisper 模型推理     │   │   │   │ 4. LLM 断句             │   │  │
│  │  │    (GPU, faster-whisper)│   │   │   │    split_by_llm() 或    │   │  │
│  │  │    长音频→ChunkedASR    │   │   │   │    ChunkedSplitter      │   │  │
│  │  └───────────┬─────────────┘   │   │   └───────────┬─────────────┘   │  │
│  │              ▼                  │   │               ▼                  │  │
│  │  ┌─────────────────────────┐   │   │   ┌─────────────────────────┐   │  │
│  │  │ 5. 保存 original_raw.srt│──────────▶│ 5. 时间戳优化            │   │  │
│  │  │    更新 cache_metadata  │   │   │   │    optimize_timing()    │   │  │
│  │  └─────────────────────────┘   │   │   └───────────┬─────────────┘   │  │
│  │                                 │   │               ▼                  │  │
│  │                                 │   │   ┌─────────────────────────┐   │  │
│  │                                 │   │   │ 6. 保存最终输出         │   │  │
│  │                                 │   │   │    original.srt         │   │  │
│  │                                 │   │   │    original.json        │   │  │
│  │                                 │   │   └─────────────────────────┘   │  │
│  └─────────────────────────────────┘   └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段定义与依赖关系

### 2.1 阶段组成

| 阶段名 | TaskStep 枚举 | 职责 | 依赖 | 输出 |
|--------|---------------|------|------|------|
| **WHISPER** | `TaskStep.WHISPER` | Whisper 模型推理 | `DOWNLOAD` | `original_raw.srt` |
| **SPLIT** | `TaskStep.SPLIT` | LLM 智能断句 | `WHISPER` | `original.srt` |

### 2.2 阶段组定义（`vat/models.py`）

```python
STAGE_GROUPS = {
    "asr": [TaskStep.WHISPER, TaskStep.SPLIT],
}

STAGE_DEPENDENCIES = {
    TaskStep.WHISPER: [TaskStep.DOWNLOAD],
    TaskStep.SPLIT: [TaskStep.WHISPER],
}
```

### 2.3 独立执行能力

重构后，每个阶段可**独立执行**：

```bash
# 只执行 Whisper（不断句）
vat pipeline --steps whisper <video_id>

# 只重跑断句（Whisper 结果已存在）
vat pipeline --steps split --force <video_id>

# 执行完整 ASR（Whisper + Split）
vat asr <video_id>
# 等价于
vat pipeline --steps whisper,split <video_id>
```

**这意味着**：
- 如果你只想调整断句参数，无需重跑 Whisper
- 如果 Whisper 结果已满意，可跳过直接断句

---

## 3. 调用链详解

### 3.1 从 CLI 到核心函数

```
CLI: vat asr -v <video_id>
    │
    ▼
commands.py: asr()
    │ 展开阶段组 "asr" → [WHISPER, SPLIT]
    ▼
scheduler.py: schedule_videos(steps=[WHISPER, SPLIT])
    │
    ▼
executor.py: VideoProcessor.process(steps=[WHISPER, SPLIT])
    │
    ├──▶ _execute_step(TaskStep.WHISPER)
    │        └──▶ _run_whisper()
    │                 ├── _find_video_file()
    │                 ├── _extract_whisper_config()
    │                 ├── _should_use_cache('whisper', ...)
    │                 └── self.asr.asr_video(...)
    │
    └──▶ _execute_step(TaskStep.SPLIT)
             └──▶ _run_split()
                      ├── ASRData.from_subtitle_file(original_raw.srt)
                      ├── _extract_split_config()
                      ├── _should_use_cache('split', ...)
                      └── split_by_llm() 或 ChunkedSplitter.split()
```

### 3.2 关键代码索引

| 组件 | 文件位置 | 函数/类 |
|------|----------|---------|
| CLI 入口 | `vat/cli/commands.py` | `asr()` |
| 阶段编排 | `vat/pipeline/executor.py` | `VideoProcessor.process()` |
| **WHISPER 阶段** | `vat/pipeline/executor.py` | `_run_whisper()` |
| **SPLIT 阶段** | `vat/pipeline/executor.py` | `_run_split()` |
| Whisper 转录器 | `vat/asr/whisper_wrapper.py` | `WhisperASR` |
| 音频提取 | `vat/asr/whisper_wrapper.py` | `_extract_audio()` |
| 分块 ASR | `vat/asr/chunked_asr.py` | `ChunkedASR` |
| Chunk 合并 | `vat/asr/chunk_merger.py` | `ChunkMerger` |
| LLM 断句 | `vat/asr/split.py` | `split_by_llm()` |
| 分块断句 | `vat/asr/chunked_split.py` | `ChunkedSplitter` |
| 缓存判定 | `vat/pipeline/executor.py` | `_should_use_cache()` |

---

## 4. WHISPER 阶段详解

### 4.1 内部流程

```
_run_whisper()
    │
    ├─ 1. _find_video_file()
    │      查找视频文件（优先 original.*，否则扫描目录）
    │
    ├─ 2. 定义路径
    │      audio_file = <video>.wav
    │      raw_srt = original_raw.srt
    │
    ├─ 3. 加载/创建 CacheMetadata
    │
    ├─ 4. _extract_whisper_config() + _should_use_cache()
    │      │
    │      ├─ 命中缓存 → 直接加载 original_raw.srt
    │      │
    │      └─ 未命中 → 执行 Whisper
    │                    │
    │                    ├─ _extract_audio() (ffmpeg)
    │                    │
    │                    ├─ WhisperASR.asr_video()
    │                    │      │
    │                    │      ├─ 短音频 → 直接转录
    │                    │      │
    │                    │      └─ 长音频 (>5min) → ChunkedASR
    │                    │              ├─ 切块 (pydub)
    │                    │              ├─ 并发转录 (ThreadPoolExecutor)
    │                    │              └─ 合并 (ChunkMerger)
    │                    │
    │                    └─ 保存 original_raw.srt
    │
    └─ 5. 更新 cache_metadata，返回
```

### 4.2 关键决策点

#### 4.2.1 是否启用 ChunkedASR？

```python
# 条件：config.asr.enable_chunked == True 且 音频时长 > 300s
if self.enable_chunked and duration_seconds > 300:
    return self._asr_with_chunking(audio_input, ...)
```

**影响**：
- ChunkedASR 会把音频切成多块并发处理
- 每块导出为 **mp3**（有损），可能影响极端情况下的识别质量
- 合并时用 `ChunkMerger` 做重叠区域对齐

#### 4.2.2 ffmpeg 音频提取命令

```bash
ffmpeg -i <video> -vn -acodec pcm_s16le -ac 1 -ar 16000 -y <output.wav>
```

- 固定 16kHz/mono/PCM，Whisper 要求
- 失败会抛 `RuntimeError("音频提取失败")`

### 4.3 易混淆/易出错点

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| "改了 beam_size 但结果不变" | `beam_size` 不在缓存 key 中 | 使用 `-f` 强制重跑，或删除 `original_raw.srt` |
| "ASR 输出为空" | 视频无语音 / language 不匹配 / vad_filter 过度过滤 | 检查配置，尝试禁用 vad_filter |
| "某些片段被过滤" | 代码过滤了特定文本（如 `作詞`、`編曲`） | 查看 `_asr_with_faster_whisper()` 的过滤逻辑 |

---

## 5. SPLIT 阶段详解

### 5.1 内部流程

```
_run_split()
    │
    ├─ 1. 检查 original_raw.srt 是否存在
    │      不存在 → 抛出 ASRError
    │
    ├─ 2. 加载 ASRData
    │      asr_data = ASRData.from_subtitle_file(original_raw.srt)
    │
    ├─ 3. 检查 split.enable
    │      │
    │      └─ 禁用 → 直接复制到 original.srt，返回
    │
    ├─ 4. _extract_split_config() + _should_use_cache()
    │      │
    │      ├─ 命中缓存 → 加载 original_split.srt
    │      │
    │      └─ 未命中 → 执行断句
    │                    │
    │                    ├─ 检查 LLM 可用性
    │                    │      不可用 → 警告并跳过断句
    │                    │
    │                    ├─ 获取场景 prompt (如有)
    │                    │
    │                    ├─ 选择断句策略：
    │                    │      │
    │                    │      ├─ 启用分块 && 片段数 >= threshold
    │                    │      │      → ChunkedSplitter.split()
    │                    │      │
    │                    │      └─ 否则
    │                    │             → _split_with_speaker_awareness()
    │                    │             → split_by_llm()
    │                    │
    │                    └─ optimize_timing() (修复时间戳重叠)
    │
    ├─ 5. 保存 original_split.srt
    │
    ├─ 6. 保存最终输出
    │      original.srt (供后续阶段使用)
    │      original.json (调试用)
    │
    └─ 7. 更新 cache_metadata，返回
```

### 5.2 关键决策点

#### 5.2.1 全文断句 vs 分块断句

```python
if (config.asr.split.enable_chunking and 
    len(asr_data.segments) >= config.asr.split.chunk_min_threshold):
    # 分块断句：适用于长视频
    splitter = ChunkedSplitter(...)
    asr_data = splitter.split(asr_data)
else:
    # 全文断句：适用于短视频
    asr_data = split_by_llm(asr_data, ...)
```

**分块断句的特点**：
- 把 segments 切成多块，每块独立调用 LLM
- overlap 区域：丢弃前块最后一句，用后块第一句替换
- 适用于长视频（>30 个 segments）

#### 5.2.2 时间戳优化（重要）

断句后会调用 `optimize_timing()`：
- 用字符级时间插值把新句子"分摊"到原时间轴
- 修复因断句导致的时间戳重叠问题

### 5.3 易混淆/易出错点

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| "断句秒出结果" | `call_llm()` 的 diskcache 命中 | 使用 `-f` 强制重跑 |
| "断句后时间戳重叠" | `optimize_timing()` 未能完全修复 | 检查 `chunk_overlap_sentences` 设置 |
| "LLM 配置不完整，跳过断句" | `llm.api_key` 或 `llm.base_url` 未设置 | 配置 `config.llm` |
| "改了 min_words_cjk 但结果不变" | 该参数不在缓存 key 中 | 使用 `-f` 强制重跑 |

---

## 6. 缓存机制详解（非常重要）

### 6.1 缓存层次

```
┌─────────────────────────────────────────────────────────────────┐
│                        缓存层次                                  │
├─────────────────────────────────────────────────────────────────┤
│ 层次 1: 步骤级缓存 (DB)                                          │
│         - 检查 tasks 表中该阶段是否已完成                         │
│         - 未传 -f 且已完成 → 整个阶段被跳过                       │
├─────────────────────────────────────────────────────────────────┤
│ 层次 2: 文件级缓存 (.cache_metadata.json)                        │
│         - 检查输出文件是否存在                                    │
│         - 检查配置快照是否一致                                    │
│         - 决定是否复用 original_raw.srt / original_split.srt     │
├─────────────────────────────────────────────────────────────────┤
│ 层次 3: LLM 调用缓存 (diskcache)                                 │
│         - call_llm() 的 memoize 缓存                             │
│         - 默认 1 小时 TTL                                        │
│         - 仅 SPLIT 阶段涉及                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 缓存 key 覆盖范围（易混淆）

**Whisper 配置快照只包含**：
- `model`, `language`, `compute_type`, `vad_filter`
- `enable_chunked`, `chunk_length_sec`, `backend`

**不包含**（改了不会触发缓存失效）：
- `beam_size`, `word_timestamps`, `condition_on_previous_text`
- `chunk_overlap_sec`, `chunk_concurrency`

**Split 配置快照只包含**：
- `enable`, `mode`, `max_words_cjk`, `max_words_english`, `model`

**不包含**：
- `min_words_*`, `enable_chunking`, `chunk_size_sentences`
- `chunk_overlap_sentences`, `chunk_min_threshold`

### 6.3 force 语义

| 标志 | 效果 |
|------|------|
| `-f` (CLI) | 跳过 DB 完成检查，强制执行阶段 |
| 文件级 | 跳过 `.cache_metadata.json` 检查 |
| LLM 缓存 | **不影响**（当前 ASR 阶段不禁用 LLM 缓存） |

---

## 7. GPU 配置与设备选择

### 7.1 配置方式

```yaml
# config/default.yaml
gpu:
  device: "auto"              # "auto" | "cuda:N" | "cpu"
  allow_cpu_fallback: false   # 是否允许 CPU 回退
  min_free_memory_mb: 2000    # 自动选择时的最小空闲显存

asr:
  device: "cuda"              # 已废弃，使用 gpu.device
```

### 7.2 GPU 选择逻辑

```python
# 新的 GPU 选择逻辑 (vat/utils/gpu.py)
if device == "auto":
    # 自动选择显存占用最低的 GPU
    gpu_id = select_best_gpu()
elif device.startswith("cuda:"):
    # 使用指定 GPU
    gpu_id = int(device.split(":")[1])
else:
    # CPU（不推荐）
    if not allow_cpu_fallback:
        raise RuntimeError("GPU 原则：禁止静默回退 CPU")
```

### 7.3 GPU 原则（项目强制）

- 默认运行在 GPU 服务器，**必须使用 GPU**
- 禁止"静默回退 CPU"
- 如果看到 `device=cpu`，应视为配置错误

---

## 8. 常见问题与 Debug Checklist

### 8.1 问题速查表

| 症状 | 可能原因 | 排查步骤 |
|------|----------|----------|
| 阶段被跳过 | DB 标记已完成 | 使用 `-f` 或检查 tasks 表 |
| 结果不变 | 缓存命中 | 检查配置是否在 key 中，使用 `-f` |
| ASR 输出为空 | 无语音/language 错误 | 检查视频内容和 `asr.language` |
| 断句被跳过 | LLM 配置不完整 | 检查 `llm.api_key` 和 `llm.base_url` |
| 时间戳重叠 | 分块断句 overlap 问题 | 调整 `chunk_overlap_sentences` |

### 8.2 Debug Checklist

1. **确认视频文件可被找到**
   - `_find_video_file()` 是否定位到期望文件
   - 优先 `original.*`，否则扫描目录

2. **确认输出目录文件状态**
   - `original_raw.srt` / `original_split.srt` / `original.srt` 是否存在
   - 文件大小是否为 0

3. **确认缓存判定**
   - `.cache_metadata.json` 中的配置快照
   - 你改动的参数是否在缓存 key 中

4. **确认 LLM 可用性**
   - `config.llm.is_available()` 是否返回 True
   - 环境变量 `VAT_LLM_APIKEY` 是否设置

5. **确认 ChunkedASR/ChunkedSplitter 触发条件**
   - 长音频 (>5min) 会触发 ChunkedASR
   - 多片段 (>=30) 会触发 ChunkedSplitter

---

## 9. 配置参考（config/default.yaml）

```yaml
gpu:
  device: "auto"
  allow_cpu_fallback: false
  min_free_memory_mb: 2000

asr:
  backend: "faster-whisper"
  model: "large-v3"
  language: "ja"
  compute_type: "float32"
  vad_filter: false
  beam_size: 7
  word_timestamps: true
  condition_on_previous_text: false

  # 长音频分块
  enable_chunked: true
  chunk_length_sec: 600
  chunk_overlap_sec: 10
  chunk_concurrency: 3

  # 智能断句
  split:
    enable: true
    mode: "sentence"
    max_words_cjk: 48
    max_words_english: 24
    min_words_cjk: 6
    min_words_english: 1
    model: "gpt-4o-mini"

    # 分块断句
    enable_chunking: true
    chunk_size_sentences: 50
    chunk_overlap_sentences: 1
    chunk_min_threshold: 30

llm:
  api_key: "${VAT_LLM_APIKEY}"
  base_url: "https://api.videocaptioner.cn"
```

---

## 10. 后处理模块（新增）

> **来源**：借鉴自 WhisperJAV 项目，针对 VTB 直播场景优化

### 10.1 功能概述

后处理模块（`vat/asr/postprocessing.py`）提供以下功能：

| 功能 | 说明 | 配置项 |
|------|------|--------|
| **幻觉检测** | 移除 Whisper 常见幻觉输出（如 "www"、"ご視聴ありがとう"） | `postprocessing.enable_hallucination_detection` |
| **重复清理** | 清理异常重复（如 "うううう" → "うう"） | `postprocessing.enable_repetition_cleaning` |
| **日语处理** | 标点标准化、相槌保护等 | `postprocessing.enable_japanese_processing` |

### 10.2 使用方法

```python
from vat.asr import ASRPostProcessor, postprocess_asr_text, is_hallucination

# 便捷函数
text = postprocess_asr_text("ほいほいほいほい")  # → "ほいほい"
is_hall = is_hallucination("www")  # → True

# 完整处理器
processor = ASRPostProcessor()
result = processor.process_text("あああああ")
print(result.processed_text)  # → "ああ"

# 批量处理字幕段
segments = [{'text': 'xxx', 'start': 0, 'end': 1}, ...]
processed, stats = processor.process_segments(segments)
```

### 10.3 配置选项

```yaml
asr:
  postprocessing:
    enable_hallucination_detection: true  # 启用幻觉检测
    enable_repetition_cleaning: true      # 启用重复清理
    enable_japanese_processing: true      # 启用日语处理
    min_confidence: 0.8                   # 幻觉检测最小置信度
    custom_blacklist: []                  # 自定义幻觉黑名单
```

---

## 11. 人声分离模块（新增）

> **来源**：集成 Mel-Band-Roformer 模型，适用于游戏/歌回直播场景

### 11.1 功能概述

人声分离模块（`vat/asr/vocal_separation.py`）使用 Mel-Band-Roformer 模型分离人声和背景音乐。

**适用场景**：
- 游戏直播（有 BGM 干扰）
- 歌回直播（背景音乐较强）
- 任何需要提取纯净人声的场景

### 11.2 使用方法

```python
from vat.asr import separate_vocals, is_vocal_separation_available

# 检查是否可用
if is_vocal_separation_available():
    result = separate_vocals("input.wav", output_dir="./output")
    if result.success:
        print(f"人声已保存: {result.vocals_path}")
```

### 11.3 配置选项

```yaml
asr:
  vocal_separation:
    enable: false                 # 是否启用（默认关闭）
    auto_detect_bgm: true         # 自动检测是否需要分离
    model_path: ""                # 模型路径（空=默认）
    save_accompaniment: false     # 是否保存伴奏
```

### 11.4 模型下载

需要手动下载模型权重文件：
```bash
# 下载到 ref/Mel-Band-Roformer-Vocal-Model/models/model.ckpt
# 模型来源: https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model
```

---

## 12. 修改指南：如果你想改某个功能...

| 如果你想... | 应该看/改哪里 |
|-------------|---------------|
| 改 Whisper 模型参数 | `config.asr.*` + `vat/asr/whisper_wrapper.py` |
| 改断句 prompt | `vat/llm/prompts/split/*.md` |
| 改断句长度限制 | `config.asr.split.max_words_*` |
| 改缓存 key 包含的字段 | `vat/utils/cache_metadata.py` `WHISPER_KEY_CONFIGS` |
| 改分块 ASR 切块逻辑 | `vat/asr/chunked_asr.py` `ChunkedASR` |
| 改分块断句合并策略 | `vat/asr/chunked_split.py` `ChunkedSplitter` |
| 改时间戳优化算法 | `vat/asr/asr_data.py` `optimize_timing()` |
| 添加新的 ASR 后端 | 继承 `BaseASR`，参考 `WhisperASR` |
| 改幻觉检测规则 | `vat/asr/postprocessing.py` `JAPANESE_HALLUCINATION_EXACT` |
| 改重复清理模式 | `vat/asr/postprocessing.py` `REPETITION_PATTERNS` |
| 改人声分离参数 | `vat/asr/vocal_separation.py` `VocalSeparator` |

---

## 13. 已修复的历史问题

以下问题已修复，记录于此供开发者参考。

### 13.1 时间戳对齐算法（chunked_split.py `_realign_timestamps`）

旧算法使用原始 ASR segment 的整体边界时间分配给断句后的段落，导致同一 segment 内拆分出的多个段落时间完全重叠。已改为基于 `difflib.SequenceMatcher` 的 diff 容错对齐：匹配字符直接取原始字符级时间戳，小范围增删改（≤3 字符）自动容错。安全检查确保段落时间不超出其字符所属原始 segment 的范围。

### 13.2 分块合并文本丢失（chunked_split.py `_trim_overlap`）

旧逻辑在合并相邻 chunk 时按固定段数跳过 overlap 部分，但 LLM 可能将 overlap segment 的文本与后续文本合并为一个段落，导致跳过时连带丢失后续文本。已改为基于字符计数的精确 overlap 移除，支持跨边界段落拆分。

### 13.3 时间戳倒挂（`_realign_timestamps` + `optimize_timing`）

`_realign_timestamps` 按行分组生成段落时，相邻行可能映射到同一原始 ASR segment，产出重叠段（`seg[i].end_time > seg[i+1].start_time`）。`optimize_timing` 对重叠输入计算 `mid_time` 时可能超过后段 `end_time`，导致 `start_time >= end_time`。修复方式：在 `_realign_timestamps` 生成段落后增加重叠消除步骤（裁剪前段 end_time 至后段 start_time）；`optimize_timing` 保留防御性检查（公式结果越界时跳过该对）。
