# VAT 模块文档：翻译阶段组（OPTIMIZE + TRANSLATE）

> **重构说明**：翻译现在是一个**阶段组**，包含两个独立可执行的细粒度阶段：
> - `OPTIMIZE`：字幕优化（错别字修正、术语统一）
> - `TRANSLATE`：LLM 翻译（多语言翻译、反思翻译）

---

## 1. 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         翻译阶段组 (translate)                               │
│                                                                              │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │     OPTIMIZE 阶段               │   │      TRANSLATE 阶段             │  │
│  │     (TaskStep.OPTIMIZE)         │   │      (TaskStep.TRANSLATE)       │  │
│  │                                 │   │                                 │  │
│  │  ┌─────────────────────────┐   │   │   ┌─────────────────────────┐   │  │
│  │  │ 1. 加载 original.srt    │   │   │   │ 1. 加载输入字幕         │   │  │
│  │  │                         │   │   │   │    优先 optimized.srt   │   │  │
│  │  └───────────┬─────────────┘   │   │   │    否则 original.srt    │   │  │
│  │              ▼                  │   │   └───────────┬─────────────┘   │  │
│  │  ┌─────────────────────────┐   │   │               ▼                  │  │
│  │  │ 2. 检查 optimize.enable │   │   │   ┌─────────────────────────┐   │  │
│  │  │    禁用→复制到输出       │   │   │   │ 2. 检查 skip_translate  │   │  │
│  │  └───────────┬─────────────┘   │   │   │    启用→复制到输出       │   │  │
│  │              ▼                  │   │   └───────────┬─────────────┘   │  │
│  │  ┌─────────────────────────┐   │   │               ▼                  │  │
│  │  │ 3. 禁用缓存 (if force)  │   │   │   ┌─────────────────────────┐   │  │
│  │  │    disable_cache()      │   │   │   │ 3. 禁用缓存 (if force)  │   │  │
│  │  └───────────┬─────────────┘   │   │   │    disable_cache()      │   │  │
│  │              ▼                  │   │   └───────────┬─────────────┘   │  │
│  │  ┌─────────────────────────┐   │   │               ▼                  │  │
│  │  │ 4. 创建 LLMTranslator   │   │   │   ┌─────────────────────────┐   │  │
│  │  │    (优化专用配置)        │   │   │   │ 4. 创建 LLMTranslator   │   │  │
│  │  └───────────┬─────────────┘   │   │   │    (翻译配置)            │   │  │
│  │              ▼                  │   │   └───────────┬─────────────┘   │  │
│  │  ┌─────────────────────────┐   │   │               ▼                  │  │
│  │  │ 5. 获取场景 prompt      │   │   │   ┌─────────────────────────┐   │  │
│  │  │    scene_prompts.optimize│   │   │   │ 5. 获取场景 prompt      │   │  │
│  │  └───────────┬─────────────┘   │   │   │    scene_prompts.translate│  │  │
│  │              ▼                  │   │   └───────────┬─────────────┘   │  │
│  │  ┌─────────────────────────┐   │   │               ▼                  │  │
│  │  │ 6. LLM 优化             │   │   │   ┌─────────────────────────┐   │  │
│  │  │    optimize_subtitles() │──────────▶│ 6. LLM 翻译             │   │  │
│  │  │    - 分 chunk 处理      │   │   │   │    translate()          │   │  │
│  │  │    - Agent Loop 验证    │   │   │   │    - 分 chunk 处理      │   │  │
│  │  └───────────┬─────────────┘   │   │   │    - 上下文传递         │   │  │
│  │              ▼                  │   │   │    - 反思翻译 (可选)    │   │  │
│  │  ┌─────────────────────────┐   │   │   └───────────┬─────────────┘   │  │
│  │  │ 7. 保存 optimized.srt   │   │   │               ▼                  │  │
│  │  │    恢复缓存             │   │   │   ┌─────────────────────────┐   │  │
│  │  └─────────────────────────┘   │   │   │ 7. 保存输出             │   │  │
│  │                                 │   │   │    translated.srt       │   │  │
│  │                                 │   │   │    translated.ass       │   │  │
│  │                                 │   │   │    恢复缓存             │   │  │
│  │                                 │   │   └─────────────────────────┘   │  │
│  └─────────────────────────────────┘   └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段定义与依赖关系

### 2.1 阶段组成

| 阶段名 | TaskStep 枚举 | 职责 | 依赖 | 输出 |
|--------|---------------|------|------|------|
| **OPTIMIZE** | `TaskStep.OPTIMIZE` | 字幕优化（错别字、术语） | `SPLIT` | `optimized.srt` |
| **TRANSLATE** | `TaskStep.TRANSLATE` | LLM 翻译 | `OPTIMIZE` | `translated.srt` |

### 2.2 阶段组定义（`vat/models.py`）

```python
STAGE_GROUPS = {
    "translate": [TaskStep.OPTIMIZE, TaskStep.TRANSLATE],
}

STAGE_DEPENDENCIES = {
    TaskStep.OPTIMIZE: [TaskStep.SPLIT],
    TaskStep.TRANSLATE: [TaskStep.OPTIMIZE],
}
```

### 2.3 独立执行能力

```bash
# 只执行优化（不翻译）
vat pipeline --steps optimize <video_id>

# 只重跑翻译（优化结果已存在）
vat pipeline --steps translate --force <video_id>

# 执行完整翻译（Optimize + Translate）
vat translate <video_id>
# 等价于
vat pipeline --steps optimize,translate <video_id>
```

**应用场景**：
- 如果你只想调整翻译 prompt，无需重跑优化
- 如果优化结果已满意，可直接翻译

---

## 3. 调用链详解

### 3.1 从 CLI 到核心函数

```
CLI: vat translate -v <video_id>
    │
    ▼
commands.py: translate()
    │ 展开阶段组 "translate" → [OPTIMIZE, TRANSLATE]
    ▼
scheduler.py: schedule_videos(steps=[OPTIMIZE, TRANSLATE])
    │
    ▼
executor.py: VideoProcessor.process(steps=[OPTIMIZE, TRANSLATE])
    │
    ├──▶ _execute_step(TaskStep.OPTIMIZE)
    │        └──▶ _run_optimize(force)
    │                 ├── ASRData.from_subtitle_file(original.srt)
    │                 ├── LLMTranslator(enable_reflect=False)
    │                 └── translator.optimize_subtitles()
    │
    └──▶ _execute_step(TaskStep.TRANSLATE)
             └──▶ _run_translate(force)
                      ├── ASRData.from_subtitle_file(optimized.srt)
                      ├── LLMTranslator(enable_reflect=config)
                      └── translator.translate()
```

### 3.2 关键代码索引

| 组件 | 文件位置 | 函数/类 |
|------|----------|---------|
| CLI 入口 | `vat/cli/commands.py` | `translate()` |
| **OPTIMIZE 阶段** | `vat/pipeline/executor.py` | `_run_optimize()` |
| **TRANSLATE 阶段** | `vat/pipeline/executor.py` | `_run_translate()` |
| LLM 翻译器 | `vat/translator/llm_translator.py` | `LLMTranslator` |
| 基础翻译器 | `vat/translator/base.py` | `BaseTranslator` |
| Chunk 缓存 | `vat/translator/base.py` | `_safe_translate_chunk()` |
| LLM 调用 | `vat/llm/client.py` | `call_llm()` |
| Prompt 加载 | `vat/llm/prompts/__init__.py` | `get_prompt()` |

---

## 4. OPTIMIZE 阶段详解

### 4.1 内部流程

```
_run_optimize(force)
    │
    ├─ 1. 检查 original.srt 是否存在
    │      不存在 → 抛出 TranslateError
    │
    ├─ 2. 检查 optimize.enable
    │      │
    │      └─ 禁用 → 复制 original.srt 到 optimized.srt，返回
    │
    ├─ 3. if force: disable_cache()
    │      禁用全局缓存，确保 LLM 调用不命中
    │
    ├─ 4. 创建 LLMTranslator
    │      enable_reflect=False (优化阶段不需要反思)
    │      custom_prompt=config.translator.llm.optimize.custom_prompt
    │
    ├─ 5. 获取场景 prompt (如有)
    │      scene_prompts.get('optimize', '')
    │
    ├─ 6. translator.optimize_subtitles(asr_data, scene_prompt)
    │      │
    │      ├─ 分 chunk 处理
    │      │      batch_size = config.translator.llm.batch_size
    │      │
    │      ├─ 每个 chunk: Agent Loop
    │      │      - 调用 LLM 优化
    │      │      - 验证输出（条数、格式）
    │      │      - 失败重试 (max_retries)
    │      │
    │      └─ 合并结果
    │
    ├─ 7. 保存 optimized.srt
    │
    └─ 8. if force: enable_cache()
          恢复缓存
```

### 4.2 优化的作用

| 优化类型 | 示例 |
|----------|------|
| 错别字修正 | "計画" → "計画" (统一写法) |
| 术语统一 | "アイテム"/"道具" → 统一为 "道具" |
| 口语整理 | 去除冗余语气词 |
| 断句优化 | 补充因 ASR 切割导致的不完整句子 |

### 4.3 配置选项

```yaml
translator:
  llm:
    optimize:
      enable: true                    # 是否启用优化
      custom_prompt: null             # 自定义 prompt 文件路径
```

### 4.4 易混淆点

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| "优化被跳过" | `optimize.enable=false` | 检查配置 |
| "优化结果和原文一样" | LLM 认为无需修改 | 检查 prompt 或输入质量 |
| "改了 prompt 但结果不变" | LLM 缓存命中 | 使用 `-f` 强制重跑 |

---

## 5. TRANSLATE 阶段详解

### 5.1 内部流程

```
_run_translate(force)
    │
    ├─ 1. 确定输入文件
    │      优先 optimized.srt，否则 original.srt
    │
    ├─ 2. 检查 skip_translate
    │      │
    │      └─ 启用 → 复制输入到 translated.srt，返回
    │
    ├─ 3. if force: disable_cache()
    │
    ├─ 4. 创建 LLMTranslator
    │      enable_reflect=config.translator.llm.enable_reflect
    │      enable_context=config.translator.llm.enable_context
    │      custom_prompt=config.translator.llm.custom_prompt
    │
    ├─ 5. 获取场景 prompt (如有)
    │      scene_prompts.get('translate', '')
    │
    ├─ 6. translator.translate(asr_data, scene_prompt)
    │      │
    │      ├─ 分 chunk 处理
    │      │      batch_size = config.translator.llm.batch_size
    │      │
    │      ├─ 每个 chunk: 
    │      │      ├─ _build_input_with_context() (上下文传递)
    │      │      ├─ _translate_chunk() (Agent Loop)
    │      │      └─ 缓存结果 (_safe_translate_chunk)
    │      │
    │      ├─ 反思翻译 (if enable_reflect)
    │      │      - 第二次 LLM 调用审查翻译质量
    │      │      - 返回改进建议
    │      │
    │      └─ 合并结果
    │
    ├─ 7. 保存输出
    │      translated.srt (SRT 格式)
    │      translated.ass (ASS 格式，带样式)
    │
    └─ 8. if force: enable_cache()
```

### 5.2 翻译流程详解

#### 5.2.1 Chunk 切分

```python
# BaseTranslator.translate_subtitle()
chunks = []
for i in range(0, len(segments), batch_size):
    chunk = segments[i:i+batch_size]
    chunks.append(chunk)
```

- `batch_size` 决定每次 LLM 调用处理的字幕条数
- 默认 30 条/batch

#### 5.2.2 上下文传递

```python
# LLMTranslator._build_input_with_context()
if enable_context and previous_batch:
    # 取前一批的最后 N 条作为上下文
    context = previous_batch[-context_size:]
    input_text = format_context(context) + format_batch(current_batch)
```

- 保持翻译的连贯性
- 避免代词指代不清等问题

#### 5.2.3 Agent Loop（核心）

```
LLM 调用
    │
    ▼
验证响应 (_validate_llm_response)
    │
    ├─ 通过 → 返回结果
    │
    └─ 失败 → 重试 (带错误信息)
             │
             ├─ 重试次数 < max_retries → 再次调用
             │
             └─ 达到上限 → 返回最佳尝试 / 抛出异常
```

验证规则：
- 输出条数必须与输入一致
- 必须为有效 JSON
- 翻译不能为空

#### 5.2.4 反思翻译（可选）

```
第一次翻译
    │
    ▼
反思 LLM 调用
    │ "请审查以下翻译，指出问题并改进"
    ▼
第二次翻译（带反思结果）
    │
    ▼
最终输出
```

- 通过 `enable_reflect: true` 启用
- 会增加 LLM 调用次数
- 提升翻译质量，但增加成本和延迟

### 5.3 skip_translate 模式

```yaml
translator:
  skip_translate: true
```

**作用**：跳过翻译，直接复制原文到 `translated.srt`

**使用场景**：
- 视频已是目标语言，只需嵌入
- 调试嵌入流程

---

## 6. 缓存机制详解

### 6.1 缓存层次

```
┌─────────────────────────────────────────────────────────────────┐
│                        缓存层次                                  │
├─────────────────────────────────────────────────────────────────┤
│ 层次 1: 步骤级缓存 (DB)                                          │
│         - 检查 tasks 表中该阶段是否已完成                         │
│         - 未传 -f 且已完成 → 整个阶段被跳过                       │
├─────────────────────────────────────────────────────────────────┤
│ 层次 2: Chunk 级缓存 (diskcache)                                 │
│         - BaseTranslator._safe_translate_chunk()                │
│         - key = class_name + chunk_hash + lang + model          │
│         - force 时通过 disable_cache() 全局禁用                  │
├─────────────────────────────────────────────────────────────────┤
│ 层次 3: LLM 调用缓存 (memoize)                                   │
│         - call_llm() 的 diskcache memoize                       │
│         - 默认 1 小时 TTL                                        │
│         - force 时同样被禁用                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Chunk 缓存 key 生成

```python
# LLMTranslator._get_cache_key()
def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
    class_name = self.__class__.__name__
    chunk_key = generate_cache_key(chunk)  # 基于内容的 hash
    lang = self.target_language.value
    model = self.model
    return f"{class_name}:{chunk_key}:{lang}:{model}"
```

**影响因素**：
- 字幕内容 (chunk 的文本)
- 目标语言
- 模型名称

**不影响**（改了也可能命中缓存）：
- prompt 内容
- batch_size（只影响切分方式）
- enable_reflect

### 6.3 force 语义（重要）

```python
# _run_translate()
if force:
    disable_cache()   # 全局禁用缓存
    self.progress_callback("强制模式：已禁用翻译缓存")

try:
    # 翻译逻辑
finally:
    if force:
        enable_cache()  # 恢复缓存
```

**效果**：
- 跳过所有缓存层（Chunk 缓存 + LLM memoize）
- 确保每个 chunk 都重新调用 LLM

---

## 7. Prompt 系统详解

### 7.1 Prompt 加载流程

```
1. 检查 custom_prompt_file
   │
   ├─ 有 → 加载自定义 prompt
   │
   └─ 无 → 加载内置 prompt
              vat/llm/prompts/{optimize,translate}/*.md

2. 变量替换 (Template.safe_substitute)
   - $source_language
   - $target_language
   - $context (上下文字幕)
   - $input (当前批次)

3. 追加场景 prompt (如有)
   scene_prompts.get('translate', '')
```

### 7.2 内置 Prompt 位置

| 用途 | 文件路径 |
|------|----------|
| 优化 | `vat/llm/prompts/optimize/system.md` |
| 翻译 | `vat/llm/prompts/translate/system.md` |
| 反思 | `vat/llm/prompts/translate/reflect.md` |

### 7.3 场景识别

```python
# 视频下载时识别场景
scene_id = SceneIdentifier().identify(title, description)
# e.g., "gaming", "tech_review", "anime"

# 翻译时获取场景 prompt
scene_prompts = SceneIdentifier().get_scene_prompts(scene_id)
translate_prompt = scene_prompts.get('translate', '')
```

场景定义：`vat/llm/scenes.yaml`

---

## 8. 并发与性能

### 8.1 多线程翻译

```python
# BaseTranslator.translate_subtitle()
with ThreadPoolExecutor(max_workers=thread_num) as executor:
    futures = [
        executor.submit(self._safe_translate_chunk, chunk)
        for chunk in chunks
    ]
    results = [f.result() for f in futures]
```

- `thread_num` 控制并发数
- 默认 3 线程

### 8.2 性能优化建议

| 场景 | 建议 |
|------|------|
| 长视频 | 增大 `batch_size` 减少 LLM 调用次数 |
| 高质量需求 | 启用 `enable_reflect` |
| 快速迭代 | 禁用 `enable_reflect`，减小 `batch_size` 方便缓存复用 |
| 避免 API 限流 | 减小 `thread_num` |

---

## 9. 常见问题与 Debug Checklist

### 9.1 问题速查表

| 症状 | 可能原因 | 排查步骤 |
|------|----------|----------|
| 阶段被跳过 | DB 标记已完成 | 使用 `-f` |
| 翻译结果不变 | 缓存命中 | 使用 `-f` 禁用缓存 |
| 翻译条数不对 | LLM 输出验证失败后的最佳尝试 | 检查日志中的验证错误 |
| 翻译质量差 | prompt 不匹配 / 无上下文 | 调整 prompt，启用 enable_context |
| API 调用失败 | 网络/配额问题 | 检查 `llm.api_key` 和网络 |

### 9.2 Debug Checklist

1. **确认输入文件存在**
   - OPTIMIZE: `original.srt` 必须存在
   - TRANSLATE: `optimized.srt` 或 `original.srt` 必须存在

2. **确认配置**
   - `translator.skip_translate` 是否误开
   - `translator.llm.optimize.enable` 是否符合预期

3. **确认缓存状态**
   - 使用 `-f` 强制重跑
   - 或清理 `~/.cache/vat/` 下的缓存

4. **确认 LLM 可用性**
   - `llm.api_key` 是否设置
   - `llm.base_url` 是否可达

5. **查看详细日志**
   - 缓存命中会显示 "缓存命中: X/Y 批次"
   - 验证失败会显示重试信息

---

## 10. 配置参考（config/default.yaml）

```yaml
translator:
  source_language: "日语"
  target_language: "简体中文"
  skip_translate: false         # 跳过翻译，直接使用原文

  llm:
    model: "gpt-4o-mini"
    batch_size: 30              # 每批处理条数
    thread_num: 3               # 并发线程数
    enable_reflect: false       # 反思翻译
    enable_context: true        # 上下文传递
    custom_prompt: null         # 自定义翻译 prompt

    optimize:
      enable: true              # 启用优化
      custom_prompt: null       # 自定义优化 prompt

llm:
  api_key: "${VAT_LLM_APIKEY}"
  base_url: "https://api.videocaptioner.cn"
```

---

## 11. 修改指南：如果你想改某个功能...

| 如果你想... | 应该看/改哪里 |
|-------------|---------------|
| 改翻译 prompt | `vat/llm/prompts/translate/*.md` 或 `custom_prompt` |
| 改优化 prompt | `vat/llm/prompts/optimize/*.md` |
| 改 Agent Loop 验证逻辑 | `vat/translator/llm_translator.py` `_validate_llm_response()` |
| 改 chunk 缓存 key | `vat/translator/llm_translator.py` `_get_cache_key()` |
| 改上下文传递逻辑 | `vat/translator/llm_translator.py` `_build_input_with_context()` |
| 添加新翻译后端 | 继承 `BaseTranslator`，参考 `LLMTranslator` |
| 改场景识别 | `vat/llm/scene_identifier.py` + `vat/llm/scenes.yaml` |
| 改缓存 TTL | `vat/utils/cache.py` |
