# VAT 模块文档：LLM（大语言模型基础设施）

> 提供统一的 LLM 调用接口、提示词管理、场景识别和视频信息翻译功能。
>
> 是 Split、Optimize、Translate 等阶段的底层依赖。

---

## 1. 模块组成

| 文件/目录 | 职责 |
|-----------|------|
| `client.py` | 统一 LLM 客户端：OpenAI 兼容 API 调用、自动重试、结果缓存、多端点支持 |
| `prompts/` | 提示词管理：Markdown 文件存储 + 模板变量替换 + LRU 缓存 |
| `scene_identifier.py` | 场景识别：基于视频标题/简介判断内容类型（游戏/聊天/唱歌等） |
| `video_info_translator.py` | 视频信息翻译：翻译标题/简介/标签 + 推荐 B 站分区 |
| `scenes.yaml` | 场景配置：定义场景类型、关键词、各阶段的场景特定提示词 |

---

## 2. LLM 客户端（client.py）

### 2.1 核心函数

| 函数 | 说明 |
|------|------|
| `call_llm(messages, model, ...)` | 调用 LLM API，带自动缓存 + 速率限制重试（最多 10 次指数退避） |
| `get_llm_client()` | 获取全局 OpenAI 客户端单例（线程安全） |
| `get_or_create_client(api_key, base_url, proxy)` | 获取或创建指定配置的客户端（按 credentials 缓存） |

### 2.2 多端点支持

项目同时使用多个 LLM 服务商，通过 per-call 参数切换：

```python
# 默认端点（环境变量 OPENAI_BASE_URL / OPENAI_API_KEY）
call_llm(messages, model="gpt-4o-mini")

# 火山引擎端点（指定 base_url + api_key）
call_llm(messages, model="kimi-k2.5",
         base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
         api_key=os.getenv("VAT_VOLC_APIKEY"))

# Google Gemini（通过代理）
call_llm(messages, model="gemini-3-flash-preview",
         base_url="...", proxy="http://localhost:7890")
```

客户端按 `(base_url, api_key, proxy)` 三元组缓存，相同配置复用同一连接。

### 2.3 特殊处理

- **火山引擎 thinking 自动关闭**：检测到 `ark.cn-beijing.volces.com` 端点时，自动注入 `thinking: {type: disabled}`。kimi-k2.5 默认走 thinking 路径（34s/call），关闭后降至 0.7s/call，且实验验证对翻译/ASR 纠错无帮助
- **缓存**：基于 diskcache 的结果缓存（`@memoize`），相同输入 1 小时内复用结果。通过 `config.storage.cache_enabled` 控制开关
- **重试**：仅对 `RateLimitError` 重试，指数退避 5-60 秒，最多 10 次
- **响应验证**：空 choices 或空 content 抛出 `ValueError`（不会被缓存）

### 2.4 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_BASE_URL` | 默认 API 端点（由 config.py 的 `LLMConfig.__post_init__` 自动设置） |
| `OPENAI_API_KEY` | 默认 API Key |
| `VAT_VOLC_APIKEY` | 火山引擎 API Key（Optimize 阶段使用） |

---

## 3. 提示词管理（prompts/）

### 3.1 目录结构

```
prompts/
├── __init__.py          # get_prompt() / list_prompts()
├── split/
│   ├── semantic.md      # 语义断句提示词
│   └── sentence.md      # 句子断句提示词
├── optimize/
│   └── subtitle.md      # 字幕优化提示词
├── translate/
│   ├── reflect.md       # 反思翻译提示词（初译→反思→重译）
│   ├── standard.md      # 标准翻译提示词
│   └── single.md        # 单条翻译提示词
├── analysis/
│   └── video.md         # 视频分析提示词
└── custom/
    ├── README.md         # 自定义提示词说明
    ├── optimize/         # 按主播定制的优化提示词
    │   ├── fubuki.md
    │   └── rurudo.md
    └── translate/        # 按主播定制的翻译提示词
        ├── fubuki.md
        └── rurudo.md
```

### 3.2 使用方式

```python
from vat.llm.prompts import get_prompt

# 加载基础提示词
prompt = get_prompt("split/semantic")

# 带模板变量替换（$variable 或 ${variable}）
prompt = get_prompt("split/semantic", max_word_count_cjk=18)
prompt = get_prompt("translate/reflect", target_language="简体中文")
```

- 提示词以 Markdown 文件存储，带 LRU 缓存（32 条）
- 变量替换使用 Python `string.Template`（`$var` / `${var}`），未定义的变量保留原样（`safe_substitute`）
- 自定义提示词通过 config 中 `custom_prompt` 字段指定（如 `fubuki`），加载路径为 `custom/optimize/fubuki.md`

---

## 4. 场景识别（scene_identifier.py）

基于视频标题和简介，通过 LLM 判断内容类型。

### 4.1 工作流程

```
标题 + 简介
    ↓
SceneIdentifier.detect_scene()
    ↓  LLM 分类（温度 0.1）
场景 ID（如 gaming/chatting/singing/asmr）
    ↓
get_scene_prompts(scene_id)
    ↓  从 scenes.yaml 获取
各阶段的场景特定提示词（split/optimize/translate）
```

### 4.2 场景配置（scenes.yaml）

每个场景包含：
- **id**：场景标识（如 `gaming`、`chatting`）
- **name**：显示名称
- **description**：场景描述（用于 LLM 分类）
- **keywords**：关键词列表
- **prompts**：各阶段的场景特定提示词（split/optimize/translate），追加到基础提示词之后

场景识别结果存入 `video.metadata['scene']`，在 Download 阶段执行。

---

## 5. 视频信息翻译（video_info_translator.py）

在 Download 阶段调用，将 YouTube 视频的标题、简介、标签翻译为中文，并推荐 B 站分区。

### 5.1 输出结构（TranslatedVideoInfo）

| 字段 | 说明 |
|------|------|
| `title_translated` | 翻译后标题（不含主播名前缀，后处理自动添加） |
| `description_summary` | 简介摘要（1-2 句话） |
| `description_translated` | 完整翻译的简介 |
| `tags_translated` | 翻译后的标签 |
| `tags_generated` | LLM 生成的额外相关标签 |
| `recommended_tid` | 推荐的 B 站分区 ID |
| `recommended_tid_name` | 分区名称 |
| `tid_reason` | 推荐理由 |

翻译结果存入 `video.metadata['translated']`。

### 5.2 B 站分区映射

内置了与 VTuber 内容相关的 B 站分区映射（`BILIBILI_ZONES`），覆盖生活、游戏、音乐、动画、舞蹈、鬼畜等大区。LLM 根据视频内容从中选择最合适的分区。

---

## 6. 关键代码索引

| 组件 | 调用者 |
|------|--------|
| `call_llm` | Split（断句）、Optimize（优化）、Translate（翻译）、SceneIdentifier、VideoInfoTranslator |
| `get_prompt` | `executor.py` 中各阶段构建 LLM 消息时 |
| `SceneIdentifier` | `executor.py` 的 `_download()` 阶段 |
| `VideoInfoTranslator` | `executor.py` 的 `_download()` 阶段 |
