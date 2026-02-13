# VAT — Video Auto Translator

> **🇬🇧 [English Documentation](README.md)**

一个端到端的视频翻译自动化系统。从 YouTube 下载视频，到语音识别、智能断句、LLM 翻译、字幕嵌入，直至上传 B 站，全流程自动完成。

<!-- TODO: 插入效果展示截图/GIF -->
<!-- ![效果展示](docs/assets/demo.gif) -->

---

## 它能做什么

VAT 的核心流程是一条 **7 阶段流水线**：

```
YouTube URL / 本地视频
    │
    ├─ 1. Download ─── 下载视频 + 字幕 + 元数据 + 场景识别
    ├─ 2. Whisper ──── faster-whisper 语音识别（支持分块并发）
    ├─ 3. Split ────── LLM 智能断句（零碎片段→完整句子）
    ├─ 4. Optimize ─── LLM 字幕优化（纠错、术语统一）
    ├─ 5. Translate ── LLM 反思翻译（初译→反思→重译）
    ├─ 6. Embed ────── 字幕嵌入视频（硬字幕 GPU 加速 / 软字幕）
    └─ 7. Upload ───── 自动上传 B 站（标题模板、封面、合集）
```

每个阶段独立可控：可以只跑其中几步，跳过已完成的步骤，或强制重跑某一步。中断后从断点继续，不需要从头开始。

---

## 主要能力

### 语音识别（ASR）

- 基于 **faster-whisper**，支持 large-v3 等模型
- **分块并发**：长视频自动切分为多段并行转录，合并时处理重叠区域
- 词级时间戳，为后续断句提供精确定位
- ASR 后处理：幻觉检测（移除 "ご視聴ありがとう" 等）、重复清理、日语标点标准化
- 可选人声分离（Mel-Band-Roformer），处理有背景音乐的视频

<!-- TODO: 插入 ASR 转录结果截图 -->
<!-- ![ASR 结果](docs/assets/asr_result.png) -->

### 智能断句

参考VideoCaptioner项目（见致谢）

Whisper 输出的片段通常是零碎的、不完整的。VAT 使用 LLM 将这些片段重组为符合人类阅读习惯的完整句子：

- 支持分块断句（长视频）和全文断句（短视频）
- 可配置的句子长度约束（CJK/英文分别控制）
- 场景感知：不同类型的视频（游戏、闲聊、歌曲等）使用不同的断句策略

### 字幕翻译

参考VideoCaptioner项目（见致谢）

- **反思翻译**（基于吴恩达方法论）：初译 → 反思 → 重译，显著提升翻译质量
- **上下文管理**：批量处理时维护前文上下文，保持术语和风格一致
- **字幕优化**：翻译前自动修正原文中的错别字、统一术语
- **场景提示词**：根据视频类型（游戏、科普、闲聊等）自动加载专用提示词
- **自定义提示词**：支持为特定频道或内容定制翻译/优化提示词
- 兼容任何 OpenAI 格式的 API（包括本地部署的 Ollama 等）

<!-- TODO: 插入翻译对比截图（原文 vs 译文） -->
<!-- ![翻译对比](docs/assets/translation_comparison.png) -->

### 字幕嵌入

- **硬字幕**：GPU 加速（H.264/H.265/AV1），支持 NVIDIA 硬件编码
- **软字幕**：快速封装，保持原画质
- 内置 ASS 样式模板（默认、科普风、番剧风、竖屏等），支持自定义样式
- 字幕根据视频分辨率自动缩放

### 视频下载

- 基于 yt-dlp，支持 YouTube 视频和播放列表
- 自动下载人工字幕（检测到人工字幕时可跳过 ASR）
- 场景自动识别（游戏、闲聊、歌曲、科普等）
- 视频信息自动翻译（标题、简介、标签）

### B 站上传

- 基于 biliup 的自动上传
- 模板系统：标题/简介支持变量替换（频道名、翻译标题等）
- 自动获取封面、推荐分区、生成标签
- 支持添加到合集

### 调度与并发

- 多 GPU 任务调度：自动分配视频到不同 GPU
- 步骤级状态追踪：每个阶段独立记录状态，支持断点续传
- 配置快照缓存：修改断句参数只重跑断句，不重跑 ASR

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA GPU（推荐，ASR 和字幕嵌入需要）
- ffmpeg（系统级安装）
- LLM API（断句、翻译、优化需要）

### 安装

```bash
git clone <repo-url> && cd vat
pip install -r requirements.txt
pip install -e .
```

**字体文件**（硬字幕渲染需要）：

字体未包含在仓库中（约 65MB）。将以下字体放入 `vat/resources/fonts/` 目录：

| 字体 | 用途 | 来源 |
|------|------|------|
| NotoSansCJKsc-VF.ttf | 默认中日韩字体 | [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+SC) |
| LXGWWenKai-Regular.ttf | 番剧风格 | [LXGW WenKai](https://github.com/lxgw/LxgwWenKai) |
| ZCOOLKuaiLe-Regular.ttf | 科普风格 | [Google Fonts](https://fonts.google.com/specimen/ZCOOL+KuaiLe) |
| AlimamaFangYuanTiVF-Thin-2.ttf | 竖屏风格 | [Alimama Fonts](https://fonts.alibabagroup.com/) |

如果不需要硬字幕或只使用默认样式，只需 NotoSansCJKsc-VF.ttf 即可。

（事实上，大部分ubuntu系统上可以找到NotoSansCJKsc-VF.ttf）

### 配置

```bash
# 设置 LLM API Key
export VAT_LLM_APIKEY="your-api-key"

# 生成配置文件
vat init

# 编辑配置（路径、模型、翻译参数等）
vim config/config.yaml
```

配置文件中的关键项：

| 配置项 | 说明 |
|--------|------|
| `storage.work_dir` | 工作目录（处理中间文件） |
| `storage.output_dir` | 输出目录（最终视频） |
| `storage.models_dir` | 模型文件目录 |
| `asr.model` | Whisper 模型（推荐 `large-v3`） |
| `asr.language` | 源语言（如 `ja`） |
| `translator.llm.model` | 翻译使用的 LLM 模型 |
| `translator.llm.enable_reflect` | 是否启用反思翻译 |
| `llm.api_key` | 全局 LLM API Key（支持 `${ENV_VAR}` 格式） |
| `llm.base_url` | 全局 LLM API 地址 |
| `asr.split.api_key/base_url` | 断句阶段可选覆写（留空继承全局） |
| `translator.llm.api_key/base_url` | 翻译阶段可选覆写（留空继承全局） |
| `translator.llm.optimize.*` | 优化阶段可选覆写 model/api_key/base_url/batch_size/thread_num（留空继承父级→全局） |
| `proxy.http_proxy` | 代理设置 |

完整配置说明参见 [`config/default.yaml`](config/default.yaml) 中的注释。

### 运行

```bash
# 一键处理单个视频（下载→识别→翻译→嵌入）
vat pipeline --url "https://www.youtube.com/watch?v=VIDEO_ID"

# 处理播放列表
vat pipeline --playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# 多 GPU 并行
vat pipeline --url "URL" --gpus 0,1

# 分阶段执行
vat process -v VIDEO_ID -s asr          # 只跑语音识别
vat process -v VIDEO_ID -s translate    # 只跑翻译
vat process -v VIDEO_ID -s embed        # 只跑字幕嵌入

# 强制重跑
vat process -v VIDEO_ID -s translate -f

# 查看状态
vat status
```

### 输出文件

处理完成后，输出目录结构：

```
data/videos/<VIDEO_ID>/
├── <video>.mp4           # 原始下载视频
├── original_raw.srt      # Whisper 原始转录
├── original.srt          # 断句后的原文字幕
├── optimized.srt         # 优化后的原文字幕
├── translated.srt        # 翻译后的字幕
├── translated.ass        # ASS 格式字幕（带样式）
└── final.mp4             # 嵌入字幕的最终视频
```

---

## Web 管理界面

VAT 提供基于 FastAPI 的 Web UI，用于查看视频状态、管理任务、编辑字幕文件。

```bash
# 启动 WebUI
vat web
# 或
python -m vat web --port 8080
```

<!-- TODO: 插入 WebUI 截图 -->
<!-- ![WebUI 首页](docs/assets/webui_index.png) -->
<!-- ![WebUI 视频详情](docs/assets/webui_detail.png) -->
<!-- ![WebUI 任务管理](docs/assets/webui_tasks.png) -->

功能包括：
- 视频列表与状态总览（支持搜索、过滤）
- 视频详情页（任务时间线、文件预览）
- 在线创建和执行处理任务
- 字幕文件在线查看与编辑
- 播放列表管理与批量操作
- B 站上传配置管理

详细操作说明参见 [WebUI 使用手册](docs/webui_manual.md)。

---

## CLI 命令速查

| 命令 | 说明 |
|------|------|
| `vat pipeline -u URL` | 完整流水线（下载到嵌入） |
| `vat process -v ID -s STAGES` | 细粒度阶段控制 |
| `vat download -u URL` | 仅下载 |
| `vat asr -v ID` | 仅语音识别 |
| `vat translate -v ID` | 仅翻译 |
| `vat embed -v ID` | 仅嵌入字幕 |
| `vat upload VIDEO_ID` | 上传到 B 站 |
| `vat playlist sync URL` | 同步播放列表 |
| `vat status` | 查看处理状态 |
| `vat clean -v ID` | 清理中间产物 |
| `vat bilibili login` | B 站登录获取 Cookie |

---

## 项目结构

```
vat/
├── asr/                  # 语音识别模块
│   ├── whisper_asr.py    #   faster-whisper 封装
│   ├── chunked_asr.py    #   分块并发 ASR
│   ├── split.py          #   LLM 智能断句
│   ├── asr_post.py       #   后处理（幻觉/重复检测）
│   └── vocal_separation/ #   人声分离
├── translator/           # 翻译模块
│   └── llm_translator.py #   LLM 反思翻译引擎
├── llm/                  # LLM 基础设施
│   ├── client.py         #   统一 LLM 调用客户端
│   ├── scene_identifier.py # 场景识别
│   └── prompts/          #   提示词管理
├── embedder/             # 字幕嵌入模块
│   └── ffmpeg_wrapper.py #   FFmpeg 封装（软/硬字幕）
├── downloaders/          # 下载器
├── uploaders/            # 上传器（B 站）
├── pipeline/             # 流水线编排
│   ├── executor.py       #   VideoProcessor（阶段调度）
│   ├── scheduler.py      #   多 GPU 调度器
│   ├── progress.py       #   进度追踪
│   └── exceptions.py     #   统一异常体系
├── web/                  # Web 管理界面
│   ├── app.py            #   FastAPI 应用
│   ├── deps.py           #   共享依赖
│   ├── routes/           #   API 路由
│   └── templates/        #   页面模板
├── cli/                  # CLI 命令
├── database.py           # SQLite 数据层
├── config.py             # 配置管理
└── models.py             # 数据模型定义
```

---

## 配置进阶

### 自定义提示词

在 `vat/llm/prompts/custom/` 下创建提示词文件，在配置中引用文件名即可：

```yaml
translator:
  llm:
    custom_prompt: "my_channel"          # 翻译提示词
    optimize:
      custom_prompt: "my_channel"        # 优化提示词
```

提示词编写指南参见 [提示词优化指南](docs/prompt_optimization_guide.md)。

### 场景识别

VAT 会根据视频标题和简介自动识别场景类型（游戏、闲聊、歌曲、科普等），并加载对应的场景提示词。场景配置定义在 `vat/llm/scenes.yaml` 中。

### ASR 参数调优

不同类型的视频可能需要不同的 ASR 参数。常见调优方向：

- 游戏/直播：关闭 VAD，降低 `no_speech_threshold`
- 纯人声（播客）：开启 VAD
- 背景音乐重的视频：考虑启用人声分离

参数详解参见 [ASR 参数指南](docs/asr_parameters_guide.md)。

### GPU 分配

多 GPU 环境下的任务分配策略说明参见 [GPU 分配规范](docs/gpu_allocation_spec.md)。

---

## 技术文档

| 文档 | 内容 |
|------|------|
| [ASR 参数指南](docs/asr_parameters_guide.md) | Whisper 参数详解与调优建议 |
| [ASR 评估报告](docs/ASR_EVALUATION_REPORT.md) | 不同参数组合的识别效果对比 |
| [提示词优化指南](docs/prompt_optimization_guide.md) | 翻译/优化提示词的编写方法 |
| [GPU 分配规范](docs/gpu_allocation_spec.md) | 多 GPU 调度策略 |
| [WebUI 手册](docs/webui_manual.md) | Web 界面操作说明 |
| [YouTube 字幕](docs/youtube_manual_subtitles.md) | YouTube 人工字幕检测与使用 |
| [项目审查报告](docs/project_review.md) | 架构审查与重构记录 |
| [开发手册](README_USAGE.md) | 分阶段运行详解与开发参考 |

---

## 致谢

本项目集成了以下开源项目的核心技术：

- [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) — 分块 ASR、智能断句、反思翻译、ASS 渲染的核心参考
- [GalTransl](https://github.com/xd2333/GalTransl) — 翻译引擎
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — 语音识别
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频下载
- [biliup](https://github.com/biliup/biliup) — B 站上传
- [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) — 人声分离模型

详细致谢信息参见 [acknowledgement.md](acknowledgement.md)。

---

## 许可证

GPLv3
