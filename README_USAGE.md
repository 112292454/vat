# VAT 开发者与内部测试手册

> **注意**：本文档面向开发者和深度用户，提供 CLI 分阶段运行指南、参数详解及验证方法。  
> 如果你更偏好可视化操作，请参考 [WebUI 使用手册](docs/webui_manual.md)。

## 🏗️ 项目核心架构

VAT (Video Auto Translator) 是一个端到端的视频翻译自动化流水线，集成了 [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) 等项目的核心能力。

### 核心能力集成
- **语音识别 (ASR)**：
  - `ChunkedASR`：长视频自动分块处理，支持多 GPU/多线程并发。
  - `Whisper`：支持 `faster-whisper` 后端，提供词级时间戳。
- **字幕处理**：
  - `Smart Split`：利用 LLM 将 Whisper 零碎的输出重组成符合人类阅读习惯的完整句子。
  - `Smart Optimize`：在翻译前对原文进行纠错、术语统一和语气润色。
- **翻译引擎**：
  - `LLMTranslator`：支持反思翻译（Reflective Translation），通过“初译-反思-润色”三阶段提升质量。
  - `GalTransl`：支持本地 GGUF 模型（如 Sakura）的高质量翻译。
- **高性能合成**：
  - `FFmpeg Wrapper`：支持硬字幕（烧录）和软字幕（外挂轨）模式。
  - `GPU 加速`：支持 H.264/H.265 的硬件加速编码。
- **智能缓存**：
  - `CacheMetadata`：基于配置快照的细粒度缓存。修改断句配置只会重跑断句，不会重跑耗时的 ASR。

---

## 🚀 快速开始：全流程运行

### 1. 环境准备
确保已安装 `ffmpeg` 并设置 API Key：
```bash
export VAT_LLM_APIKEY="your-api-key-here"
```

### 2. 一键运行 (Pipeline)
最简单的方式，自动完成下载、转录、翻译、合成。
```bash
# 处理单个 URL
vat pipeline --url "https://www.youtube.com/watch?v=xxxx"

# 使用多个 GPU 并行处理（如果配置了多个 GPU）
vat pipeline --url "URL" --gpus 0,1
```

### 3. 检查结果
输出文件位于 `data/videos/<VIDEO_ID>/` 目录下：
- `final.mkv`：最终合成的视频（带字幕）。
- `translated.srt` / `translated.ass`：翻译后的字幕文件。

---

## 🛠️ 分阶段详解与参数说明

你可以独立运行每个阶段进行调试。系统会自动识别上一步的产物。

### 阶段 1：下载 (Download)
将视频下载到本地，并提取元数据。
```bash
vat download --url "URL"
```
- **关键产物**：`original.mp4` (或 webm/mkv)
- **参数**：
  - `--url`: 支持多个 URL。
  - `--file`: 从文本文件读取 URL 列表。
  - `--playlist`: 下载整个 YouTube 播放列表。

### 阶段 2：转录与断句 (asr)
包含 Whisper 识别和 LLM 智能断句。
```bash
vat asr --video-id <ID>
```
- **关键产物**：
  - `original_raw.srt`: Whisper 原始输出（通常很碎）。
  - `original_split.srt`: LLM 断句后的输出（句子完整，阅读感好）。
  - `original.srt`: 最终送入翻译阶段的原文。
- **关键配置** (`config/default.yaml`):
  - `asr.model`: 推荐 `large-v3`。
  - `asr.split.enable`: 必须开启以获得高质量断句。
  - `asr.split.max_words_cjk`: 中日文字符数限制（默认 24）。

### 阶段 3：翻译与优化 (Translate)
包含原文优化和多阶段翻译。
```bash
vat translate --video-id <ID> --backend llm
```
- **关键产物**：
  - `translated.srt`: 最终译文。
  - `translated.ass`: 带样式的 ASS 字幕。
- **关键配置**:
  - `translator.llm.optimize.enable`: 开启原文优化，修正 ASR 错误。
  - `translator.llm.enable_reflect`: 开启反思翻译（质量飞跃，但消耗更多 Token）。
  - `translator.llm.batch_size`: 每批处理行数（默认 10）。

### 阶段 4：字幕嵌入 (Embed)
将字幕合成到视频中。
```bash
vat embed --video-id <ID>
```
- **关键产物**：`final.mkv`
- **模式选择**:
  - **软字幕 (`soft`)**: 极快，不重编码视频，支持多语言切换。
  - **硬字幕 (`hard`)**: 将字幕烧录进画面，兼容性最好。
- **参数**:
  - `embedder.use_gpu`: 强烈建议开启，加速硬字幕编码。
  - `embedder.subtitle_style`: 选择样式模板（如 `毕导科普风`）。

---

## 🧪 开发者验证指南 (如何检查效果)

作为开发者，你需要确保每一步都达到了预期效果：

### 1. 检查断句质量
对比 `original_raw.srt` 和 `original_split.srt`。
- **预期**：`split` 版本应该将“半句话”合并成完整的句子，且时间戳对齐准确。

### 2. 验证缓存逻辑 (核心测试)
1. 运行一次完整的 `vat asr`。
2. 修改 `config/default.yaml` 中的 `split.max_words_cjk`。
3. 再次运行 `vat asr`。
4. **验证点**：日志应显示 `复用 Whisper 缓存`，仅重新运行 `智能断句`。这证明 `CacheMetadata` 工作正常。

### 3. 检查翻译“反思”效果
查看日志或 `vat.log`。
- **验证点**：如果开启了 `enable_reflect`，你应该能看到 LLM 对初译结果的自我评价和修正过程。

### 4. 检查 ASS 样式
使用支持 ASS 的播放器（如 VLC, IINA, PotPlayer）打开 `final.mkv`。
- **验证点**：字幕是否符合 `subtitle_style` 定义的颜色、位置和边框。

---

## ⚙️ 核心参数速查表

| 模块 | 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **存储** | `storage.models_dir` | - | 所有模型（Whisper, GGUF）的存放根目录 |
| **识别** | `asr.enable_chunked` | `true` | 长视频必须开启，否则显存溢出 |
| **断句** | `asr.split.model` | `gpt-4o-mini` | 建议使用速度快且便宜的模型 |
| **翻译** | `translator.llm.enable_reflect` | `true` | 追求质量建议开启 |
| **优化** | `translator.llm.optimize.enable` | `true` | 修正 ASR 错别字的关键 |
| **合成** | `embedder.embed_mode` | `soft` | 调试阶段建议用 `soft` 极速出片 |
| **并发** | `concurrency.gpu_devices` | `[0]` | 指定使用的 GPU 编号列表 |

---

## 🆘 常见问题与调试

1. **API 调用失败**：检查 `VAT_LLM_APIKEY` 环境变量，或在 `config/default.yaml` 中直接填写（不推荐）。
2. **GPU 内存不足**：减小 `asr.chunk_concurrency` 或 `translator.llm.thread_num`。
3. **强制重跑**：如果想忽略所有缓存从头开始，在命令后加上 `--force` (或 `-f`)。
4. **查看详细日志**：`tail -f vat.log`。
5. **数据库检查**：`vat status -v <ID>` 查看每个步骤的耗时和错误信息。
