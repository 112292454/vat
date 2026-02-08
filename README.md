# VAT - Video Handling & Translation

**视频自动化翻译流水线系统**

VAT 是一个完整的视频自动化翻译流水线系统，支持从 YouTube 下载视频、语音识别、字幕翻译、字幕嵌入到视频上传的全流程自动化处理。

## ✨ 主要特性

- 🎬 **多平台下载**: 支持 YouTube 视频/播放列表下载（可扩展至其他平台）
- 🎤 **语音识别**: 基于 faster-whisper 的高质量语音转文字
  - ⭐ **分块处理**: 自动处理超长音频，支持并发分块转录和智能合并
  - ⭐ **词级时间戳**: 精确的时间轴控制
- 🧠 **智能字幕处理**:
  - ⭐ **智能断句**: 使用 LLM 将零碎识别结果重组为完整句子
  - ⭐ **字幕优化**: 自动修正错别字、统一术语、格式化数学公式
  - ⭐ **Agent Loop**: 自动验证和修正，确保质量
- 🌐 **高质量翻译**:
  - ⭐ **反思翻译**: 基于吴恩达方法论的三阶段翻译（初译-反思-重译）
  - ⭐ **上下文管理**: 批量处理保持上下文连贯性
  - ⭐ **离线支持**: 支持通过本地 OpenAI 兼容服务（如 Ollama）实现完全离线翻译
- 📝 **字幕美化**: 内置多种样式模板（科普风、番剧风等），支持专业级 ASS 渲染
- 🎞️ **视频合成**: 自动将翻译字幕嵌入视频，支持硬字幕 GPU 加速
- 📤 **自动上传**: 支持 B 站自动上传
- 🚀 **多 GPU 并发**: 智能任务调度，充分利用多卡环境
- 💾 **断点续传**: 步骤级状态追踪，支持中断后继续

## 📋 系统架构

```
输入层 (YouTube URL/本地文件)
    ↓
下载器 (youtube-dl) → 数据库状态记录
    ↓
语音识别 (faster-whisper + ChunkedASR) → 原始转录
    ↓
智能断句 (LLM) → 完整句子
    ↓
字幕优化 (LLM) → 修正错误
    ↓
翻译器 (LLM反思翻译 / GalTransl) → 中文字幕
    ↓
字幕嵌入 (ffmpeg + ASS 模板) → 最终视频
    ↓
上传器 (biliup) → B 站发布
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. 配置 API (使用 LLM 功能需要)
```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-key"
```

### 3. 初始化并运行
```bash
vat init  # 生成 config/config.yaml
vat pipeline --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

## 📚 文档指南

- **[使用示例](docs/USAGE_EXAMPLES.md)**: 涵盖各种场景的配置和命令示例
- **[升级指南](docs/UPGRADE_GUIDE.md)**: 从旧版本迁移到 v0.2.0 的说明
- **[功能对比](docs/FEATURE_COMPARISON.md)**: VAT 与 VideoCaptioner 的功能差异说明
- **[更新日志](CHANGELOG.md)**: 项目版本演进历史

## ⚙️ 核心配置简述

配置文件位于 `config/config.yaml`。**注意：v0.2.0 之后所有路径配置必须使用绝对路径。**

```yaml
storage:
  work_dir: "/path/to/work"
  output_dir: "/path/to/output"
  models_dir: "/path/to/models"

translator:
  backend_type: "llm"  # 使用 LLM 翻译
  llm:
    model: "gpt-4o-mini"
    enable_reflect: true  # 启用反思翻译
```

## 🙏 致谢

本项目集成了以下优秀开源项目的核心技术：

- [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) - 核心技术参考（分块 ASR、智能断句、反思翻译、ASS 渲染）
- [VoiceTransl](https://github.com/shinnpuru/VoiceTransl) - 早期翻译功能参考
- [GalTransl](https://github.com/xd2333/GalTransl) - 翻译引擎
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 语音识别
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - 视频下载
- [biliup](https://github.com/biliup/biliup) - B 站上传

## 📄 许可证

本项目基于 GPLv3 许可证开源。
