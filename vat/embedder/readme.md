# VAT 模块文档：EMBED（字幕嵌入阶段）

> **阶段定义**：`TaskStep.EMBED` 是一个单一阶段（非阶段组）
> 
> 职责：将翻译后的字幕嵌入到视频中，生成最终可分发的视频文件

---

## 1. 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBED 阶段                                         │
│                           (TaskStep.EMBED)                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. 查找输入文件                                                      │    │
│  │    ├── 视频文件: _find_video_file()                                 │    │
│  │    └── 字幕文件: translated.srt → translated.ass                    │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. ASS 字幕生成/更新                                                 │    │
│  │    ├── 加载 translated.srt                                          │    │
│  │    ├── 加载字幕样式 (get_subtitle_style)                            │    │
│  │    └── 生成 translated.ass                                          │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. 选择嵌入模式                                                      │    │
│  │    ├── soft: 软字幕（快速，可切换）                                  │    │
│  │    └── hard: 硬字幕（烧录，不可关闭）                                │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. FFmpeg 处理                                                       │    │
│  │                                                                      │    │
│  │    ┌─────────────────────┐     ┌─────────────────────────────────┐  │    │
│  │    │   软字幕模式        │     │   硬字幕模式                     │  │    │
│  │    │                     │     │                                 │  │    │
│  │    │ embed_subtitle_soft │     │ embed_subtitle_hard             │  │    │
│  │    │ - stream copy       │     │ - 获取视频分辨率                │  │    │
│  │    │ - 添加字幕流        │     │ - 缩放样式（字号/边距）         │  │    │
│  │    │ - 极快（秒级）      │     │ - 自动换行处理                  │  │    │
│  │    │                     │     │ - GPU 硬件编码 (NVENC)          │  │    │
│  │    │                     │     │ - 进度回调                      │  │    │
│  │    └─────────────────────┘     └─────────────────────────────────┘  │    │
│  │                                                                      │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 输出文件                                                            │    │
│  │   <output_dir>/<video_id>/                                          │    │
│  │       └─ final.mp4 (或 final.mkv)                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段定义与依赖

| 阶段名 | TaskStep 枚举 | 依赖 | 输出 |
|--------|---------------|------|------|
| **EMBED** | `TaskStep.EMBED` | `TRANSLATE` | `final.mp4` / `final.mkv` |

```python
# vat/models.py
STAGE_DEPENDENCIES = {
    TaskStep.EMBED: [TaskStep.TRANSLATE],
}
```

---

## 3. 调用链详解

### 3.1 从 CLI 到核心函数

```
CLI: vat embed -v <video_id>
    │
    ▼
commands.py: embed()
    │
    ▼
scheduler.py: schedule_videos(steps=[EMBED])
    │
    ▼
executor.py: VideoProcessor.process(steps=[EMBED])
    │
    └──▶ _execute_step(TaskStep.EMBED)
             └──▶ _embed(force)
                      ├── _find_video_file()
                      ├── ASRData.from_subtitle_file(translated.srt)
                      ├── get_subtitle_style() → translated.ass
                      └── FFmpegWrapper.embed_subtitle_soft/hard()
```

### 3.2 关键代码索引

| 组件 | 文件位置 | 函数/类 |
|------|----------|---------|
| CLI 入口 | `vat/cli/commands.py` | `embed()` |
| 阶段实现 | `vat/pipeline/executor.py` | `_embed()` |
| FFmpeg 封装 | `vat/embedder/ffmpeg_wrapper.py` | `FFmpegWrapper` |
| 软字幕嵌入 | `vat/embedder/ffmpeg_wrapper.py` | `embed_subtitle_soft()` |
| 硬字幕嵌入 | `vat/embedder/ffmpeg_wrapper.py` | `embed_subtitle_hard()` |
| 字幕样式 | `vat/asr/subtitle/__init__.py` | `get_subtitle_style()` |
| 自动换行 | `vat/asr/subtitle/__init__.py` | `auto_wrap_ass_file()` |
| GPU 选择 | `vat/utils/gpu.py` | `resolve_gpu_device()` |

---

## 4. 嵌入模式详解

### 4.1 软字幕 (soft)

```
特点：
├── 极快（秒级完成）
├── 文件大小几乎不变
├── 保持原始视频质量
├── 用户可以开关字幕
└── 不支持所有播放器（某些平台不显示）

适用场景：
├── 本地播放
├── 上传到支持软字幕的平台
└── 需要保留原画质
```

**FFmpeg 命令示例**（MKV 容器）：
```bash
ffmpeg -i video.mp4 -i subtitle.ass \
  -c:v copy -c:a copy -c:s copy \
  -metadata:s:s:0 language=chi \
  -metadata:s:s:0 title=中文 \
  -disposition:s:0 default \
  -y output.mkv
```

### 4.2 硬字幕 (hard)

```
特点：
├── 字幕烧录到视频画面
├── 所有播放器都能显示
├── 不可关闭字幕
├── 需要重新编码（较慢）
└── 使用 GPU 硬件加速 (NVENC)

适用场景：
├── 上传到 B 站等平台
├── 需要确保字幕显示
└── 分享给他人
```

**硬字幕处理流程**：
```
1. 获取视频分辨率
       │
       ▼
2. 样式缩放（根据视频高度）
   scale_factor = height / 720
   - 字号缩放
   - 边距缩放
   - Outline 缩放
       │
       ▼
3. 生成临时 ASS 文件
   - 固定布局：原文在上，译文在下
   - 应用缩放后的样式
       │
       ▼
4. 自动换行处理
   auto_wrap_ass_file()
       │
       ▼
5. GPU 硬件编码 (NVENC)
   - hevc_nvenc (H.265)
   - h264_nvenc (H.264)
   - av1_nvenc (AV1，如支持)
```

---

## 5. GPU 使用与编码器选择

### 5.1 GPU 原则（项目强制）

- **EMBED 阶段必须使用 GPU**
- 禁止 CPU 回退
- 如果 GPU 不可用，阶段将失败并报错

```python
# vat/embedder/ffmpeg_wrapper.py
device_str, gpu_id = resolve_gpu_device(
    gpu_device,
    allow_cpu_fallback=False,  # 遵循 GPU 原则
    min_free_memory_mb=1000
)
if not use_gpu:
    raise RuntimeError("Embed 阶段需要 GPU，按 GPU 原则禁止 CPU 回退")
```

### 5.2 编码器映射

| 配置值 | 实际编码器 | 说明 |
|--------|-----------|------|
| `libx265` / `hevc` | `hevc_nvenc` | H.265 GPU 编码 |
| `libx264` / `h264` | `h264_nvenc` | H.264 GPU 编码 |
| `av1` | `av1_nvenc` | AV1 GPU 编码（需 RTX 40 系列） |

### 5.3 码率控制

```python
# 获取原视频码率
original_bitrate = get_video_info(video_path)['bit_rate']

# VBR 模式：目标 1.1x，最大 1.5x
target_bitrate = original_bitrate * 1.1
max_bitrate = original_bitrate * 1.5

codec_params = [
    '-rc', 'vbr',
    '-cq', str(crf),
    '-b:v', str(target_bitrate),
    '-maxrate', str(max_bitrate),
]
```

---

## 6. 字幕样式系统

### 6.1 样式加载流程

```
1. 读取样式名称
   config.embedder.subtitle_style (如 "default", "bilibili")
       │
       ▼
2. 加载样式文件
   get_subtitle_style(style_name, style_dir)
   → 从 storage.subtitle_style_dir 加载 .ass 样式
       │
       ▼
3. 样式缩放
   _scale_ass_style(style_str, scale_factor)
   - Fontsize: 按比例缩放
   - MarginV: 按比例缩放
   - Outline: 按比例缩放
       │
       ▼
4. 应用到 ASS 文件
```

### 6.2 样式文件位置

```
{storage.subtitle_style_dir}/
├── default.ass
├── bilibili.ass
└── custom.ass
```

### 6.3 字体处理

硬字幕需要字体文件才能正确渲染：

```yaml
# config/default.yaml
storage:
  fonts_dir: "/path/to/fonts"    # 字体目录
  subtitle_style_dir: "/path/to/styles"  # 样式目录
```

FFmpeg 命令中会添加 `fontsdir` 参数：
```bash
-vf "ass='subtitle.ass':fontsdir='/path/to/fonts'"
```

---

## 7. 输入与输出

### 7.1 输入文件

| 文件 | 来源 | 说明 |
|------|------|------|
| 视频文件 | DOWNLOAD 阶段 | `_find_video_file()` 定位 |
| `translated.srt` | TRANSLATE 阶段 | 翻译后的字幕 |
| `translated.ass` | 本阶段生成 | 带样式的字幕 |

### 7.2 输出文件

| 文件 | 说明 |
|------|------|
| `final.mp4` | 嵌入字幕后的视频（MP4 容器） |
| `final.mkv` | 嵌入字幕后的视频（MKV 容器） |
| `translated.ass` | 中间产物：带样式的 ASS 字幕 |

### 7.3 ASS 生成条件

```python
need_regenerate_ass = (
    force  # 强制模式
    or (not subtitle_ass.exists())  # ASS 不存在
    or subtitle_ass.stat().st_mtime < subtitle_srt.stat().st_mtime  # SRT 更新了
)
```

---

## 8. 配置参考

```yaml
embedder:
  embed_mode: "hard"           # "soft" | "hard"
  output_container: "mp4"      # "mp4" | "mkv"
  subtitle_style: "default"    # 样式模板名称
  
  # 硬字幕编码参数
  use_gpu: true               # 已废弃，现在强制使用 GPU
  video_codec: "hevc"         # "hevc" | "h264" | "av1"
  audio_codec: "copy"         # "copy" | "aac"
  crf: 28                     # 质量参数 (0-51，越小越好)
  preset: "p4"                # NVENC 预设 (p1-p7)

gpu:
  device: "auto"              # "auto" | "cuda:N" | "cpu"
  allow_cpu_fallback: false   # 禁止 CPU 回退

storage:
  fonts_dir: "/path/to/fonts"
  subtitle_style_dir: "/path/to/styles"
```

---

## 9. 常见问题与 Debug Checklist

### 9.1 问题速查表

| 症状 | 可能原因 | 排查步骤 |
|------|----------|----------|
| "找不到视频文件" | 视频文件被删除/移动 | 检查输出目录 |
| "找不到字幕文件" | TRANSLATE 阶段未完成 | 确认 translated.srt 存在 |
| "GPU 不可用" | CUDA 未安装/显存不足 | 检查 nvidia-smi |
| "NVENC 不支持" | 驱动版本过低/显卡不支持 | 更新驱动，检查显卡型号 |
| "字幕样式丢失" | 字体文件缺失 | 检查 fonts_dir 配置 |
| "输出文件过大" | 码率参数不当 | 调整 crf / 检查原视频码率 |

### 9.2 Debug Checklist

1. **确认输入文件存在**
   - `_find_video_file()` 能否找到视频
   - `translated.srt` 是否存在且非空

2. **确认 GPU 可用**
   - `nvidia-smi` 是否正常
   - 显存是否足够（至少 1GB）

3. **确认 FFmpeg 配置**
   - `ffmpeg -encoders | grep nvenc` 检查 NVENC 支持
   - 字体目录和样式目录是否配置正确

4. **查看详细日志**
   - FFmpeg 的 stderr 输出
   - 进度回调信息

---

## 10. 修改指南：如果你想改某个功能...

| 如果你想... | 应该看/改哪里 |
|-------------|---------------|
| 改默认嵌入模式 | `config.embedder.embed_mode` |
| 改编码器参数 | `FFmpegWrapper.embed_subtitle_hard()` |
| 改样式缩放逻辑 | `FFmpegWrapper._scale_ass_style()` |
| 添加新容器格式 | `embed_subtitle_soft()` / `embed_subtitle_hard()` |
| 改字幕布局 | `ASRData.to_ass()` 方法 |
| 改自动换行逻辑 | `vat/asr/subtitle/__init__.py` `auto_wrap_ass_file()` |
| 添加新编码器 | `embed_subtitle_hard()` 中的编码器映射 |
