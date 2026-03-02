# VAT 模块文档：Subtitle Utils（字幕工具）

> 字幕处理通用工具模块，提供文本对齐和格式/枚举定义。
>
> 被 Translator、Embedder 等模块依赖。

---

## 1. 模块组成

| 文件 | 职责 |
|------|------|
| `alignment.py` | 字幕文本对齐器 `SubtitleAligner`：基于 difflib 的序列匹配 |
| `entities.py` | 枚举和数据类定义：字幕格式、音视频格式、布局模式、质量等级等 |

---

## 2. SubtitleAligner（alignment.py）

基于 `difflib.ndiff` 的文本序列对齐器，用于对齐翻译前后的字幕行。

### 使用场景

翻译阶段可能改变字幕行数（合并/拆分），需要将翻译结果对齐回原始时间戳。当目标序列缺少某项时，使用上一项填充。

### 示例

```python
from vat.subtitle_utils import SubtitleAligner

aligner = SubtitleAligner()
source = ['ab', 'b', 'c', 'd', 'e', 'f']
target = ['a',  'b', 'c', 'd', 'f']      # 缺少 'e'

aligned_src, aligned_tgt = aligner.align_texts(source, target)
# aligned_tgt: ['a', 'b', 'c', 'd', 'd', 'f']  ← 缺失的 'e' 由 'd' 填充
```

---

## 3. 枚举定义（entities.py）

### 数据类

| 类 | 说明 |
|----|------|
| `SubtitleProcessData` | 字幕处理数据（index, original_text, translated_text, optimized_text） |

### 格式枚举

| 枚举 | 说明 |
|------|------|
| `SupportedAudioFormats` | 支持的音频格式（wav, mp3, flac 等 18 种） |
| `SupportedVideoFormats` | 支持的视频格式（mp4, mkv, webm 等 20 种） |
| `SupportedSubtitleFormats` | 支持的字幕格式（srt, ass, vtt） |
| `OutputSubtitleFormatEnum` | 字幕输出格式（srt, ass, vtt, json, txt） |
| `ASROutputFormatEnum` | 转录输出格式 |

### 配置枚举

| 枚举 | 说明 |
|------|------|
| `SubtitleLayoutEnum` | 字幕布局（译文在上/原文在上/仅原文/仅译文） |
| `SubtitleRenderModeEnum` | 渲染模式（ASS 样式 / 圆角背景） |
| `VideoQualityEnum` | 视频质量（极高/高/中/低，映射到 CRF 值） |
| `VadMethodEnum` | VAD 方法（silero v3/v4/v5, pyannote, webrtc, auditok） |
| `ASRModelEnum` | ASR 模型选项 |
| `LLMServiceEnum` | LLM 服务商 |
| `TranslatorServiceEnum` | 翻译器服务 |
