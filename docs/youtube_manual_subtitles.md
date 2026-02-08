# YouTube人工字幕机制调研

## 1. 字幕类型区分

YouTube视频字幕分为两类：

| 类型 | yt-dlp显示 | 来源 | 质量 |
|------|-----------|------|------|
| **人工字幕** | `Available subtitles` | 上传者/贡献者添加 | 高 |
| **自动字幕** | `Available automatic captions` | YouTube AI生成 | 中低 |

### yt-dlp输出示例

```
[info] Available automatic captions for xxx:
Language Name                  Formats
ja       Japanese              vtt, srt, ...
en-ja    English from Japanese vtt, srt, ...  # 自动翻译

[info] Available subtitles for xxx:
Language Name     Formats
ja       Japanese vtt, srt, ...  # 人工字幕
```

---

## 2. 人工字幕的来源

### 2.1 创作者上传
- 视频上传者自己添加字幕
- 通过YouTube Studio上传SRT/VTT文件
- 质量最高，与视频完全匹配

### 2.2 社区贡献 (已停止)
- **注意**: YouTube于2020年9月停止了社区字幕贡献功能
- 之前的社区贡献字幕仍然保留
- 新视频不再有社区贡献字幕

### 2.3 第三方服务
- 部分创作者使用专业字幕服务
- 如: Rev, Amara等

---

## 3. 检测与使用方法

### 3.1 检测是否有人工字幕
```bash
# 方法1: 使用yt-dlp
yt-dlp --list-subs "URL" 2>&1 | grep "Available subtitles"

# 方法2: Python代码
import subprocess
result = subprocess.run(
    ['yt-dlp', '--list-subs', url],
    capture_output=True, text=True
)
has_manual_subs = 'Available subtitles' in result.stdout
```

### 3.2 只下载人工字幕
```bash
# --write-sub 只下载人工字幕（不包括自动生成）
yt-dlp --write-sub --sub-lang ja -o "%(id)s.%(ext)s" "URL"

# 如果没有人工字幕，不会下载任何字幕
```

### 3.3 下载自动字幕（备选）
```bash
# --write-auto-sub 下载自动生成字幕
yt-dlp --write-auto-sub --sub-lang ja -o "%(id)s.%(ext)s" "URL"
```

---

## 4. 系统集成建议

### 4.1 下载流程优化

```
下载视频
    │
    ▼
检查是否有人工字幕 (--list-subs)
    │
    ├─ 有 ──► 下载人工字幕 ──► 标记为 "manual_subtitle"
    │                              │
    │                              ▼
    │                         跳过ASR识别
    │                              │
    │                              ▼
    │                         直接用于翻译
    │
    └─ 无 ──► 运行ASR识别 ──► 标记为 "asr_generated"
                                   │
                                   ▼
                              Split断句 + 翻译
```

### 4.2 数据库字段建议

```yaml
video:
  subtitle_source: "manual" | "asr" | "auto_caption"
  has_manual_subtitle: true | false
  subtitle_language: "ja"
```

### 4.3 代码实现要点

```python
def check_manual_subtitles(video_url: str, lang: str = "ja") -> bool:
    """检查视频是否有指定语言的人工字幕"""
    result = subprocess.run(
        ['yt-dlp', '--list-subs', video_url],
        capture_output=True, text=True
    )
    
    # 解析输出
    lines = result.stdout.split('\n')
    in_manual_section = False
    
    for line in lines:
        if 'Available subtitles' in line:
            in_manual_section = True
        elif 'Available automatic' in line:
            in_manual_section = False
        elif in_manual_section and line.strip().startswith(lang):
            return True
    
    return False
```

---

## 5. 注意事项

### 5.1 人工字幕不一定是原文
- 有些视频的"人工字幕"是翻译字幕（如英文视频配日文翻译）
- 需要检查字幕语言与视频语言是否匹配

### 5.2 字幕质量差异
- 不同创作者的字幕质量参差不齐
- 部分可能有错别字或时间轴偏差
- 但整体仍优于自动生成

### 5.3 可用性
- 大部分日文VTuber视频没有人工字幕
- 有人工字幕的通常是：
  - 官方频道的精选视频
  - 专业剪辑频道
  - 教育类/新闻类视频

---

## 6. 实现状态

### 已实现 ✓
1. **downloader检测** (`vat/downloaders/youtube.py`)
   - `check_manual_subtitles()` 方法检测人工字幕
   - 下载时自动获取字幕类型信息

2. **metadata记录** (`vat/pipeline/executor.py`)
   - `subtitle_source`: "manual" | "auto" | "asr"
   - `manual_subtitle_path`: 人工字幕文件路径

3. **ASR跳过逻辑** (`vat/pipeline/executor.py::_run_whisper`)
   - 检测到人工字幕时直接使用，跳过Whisper
   - 自动将VTT转换为SRT格式

### 待实现
- **Phase 4**: Web UI显示字幕来源标识

---

## 7. 参考资料

- YouTube官方帮助: https://support.google.com/youtube/answer/2734796
- yt-dlp字幕选项: https://github.com/yt-dlp/yt-dlp#subtitle-options
- YouTube社区字幕停止公告: https://support.google.com/youtube/answer/6052538
