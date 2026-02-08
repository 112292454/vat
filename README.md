# VAT — Video Auto Translator

> **🇨🇳 [中文文档 / Chinese Documentation](README_CN.md)** — 完整的中文使用说明与功能介绍

An end-to-end video translation automation system. Downloads videos from YouTube, performs speech recognition, intelligent sentence segmentation, LLM translation, subtitle embedding, and uploads to Bilibili — all fully automated.

<!-- TODO: Insert demo screenshot/GIF -->
<!-- ![Demo](docs/assets/demo.gif) -->

---

## What It Does

VAT's core is a **7-stage pipeline**:

```
YouTube URL / Local Video
    │
    ├─ 1. Download ─── Video + subtitles + metadata + scene detection
    ├─ 2. Whisper ──── faster-whisper ASR (chunked concurrent processing)
    ├─ 3. Split ────── LLM smart segmentation (fragments → complete sentences)
    ├─ 4. Optimize ─── LLM subtitle optimization (typo correction, term unification)
    ├─ 5. Translate ── LLM reflective translation (draft → reflect → refine)
    ├─ 6. Embed ────── Subtitle embedding (hardcoded GPU-accelerated / soft subs)
    └─ 7. Upload ───── Auto-upload to Bilibili (title templates, covers, collections)
```

Each stage is independently controllable: run only specific steps, skip completed ones, or force re-run. Resume from breakpoints after interruption.

---

## Key Features

### Speech Recognition (ASR)

- Based on **faster-whisper**, supports large-v3 and other models
- **Chunked concurrency**: Long videos are automatically split into segments for parallel transcription, with overlap handling during merge
- Word-level timestamps for precise downstream segmentation
- ASR post-processing: hallucination detection, repetition cleaning, Japanese punctuation normalization
- Optional vocal separation (Mel-Band-Roformer) for videos with background music

### Smart Segmentation

Based on [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) (see Acknowledgments).

Whisper output is typically fragmented and incomplete. VAT uses LLM to reorganize these fragments into complete, human-readable sentences:

- Chunked segmentation (long videos) and full-text segmentation (short videos)
- Configurable sentence length constraints (CJK/English separately)
- Scene-aware: different video types (gaming, chatting, singing, etc.) use different segmentation strategies

### Subtitle Translation

Based on [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) (see Acknowledgments).

- **Reflective translation** (based on Andrew Ng's methodology): draft → reflect → refine, significantly improving quality
- **Context management**: maintains context across batches for consistent terminology and style
- **Subtitle optimization**: auto-corrects typos and unifies terminology before translation
- **Scene prompts**: automatically loads specialized prompts based on video type (gaming, educational, chatting, etc.)
- **Custom prompts**: supports per-channel or per-content translation/optimization prompts
- Compatible with any OpenAI-format API (including locally deployed Ollama, etc.)

### Subtitle Embedding

- **Hardcoded subtitles**: GPU-accelerated (H.264/H.265/AV1), supports NVIDIA hardware encoding
- **Soft subtitles**: fast muxing, preserves original quality
- Built-in ASS style templates (default, educational, anime, vertical, etc.), supports custom styles
- Subtitles auto-scale based on video resolution

### Video Download

- Based on yt-dlp, supports YouTube videos and playlists
- Auto-downloads manual subtitles (can skip ASR when manual subtitles are detected)
- Automatic scene detection (gaming, chatting, singing, educational, etc.)
- Auto-translates video metadata (title, description, tags)

### Bilibili Upload

- Automated upload based on biliup
- Template system: title/description support variable substitution (channel name, translated title, etc.)
- Auto-fetches covers, recommends categories, generates tags
- Supports adding to collections

### Scheduling & Concurrency

- Multi-GPU task scheduling: automatically distributes videos across GPUs
- Step-level state tracking: each stage independently tracked, supports resumption
- Config snapshot caching: changing segmentation params only re-runs segmentation, not ASR

---

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (recommended; required for ASR and subtitle embedding)
- ffmpeg (system-level installation)
- LLM API (required for segmentation, translation, optimization)

### Installation

```bash
git clone <repo-url> && cd vat
pip install -r requirements.txt
pip install -e .
```

**Font files** (required for hardcoded subtitle rendering):

Fonts are not included in the repository (~65MB). Place the following fonts in `vat/resources/fonts/`:

| Font | Usage | Source |
|------|-------|--------|
| NotoSansCJKsc-VF.ttf | Default CJK font | [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+SC) |
| LXGWWenKai-Regular.ttf | Anime style | [LXGW WenKai](https://github.com/lxgw/LxgwWenKai) |
| ZCOOLKuaiLe-Regular.ttf | Educational style | [Google Fonts](https://fonts.google.com/specimen/ZCOOL+KuaiLe) |
| AlimamaFangYuanTiVF-Thin-2.ttf | Vertical style | [Alimama Fonts](https://fonts.alibabagroup.com/) |

Only NotoSansCJKsc-VF.ttf is needed if you only use the default style. (It is pre-installed on most Ubuntu systems.)

### Configuration

```bash
# Set LLM API Key
export VAT_LLM_APIKEY="your-api-key"

# Generate config file
vat init

# Edit configuration (paths, models, translation params, etc.)
vim config/config.yaml
```

Key configuration items:

| Config | Description |
|--------|-------------|
| `storage.work_dir` | Working directory (intermediate files) |
| `storage.output_dir` | Output directory (final videos) |
| `storage.models_dir` | Model files directory |
| `asr.model` | Whisper model (recommended: `large-v3`) |
| `asr.language` | Source language (e.g. `ja`) |
| `translator.llm.model` | LLM model for translation |
| `translator.llm.enable_reflect` | Enable reflective translation |
| `llm.api_key` | LLM API Key (supports `${ENV_VAR}` format) |
| `llm.base_url` | LLM API endpoint |
| `proxy.http_proxy` | Proxy settings |

See [`config/default.yaml`](config/default.yaml) for full configuration reference with comments.

### Usage

```bash
# One-click full pipeline (download → ASR → translate → embed)
vat pipeline --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Process a playlist
vat pipeline --playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Multi-GPU parallel processing
vat pipeline --url "URL" --gpus 0,1

# Run specific stages
vat process -v VIDEO_ID -s asr          # ASR only
vat process -v VIDEO_ID -s translate    # Translation only
vat process -v VIDEO_ID -s embed        # Embedding only

# Force re-run
vat process -v VIDEO_ID -s translate -f

# Check status
vat status
```

### Output Files

```
data/videos/<VIDEO_ID>/
├── <video>.mp4           # Original downloaded video
├── original_raw.srt      # Raw Whisper transcription
├── original.srt          # Segmented source subtitles
├── optimized.srt         # Optimized source subtitles
├── translated.srt        # Translated subtitles
├── translated.ass        # ASS format subtitles (styled)
└── final.mp4             # Final video with embedded subtitles
```

---

## Web UI

VAT provides a FastAPI-based Web UI for viewing video status, managing tasks, and editing subtitle files.

```bash
# Start WebUI
vat web
# or
python -m vat web --port 8080
```

<!-- TODO: Insert WebUI screenshots -->
<!-- ![WebUI Index](docs/assets/webui_index.png) -->
<!-- ![WebUI Detail](docs/assets/webui_detail.png) -->
<!-- ![WebUI Tasks](docs/assets/webui_tasks.png) -->

Features include:
- Video list with status overview (search, filter)
- Video detail page (task timeline, file preview)
- Online task creation and execution
- Subtitle file viewing and editing
- Playlist management and batch operations
- Bilibili upload configuration

See [WebUI Manual](docs/webui_manual.md) for detailed instructions.

---

## CLI Quick Reference

| Command | Description |
|---------|-------------|
| `vat pipeline -u URL` | Full pipeline (download to embed) |
| `vat process -v ID -s STAGES` | Fine-grained stage control |
| `vat download -u URL` | Download only |
| `vat asr -v ID` | ASR only |
| `vat translate -v ID` | Translation only |
| `vat embed -v ID` | Subtitle embedding only |
| `vat upload VIDEO_ID` | Upload to Bilibili |
| `vat playlist sync URL` | Sync playlist |
| `vat status` | View processing status |
| `vat clean -v ID` | Clean intermediate files |
| `vat bilibili login` | Bilibili login for cookies |

---

## Project Structure

```
vat/
├── asr/                  # Speech recognition module
│   ├── whisper_asr.py    #   faster-whisper wrapper
│   ├── chunked_asr.py    #   Chunked concurrent ASR
│   ├── split.py          #   LLM smart segmentation
│   ├── asr_post.py       #   Post-processing (hallucination/repetition)
│   └── vocal_separation/ #   Vocal separation
├── translator/           # Translation module
│   └── llm_translator.py #   LLM reflective translation engine
├── llm/                  # LLM infrastructure
│   ├── client.py         #   Unified LLM client
│   ├── scene_identifier.py # Scene detection
│   └── prompts/          #   Prompt management
├── embedder/             # Subtitle embedding module
│   └── ffmpeg_wrapper.py #   FFmpeg wrapper (soft/hard subs)
├── downloaders/          # Downloaders
├── uploaders/            # Uploaders (Bilibili)
├── pipeline/             # Pipeline orchestration
│   ├── executor.py       #   VideoProcessor (stage scheduling)
│   ├── scheduler.py      #   Multi-GPU scheduler
│   ├── progress.py       #   Progress tracking
│   └── exceptions.py     #   Unified exception hierarchy
├── web/                  # Web management UI
│   ├── app.py            #   FastAPI application
│   ├── deps.py           #   Shared dependencies
│   ├── routes/           #   API routes
│   └── templates/        #   Page templates
├── cli/                  # CLI commands
├── database.py           # SQLite data layer
├── config.py             # Configuration management
└── models.py             # Data model definitions
```

---

## Advanced Configuration

### Custom Prompts

Create prompt files under `vat/llm/prompts/custom/` and reference them in config:

```yaml
translator:
  llm:
    custom_prompt: "my_channel"          # Translation prompt
    optimize:
      custom_prompt: "my_channel"        # Optimization prompt
```

See [Prompt Optimization Guide](docs/prompt_optimization_guide.md) for writing guidelines.

### Scene Detection

VAT automatically identifies scene types (gaming, chatting, singing, educational, etc.) based on video title and description, loading corresponding scene prompts. Scene configs are defined in `vat/llm/scenes.yaml`.

### ASR Parameter Tuning

Different video types may require different ASR parameters:

- Gaming/Livestream: disable VAD, lower `no_speech_threshold`
- Voice-only (podcasts): enable VAD
- Heavy background music: consider enabling vocal separation

See [ASR Parameters Guide](docs/asr_parameters_guide.md) for details.

### GPU Allocation

See [GPU Allocation Spec](docs/gpu_allocation_spec.md) for multi-GPU task distribution strategies.

---

## Documentation

| Document | Content |
|----------|---------|
| [ASR Parameters Guide](docs/asr_parameters_guide.md) | Whisper parameter details and tuning |
| [ASR Evaluation Report](docs/ASR_EVALUATION_REPORT.md) | Recognition quality comparison across parameter sets |
| [Prompt Optimization Guide](docs/prompt_optimization_guide.md) | Translation/optimization prompt writing |
| [GPU Allocation Spec](docs/gpu_allocation_spec.md) | Multi-GPU scheduling strategy |
| [WebUI Manual](docs/webui_manual.md) | Web UI operation guide |
| [YouTube Subtitles](docs/youtube_manual_subtitles.md) | YouTube manual subtitle detection and usage |
| [Project Review](docs/project_review.md) | Architecture review and refactoring log |
| [Developer Manual](README_USAGE.md) | Per-stage execution details and dev reference |

---

## Acknowledgments

This project integrates core technologies from the following open-source projects:

- [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) — Core reference for chunked ASR, smart segmentation, reflective translation, ASS rendering
- [GalTransl](https://github.com/xd2333/GalTransl) — Translation engine
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — Speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — Video download
- [biliup](https://github.com/biliup/biliup) — Bilibili upload
- [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) — Vocal separation model

See [acknowledgement.md](acknowledgement.md) for detailed credits.

---

## License

GPLv3
