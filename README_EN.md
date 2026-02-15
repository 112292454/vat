# VAT — Video Auto Translator

> **🇨🇳 [中文文档 / Chinese Documentation](README.md)**

An end-to-end video translation automation pipeline. From YouTube download to speech recognition, intelligent sentence segmentation, LLM translation, subtitle embedding, and Bilibili upload — all fully automated.

Supports both **CLI** and **Web UI**. The CLI is the core capability layer; the Web UI is an enhancement for visual management — similar to Clash vs its Dashboard: all processing capabilities are fully independent of the Web UI.

<p align="center">
  <img src="docs/assets/webui_index.png" alt="Video Management Overview" width="80%">
</p>

---

## Development & Runtime Environment

> ⚠️ VAT is primarily developed and tested on **Linux multi-GPU servers** (Ubuntu 22.04, CUDA 12.x, multiple RTX 4090s).
>
> If you are running on a **Windows home PC**, you may encounter:
> - Path and environment variable differences for CUDA / FFmpeg / yt-dlp
> - Multi-GPU scheduling features have not been tested on Windows
> - Some dependencies (e.g., faster-whisper) may require extra steps to install on Windows
>
> Feedback on Windows compatibility issues is welcome.

---

## Design Goals

VAT is designed for **server-side batch video translation**, not as a single-video desktop tool.

### Architecture

- **CLI is the core**: all processing capabilities are exposed via the `vat` command, scriptable and integrable
- **WebUI is an enhancement**: WebUI executes tasks by spawning CLI subprocesses; the system works fully without it
- **Modular pipeline**: 7 independently controllable stages — run specific steps, skip completed ones, or force re-run. Resume from breakpoints after interruption
- **Embeddable**: the pipeline module can be imported directly by other Python projects

### Compared to Single-Video Tools

- **Batch management**: playlist-level incremental sync, time-ordered, batch processing for thousands of videos
- **Stage-level tracking**: each processing stage independently tracked, supports resumption and selective re-run
- **Concurrent scheduling**: multi-GPU task scheduling, parallel video processing
- **Crash recovery**: auto-detects orphaned tasks after process crashes, resumes on restart

---

## Pipeline

```
YouTube URL / Local Video
    │
    ├─ 1. Download ─── Video + subtitles + metadata + scene detection + info translation
    ├─ 2. Whisper ──── faster-whisper ASR (chunked concurrency, vocal separation)
    ├─ 3. Split ────── LLM smart segmentation (fragments → complete sentences, timestamp alignment)
    ├─ 4. Optimize ─── LLM subtitle optimization (typo correction, term unification)
    ├─ 5. Translate ── LLM reflective translation (draft → reflect → refine)
    ├─ 6. Embed ────── Subtitle embedding (GPU-accelerated hardcoded / soft subs)
    └─ 7. Upload ───── Auto-upload to Bilibili (title templates, covers, category recommendation)
```

---

## Stage Capabilities

### Speech Recognition (ASR)

Based on [faster-whisper](https://github.com/guillaumekln/faster-whisper), with extensive parameter testing for Japanese spoken-language scenarios (VTuber streams, gaming, etc.). See [ASR Evaluation Report](docs/ASR_EVALUATION_REPORT.md).

- **Chunked concurrency**: long videos are automatically split into segments for parallel transcription with overlap handling
- Word-level timestamps for precise downstream segmentation
- **ASR post-processing**: hallucination detection (removing repetitions, meaningless fixed phrases), Japanese punctuation normalization. Post-processing implementation references [WhisperJAV](https://github.com/meizhong986/WhisperJAV.git)
- **Vocal separation** (optional, disabled by default): based on [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) for videos with background music

### Smart Segmentation

Whisper output is typically fragmented and incomplete. This stage uses LLM to reorganize fragments into complete, human-readable sentences.

The segmentation approach is based on [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner), with significant changes to the timestamp alignment algorithm — the original implementation suffered from time drift on long videos; the current version has notably improved alignment accuracy.

- Chunked segmentation (long videos) and full-text segmentation (short videos)
- Configurable sentence length constraints (CJK / English separately)
- Scene-aware: different video types use different segmentation strategies

### Subtitle Optimization & Translation

The translation engine core comes from [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) (whose translation engine is based on [GalTransl](https://github.com/xd2333/GalTransl)):

- **Reflective translation** (based on Andrew Ng's methodology): draft → reflect → refine, significantly improving quality
- **Subtitle optimization**: auto-corrects typos and unifies terminology before translation
- **Scene prompts**: automatically loads specialized prompts based on video type (gaming, educational, chatting, etc.)
- **Custom prompts**: supports per-channel or per-content translation/optimization prompts

VAT's additions:
- **Context-aware processing**: maintains previous context across sequential chunks for consistent terminology and style
- More flexible model configuration

> **About local models**: configuration supports local LLM endpoints (e.g., Ollama), but no dedicated adaptation or testing has been done. Auto-launching local services (similar to GalTransl) is not implemented. Currently gpt-4o-mini is insufficient; gpt-5-nano is barely adequate — local models reaching this performance level remains difficult.

### Subtitle Embedding

- **Hardcoded subtitles**: GPU-accelerated (H.264/H.265/AV1), wrapping FFmpeg's NVIDIA hardware encoding
- **Soft subtitles**: fast muxing, preserves original quality
- Built-in ASS subtitle style templates (default, educational, anime, vertical, etc.), auto-scaled by video resolution

### Video Download

Based on [yt-dlp](https://github.com/yt-dlp/yt-dlp):

- YouTube videos and playlists
- Auto-downloads manual subtitles (can skip ASR when detected)
- **Playlist incremental sync**: time-ordered video management, subsequent syncs only fetch new videos
- LLM automatic scene detection (gaming, chatting, singing, educational, etc.)
- LLM automatic video info translation (title, description, tags)

### Bilibili Upload

Based on [biliup](https://github.com/biliup/biliup):

- Template system: title/description support variable substitution (channel name, translated title, etc.)
- Auto-fetches covers, generates tags
- LLM-recommended Bilibili categories
- Supports adding to collections (⚠️ Known issue: video upload works, but adding to collections is sometimes unstable)
- Scheduled upload: cron expression support for timed uploads, one video per trigger (CLI `--upload-cron` / WebUI visual config)

### Scheduling & Concurrency

- Multi-GPU task scheduling: auto-detects GPUs, distributes videos across them
- Step-level state tracking: each stage independently tracked, supports resumption
- Config snapshot caching: changing segmentation params only re-runs segmentation, not ASR
- Multi-video parallel processing (configurable concurrency)

---

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (recommended; required for ASR and hardcoded subtitle embedding)
- ffmpeg (system-level installation)
- LLM API (required for segmentation, translation, optimization; any OpenAI-format API)

### Installation

```bash
git clone <repo-url> && cd vat
pip install -r requirements.txt
pip install -e .
```

> **About `vat` vs `python -m vat`**: after installation, the `vat` command is available directly. If `vat` is not available (PATH issues or frequent dev iterations), use `python -m vat` instead — they are equivalent. Examples in this document use `vat` but can be replaced with `python -m vat`.

**Font files** (optional):

Place fonts in `vat/resources/fonts/`. Most Ubuntu systems already include NotoSansCJK used by the default style.

| Font | Usage | Source |
|------|-------|--------|
| NotoSansCJKsc-VF.ttf | Default CJK font | [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+SC) |
| LXGWWenKai-Regular.ttf | Anime style (optional) | [LXGW WenKai](https://github.com/lxgw/LxgwWenKai) |
| ZCOOLKuaiLe-Regular.ttf | Educational style (optional) | [Google Fonts](https://fonts.google.com/specimen/ZCOOL+KuaiLe) |
| AlimamaFangYuanTiVF-Thin-2.ttf | Vertical style (optional) | [Alimama Fonts](https://fonts.alibabagroup.com/) |

### Configuration

```bash
# Set LLM API Key (environment variable)
export VAT_LLM_APIKEY="your-api-key"

# Generate config file
vat init

# Edit configuration (paths, models, translation params, etc.)
nano config/config.yaml
```

Key configuration items:

| Config | Description |
|--------|-------------|
| `storage.work_dir` | Working directory (intermediate files) |
| `storage.output_dir` | Output directory (final videos) |
| `storage.models_dir` | Model files directory |
| `asr.model` | Whisper model (recommended: `large-v3`) |
| `asr.language` | Source language (e.g., `ja`) |
| `llm.api_key` | Global LLM API Key (supports `${ENV_VAR}` env var reference) |
| `llm.base_url` | Global LLM API endpoint |
| `translator.llm.model` | LLM model for translation |
| `translator.llm.enable_reflect` | Enable reflective translation |

Each stage (split, translate, optimize) supports independent `api_key` / `base_url` / `model` overrides; empty values inherit from global config. See [`config/default.yaml`](config/default.yaml) for full reference.

---

## CLI Usage

### One-Click Processing

```bash
# Full pipeline (download → ASR → split → optimize → translate → embed)
vat pipeline --url "https://www.youtube.com/watch?v=VIDEO_ID"
# Or equivalently: python -m vat pipeline --url "..."

# Process a playlist
vat pipeline --playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Multi-GPU parallel
vat pipeline --url "URL" --gpus 0,1
```

### Per-Stage Execution

```bash
# Option 1: specify stage list (comma-separated)
vat process -v VIDEO_ID -s download,whisper,split

# Option 2: single stage
vat process -v VIDEO_ID -s translate

# Option 3: shortcut commands
vat download -u URL              # Download only
vat asr -v VIDEO_ID              # ASR only
vat translate -v VIDEO_ID        # Translation only
vat embed -v VIDEO_ID            # Subtitle embedding only

# Force re-run (ignore completed status)
vat process -v VIDEO_ID -s translate -f

# Specify GPU
vat process -v VIDEO_ID -s whisper -g cuda:1

# Multi-video parallel
vat process -v VID1 -v VID2 -v VID3 -s download,whisper -c 3
```

> **Stage skipping**: if non-consecutive stages are specified (e.g., `whisper,embed` to test Japanese source embedding), the system auto-fills intermediate stages in "passthrough mode" (copying input to output), marked as `SKIPPED`.

### Playlist Management

```bash
# Sync playlist
vat playlist sync "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Check status
vat status
```

### Command Quick Reference

| Command | Description |
|---------|-------------|
| `vat pipeline -u URL` | Full pipeline (download to embed) |
| `vat process -v ID -s STAGES` | Fine-grained stage control |
| `vat download -u URL` | Download only |
| `vat asr -v ID` | ASR only |
| `vat translate -v ID` | Translation only |
| `vat embed -v ID` | Subtitle embedding only |
| `vat upload VIDEO_ID` | Upload to Bilibili |
| `vat process -v ID -s upload --upload-cron "0 12,18 * * *"` | Scheduled upload (daily 12/18) |
| `vat playlist sync URL` | Sync playlist |
| `vat status` | View processing status |
| `vat clean -v ID` | Clean intermediate files |
| `vat bilibili login` | Bilibili login for cookies |

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

The Web UI is a visual wrapper around CLI capabilities. All tasks execute via CLI subprocesses, fully decoupled from the web server lifecycle — restarting the web server does not affect running tasks.

```bash
# Start (default port 8094)
vat web
# Or specify port
vat web --port 8080
```

### Video Management

The video list page provides a global overview: status statistics, search/filter (by title/channel/status/stage/playlist), and paginated browsing. Each video shows thumbnail, source, duration, 7-stage status, progress, and publish date.

<p align="center">
  <img src="docs/assets/webui_index.png" alt="Video List" width="90%">
</p>

The video detail page shows the complete processing timeline, translation info, and related files (with inline viewing, subtitle editing, and video playback). You can directly trigger a specific stage or force re-run.

<p align="center">
  <img src="docs/assets/webui_video_detail.png" alt="Video Detail" width="90%">
</p>
<p align="center">
  <img src="docs/assets/webui_video_detail2.png" alt="Video Detail" width="90%">
</p>

### Playlist Management

Supports adding YouTube Playlists with incremental sync. Videos are sorted by publish date, with batch processing and range selection. Each Playlist can have independent translation/optimization prompts and Bilibili upload parameters.

<p align="center">
  <img src="docs/assets/webui_playlists.png" alt="Playlist List" width="90%">
</p>

<p align="center">
  <img src="docs/assets/webui_playlist_detail.png" alt="Playlist Detail" width="90%">
</p>

### Task Management

When creating tasks, you can select videos, execution stages, GPU device, and concurrency. Running tasks provide real-time logs (SSE push) and progress tracking. Supports cancel, retry, and batch delete.

<p align="center">
  <img src="docs/assets/webui_tasks.png" alt="Task List" width="90%">
</p>

<p align="center">
  <img src="docs/assets/webui_task_new.png" alt="New Task" width="90%">
</p>

See [WebUI Manual](docs/webui_manual.md) for detailed instructions.

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
│   └── prompts/          #   Prompt management (built-in + custom)
├── embedder/             # Subtitle embedding module
│   └── ffmpeg_wrapper.py #   FFmpeg wrapper (soft/hard subs + GPU accel)
├── downloaders/          # Downloaders (yt-dlp)
├── uploaders/            # Uploaders (Bilibili biliup)
├── pipeline/             # Pipeline orchestration
│   ├── executor.py       #   VideoProcessor (stage scheduling)
│   ├── scheduler.py      #   Multi-GPU scheduler
│   └── progress.py       #   Progress tracking
├── web/                  # Web management UI
│   ├── app.py            #   FastAPI app + page routes
│   ├── jobs.py           #   Job manager (subprocess scheduling)
│   ├── routes/           #   API routes
│   └── templates/        #   Jinja2 + TailwindCSS templates
├── cli/                  # CLI commands (click)
├── services/             # Business logic (Playlist service, etc.)
├── database.py           # SQLite data layer (WAL mode)
├── config.py             # Configuration management (YAML + env vars)
└── models.py             # Data model definitions
```

Each module directory contains its own documentation.

---

## Advanced Configuration

### Custom Prompts

Create prompt files under `vat/llm/prompts/custom/` and reference the filename in config:

```yaml
translator:
  llm:
    custom_prompt: "my_channel"          # Translation prompt
    optimize:
      custom_prompt: "my_channel"        # Optimization prompt
```

Prompts can also be created and edited via the WebUI Prompts management page. See [Prompt Optimization Guide](docs/prompt_optimization_guide.md).

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
| [ASR Evaluation Report](docs/ASR_EVALUATION_REPORT.md) | Parameter evaluation across 350 VTuber videos |
| [Prompt Optimization Guide](docs/prompt_optimization_guide.md) | Translation/optimization prompt writing |
| [GPU Allocation Spec](docs/gpu_allocation_spec.md) | Multi-GPU scheduling strategy |
| [WebUI Manual](docs/webui_manual.md) | Web UI operation guide |
| [YouTube Subtitles](docs/youtube_manual_subtitles.md) | YouTube manual subtitle detection and usage |
| [Subtitle Style Guide](docs/subtitle_style_guide.md) | ASS subtitle style template guide |
| [Known Issues](docs/known_issues.md) | Known limitations and LLM cost reference |

---

## Acknowledgments

This project integrates content from or wraps calls to the following open-source projects:

- [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) — Core reference for chunked ASR, smart segmentation, reflective translation, ASS rendering. VAT's segmentation and translation modules are modified and extended from this project
- [GalTransl](https://github.com/xd2333/GalTransl) — Inspiration source
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — Speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — Video download
- [biliup](https://github.com/biliup/biliup) — Bilibili upload
- [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) — Vocal separation model
- [WhisperJAV](https://github.com/meizhong986/WhisperJAV.git) — ASR post-processing reference (hallucination detection, repetition cleaning)

See [acknowledgement.md](docs/acknowledgement.md) for detailed credits.

---

## Support

If you find this project useful, consider buying me a coffee:

<a href="https://buymeacoffee.com/112292454" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="180"></a>

---

## License

GPLv3
