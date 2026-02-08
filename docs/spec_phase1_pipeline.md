# VAT Pipeline 改动需求规划文档

> **文档版本**: v1.0  
> **创建日期**: 2026-02-02  
> **状态**: 待实施

---

## 目录

1. [概述](#一概述)
2. [GPU 调度机制改造](#二gpu-调度机制改造)
3. [子阶段独立化](#三子阶段独立化)
4. [Playlist 管理功能](#四playlist-管理功能)
5. [实施计划](#五实施计划)

---

## 一、概述

### 1.1 本阶段目标

Pipeline 阶段的改动目标是**优化核心处理逻辑**，使其：
1. GPU 调度更智能、更可控
2. 子阶段（如 ASR 的 split、翻译的 optimize）成为独立可执行的 stage
3. Playlist 管理功能完善，支持增量同步和正确的排序

### 1.2 设计原则

1. **Pipeline 自洽**：核心处理逻辑不因外部调用方式（CLI/Web）而改变
2. **接口统一**：CLI 和 Web 使用完全相同的底层接口，保证行为等价
3. **最小侵入**：不在核心代码中添加 `if web_mode` 之类的条件判断

---

## 二、GPU 调度机制改造

### 2.1 现状问题

**问题描述**：
- 无论如何设置 `CUDA_VISIBLE_DEVICES` 或命令行参数，始终使用 GPU 0
- GPU 调用发生在两个阶段：ASR（Whisper 模型推理）和 Embed（FFmpeg 硬件加速）
- 当前的 GPU 设置方式不够灵活，无法自动选择空闲 GPU

**涉及代码**：
- `vat/pipeline/executor.py`: `VideoProcessor.__init__()` 设置 `CUDA_VISIBLE_DEVICES`
- `vat/pipeline/scheduler.py`: `MultiGPUScheduler._worker()` 设置 `CUDA_VISIBLE_DEVICES`
- `vat/asr/whisper_asr.py`: Whisper 模型加载
- `vat/embedder/ffmpeg_wrapper.py`: FFmpeg 硬件加速调用

### 2.2 改造方案

#### 2.2.1 GPU 选择策略

```python
# 新增文件：vat/utils/gpu.py

import subprocess
from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class GPUInfo:
    """GPU 信息"""
    index: int
    name: str
    memory_total: int      # MB
    memory_used: int       # MB
    memory_free: int       # MB
    utilization: int       # 0-100%

def get_available_gpus() -> List[GPUInfo]:
    """
    获取所有可用 GPU 的信息
    
    Returns:
        GPU 信息列表，按 index 排序
    
    Raises:
        RuntimeError: 如果无法获取 GPU 信息（CUDA 不可用）
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi 不可用，请确认 CUDA 环境已正确安装")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi 执行失败: {e.stderr}")
    
    gpus = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        gpus.append(GPUInfo(
            index=int(parts[0]),
            name=parts[1],
            memory_total=int(parts[2]),
            memory_used=int(parts[3]),
            memory_free=int(parts[4]),
            utilization=int(parts[5]),
        ))
    return gpus

def select_best_gpu(exclude: Optional[List[int]] = None) -> int:
    """
    选择最佳 GPU（显存利用率最低）
    
    Args:
        exclude: 要排除的 GPU index 列表
        
    Returns:
        最佳 GPU 的 index
        
    Raises:
        RuntimeError: 如果没有可用 GPU
    """
    gpus = get_available_gpus()
    exclude = exclude or []
    
    available = [g for g in gpus if g.index not in exclude]
    if not available:
        raise RuntimeError(f"没有可用的 GPU（排除列表: {exclude}）")
    
    # 按显存利用率排序（低到高），利用率相同则按 index
    best = min(available, key=lambda g: (g.memory_used / g.memory_total, g.index))
    return best.index

def validate_gpu(gpu_id: int) -> bool:
    """
    验证指定 GPU 是否可用
    
    Args:
        gpu_id: GPU index
        
    Returns:
        是否可用
    """
    try:
        gpus = get_available_gpus()
        return any(g.index == gpu_id for g in gpus)
    except RuntimeError:
        return False
```

#### 2.2.2 GPU 调用点改造

**原则**：
- 每个实际使用 GPU 的操作点（ASR、Embed）独立决定使用哪个 GPU
- 支持显式指定 GPU，也支持自动选择
- 不依赖全局的 `CUDA_VISIBLE_DEVICES` 环境变量

**ASR 阶段改造** (`vat/asr/whisper_asr.py`)：

```python
class WhisperASR:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",  # "auto" | "cuda" | "cuda:0" | "cpu"
        # ... 其他参数
    ):
        """
        初始化 Whisper ASR
        
        Args:
            device: 设备选择
                - "auto": 自动选择显存最空闲的 GPU
                - "cuda": 等同于 "auto"
                - "cuda:N": 使用指定的 GPU N
                - "cpu": 使用 CPU（不推荐）
        """
        self.device = self._resolve_device(device)
        # ...
    
    def _resolve_device(self, device: str) -> str:
        """解析设备字符串，返回实际使用的设备"""
        if device == "cpu":
            return "cpu"
        
        if device in ("auto", "cuda"):
            from vat.utils.gpu import select_best_gpu
            gpu_id = select_best_gpu()
            return f"cuda:{gpu_id}"
        
        if device.startswith("cuda:"):
            # 验证指定的 GPU 是否可用
            gpu_id = int(device.split(":")[1])
            from vat.utils.gpu import validate_gpu
            if not validate_gpu(gpu_id):
                raise RuntimeError(f"指定的 GPU {gpu_id} 不可用")
            return device
        
        raise ValueError(f"无效的 device 参数: {device}")
```

**Embed 阶段改造** (`vat/embedder/ffmpeg_wrapper.py`)：

```python
class FFmpegWrapper:
    def __init__(self, gpu_id: Optional[int] = None):
        """
        初始化 FFmpeg 封装器
        
        Args:
            gpu_id: 使用的 GPU ID，None 表示自动选择
        """
        self.gpu_id = gpu_id
    
    def _get_gpu_for_encoding(self) -> int:
        """获取用于硬件编码的 GPU"""
        if self.gpu_id is not None:
            return self.gpu_id
        
        from vat.utils.gpu import select_best_gpu
        return select_best_gpu()
    
    def embed_subtitle(self, video_path, subtitle_path, output_path, **kwargs):
        """嵌入字幕"""
        gpu_id = self._get_gpu_for_encoding()
        
        # 使用 NVENC 硬件编码
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-i', str(video_path),
            '-vf', f"subtitles={subtitle_path}",
            '-c:v', 'h264_nvenc',
            '-gpu', str(gpu_id),
            # ... 其他参数
        ]
        # ...
```

#### 2.2.3 配置层改造

**config.yaml 增加 GPU 配置**：

```yaml
# GPU 配置
gpu:
  # 设备选择模式: "auto" | "manual"
  # auto: 自动选择显存最空闲的 GPU
  # manual: 使用 devices 列表中指定的设备
  mode: auto
  
  # 手动模式下使用的 GPU 列表
  devices: [0, 1]
  
  # 是否允许回退到 CPU（不推荐）
  allow_cpu_fallback: false
```

**Config 类增加**：

```python
@dataclass
class GPUConfig:
    """GPU 配置"""
    mode: str = "auto"  # "auto" | "manual"
    devices: List[int] = field(default_factory=lambda: [0])
    allow_cpu_fallback: bool = False
```

#### 2.2.4 CLI 参数支持

```bash
# 指定 GPU
vat pipeline --gpu 1 video_id

# 自动选择（默认）
vat pipeline video_id

# 多 GPU 并行（使用指定的 GPU 列表）
vat pipeline --gpu 0,1,2 --multi-gpu video_id1 video_id2 video_id3
```

### 2.3 验证要点

1. **功能验证**：
   - [ ] `nvidia-smi` 不可用时的错误处理
   - [ ] 指定不存在的 GPU 时的错误处理
   - [ ] 自动选择能正确选中显存最空闲的 GPU
   - [ ] 手动指定 GPU 能正确生效

2. **集成验证**：
   - [ ] ASR 阶段能在指定 GPU 上运行
   - [ ] Embed 阶段能在指定 GPU 上运行
   - [ ] 多视频并行时各进程使用不同 GPU

---

## 三、子阶段独立化

### 3.1 现状问题

**当前结构**：
```
TaskStep (主阶段)          SubPhase (子阶段，仅用于失败标识)
├─ DOWNLOAD                 └─ MAIN
├─ ASR                      ├─ WHISPER
│                           └─ SPLIT
├─ TRANSLATE                ├─ OPTIMIZE
│                           └─ TRANSLATE_LLM
├─ EMBED                    └─ MAIN
└─ UPLOAD                   └─ MAIN
```

**问题**：
- 子阶段（SubPhase）仅用于记录失败位置，不能独立执行
- 无法单独重跑 `split` 或 `optimize`
- 状态只在主阶段级别追踪，子阶段进度不可见

### 3.2 改造方案

#### 3.2.1 新的阶段定义

将子阶段提升为**一级阶段**，可独立执行：

```python
# vat/models.py

class TaskStep(Enum):
    """处理步骤（一级阶段）"""
    # 下载
    DOWNLOAD = "download"
    
    # ASR 阶段（原来的子阶段独立出来）
    WHISPER = "whisper"      # Whisper 语音识别
    SPLIT = "split"          # 智能断句
    
    # 翻译阶段（原来的子阶段独立出来）
    OPTIMIZE = "optimize"    # 字幕优化
    TRANSLATE = "translate"  # LLM 翻译
    
    # 嵌入和上传
    EMBED = "embed"
    UPLOAD = "upload"


# 阶段组定义（用于简化操作）
STAGE_GROUPS = {
    "asr": [TaskStep.WHISPER, TaskStep.SPLIT],
    "translate": [TaskStep.OPTIMIZE, TaskStep.TRANSLATE],
    "all": [TaskStep.DOWNLOAD, TaskStep.WHISPER, TaskStep.SPLIT, 
            TaskStep.OPTIMIZE, TaskStep.TRANSLATE, TaskStep.EMBED, TaskStep.UPLOAD],
}

# 阶段依赖关系
STAGE_DEPENDENCIES = {
    TaskStep.DOWNLOAD: [],
    TaskStep.WHISPER: [TaskStep.DOWNLOAD],
    TaskStep.SPLIT: [TaskStep.WHISPER],
    TaskStep.OPTIMIZE: [TaskStep.SPLIT],
    TaskStep.TRANSLATE: [TaskStep.OPTIMIZE],
    TaskStep.EMBED: [TaskStep.TRANSLATE],
    TaskStep.UPLOAD: [TaskStep.EMBED],
}

# 默认阶段序列（完整流程）
DEFAULT_STAGE_SEQUENCE = [
    TaskStep.DOWNLOAD,
    TaskStep.WHISPER,
    TaskStep.SPLIT,
    TaskStep.OPTIMIZE,
    TaskStep.TRANSLATE,
    TaskStep.EMBED,
    TaskStep.UPLOAD,
]
```

#### 3.2.2 数据库表结构调整

**tasks 表保持不变**，但 `step` 字段的值变为新的细粒度阶段：

```sql
-- 原来的值: download, asr, translate, embed, upload
-- 新的值: download, whisper, split, optimize, translate, embed, upload
```

**数据迁移**：
```sql
-- 将旧的 asr 阶段根据 sub_phase 拆分
UPDATE tasks SET step = 'whisper' WHERE step = 'asr' AND (sub_phase = 'whisper' OR sub_phase IS NULL);
UPDATE tasks SET step = 'split' WHERE step = 'asr' AND sub_phase = 'split';

-- 将旧的 translate 阶段根据 sub_phase 拆分
UPDATE tasks SET step = 'optimize' WHERE step = 'translate' AND sub_phase = 'optimize';
UPDATE tasks SET step = 'translate' WHERE step = 'translate' AND (sub_phase = 'translate_llm' OR sub_phase IS NULL);

-- sub_phase 字段可以废弃或保留用于更细粒度的标识
```

#### 3.2.3 Executor 改造

```python
# vat/pipeline/executor.py

class VideoProcessor:
    """单个视频的处理器"""
    
    def process(
        self,
        steps: Optional[List[str]] = None,
        force: bool = False,
        expand_groups: bool = True,  # 是否展开阶段组
    ) -> bool:
        """
        执行处理流程
        
        Args:
            steps: 要执行的步骤列表
                - None: 执行所有未完成的步骤
                - ["asr"]: 展开为 ["whisper", "split"]（如果 expand_groups=True）
                - ["whisper"]: 只执行 whisper
            force: 是否强制重新执行
            expand_groups: 是否将阶段组展开为子阶段
        """
        # 解析和展开步骤
        if steps is None:
            steps = self._get_pending_steps()
        else:
            steps = self._resolve_steps(steps, expand_groups)
        
        # 验证依赖关系
        self._validate_dependencies(steps)
        
        # 执行
        for step in steps:
            success = self._execute_step(step, force)
            if not success:
                return False
        return True
    
    def _resolve_steps(
        self,
        steps: List[str],
        expand_groups: bool
    ) -> List[TaskStep]:
        """
        解析步骤名称，支持阶段组展开
        
        Examples:
            ["asr"] -> [WHISPER, SPLIT]  (expand_groups=True)
            ["asr"] -> ValueError        (expand_groups=False, "asr" 不是有效阶段)
            ["whisper"] -> [WHISPER]
            ["whisper", "split"] -> [WHISPER, SPLIT]
        """
        result = []
        for step_name in steps:
            if expand_groups and step_name in STAGE_GROUPS:
                result.extend(STAGE_GROUPS[step_name])
            else:
                try:
                    result.append(TaskStep(step_name))
                except ValueError:
                    raise ValueError(
                        f"无效的阶段名称: {step_name}。"
                        f"有效值: {[s.value for s in TaskStep]} 或阶段组 {list(STAGE_GROUPS.keys())}"
                    )
        return result
    
    def _validate_dependencies(self, steps: List[TaskStep]) -> None:
        """
        验证阶段依赖关系
        
        Raises:
            ValueError: 如果依赖的阶段未完成
        """
        for step in steps:
            deps = STAGE_DEPENDENCIES.get(step, [])
            for dep in deps:
                if dep not in steps and not self.db.is_step_completed(self.video_id, dep):
                    raise ValueError(
                        f"阶段 {step.value} 依赖于 {dep.value}，"
                        f"但 {dep.value} 尚未完成。请先执行 {dep.value} 或使用阶段组。"
                    )
    
    def _execute_step(self, step: TaskStep, force: bool = False) -> bool:
        """执行单个步骤"""
        handlers = {
            TaskStep.DOWNLOAD: self._download,
            TaskStep.WHISPER: self._whisper,
            TaskStep.SPLIT: self._split,
            TaskStep.OPTIMIZE: self._optimize,
            TaskStep.TRANSLATE: self._translate_llm,
            TaskStep.EMBED: self._embed,
            TaskStep.UPLOAD: self._upload,
        }
        
        handler = handlers.get(step)
        if handler is None:
            raise ValueError(f"未知步骤: {step}")
        
        return handler(force=force)
    
    # ==================== 各阶段实现 ====================
    
    def _whisper(self, force: bool = False) -> bool:
        """Whisper 语音识别"""
        # 原 _run_asr 的 whisper 部分
        # 输出: asr_result.json (原始 whisper 结果)
        pass
    
    def _split(self, force: bool = False) -> bool:
        """智能断句"""
        # 原 _run_asr 的 split 部分
        # 输入: asr_result.json
        # 输出: original.srt
        pass
    
    def _optimize(self, force: bool = False) -> bool:
        """字幕优化"""
        # 原 _translate 的 optimize 部分
        # 输入: original.srt
        # 输出: optimized.srt (优化后的原文字幕)
        pass
    
    def _translate_llm(self, force: bool = False) -> bool:
        """LLM 翻译"""
        # 原 _translate 的翻译部分
        # 输入: optimized.srt
        # 输出: translated.srt
        pass
```

#### 3.2.4 CLI 改造

```bash
# 执行完整 ASR（whisper + split）
vat asr video_id
# 等价于
vat pipeline --steps whisper,split video_id

# 只执行 whisper
vat pipeline --steps whisper video_id

# 只重跑 split（whisper 结果已存在）
vat pipeline --steps split --force video_id

# 执行完整翻译（optimize + translate）
vat translate video_id
# 等价于
vat pipeline --steps optimize,translate video_id

# 只重跑翻译（跳过 optimize）
vat pipeline --steps translate --force video_id

# 查看阶段依赖关系
vat stages --show-deps
# 输出:
# download     -> (无依赖)
# whisper      -> download
# split        -> whisper
# optimize     -> split
# translate    -> optimize
# embed        -> translate
# upload       -> embed
```

#### 3.2.5 中间产物文件约定

```
output_dir/
├── video.mp4                    # download 阶段产物
├── asr_result.json              # whisper 阶段产物（原始识别结果）
├── original.srt                 # split 阶段产物（断句后的原文字幕）
├── optimized.srt                # optimize 阶段产物（优化后的原文字幕）
├── translated.srt               # translate 阶段产物（翻译后的字幕）
├── translated.ass               # embed 阶段预产物（ASS 格式）
├── embedded.mp4                 # embed 阶段产物（嵌入字幕的视频）
└── cache/                       # 各阶段的缓存文件
    ├── whisper_cache.json
    ├── split_cache.json
    └── translate_cache.json
```

### 3.3 验证要点

1. **功能验证**：
   - [ ] 各阶段可独立执行
   - [ ] 阶段组（asr, translate）能正确展开
   - [ ] 依赖验证能正确阻止无效操作
   - [ ] force 参数能正确触发重新执行

2. **兼容性验证**：
   - [ ] 旧数据库迁移后状态正确
   - [ ] CLI 命令向后兼容

---

## 四、Playlist 管理功能

### 4.1 需求描述

1. **导入 Playlist**：从 YouTube 播放列表 URL 导入所有视频
2. **正确排序**：视频按时间顺序排列（旧 → 新）
3. **增量同步**：只添加新视频，不重复导入
4. **范围处理**：支持处理指定范围的视频
5. **Playlist 级别管理**：可以查看、管理整个 Playlist

### 4.2 数据模型

#### 4.2.1 Playlist 模型

```python
# vat/models.py

@dataclass
class Playlist:
    """播放列表信息"""
    id: str                              # playlist URL 的 hash
    source_type: SourceType              # youtube
    source_url: str                      # playlist URL
    title: Optional[str] = None          # playlist 标题
    channel: Optional[str] = None        # 频道名
    video_count: int = 0                 # 视频数量
    last_synced_at: Optional[datetime] = None  # 最后同步时间
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
```

#### 4.2.2 Video 模型扩展

```python
# vat/models.py

@dataclass
class Video:
    """视频信息"""
    id: str
    source_type: SourceType
    source_url: str
    title: Optional[str] = None
    output_dir: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 新增字段
    playlist_id: Optional[str] = None    # 关联的 playlist
    playlist_index: Optional[int] = None # 在 playlist 中的序号（1-based，旧→新）
```

### 4.3 数据库改造

#### 4.3.1 新增 playlists 表

```sql
CREATE TABLE IF NOT EXISTS playlists (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_url TEXT NOT NULL UNIQUE,
    title TEXT,
    channel TEXT,
    video_count INTEGER DEFAULT 0,
    last_synced_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON
);
```

#### 4.3.2 videos 表增加字段

```sql
ALTER TABLE videos ADD COLUMN playlist_id TEXT REFERENCES playlists(id);
ALTER TABLE videos ADD COLUMN playlist_index INTEGER;

-- 索引
CREATE INDEX IF NOT EXISTS idx_videos_playlist ON videos(playlist_id, playlist_index);
```

#### 4.3.3 Database 类新增方法

```python
# vat/database.py

class Database:
    # ==================== Playlist 操作 ====================
    
    def add_playlist(self, playlist: Playlist) -> None:
        """添加或更新 Playlist"""
        pass
    
    def get_playlist(self, playlist_id: str) -> Optional[Playlist]:
        """获取 Playlist"""
        pass
    
    def get_playlist_by_url(self, url: str) -> Optional[Playlist]:
        """根据 URL 获取 Playlist"""
        pass
    
    def list_playlists(self) -> List[Playlist]:
        """列出所有 Playlist"""
        pass
    
    def delete_playlist(self, playlist_id: str, delete_videos: bool = False) -> None:
        """
        删除 Playlist
        
        Args:
            playlist_id: Playlist ID
            delete_videos: 是否同时删除关联的视频
        """
        pass
    
    def get_videos_by_playlist(
        self,
        playlist_id: str,
        order_by: str = "playlist_index"
    ) -> List[Video]:
        """
        获取 Playlist 下的所有视频
        
        Args:
            playlist_id: Playlist ID
            order_by: 排序方式，"playlist_index" 或 "created_at"
        """
        pass
    
    def update_video_playlist_info(
        self,
        video_id: str,
        playlist_id: str,
        playlist_index: int
    ) -> None:
        """更新视频的 Playlist 关联信息"""
        pass
    
    def get_playlist_video_ids(self, playlist_id: str) -> Set[str]:
        """获取 Playlist 下所有视频的 ID 集合（用于增量同步）"""
        pass
```

### 4.4 Downloader 改造

```python
# vat/downloaders/youtube.py

class YouTubeDownloader:
    
    def get_playlist_info(self, playlist_url: str) -> Dict[str, Any]:
        """
        获取 Playlist 元信息
        
        Returns:
            {
                'id': 'PLxxxxxx',
                'title': 'Playlist 标题',
                'channel': '频道名',
                'channel_id': 'UCxxxxxx',
                'video_count': 100,
                'videos': [
                    {
                        'id': 'video_id_1',
                        'title': '视频标题',
                        'upload_date': '20240101',  # YYYYMMDD
                        'duration': 3600,
                    },
                    # ...
                ]
            }
        """
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'logger': YtDlpLogger(),
        }
        if self.proxy:
            ydl_opts['proxy'] = self.proxy
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            
            if info is None:
                raise RuntimeError(f"无法获取 Playlist 信息: {playlist_url}")
            
            videos = []
            for entry in info.get('entries', []):
                if entry and 'id' in entry:
                    videos.append({
                        'id': entry['id'],
                        'title': entry.get('title', ''),
                        'upload_date': entry.get('upload_date', ''),
                        'duration': entry.get('duration', 0),
                    })
            
            return {
                'id': info.get('id', ''),
                'title': info.get('title', ''),
                'channel': info.get('uploader', ''),
                'channel_id': info.get('uploader_id', ''),
                'video_count': len(videos),
                'videos': videos,
            }
    
    def get_playlist_urls(
        self,
        playlist_url: str,
        sort_by: str = "upload_date",
        ascending: bool = True,
    ) -> List[Tuple[str, int]]:
        """
        获取 Playlist 中的所有视频 URL，按指定方式排序
        
        Args:
            playlist_url: Playlist URL
            sort_by: 排序字段 "upload_date" | "playlist_order"
            ascending: 是否升序（True = 旧→新）
            
        Returns:
            [(video_url, playlist_index), ...]
            playlist_index 从 1 开始，表示排序后的位置
        """
        info = self.get_playlist_info(playlist_url)
        videos = info['videos']
        
        # 排序
        if sort_by == "upload_date":
            videos.sort(key=lambda v: v.get('upload_date', ''), reverse=not ascending)
        # playlist_order 保持原顺序
        
        # 生成结果
        result = []
        for idx, video in enumerate(videos, start=1):
            url = f"https://www.youtube.com/watch?v={video['id']}"
            result.append((url, idx))
        
        return result
```

### 4.5 Playlist 服务层

```python
# vat/services/playlist_service.py

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

from vat.models import Playlist, Video, SourceType
from vat.database import Database
from vat.downloaders import YouTubeDownloader
from vat.config import Config
from vat.utils.logger import setup_logger

logger = setup_logger("playlist_service")

@dataclass
class SyncResult:
    """同步结果"""
    total_in_playlist: int      # Playlist 中的视频总数
    already_exists: int         # 已存在的视频数
    newly_added: int            # 新添加的视频数
    new_video_ids: List[str]    # 新添加的视频 ID 列表


class PlaylistService:
    """Playlist 管理服务"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.storage.database_path)
        self.downloader = YouTubeDownloader(
            proxy=config.downloader.youtube.proxy
        )
    
    def import_playlist(
        self,
        playlist_url: str,
        sync: bool = True,
    ) -> Tuple[Playlist, SyncResult]:
        """
        导入 Playlist
        
        Args:
            playlist_url: Playlist URL
            sync: 是否执行增量同步（True）或全量导入（False）
            
        Returns:
            (Playlist, SyncResult)
        """
        # 获取 Playlist 信息
        logger.info(f"获取 Playlist 信息: {playlist_url}")
        info = self.downloader.get_playlist_info(playlist_url)
        
        # 创建或获取 Playlist 记录
        playlist_id = self._generate_playlist_id(playlist_url)
        existing = self.db.get_playlist(playlist_id)
        
        playlist = Playlist(
            id=playlist_id,
            source_type=SourceType.YOUTUBE,
            source_url=playlist_url,
            title=info['title'],
            channel=info['channel'],
            video_count=info['video_count'],
        )
        
        # 获取已存在的视频 ID
        existing_video_ids: Set[str] = set()
        if sync and existing:
            existing_video_ids = self.db.get_playlist_video_ids(playlist_id)
        
        # 获取视频列表（按时间排序，旧→新）
        video_list = self.downloader.get_playlist_urls(
            playlist_url,
            sort_by="upload_date",
            ascending=True,
        )
        
        # 导入视频
        newly_added = []
        for video_url, playlist_index in video_list:
            video_id = self.downloader.extract_video_id(video_url)
            
            if video_id in existing_video_ids:
                # 已存在，只更新 playlist_index
                self.db.update_video_playlist_info(video_id, playlist_id, playlist_index)
                continue
            
            # 新视频，创建记录
            video = Video(
                id=video_id,
                source_type=SourceType.YOUTUBE,
                source_url=video_url,
                playlist_id=playlist_id,
                playlist_index=playlist_index,
            )
            self.db.add_video(video)
            newly_added.append(video_id)
            logger.info(f"添加视频 [{playlist_index}]: {video_id}")
        
        # 更新 Playlist 记录
        playlist.last_synced_at = datetime.now()
        self.db.add_playlist(playlist)
        
        result = SyncResult(
            total_in_playlist=len(video_list),
            already_exists=len(existing_video_ids),
            newly_added=len(newly_added),
            new_video_ids=newly_added,
        )
        
        logger.info(
            f"同步完成: 总计 {result.total_in_playlist}, "
            f"已存在 {result.already_exists}, 新增 {result.newly_added}"
        )
        
        return playlist, result
    
    def _generate_playlist_id(self, url: str) -> str:
        """生成 Playlist ID"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    def get_videos_in_range(
        self,
        playlist_id: str,
        range_spec: Optional[str] = None,
    ) -> List[Video]:
        """
        获取指定范围的视频
        
        Args:
            playlist_id: Playlist ID
            range_spec: 范围规格，如 "1-10", "5-", "-20", None（全部）
            
        Returns:
            视频列表
        """
        videos = self.db.get_videos_by_playlist(playlist_id)
        
        if range_spec is None:
            return videos
        
        start, end = self._parse_range(range_spec, len(videos))
        return videos[start:end]
    
    def _parse_range(self, range_spec: str, total: int) -> Tuple[int, int]:
        """
        解析范围规格
        
        Args:
            range_spec: "1-10", "5-", "-20"
            total: 总数
            
        Returns:
            (start_index, end_index) 0-based, 用于切片
        """
        if '-' not in range_spec:
            # 单个数字，如 "5" -> 第5个视频
            idx = int(range_spec)
            return (idx - 1, idx)
        
        parts = range_spec.split('-')
        start = int(parts[0]) - 1 if parts[0] else 0
        end = int(parts[1]) if parts[1] else total
        
        return (max(0, start), min(total, end))
```

### 4.6 CLI 命令

```python
# vat/cli/commands.py

@cli.group()
def playlist():
    """Playlist 管理命令"""
    pass


@playlist.command('add')
@click.argument('url')
@click.option('--no-sync', is_flag=True, help='不执行增量同步，全量导入')
@pass_config
def playlist_add(config, url, no_sync):
    """
    导入 YouTube Playlist
    
    URL: Playlist URL
    """
    from vat.services.playlist_service import PlaylistService
    
    service = PlaylistService(config)
    playlist, result = service.import_playlist(url, sync=not no_sync)
    
    click.echo(f"\n{'='*50}")
    click.echo(f"Playlist: {playlist.title}")
    click.echo(f"频道: {playlist.channel}")
    click.echo(f"{'='*50}")
    click.echo(f"视频总数: {result.total_in_playlist}")
    click.echo(f"已存在: {result.already_exists}")
    click.echo(f"新增: {result.newly_added}")
    
    if result.new_video_ids:
        click.echo(f"\n新增视频 ID:")
        for vid in result.new_video_ids[:10]:
            click.echo(f"  - {vid}")
        if len(result.new_video_ids) > 10:
            click.echo(f"  ... 共 {len(result.new_video_ids)} 个")


@playlist.command('list')
@pass_config
def playlist_list(config):
    """列出所有 Playlist"""
    db = Database(config.storage.database_path)
    playlists = db.list_playlists()
    
    if not playlists:
        click.echo("没有已导入的 Playlist")
        return
    
    for pl in playlists:
        synced = pl.last_synced_at.strftime('%Y-%m-%d %H:%M') if pl.last_synced_at else '从未'
        click.echo(f"\n[{pl.id[:8]}] {pl.title}")
        click.echo(f"  频道: {pl.channel}")
        click.echo(f"  视频数: {pl.video_count}")
        click.echo(f"  最后同步: {synced}")


@playlist.command('show')
@click.argument('playlist_id')
@click.option('--range', 'video_range', help='显示指定范围，如 1-10')
@pass_config
def playlist_show(config, playlist_id, video_range):
    """显示 Playlist 详情和视频列表"""
    from vat.services.playlist_service import PlaylistService
    
    service = PlaylistService(config)
    videos = service.get_videos_in_range(playlist_id, video_range)
    
    if not videos:
        click.echo("没有找到视频")
        return
    
    for v in videos:
        status = "✓" if _is_completed(config, v.id) else "○"
        click.echo(f"[{v.playlist_index:3d}] {status} {v.id} - {v.title or '(未获取标题)'}")


@playlist.command('process')
@click.argument('playlist_id')
@click.option('--range', 'video_range', help='处理指定范围，如 1-10, 5-, -20')
@click.option('--steps', help='执行的阶段，如 asr,translate')
@click.option('--force', is_flag=True, help='强制重新处理')
@pass_config
def playlist_process(config, playlist_id, video_range, steps, force):
    """处理 Playlist 中的视频"""
    from vat.services.playlist_service import PlaylistService
    from vat.pipeline import schedule_videos
    
    service = PlaylistService(config)
    videos = service.get_videos_in_range(playlist_id, video_range)
    
    if not videos:
        click.echo("没有找到要处理的视频")
        return
    
    video_ids = [v.id for v in videos]
    step_list = steps.split(',') if steps else None
    
    click.echo(f"将处理 {len(video_ids)} 个视频")
    schedule_videos(config, video_ids, step_list, force=force)


@playlist.command('sync')
@click.argument('playlist_id')
@pass_config
def playlist_sync(config, playlist_id):
    """同步更新 Playlist（增量）"""
    from vat.services.playlist_service import PlaylistService
    
    db = Database(config.storage.database_path)
    playlist = db.get_playlist(playlist_id)
    
    if not playlist:
        click.echo(f"未找到 Playlist: {playlist_id}")
        return
    
    service = PlaylistService(config)
    _, result = service.import_playlist(playlist.source_url, sync=True)
    
    click.echo(f"同步完成: 新增 {result.newly_added} 个视频")
```

### 4.7 验证要点

1. **功能验证**：
   - [ ] Playlist 导入能正确获取所有视频
   - [ ] 视频排序正确（旧→新）
   - [ ] 增量同步只添加新视频
   - [ ] playlist_index 正确分配
   - [ ] 范围处理正确

2. **边界情况**：
   - [ ] 空 Playlist
   - [ ] 私有/删除的视频
   - [ ] 超长 Playlist（1000+ 视频）

---

## 五、实施计划

### 5.1 实施顺序

```
Week 1: 子阶段独立化
├─ Day 1-2: 模型和数据库改造
│   ├─ 更新 TaskStep 枚举
│   ├─ 添加 STAGE_GROUPS 和 STAGE_DEPENDENCIES
│   ├─ 数据库迁移脚本
│   └─ 单元测试
├─ Day 3-4: Executor 改造
│   ├─ 拆分 _run_asr 为 _whisper + _split
│   ├─ 拆分 _translate 为 _optimize + _translate_llm
│   ├─ 实现依赖验证
│   └─ 集成测试
└─ Day 5: CLI 改造和测试
    ├─ 更新 CLI 命令
    └─ 端到端测试

Week 2: GPU 调度改造
├─ Day 1: GPU 工具模块
│   ├─ 实现 vat/utils/gpu.py
│   └─ 单元测试
├─ Day 2-3: ASR 和 Embed 改造
│   ├─ WhisperASR 设备选择
│   ├─ FFmpegWrapper GPU 选择
│   └─ 集成测试
└─ Day 4-5: 配置和 CLI
    ├─ 添加 GPU 配置
    ├─ CLI 参数支持
    └─ 多 GPU 验证

Week 3: Playlist 功能
├─ Day 1-2: 数据模型和数据库
│   ├─ Playlist 模型
│   ├─ 数据库表和方法
│   └─ 单元测试
├─ Day 3: Downloader 改造
│   ├─ get_playlist_info
│   ├─ 排序逻辑
│   └─ 测试
└─ Day 4-5: 服务层和 CLI
    ├─ PlaylistService
    ├─ CLI 命令
    └─ 集成测试
```

### 5.2 验收标准

#### 子阶段独立化
- [ ] `vat pipeline --steps whisper video_id` 只执行 whisper
- [ ] `vat pipeline --steps split video_id` 只执行 split（whisper 已完成）
- [ ] `vat asr video_id` 执行 whisper + split
- [ ] 依赖未满足时报错提示清晰

#### GPU 调度
- [ ] 不指定 GPU 时自动选择显存最空闲的
- [ ] `--gpu 1` 能正确使用 GPU 1
- [ ] GPU 不可用时报错而非静默回退 CPU

#### Playlist
- [ ] `vat playlist add <url>` 导入成功
- [ ] 视频按旧→新排序
- [ ] 重复执行只添加新视频
- [ ] `vat playlist process <id> --range 1-10` 正确处理

---

## 附录

### A. 现有代码文件清单

| 文件 | 需要改动 | 改动内容 |
|-----|---------|---------|
| `vat/models.py` | 是 | TaskStep 扩展，新增 Playlist 模型 |
| `vat/database.py` | 是 | 新增 Playlist 表和方法 |
| `vat/pipeline/executor.py` | 是 | 拆分阶段，GPU 设备参数 |
| `vat/pipeline/scheduler.py` | 是 | GPU 分配逻辑 |
| `vat/asr/whisper_asr.py` | 是 | 设备选择逻辑 |
| `vat/embedder/ffmpeg_wrapper.py` | 是 | GPU 选择逻辑 |
| `vat/downloaders/youtube.py` | 是 | Playlist 信息获取 |
| `vat/cli/commands.py` | 是 | 新增 playlist 命令组 |
| `vat/config.py` | 是 | 新增 GPU 配置 |
| `vat/utils/gpu.py` | 新增 | GPU 工具函数 |
| `vat/services/playlist_service.py` | 新增 | Playlist 服务 |

### B. 数据库迁移脚本

```python
# migrations/001_stage_split.py

def upgrade(db):
    """将旧的 asr/translate 阶段拆分为子阶段"""
    cursor = db.cursor()
    
    # 备份
    cursor.execute("CREATE TABLE tasks_backup AS SELECT * FROM tasks")
    
    # 拆分 asr
    cursor.execute("""
        UPDATE tasks SET step = 'whisper' 
        WHERE step = 'asr' AND (sub_phase = 'whisper' OR sub_phase IS NULL)
    """)
    cursor.execute("""
        UPDATE tasks SET step = 'split' 
        WHERE step = 'asr' AND sub_phase = 'split'
    """)
    
    # 拆分 translate
    cursor.execute("""
        UPDATE tasks SET step = 'optimize' 
        WHERE step = 'translate' AND sub_phase = 'optimize'
    """)
    cursor.execute("""
        UPDATE tasks SET step = 'translate' 
        WHERE step = 'translate' AND (sub_phase = 'translate_llm' OR sub_phase IS NULL)
    """)
    
    db.commit()

def downgrade(db):
    """回滚"""
    cursor = db.cursor()
    cursor.execute("DROP TABLE tasks")
    cursor.execute("ALTER TABLE tasks_backup RENAME TO tasks")
    db.commit()
```

---

**文档结束**
