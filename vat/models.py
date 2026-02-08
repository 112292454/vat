"""
数据模型定义

子阶段独立化设计：
- 每个子阶段都是独立可执行的 stage
- 支持细粒度的任务控制和断点续传
- 阶段组定义了默认执行顺序
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from enum import Enum


class SourceType(Enum):
    """视频来源类型"""
    YOUTUBE = "youtube"
    LOCAL = "local"
    BILIBILI = "bilibili"


class TaskStep(Enum):
    """
    处理步骤（细粒度阶段）
    
    每个子阶段都是独立可执行的 stage，支持：
    - 单独执行某个子阶段
    - 从某个子阶段继续执行
    - 阶段组批量执行
    """
    # 下载阶段（单一步骤）
    DOWNLOAD = "download"
    
    # ASR 阶段组
    WHISPER = "whisper"      # 语音识别（Whisper 模型推理）
    SPLIT = "split"          # 智能断句（LLM 辅助）
    
    # 翻译阶段组
    OPTIMIZE = "optimize"    # 字幕优化（错别字、术语统一）
    TRANSLATE = "translate"  # LLM 翻译
    
    # 嵌入阶段（单一步骤）
    EMBED = "embed"          # 字幕嵌入（FFmpeg）
    
    # 上传阶段（单一步骤）
    UPLOAD = "upload"        # 上传到平台


# 阶段组定义：将子阶段组织成逻辑分组
STAGE_GROUPS = {
    "download": [TaskStep.DOWNLOAD],
    "asr": [TaskStep.WHISPER, TaskStep.SPLIT],
    "translate": [TaskStep.OPTIMIZE, TaskStep.TRANSLATE],
    "embed": [TaskStep.EMBED],
    "upload": [TaskStep.UPLOAD],
}

# 阶段依赖关系：每个阶段需要哪些前置阶段完成
STAGE_DEPENDENCIES: Dict[TaskStep, List[TaskStep]] = {
    TaskStep.DOWNLOAD: [],
    TaskStep.WHISPER: [TaskStep.DOWNLOAD],
    TaskStep.SPLIT: [TaskStep.WHISPER],
    TaskStep.OPTIMIZE: [TaskStep.SPLIT],
    TaskStep.TRANSLATE: [TaskStep.OPTIMIZE],
    TaskStep.EMBED: [TaskStep.TRANSLATE],
    TaskStep.UPLOAD: [TaskStep.EMBED],
}

# 默认执行顺序
DEFAULT_STAGE_SEQUENCE = [
    TaskStep.DOWNLOAD,
    TaskStep.WHISPER,
    TaskStep.SPLIT,
    TaskStep.OPTIMIZE,
    TaskStep.TRANSLATE,
    TaskStep.EMBED,
    TaskStep.UPLOAD,
]


def expand_stage_group(stage_or_group: str) -> List[TaskStep]:
    """
    展开阶段组为子阶段列表
    
    Args:
        stage_or_group: 阶段名或阶段组名
            - "asr" -> [WHISPER, SPLIT]
            - "whisper" -> [WHISPER]
            
    Returns:
        TaskStep 列表
    """
    # 先尝试作为阶段组
    if stage_or_group.lower() in STAGE_GROUPS:
        return STAGE_GROUPS[stage_or_group.lower()]
    
    # 尝试作为单个阶段
    try:
        return [TaskStep(stage_or_group.lower())]
    except ValueError:
        raise ValueError(f"未知的阶段或阶段组: {stage_or_group}")


def get_required_stages(target_stages: List[TaskStep]) -> List[TaskStep]:
    """
    获取执行目标阶段所需的完整阶段列表（包含依赖）
    
    Args:
        target_stages: 目标阶段列表
        
    Returns:
        按执行顺序排列的完整阶段列表
    """
    required: Set[TaskStep] = set()
    
    def add_with_deps(stage: TaskStep):
        if stage in required:
            return
        for dep in STAGE_DEPENDENCIES.get(stage, []):
            add_with_deps(dep)
        required.add(stage)
    
    for stage in target_stages:
        add_with_deps(stage)
    
    # 按默认顺序排序
    return [s for s in DEFAULT_STAGE_SEQUENCE if s in required]


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # 跳过（依赖未满足或显式跳过）


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
    # Playlist 关联
    playlist_id: Optional[str] = None  # 所属 Playlist ID
    playlist_index: Optional[int] = None  # 在 Playlist 中的索引（按时间排序，最早=1）
    
    def __post_init__(self):
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Task:
    """
    处理任务
    
    每个 Task 对应一个细粒度的阶段（如 WHISPER, SPLIT, OPTIMIZE 等）
    """
    video_id: str
    step: TaskStep  # 细粒度阶段
    status: TaskStatus
    id: Optional[int] = None
    gpu_id: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.step, str):
            self.step = TaskStep(self.step)
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)


@dataclass
class Playlist:
    """Playlist 信息"""
    id: str  # Playlist ID (YouTube playlist ID)
    title: str
    source_url: str
    channel: Optional[str] = None
    channel_id: Optional[str] = None
    video_count: int = 0
    last_synced_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
