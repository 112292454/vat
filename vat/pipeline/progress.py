"""
Pipeline 进度追踪系统

设计原则：
1. 总进度 = 各阶段进度之和（按阶段数量平均分配权重）
2. 每个阶段有独立的内部进度（0-100%）
3. 阶段内部进度基于标志性事件计算
4. 支持动态阶段列表（任务可能只执行部分阶段）
"""
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass, field
from enum import Enum


class ProgressEvent(Enum):
    """进度事件类型"""
    # Download 阶段事件
    DOWNLOAD_INFO_FETCHED = "download_info_fetched"      # 获取视频信息 (20%)
    DOWNLOAD_VIDEO_DONE = "download_video_done"          # 视频下载完成 (60%)
    DOWNLOAD_TRANSLATE_DONE = "download_translate_done"  # 视频信息翻译完成 (20%)
    
    # Whisper 阶段事件
    WHISPER_MODEL_LOADED = "whisper_model_loaded"        # 模型加载完成 (10%)
    WHISPER_AUDIO_EXTRACTED = "whisper_audio_extracted"  # 音频提取完成 (10%)
    WHISPER_CHUNK_DONE = "whisper_chunk_done"            # 单个chunk完成 (动态)
    WHISPER_TRANSCRIBE_DONE = "whisper_transcribe_done"  # 转录完成 (剩余)
    
    # Split 阶段事件
    SPLIT_CHUNK_DONE = "split_chunk_done"                # 分块完成 (动态)
    
    # Optimize 阶段事件
    OPTIMIZE_BATCH_DONE = "optimize_batch_done"          # 批次完成 (动态)
    
    # Translate 阶段事件
    TRANSLATE_BATCH_DONE = "translate_batch_done"        # 批次完成 (动态)
    
    # Embed 阶段事件
    EMBED_PROGRESS = "embed_progress"                    # FFmpeg 进度 (动态)


@dataclass
class StageProgress:
    """单个阶段的进度信息"""
    stage: str                      # 阶段名称
    total_items: int = 1            # 总项目数（用于动态计算）
    completed_items: int = 0        # 已完成项目数
    base_progress: float = 0.0      # 基础进度（固定事件累计）
    
    def get_progress(self) -> float:
        """获取阶段进度 (0-1)"""
        if self.total_items <= 0:
            return self.base_progress
        
        # 动态进度 = 已完成 / 总数 * 剩余空间
        remaining = 1.0 - self.base_progress
        dynamic_progress = (self.completed_items / self.total_items) * remaining
        return min(1.0, self.base_progress + dynamic_progress)


class ProgressTracker:
    """
    Pipeline 进度追踪器
    
    使用示例：
        tracker = ProgressTracker(
            stages=['download', 'whisper', 'split', 'translate', 'embed'],
            callback=lambda p, msg: print(f"{p:.1%} - {msg}")
        )
        
        tracker.start_stage('download')
        tracker.report_event(ProgressEvent.DOWNLOAD_INFO_FETCHED)
        tracker.report_event(ProgressEvent.DOWNLOAD_VIDEO_DONE)
        tracker.complete_stage('download')
        
        tracker.start_stage('whisper')
        tracker.set_total_items(3)  # 3个chunk
        tracker.report_event(ProgressEvent.WHISPER_CHUNK_DONE)
        ...
    """
    
    def __init__(
        self,
        stages: List[str],
        callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        初始化进度追踪器
        
        Args:
            stages: 要执行的阶段列表（决定权重分配）
            callback: 进度回调函数 (progress: 0-1, message: str)
        """
        self.stages = stages
        self.callback = callback
        
        # 每个阶段的权重（平均分配）
        self.stage_weight = 1.0 / len(stages) if stages else 0
        
        # 阶段进度状态
        self._stage_progress: Dict[str, StageProgress] = {}
        self._completed_stages: List[str] = []
        self._current_stage: Optional[str] = None
        
        # 初始化所有阶段
        for stage in stages:
            self._stage_progress[stage] = StageProgress(stage=stage)
    
    def get_overall_progress(self) -> float:
        """获取总体进度 (0-1)"""
        if not self.stages:
            return 0.0
        
        total = 0.0
        for i, stage in enumerate(self.stages):
            if stage in self._completed_stages:
                # 已完成阶段贡献满权重
                total += self.stage_weight
            elif stage == self._current_stage:
                # 当前阶段贡献部分权重
                stage_prog = self._stage_progress.get(stage)
                if stage_prog:
                    total += self.stage_weight * stage_prog.get_progress()
        
        return min(1.0, total)
    
    def start_stage(self, stage: str):
        """开始一个阶段"""
        self._current_stage = stage
        if stage not in self._stage_progress:
            self._stage_progress[stage] = StageProgress(stage=stage)
        self._notify(f"开始阶段: {stage}")
    
    def complete_stage(self, stage: str):
        """完成一个阶段"""
        if stage not in self._completed_stages:
            self._completed_stages.append(stage)
        
        # 确保阶段进度为100%
        if stage in self._stage_progress:
            self._stage_progress[stage].base_progress = 1.0
            self._stage_progress[stage].completed_items = self._stage_progress[stage].total_items
        
        self._notify(f"阶段完成: {stage}")
        
        if stage == self._current_stage:
            self._current_stage = None
    
    def set_total_items(self, total: int):
        """设置当前阶段的总项目数（用于动态进度）"""
        if self._current_stage and self._current_stage in self._stage_progress:
            self._stage_progress[self._current_stage].total_items = max(1, total)
    
    def increment_completed(self, count: int = 1):
        """增加已完成项目数"""
        if self._current_stage and self._current_stage in self._stage_progress:
            prog = self._stage_progress[self._current_stage]
            prog.completed_items = min(prog.total_items, prog.completed_items + count)
            self._notify()
    
    def set_stage_progress(self, progress: float, message: str = ""):
        """直接设置当前阶段的进度 (0-1)"""
        if self._current_stage and self._current_stage in self._stage_progress:
            prog = self._stage_progress[self._current_stage]
            prog.base_progress = min(1.0, max(0.0, progress))
            prog.completed_items = 0
            prog.total_items = 1
            self._notify(message)
    
    def report_event(self, event: ProgressEvent, message: str = ""):
        """
        报告进度事件
        
        Args:
            event: 进度事件类型
            message: 可选的进度消息
        """
        if not self._current_stage:
            return
        
        prog = self._stage_progress.get(self._current_stage)
        if not prog:
            return
        
        # 根据事件类型更新进度
        event_progress = self._get_event_progress(event)
        if event_progress > 0:
            prog.base_progress = min(1.0, prog.base_progress + event_progress)
        
        # 对于动态事件，增加完成计数
        if event in [
            ProgressEvent.WHISPER_CHUNK_DONE,
            ProgressEvent.SPLIT_CHUNK_DONE,
            ProgressEvent.OPTIMIZE_BATCH_DONE,
            ProgressEvent.TRANSLATE_BATCH_DONE,
        ]:
            prog.completed_items += 1
        
        self._notify(message)
    
    def report_embed_progress(self, percent: float, message: str = ""):
        """报告 FFmpeg 嵌入进度"""
        if self._current_stage and self._current_stage in self._stage_progress:
            prog = self._stage_progress[self._current_stage]
            prog.base_progress = min(1.0, percent / 100.0)
            self._notify(message)
    
    def _get_event_progress(self, event: ProgressEvent) -> float:
        """获取事件对应的进度增量"""
        # Download 阶段事件进度
        download_events = {
            ProgressEvent.DOWNLOAD_INFO_FETCHED: 0.2,
            ProgressEvent.DOWNLOAD_VIDEO_DONE: 0.6,
            ProgressEvent.DOWNLOAD_TRANSLATE_DONE: 0.2,
        }
        
        # Whisper 阶段固定事件进度
        whisper_events = {
            ProgressEvent.WHISPER_MODEL_LOADED: 0.1,
            ProgressEvent.WHISPER_AUDIO_EXTRACTED: 0.1,
            # WHISPER_CHUNK_DONE 是动态的，不在这里处理
        }
        
        if event in download_events:
            return download_events[event]
        if event in whisper_events:
            return whisper_events[event]
        
        return 0.0
    
    def _notify(self, message: str = ""):
        """通知进度更新"""
        if self.callback:
            progress = self.get_overall_progress()
            self.callback(progress, message)
    
    def get_progress_info(self) -> Dict:
        """获取详细进度信息"""
        return {
            "overall": self.get_overall_progress(),
            "current_stage": self._current_stage,
            "completed_stages": self._completed_stages.copy(),
            "stages": {
                stage: {
                    "progress": prog.get_progress(),
                    "completed_items": prog.completed_items,
                    "total_items": prog.total_items,
                }
                for stage, prog in self._stage_progress.items()
            }
        }
