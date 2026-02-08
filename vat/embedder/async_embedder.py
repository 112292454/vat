"""
异步字幕嵌入队列管理系统

用于将嵌字任务与主流程解耦，实现：
1. 主流程完成下载、转录、翻译后立即进入下一个任务
2. 嵌字任务在后台异步处理
3. 支持多GPU并行嵌字
"""
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum
import json

from ..embedder.ffmpeg_wrapper import FFmpegWrapper
from ..database import SessionLocal, Video, ProcessingStatus


class EmbedTaskStatus(str, Enum):
    """嵌字任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EmbedTask:
    """嵌字任务"""
    video_id: int
    video_path: Path
    subtitle_path: Path
    output_path: Path
    
    # 编码参数
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 23
    preset: str = "medium"
    use_gpu: bool = True
    gpu_id: int = 0
    
    # 任务状态
    status: EmbedTaskStatus = EmbedTaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'video_id': self.video_id,
            'video_path': str(self.video_path),
            'subtitle_path': str(self.subtitle_path),
            'output_path': str(self.output_path),
            'video_codec': self.video_codec,
            'audio_codec': self.audio_codec,
            'crf': self.crf,
            'preset': self.preset,
            'use_gpu': self.use_gpu,
            'gpu_id': self.gpu_id,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }


class AsyncEmbedderQueue:
    """异步嵌字队列管理器"""
    
    def __init__(
        self,
        gpu_devices: List[int] = [0],
        max_concurrent_per_gpu: int = 1,
        persist_file: Optional[Path] = None
    ):
        """
        初始化异步嵌字队列
        
        Args:
            gpu_devices: 可用GPU设备列表
            max_concurrent_per_gpu: 每个GPU同时处理的任务数
            persist_file: 任务持久化文件路径（用于重启后恢复任务）
        """
        self.gpu_devices = gpu_devices
        self.max_concurrent_per_gpu = max_concurrent_per_gpu
        self.persist_file = persist_file
        
        # 任务队列
        self.pending_tasks: asyncio.Queue = asyncio.Queue()
        self.processing_tasks: Dict[str, EmbedTask] = {}
        self.completed_tasks: Dict[str, EmbedTask] = {}
        
        # GPU工作器
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        # 统计信息
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_processing_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.ffmpeg = FFmpegWrapper()
    
    async def start(self):
        """启动队列处理"""
        if self.running:
            self.logger.warning("队列已在运行中")
            return
        
        self.running = True
        self.logger.info(f"启动异步嵌字队列，GPU设备: {self.gpu_devices}")
        
        # 从持久化文件恢复未完成的任务
        if self.persist_file and self.persist_file.exists():
            await self._load_persisted_tasks()
        
        # 为每个GPU创建工作器
        for gpu_id in self.gpu_devices:
            for i in range(self.max_concurrent_per_gpu):
                worker = asyncio.create_task(
                    self._worker(gpu_id, worker_id=f"GPU{gpu_id}-{i}")
                )
                self.workers.append(worker)
        
        self.logger.info(f"已启动 {len(self.workers)} 个工作器")
    
    async def stop(self):
        """停止队列处理"""
        if not self.running:
            return
        
        self.logger.info("正在停止异步嵌字队列...")
        self.running = False
        
        # 等待所有工作器完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # 持久化未完成的任务
        if self.persist_file:
            await self._persist_tasks()
        
        self.logger.info("异步嵌字队列已停止")
    
    async def submit_task(self, task: EmbedTask) -> str:
        """
        提交嵌字任务
        
        Args:
            task: 嵌字任务
            
        Returns:
            任务ID
        """
        task_id = f"{task.video_id}_{int(time.time())}"
        await self.pending_tasks.put((task_id, task))
        
        self.stats['total_submitted'] += 1
        self.logger.info(f"已提交嵌字任务: {task_id}, 队列大小: {self.pending_tasks.qsize()}")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[EmbedTask]:
        """获取任务状态"""
        if task_id in self.processing_tasks:
            return self.processing_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务（仅能取消待处理的任务）"""
        # TODO: 实现取消逻辑
        self.logger.warning("任务取消功能尚未实现")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'pending_count': self.pending_tasks.qsize(),
            'processing_count': len(self.processing_tasks),
            'completed_count': len(self.completed_tasks),
            'avg_processing_time': (
                self.stats['total_processing_time'] / self.stats['total_completed']
                if self.stats['total_completed'] > 0 else 0
            )
        }
    
    async def _worker(self, gpu_id: int, worker_id: str):
        """
        工作器协程
        
        Args:
            gpu_id: GPU设备ID
            worker_id: 工作器ID
        """
        self.logger.info(f"工作器 {worker_id} 已启动")
        
        while self.running:
            try:
                # 从队列获取任务（超时1秒以便检查running状态）
                try:
                    task_id, task = await asyncio.wait_for(
                        self.pending_tasks.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 处理任务
                task.gpu_id = gpu_id
                task.status = EmbedTaskStatus.PROCESSING
                task.started_at = datetime.now()
                self.processing_tasks[task_id] = task
                
                self.logger.info(f"[{worker_id}] 开始处理任务: {task_id}")
                
                # 执行嵌字
                start_time = time.time()
                success = await self._process_task(task)
                processing_time = time.time() - start_time
                
                # 更新任务状态
                task.completed_at = datetime.now()
                if success:
                    task.status = EmbedTaskStatus.COMPLETED
                    self.stats['total_completed'] += 1
                    self.stats['total_processing_time'] += processing_time
                    self.logger.info(
                        f"[{worker_id}] 任务完成: {task_id}, "
                        f"耗时: {processing_time:.2f}秒"
                    )
                else:
                    task.status = EmbedTaskStatus.FAILED
                    self.stats['total_failed'] += 1
                    self.logger.error(f"[{worker_id}] 任务失败: {task_id}")
                
                # 移动到完成列表
                self.processing_tasks.pop(task_id)
                self.completed_tasks[task_id] = task
                
                # 更新数据库状态
                await self._update_database_status(task)
                
            except Exception as e:
                self.logger.error(f"[{worker_id}] 工作器异常: {e}", exc_info=True)
        
        self.logger.info(f"工作器 {worker_id} 已停止")
    
    async def _process_task(self, task: EmbedTask) -> bool:
        """
        处理嵌字任务
        
        Args:
            task: 嵌字任务
            
        Returns:
            是否成功
        """
        try:
            # 在线程池中执行同步的 FFmpeg 操作
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                self.ffmpeg.embed_subtitle_hard,
                task.video_path,
                task.subtitle_path,
                task.output_path,
                task.video_codec,
                task.audio_codec,
                task.crf,
                task.preset,
                task.use_gpu,
                task.gpu_id
            )
            
            if not success:
                task.error_message = "FFmpeg嵌字失败"
            
            return success
            
        except Exception as e:
            task.error_message = str(e)
            self.logger.error(f"处理任务异常: {e}", exc_info=True)
            return False
    
    async def _update_database_status(self, task: EmbedTask):
        """更新数据库中的视频状态"""
        try:
            db = SessionLocal()
            video = db.query(Video).filter(Video.id == task.video_id).first()
            
            if video:
                if task.status == EmbedTaskStatus.COMPLETED:
                    video.status = ProcessingStatus.COMPLETED
                    video.final_video_path = str(task.output_path)
                elif task.status == EmbedTaskStatus.FAILED:
                    video.status = ProcessingStatus.FAILED
                    video.error_message = task.error_message
                
                db.commit()
            
            db.close()
        except Exception as e:
            self.logger.error(f"更新数据库状态失败: {e}", exc_info=True)
    
    async def _persist_tasks(self):
        """持久化未完成的任务"""
        if not self.persist_file:
            return
        
        try:
            tasks_to_save = []
            
            # 保存待处理和正在处理的任务
            while not self.pending_tasks.empty():
                task_id, task = await self.pending_tasks.get()
                tasks_to_save.append({'id': task_id, 'task': task.to_dict()})
            
            for task_id, task in self.processing_tasks.items():
                tasks_to_save.append({'id': task_id, 'task': task.to_dict()})
            
            self.persist_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_file, 'w', encoding='utf-8') as f:
                json.dump(tasks_to_save, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"已持久化 {len(tasks_to_save)} 个未完成任务")
        
        except Exception as e:
            self.logger.error(f"持久化任务失败: {e}", exc_info=True)
    
    async def _load_persisted_tasks(self):
        """从持久化文件加载任务"""
        if not self.persist_file or not self.persist_file.exists():
            return
        
        try:
            with open(self.persist_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
            
            for item in tasks_data:
                task_id = item['id']
                task_dict = item['task']
                
                # 重建任务对象
                task = EmbedTask(
                    video_id=task_dict['video_id'],
                    video_path=Path(task_dict['video_path']),
                    subtitle_path=Path(task_dict['subtitle_path']),
                    output_path=Path(task_dict['output_path']),
                    video_codec=task_dict['video_codec'],
                    audio_codec=task_dict['audio_codec'],
                    crf=task_dict['crf'],
                    preset=task_dict['preset'],
                    use_gpu=task_dict['use_gpu'],
                    gpu_id=task_dict['gpu_id']
                )
                task.status = EmbedTaskStatus.PENDING
                
                await self.pending_tasks.put((task_id, task))
            
            self.logger.info(f"已加载 {len(tasks_data)} 个持久化任务")
            
            # 删除持久化文件
            self.persist_file.unlink()
        
        except Exception as e:
            self.logger.error(f"加载持久化任务失败: {e}", exc_info=True)


# 全局队列实例
_global_queue: Optional[AsyncEmbedderQueue] = None


def get_global_queue() -> AsyncEmbedderQueue:
    """获取全局队列实例"""
    global _global_queue
    if _global_queue is None:
        raise RuntimeError("异步嵌字队列未初始化，请先调用 init_global_queue()")
    return _global_queue


def init_global_queue(
    gpu_devices: List[int] = [0],
    max_concurrent_per_gpu: int = 1,
    persist_file: Optional[Path] = None
):
    """初始化全局队列实例"""
    global _global_queue
    _global_queue = AsyncEmbedderQueue(
        gpu_devices=gpu_devices,
        max_concurrent_per_gpu=max_concurrent_per_gpu,
        persist_file=persist_file
    )
    return _global_queue
