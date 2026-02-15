"""
多GPU任务调度器
"""
import os
import traceback
from multiprocessing import Process
from typing import List, Optional

from ..config import Config
from ..database import Database
from ..utils.logger import setup_logger
from .executor import VideoProcessor

logger = setup_logger("pipeline.scheduler")


class MultiGPUScheduler:
    """多进程调度器，每个GPU运行独立的Python进程"""
    
    def __init__(self, config: Config):
        """
        初始化调度器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.gpu_devices = config.concurrency.gpu_devices
        self.db = Database(config.storage.database_path)
    
    def run(
        self,
        video_ids: List[str],
        steps: Optional[List[str]] = None,
        force: bool = False
    ) -> None:
        """
        将视频列表分配到多个GPU并行处理
        
        Args:
            video_ids: 视频ID列表
            steps: 要执行的步骤列表（None表示所有待处理步骤）
            force: 是否强制重新处理（即使已完成）
        """
        assert isinstance(video_ids, list), f"video_ids 必须是列表, 得到 {type(video_ids)}"
        if not video_ids:
            logger.info("没有待处理的视频")
            return
        
        assert self.gpu_devices, "没有可用的 GPU 设备"
        
        # 按GPU数量分割任务
        chunks = self._split_tasks(video_ids, len(self.gpu_devices))
        
        logger.info(f"将 {len(video_ids)} 个视频分配到 {len(self.gpu_devices)} 个GPU")
        for i, (gpu_id, chunk) in enumerate(zip(self.gpu_devices, chunks)):
            logger.info(f"  GPU {gpu_id}: {len(chunk)} 个视频")
        
        # 为每个GPU启动独立进程
        processes = []
        for gpu_id, video_chunk in zip(self.gpu_devices, chunks):
            if not video_chunk:
                continue
            
            p = Process(
                target=self._worker,
                args=(gpu_id, video_chunk, steps, force)
            )
            p.start()
            processes.append((gpu_id, p))
        
        # 等待所有进程完成
        logger.info("等待所有GPU完成处理...")
        for gpu_id, p in processes:
            p.join()
            logger.info(f"GPU {gpu_id} 处理完成")
        
        logger.info("所有GPU处理完成")
    
    def _split_tasks(
        self,
        video_ids: List[str],
        num_chunks: int
    ) -> List[List[str]]:
        """
        将视频列表分割成多个块
        
        Args:
            video_ids: 视频ID列表
            num_chunks: 块数量
            
        Returns:
            分割后的列表
        """
        chunks = [[] for _ in range(num_chunks)]
        for i, video_id in enumerate(video_ids):
            chunks[i % num_chunks].append(video_id)
        return chunks
    
    def _worker(
        self,
        gpu_id: int,
        video_ids: List[str],
        steps: Optional[List[str]],
        force: bool = False
    ):
        """
        单个GPU的工作进程
        
        Args:
            gpu_id: GPU编号
            video_ids: 该GPU要处理的视频ID列表
            steps: 要执行的步骤
            force: 是否强制重新处理
        """
        # 设置GPU环境变量（在子进程中设置）
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 子进程中需要重新初始化 logger
        worker_logger = setup_logger(f"pipeline.scheduler.gpu{gpu_id}")
        worker_logger.info(f"[GPU {gpu_id}] 开始处理 {len(video_ids)} 个视频")
        
        # 逐个处理视频
        for i, video_id in enumerate(video_ids):
            worker_logger.info(f"[GPU {gpu_id}] 处理视频 {i+1}/{len(video_ids)}: {video_id}")
            
            try:
                processor = VideoProcessor(
                    video_id=video_id,
                    config=self.config,
                    gpu_id=gpu_id,
                    force=force,
                    video_index=i,
                    total_videos=len(video_ids)
                )
                
                success = processor.process(steps)
                
                if success:
                    worker_logger.info(f"[GPU {gpu_id}] 视频处理成功: {video_id}")
                else:
                    worker_logger.warning(f"[GPU {gpu_id}] 视频处理失败: {video_id}")
                    
            except KeyboardInterrupt:
                worker_logger.warning(f"[GPU {gpu_id}] 用户中止操作")
                raise
            except Exception as e:
                worker_logger.error(f"[GPU {gpu_id}] 视频处理异常: {video_id} - {e}")
                worker_logger.debug(traceback.format_exc())
        
        worker_logger.info(f"[GPU {gpu_id}] 所有视频处理完成")


class SingleGPUScheduler:
    """单GPU顺序处理调度器"""
    
    def __init__(self, config: Config, gpu_id: int = 0):
        """
        初始化调度器
        
        Args:
            config: 配置对象
            gpu_id: GPU编号
        """
        self.config = config
        self.gpu_id = gpu_id
        self.db = Database(config.storage.database_path)
    
    def run(
        self,
        video_ids: List[str],
        steps: Optional[List[str]] = None,
        force: bool = False
    ) -> None:
        """
        顺序处理视频列表
        
        Args:
            video_ids: 视频ID列表
            steps: 要执行的步骤列表
            force: 是否强制重新处理
        """
        assert isinstance(video_ids, list), f"video_ids 必须是列表, 得到 {type(video_ids)}"
        if not video_ids:
            logger.info("没有待处理的视频")
            return
        
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        logger.info(f"使用 GPU {self.gpu_id} 顺序处理 {len(video_ids)} 个视频")
        
        # 视频间延迟（从配置读取）
        download_delay = self.config.downloader.youtube.download_delay
        
        # 逐个处理
        for i, video_id in enumerate(video_ids):
            assert video_id, f"第 {i+1} 个视频ID为空"
            
            if i > 0 and download_delay > 0:
                logger.info(f"等待 {download_delay:.0f} 秒后处理下一个视频...")
                import time
                time.sleep(download_delay)
            
            logger.info(f"处理视频 [{i+1}/{len(video_ids)}]: {video_id}")
            
            try:
                processor = VideoProcessor(
                    video_id=video_id,
                    config=self.config,
                    gpu_id=self.gpu_id,
                    force=force,
                    video_index=i,
                    total_videos=len(video_ids)
                )
                
                success = processor.process(steps)
                
                if success:
                    logger.info(f"视频处理成功: {video_id}")
                else:
                    logger.warning(f"视频处理失败: {video_id}")
                    
            except KeyboardInterrupt:
                logger.warning(f"用户中止操作")
                raise
            except Exception as e:
                logger.error(f"视频处理异常: {video_id} - {e}")
                logger.debug(traceback.format_exc())
        
        logger.info(f"所有视频处理完成 (共 {len(video_ids)} 个)")


def schedule_videos(
    config: Config,
    video_ids: List[str],
    steps: Optional[List[str]] = None,
    use_multi_gpu: bool = True,
    force: bool = False
) -> None:
    """
    调度视频处理
    
    Args:
        config: 配置对象
        video_ids: 视频ID列表
        steps: 要执行的步骤
        use_multi_gpu: 是否使用多GPU
        force: 是否强制重新处理（即使已完成）
    """
    if use_multi_gpu and len(config.concurrency.gpu_devices) > 1:
        scheduler = MultiGPUScheduler(config)
    else:
        gpu_id = config.concurrency.gpu_devices[0] if config.concurrency.gpu_devices else 0
        scheduler = SingleGPUScheduler(config, gpu_id)
    
    scheduler.run(video_ids, steps, force=force)
