"""
多GPU任务调度器
"""
import os
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process
from typing import List, Optional

from ..config import Config
from ..database import Database
from ..utils.logger import setup_logger
from .executor import VideoProcessor

logger = setup_logger("pipeline.scheduler")


@dataclass
class BatchRunResult:
    """批处理运行结果"""
    total_videos: int
    failed_video_ids: List[str] = field(default_factory=list)
    stopped_early: bool = False


def run_video_batch(
    *,
    config: Config,
    video_ids: List[str],
    steps: Optional[List[str]] = None,
    force: bool = False,
    gpu_id: Optional[int] = None,
    concurrency: int = 1,
    playlist_id: Optional[str] = None,
    fail_fast: bool = False,
    delay_seconds: Optional[float] = None,
    max_retry_rounds: int = 0,
    logger_override=None,
    db: Optional[Database] = None,
    processor_cls=None,
) -> BatchRunResult:
    """
    共享批处理运行时。

    给 `cli process` 与 `pipeline.scheduler` 统一使用，避免双轨控制面。
    """
    batch_logger = logger_override or logger
    assert isinstance(video_ids, list), f"video_ids 必须是列表, 得到 {type(video_ids)}"
    if not video_ids:
        batch_logger.info("没有待处理的视频")
        return BatchRunResult(total_videos=0)

    if db is None:
        db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    if processor_cls is None:
        processor_cls = VideoProcessor
    total = len(video_ids)
    effective_delay = (
        config.downloader.youtube.download_delay
        if delay_seconds is None else delay_seconds
    )

    def process_one_video(args):
        idx, vid = args
        video = db.get_video(vid)
        if not video:
            batch_logger.warning(f"视频不存在: {vid}")
            return vid, False, "视频不存在"

        title = video.title[:30] if video.title else vid
        batch_logger.info(f"[{idx + 1}/{total}] 开始处理: {title}")

        try:
            processor = processor_cls(
                video_id=vid,
                config=deepcopy(config),
                gpu_id=gpu_id,
                force=force,
                video_index=idx,
                total_videos=total,
                playlist_id=playlist_id,
            )
            success = processor.process(steps=steps)
            if success:
                batch_logger.info(f"[{idx + 1}/{total}] 完成: {title}")
                return vid, True, None
            batch_logger.warning(f"[{idx + 1}/{total}] 失败: {title}")
            return vid, False, "处理返回失败"
        except Exception as e:
            batch_logger.error(
                f"[{idx + 1}/{total}] 失败: {title} - {e}\n{traceback.format_exc()}"
            )
            return vid, False, str(e)

    def _run_batch(video_list):
        failed_vids = []
        stopped_early = False

        if concurrency <= 1:
            for i, (idx, vid) in enumerate(video_list):
                if i > 0 and effective_delay > 0:
                    batch_logger.info(f"等待 {effective_delay:.0f} 秒后处理下一个视频...")
                    time.sleep(effective_delay)
                _, success, _ = process_one_video((idx, vid))
                if not success:
                    failed_vids.append(vid)
                    if fail_fast:
                        remaining = len(video_list) - i - 1
                        if remaining > 0:
                            batch_logger.warning(
                                f"fail-fast: 视频 {vid} 处理失败，跳过剩余 {remaining} 个视频"
                            )
                        stopped_early = True
                        break
        else:
            fail_fast_triggered = False
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                pending_futures = {}
                video_iter = iter(video_list)

                for _ in range(min(concurrency, len(video_list))):
                    try:
                        idx, vid = next(video_iter)
                        future = executor.submit(process_one_video, (idx, vid))
                        pending_futures[future] = vid
                    except StopIteration:
                        break

                while pending_futures:
                    done_future = None
                    for future in as_completed(pending_futures):
                        done_future = future
                        break

                    if done_future is None:
                        break

                    vid = pending_futures.pop(done_future)
                    try:
                        _, success, _ = done_future.result()
                        if not success:
                            failed_vids.append(vid)
                            if fail_fast:
                                fail_fast_triggered = True
                    except Exception as e:
                        batch_logger.error(f"并发处理异常: {vid} - {e}")
                        failed_vids.append(vid)
                        if fail_fast:
                            fail_fast_triggered = True

                    if not fail_fast_triggered:
                        try:
                            idx, vid = next(video_iter)
                            future = executor.submit(process_one_video, (idx, vid))
                            pending_futures[future] = vid
                        except StopIteration:
                            pass
                    elif not pending_futures:
                        break

                if fail_fast_triggered:
                    stopped_early = True
                    batch_logger.warning("fail-fast: 处理失败，不再启动新任务（已等待运行中的任务完成）")

        return failed_vids, stopped_early

    batch_logger.info(f"开始处理 {total} 个视频（并发: {concurrency}）")
    failed_vids, stopped_early = _run_batch(list(enumerate(video_ids)))

    if not (fail_fast and stopped_early):
        for retry_round in range(1, max_retry_rounds + 1):
            if not failed_vids:
                break
            batch_logger.info(f"第 {retry_round} 轮重试: {len(failed_vids)} 个失败视频")
            retry_list = [(video_ids.index(vid), vid) for vid in failed_vids]
            failed_vids, stopped_early = _run_batch(retry_list)
            if fail_fast and stopped_early:
                break
    elif failed_vids:
        batch_logger.info("fail-fast 模式：跳过重试")

    if failed_vids:
        batch_logger.warning(f"处理完成，{len(failed_vids)} 个视频最终失败: {', '.join(failed_vids[:5])}")
    else:
        batch_logger.info("处理完成，全部成功")

    return BatchRunResult(
        total_videos=total,
        failed_video_ids=failed_vids,
        stopped_early=stopped_early,
    )


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
        self.db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
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

        try:
            run_video_batch(
                config=self.config,
                video_ids=video_ids,
                steps=steps,
                force=force,
                gpu_id=gpu_id,
                concurrency=1,
                delay_seconds=self.config.downloader.youtube.download_delay,
                max_retry_rounds=0,
                logger_override=worker_logger,
                db=self.db,
                processor_cls=VideoProcessor,
            )
        except KeyboardInterrupt:
            worker_logger.warning(f"[GPU {gpu_id}] 用户中止操作")
            raise
        except Exception as e:
            worker_logger.error(f"[GPU {gpu_id}] 批处理异常: {e}")
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
        self.db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
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

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        logger.info(f"使用 GPU {self.gpu_id} 顺序处理 {len(video_ids)} 个视频")

        try:
            run_video_batch(
                config=self.config,
                video_ids=video_ids,
                steps=steps,
                force=force,
                gpu_id=self.gpu_id,
                concurrency=1,
                delay_seconds=self.config.downloader.youtube.download_delay,
                max_retry_rounds=0,
                logger_override=logger,
                db=self.db,
                processor_cls=VideoProcessor,
            )
        except KeyboardInterrupt:
            logger.warning("用户中止操作")
            raise

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
