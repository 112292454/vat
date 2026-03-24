"""
业务层的视频处理任务提交通道。

用于让 service 依赖一个轻量的“提交 process job”能力，
而不是直接依赖 Web 层的 JobManager。
"""
from pathlib import Path
from typing import Callable, List, Optional

from vat.config import Config


def build_process_job_submitter(config: Config) -> Callable[..., Optional[str]]:
    """
    构造一个基于 JobManager 的 process job submitter。

    返回的 callable 只暴露业务层需要的参数。
    """
    from vat.web.jobs import JobManager

    log_dir = Path(config.storage.database_path).parent / "job_logs"
    job_manager = JobManager(str(config.storage.database_path), str(log_dir))

    def submitter(
        *,
        video_ids: List[str],
        steps: List[str],
        gpu_device: str,
        force: bool,
        concurrency: int,
        playlist_id: str,
        fail_fast: bool,
    ) -> Optional[str]:
        return job_manager.submit_job(
            video_ids=video_ids,
            steps=steps,
            gpu_device=gpu_device,
            force=force,
            concurrency=concurrency,
            playlist_id=playlist_id,
            fail_fast=fail_fast,
            task_type='process',
        )

    return submitter
