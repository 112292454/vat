"""
任务执行 API

任务通过子进程执行 CLI 命令，与 Web UI 完全解耦
"""
import asyncio
import re
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vat.models import expand_stage_group
from vat.web.jobs import JobManager, JobStatus

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

# 全局 JobManager 实例
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """获取 JobManager 实例"""
    global _job_manager
    if _job_manager is None:
        from vat.config import load_config
        config = load_config()
        log_dir = Path(config.storage.database_path).parent / "job_logs"
        _job_manager = JobManager(config.storage.database_path, str(log_dir))
    return _job_manager


class ExecuteRequest(BaseModel):
    """执行任务请求"""
    video_ids: List[str]
    steps: List[str]  # TaskStep 名称列表或阶段组
    gpu_device: str = "auto"  # "auto", "cuda:0", "cpu"
    force: bool = False
    concurrency: int = 1  # 并发处理的视频数量（默认1=串行）
    playlist_id: Optional[str] = None  # playlist context（用于 custom prompt 覆写）
    upload_cron: Optional[str] = None  # 定时上传 cron 表达式（仅 steps=["upload"] 时可用）
    
    # 可选：生成等价 CLI 命令
    generate_cli: bool = False


class ExecuteResponse(BaseModel):
    """执行任务响应"""
    task_id: str
    status: str
    cli_command: Optional[str] = None


class TaskResponse(BaseModel):
    """任务信息响应"""
    task_id: str
    video_ids: List[str]
    steps: List[str]
    status: str
    progress: float
    current_step: Optional[str]
    current_video: Optional[str]
    current_video_title: Optional[str]
    error: Optional[str]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]


def parse_steps(step_names: List[str]) -> List[str]:
    """解析步骤参数，返回步骤名称列表"""
    result = []
    for name in step_names:
        name = name.strip().lower()
        if not name:
            continue
        
        # 尝试作为阶段组展开
        try:
            expanded = expand_stage_group(name)
            result.extend([s.value for s in expanded])
        except ValueError:
            result.append(name)
    
    # 去重并保持顺序
    seen = set()
    unique = []
    for step in result:
        if step not in seen:
            seen.add(step)
            unique.append(step)
    
    return unique


@router.post("/execute", response_model=ExecuteResponse)
async def execute_task(
    request: ExecuteRequest,
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    执行处理任务
    
    - 通过子进程执行 CLI 命令，与 Web UI 解耦
    - 立即返回 job_id
    """
    # 解析 steps
    try:
        steps = parse_steps(request.steps)
    except Exception as e:
        raise HTTPException(400, f"Invalid step: {e}")
    
    if not steps:
        raise HTTPException(400, "No valid steps provided")
    
    if not request.video_ids:
        raise HTTPException(400, "No video_ids provided")
    
    # ========== upload_cron 校验 ==========
    if request.upload_cron:
        # 校验1: steps 必须仅为 upload
        if steps != ["upload"]:
            raise HTTPException(
                400,
                "定时上传仅可用于 upload 阶段。请将执行阶段设为仅 'upload'，"
                "或去掉定时上传设置。"
            )
        
        # 校验2: cron 表达式合法性
        try:
            from croniter import croniter
            if not croniter.is_valid(request.upload_cron):
                raise HTTPException(400, f"无效的 cron 表达式: {request.upload_cron}")
        except ImportError:
            raise HTTPException(500, "服务端缺少 croniter 依赖，请安装: pip install croniter")
        
        # 校验3: 所有视频的 embed 阶段已完成
        from vat.web.deps import get_db as _get_db_for_check
        from vat.models import TaskStep
        check_db = _get_db_for_check()
        not_ready = []
        for vid in request.video_ids:
            if not check_db.is_step_completed(vid, TaskStep.EMBED):
                v = check_db.get_video(vid)
                name = (v.title[:30] if v and v.title else vid)
                not_ready.append(name)
        if not_ready:
            detail = f"{len(not_ready)} 个视频尚未完成 embed 阶段: {', '.join(not_ready[:5])}"
            if len(not_ready) > 5:
                detail += f" ...等 {len(not_ready)} 个"
            raise HTTPException(400, f"无法创建定时上传任务: {detail}")
    
    # 单任务约束：同一视频同时只能有一个 running job（阶段顺序依赖，多 task 无意义）
    running_vids = job_manager.get_running_video_ids()
    conflict_vids = [vid for vid in request.video_ids if vid in running_vids]
    if conflict_vids:
        raise HTTPException(
            409,
            f"以下视频正在处理中，请等待完成或取消后重试: {', '.join(conflict_vids[:5])}"
            + (f" ...等 {len(conflict_vids)} 个" if len(conflict_vids) > 5 else "")
        )
    
    # 如果强制重新处理，重置后续阶段的任务状态
    if request.force and steps:
        from vat.web.deps import get_db as _get_db
        from vat.models import TaskStep
        
        db = _get_db()
        
        # 找到执行步骤中最早的那个，重置它之后的所有阶段
        first_step = None
        for step_name in steps:
            try:
                step = TaskStep(step_name)
                if first_step is None:
                    first_step = step
                else:
                    # 比较阶段顺序
                    from vat.models import DEFAULT_STAGE_SEQUENCE
                    if DEFAULT_STAGE_SEQUENCE.index(step) < DEFAULT_STAGE_SEQUENCE.index(first_step):
                        first_step = step
            except ValueError:
                pass
        
        if first_step:
            for video_id in request.video_ids:
                db.invalidate_downstream_tasks(video_id, first_step)
    
    # 提交任务（启动子进程）
    job_id = job_manager.submit_job(
        video_ids=request.video_ids,
        steps=steps,
        gpu_device=request.gpu_device,
        force=request.force,
        concurrency=request.concurrency,
        playlist_id=request.playlist_id,
        upload_cron=request.upload_cron
    )
    
    # 生成等价 CLI 命令（可选）
    cli_command = None
    if request.generate_cli:
        cli_command = _generate_cli_command(request)
    
    return ExecuteResponse(
        task_id=job_id,
        status="submitted",
        cli_command=cli_command
    )


def _generate_cli_command(request: ExecuteRequest) -> str:
    """生成与 _start_job_process 完全等价的 CLI 命令"""
    parts = ["python -m vat process"]
    
    # 视频 ID（全部列出，不省略）
    for vid in request.video_ids:
        parts.append(f"-v {vid}")
    
    # 步骤
    steps_str = ",".join(request.steps)
    if steps_str != "all":
        parts.append(f"-s {steps_str}")
    
    # GPU
    if request.gpu_device != "auto":
        parts.append(f"-g {request.gpu_device}")
    
    # Force
    if request.force:
        parts.append("-f")
    
    # Playlist context
    if request.playlist_id:
        parts.append(f"-p {request.playlist_id}")
    
    # 并发
    if request.concurrency > 1:
        parts.append(f"-c {request.concurrency}")
    
    # 定时上传
    if request.upload_cron:
        parts.append(f'--upload-cron "{request.upload_cron}"')
    
    return " ".join(parts)


@router.get("")
async def list_tasks(
    limit: int = 50,
    job_manager: JobManager = Depends(get_job_manager)
):
    """列出任务历史"""
    jobs = job_manager.list_jobs(limit=limit)
    return [job.to_dict() for job in jobs]


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    """获取任务详情"""
    # 先更新任务状态
    job_manager.update_job_status(task_id)
    
    job = job_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Task not found")
    
    return job.to_dict()


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    """取消任务（发送 SIGTERM 到子进程）"""
    success = job_manager.cancel_job(task_id)
    if not success:
        raise HTTPException(400, "Task cannot be cancelled")
    return {"status": "cancelled", "task_id": task_id}


@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    """删除任务记录（仅已完成/失败/取消的任务）"""
    job = job_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Task not found")
    
    if job.status == JobStatus.RUNNING:
        raise HTTPException(400, "Cannot delete running task. Cancel it first.")
    
    success = job_manager.delete_job(task_id)
    if not success:
        raise HTTPException(500, "Failed to delete task")
    return {"status": "deleted", "task_id": task_id}


@router.post("/{task_id}/retry")
async def retry_task(
    task_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    """重新运行任务（基于原任务参数创建新任务）"""
    job = job_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Task not found")
    
    if job.status == JobStatus.RUNNING:
        raise HTTPException(400, "Task is still running")
    
    # 使用原任务的参数创建新任务
    new_job_id = job_manager.submit_job(
        video_ids=job.video_ids,
        steps=job.steps,
        gpu_device=job.gpu_device,
        force=job.force
    )
    
    return {
        "status": "submitted",
        "new_task_id": new_job_id,
        "original_task_id": task_id
    }


@router.get("/{task_id}/log-content")
async def get_log_content(
    task_id: str,
    tail: int = Query(500, ge=0, description="只返回最后 N 行，0=全部"),
    job_manager: JobManager = Depends(get_job_manager)
):
    """获取任务日志内容（非流式），返回尾部 N 行和文件 byte offset（供 SSE 续读）"""
    job = job_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Task not found")
    
    content = ""
    file_offset = 0  # SSE 从此处开始读增量
    if job.log_file:
        from pathlib import Path
        log_path = Path(job.log_file)
        if log_path.exists():
            raw = log_path.read_text(encoding="utf-8", errors="replace")
            file_offset = len(raw.encode("utf-8"))  # 文件末尾的 byte 位置
            clean = re.sub(r'\x1b\[[0-9;]*m', '', raw)
            if tail > 0:
                lines = clean.splitlines()
                if len(lines) > tail:
                    content = f"... (已省略 {len(lines) - tail} 条早期日志) ...\n" + "\n".join(lines[-tail:])
                else:
                    content = clean
            else:
                content = clean
    
    return {"task_id": task_id, "content": content, "file_offset": file_offset}


@router.get("/{task_id}/logs")
async def stream_logs(
    task_id: str,
    offset: int = Query(0, ge=0, description="文件 byte offset，从此处开始读增量"),
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    SSE 实时日志流（增量推送）
    
    offset 参数由前端从 /log-content 的 file_offset 获取，
    确保 SSE 只推送 loadExistingLogs 之后新产生的日志行。
    """
    job = job_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Task not found")
    
    async def event_generator():
        last_pos = offset
        
        while True:
            # 更新任务状态
            job_manager.update_job_status(task_id)
            current_job = job_manager.get_job(task_id)
            
            # 读取新的日志内容
            if current_job and current_job.log_file:
                try:
                    from pathlib import Path
                    log_path = Path(current_job.log_file)
                    if log_path.exists():
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_pos)
                            new_content = f.read()
                            last_pos = f.tell()
                            
                            for line in new_content.splitlines():
                                if line.strip():
                                    # 过滤 ANSI 转义序列
                                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                                    yield f"data: {clean_line}\n\n"
                except Exception as e:
                    yield f"data: [读取日志失败: {e}]\n\n"
            
            # 检查任务状态
            if current_job and current_job.status in (JobStatus.COMPLETED, JobStatus.PARTIAL_COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                yield f"event: complete\ndata: {current_job.status.value}\n\n"
                break
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
