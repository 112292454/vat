"""
任务执行 API

任务通过子进程执行 CLI 命令，与 Web UI 完全解耦
"""
import asyncio
import re
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
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
        concurrency=request.concurrency
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
    """生成等价的 CLI 命令"""
    parts = ["vat", "process"]
    
    # 视频 ID
    for vid in request.video_ids[:3]:
        parts.append(f"-v {vid}")
    
    if len(request.video_ids) > 3:
        parts.append(f"# ... 还有 {len(request.video_ids) - 3} 个视频")
    
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
    job_manager: JobManager = Depends(get_job_manager)
):
    """获取任务日志内容（非流式）"""
    job = job_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Task not found")
    
    content = ""
    if job.log_file:
        from pathlib import Path
        log_path = Path(job.log_file)
        if log_path.exists():
            content = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
    
    return {"task_id": task_id, "content": content}


@router.get("/{task_id}/logs")
async def stream_logs(
    task_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    SSE 实时日志流
    
    从日志文件读取并推送，直到任务完成
    """
    job = job_manager.get_job(task_id)
    if not job:
        raise HTTPException(404, "Task not found")
    
    async def event_generator():
        last_pos = 0
        
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
                        with open(log_path, "r") as f:
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
