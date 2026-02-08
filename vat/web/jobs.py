"""
Web 任务持久化

任务通过子进程执行 CLI 命令，与 Web UI 完全解耦
"""
import os
import signal
import subprocess
import json
import sqlite3
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from vat.utils.logger import setup_logger

logger = setup_logger("web.jobs")


class JobStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WebJob:
    """Web 任务记录"""
    job_id: str
    video_ids: List[str]
    steps: List[str]
    gpu_device: str
    force: bool
    status: JobStatus
    pid: Optional[int]
    log_file: Optional[str]
    progress: float
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "task_id": self.job_id,  # 兼容模板使用 task_id
            "video_ids": self.video_ids,
            "steps": self.steps,
            "gpu_device": self.gpu_device,
            "force": self.force,
            "status": self.status.value,
            "pid": self.pid,
            "log_file": self.log_file,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class JobManager:
    """
    任务管理器
    
    任务通过子进程执行，与 Web 服务器生命周期解耦
    """
    
    def __init__(self, db_path: str, log_dir: str):
        self.db_path = db_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_table()
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_table(self):
        """初始化 web_jobs 表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS web_jobs (
                    job_id TEXT PRIMARY KEY,
                    video_ids TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    gpu_device TEXT DEFAULT 'auto',
                    force INTEGER DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    pid INTEGER,
                    log_file TEXT,
                    progress REAL DEFAULT 0.0,
                    error TEXT,
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON web_jobs(status)")
    
    def submit_job(
        self,
        video_ids: List[str],
        steps: List[str],
        gpu_device: str = "auto",
        force: bool = False
    ) -> str:
        """
        提交任务并立即启动子进程执行
        
        Returns:
            job_id
        """
        import uuid
        job_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        log_file = str(self.log_dir / f"job_{job_id}.log")
        
        # 写入数据库
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO web_jobs 
                (job_id, video_ids, steps, gpu_device, force, status, log_file, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                json.dumps(video_ids),
                json.dumps(steps),
                gpu_device,
                1 if force else 0,
                JobStatus.PENDING.value,
                log_file,
                now
            ))
        
        # 启动子进程
        self._start_job_process(job_id, video_ids, steps, gpu_device, force, log_file)
        
        logger.info(f"任务已提交: {job_id}, 视频数: {len(video_ids)}, 步骤: {steps}")
        return job_id
    
    def _start_job_process(
        self,
        job_id: str,
        video_ids: List[str],
        steps: List[str],
        gpu_device: str,
        force: bool,
        log_file: str
    ):
        """启动子进程执行任务"""
        # 构建 CLI 命令
        cmd = ["python", "-m", "vat", "process"]
        
        for vid in video_ids:
            cmd.extend(["-v", vid])
        
        if steps:
            cmd.extend(["-s", ",".join(steps)])
        
        if gpu_device != "auto":
            cmd.extend(["-g", gpu_device])
        
        if force:
            cmd.append("-f")
        
        # 打开日志文件
        log_fd = open(log_file, "w", buffering=1)  # 行缓冲
        
        # 启动子进程
        process = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # 独立进程组，不受父进程影响
            cwd=str(Path(__file__).parent.parent.parent)  # VAT 项目根目录
        )
        
        # 更新数据库
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE web_jobs 
                SET status = ?, pid = ?, started_at = ?
                WHERE job_id = ?
            """, (JobStatus.RUNNING.value, process.pid, datetime.now(), job_id))
        
        logger.info(f"任务进程已启动: {job_id}, PID: {process.pid}")
    
    def get_job(self, job_id: str) -> Optional[WebJob]:
        """获取任务信息"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM web_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_job(row)
        return None
    
    def list_jobs(self, limit: int = 50) -> List[WebJob]:
        """列出任务"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM web_jobs 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            return [self._row_to_job(row) for row in cursor.fetchall()]
    
    def cancel_job(self, job_id: str) -> bool:
        """取消任务（发送 SIGTERM）"""
        job = self.get_job(job_id)
        if not job or job.status != JobStatus.RUNNING or not job.pid:
            return False
        
        try:
            os.kill(job.pid, signal.SIGTERM)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE web_jobs 
                    SET status = ?, finished_at = ?
                    WHERE job_id = ?
                """, (JobStatus.CANCELLED.value, datetime.now(), job_id))
            
            logger.info(f"任务已取消: {job_id}, PID: {job.pid}")
            return True
        except ProcessLookupError:
            # 进程已结束
            return False
    
    def _parse_progress_from_log(self, log_file: str) -> float:
        """从日志中解析实时进度
        
        优先解析批次总进度 [TOTAL:N%]，回退到单视频进度 [N%]。
        批次总进度考虑了多视频处理场景，不会在每个视频间反复 0→100%。
        """
        if not log_file or not Path(log_file).exists():
            return 0.0
        
        try:
            log_content = Path(log_file).read_text()
            lines = log_content.strip().split("\n")
            
            import re
            for line in reversed(lines):
                # 优先匹配批次总进度：[TOTAL:50%]
                total_match = re.search(r'\[TOTAL:(\d+)%\]', line)
                if total_match:
                    return float(total_match.group(1)) / 100.0
                # 回退：匹配单视频进度 [50%]
                match = re.search(r'\[(\d+)%\]', line)
                if match:
                    return float(match.group(1)) / 100.0
        except Exception:
            pass
        
        return 0.0
    
    def update_job_status(self, job_id: str):
        """检查并更新任务状态（通过检查进程是否存在）"""
        job = self.get_job(job_id)
        if not job or job.status != JobStatus.RUNNING or not job.pid:
            return
        
        # 检查进程是否真正结束（包括僵尸进程）
        process_ended = False
        try:
            os.kill(job.pid, 0)  # 检查进程是否存在
            # 进程存在，但可能是僵尸进程，检查 /proc/pid/status
            try:
                with open(f"/proc/{job.pid}/status", "r") as f:
                    status_content = f.read()
                    if "State:\tZ" in status_content:  # Z = zombie
                        process_ended = True
                        # 尝试回收僵尸进程
                        try:
                            os.waitpid(job.pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
            except FileNotFoundError:
                process_ended = True
        except ProcessLookupError:
            process_ended = True
        
        if not process_ended:
            # 进程仍在运行，更新实时进度
            progress = self._parse_progress_from_log(job.log_file)
            if progress > job.progress:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE web_jobs SET progress = ? WHERE job_id = ?
                    """, (progress, job_id))
            return
        
        # 进程已结束，检查日志确定状态
        status = JobStatus.COMPLETED
        error = None
        progress = 1.0  # 完成时设为 100%
        
        if job.log_file and Path(job.log_file).exists():
            log_content = Path(job.log_file).read_text()
            lines = log_content.strip().split("\n")
            
            # 只检查 ERROR 级别的日志，忽略 WARNING
            # 格式：时间 | ERROR | ... 或 时间 | INFO | ... | 步骤失败:
            for line in reversed(lines):
                # 检查是否是 ERROR 级别日志
                if "| ERROR |" in line:
                    status = JobStatus.FAILED
                    error = line[:200]
                    # 失败时保留当前进度，不设为100%
                    progress = self._parse_progress_from_log(job.log_file)
                    break
                # 检查 pipeline 步骤失败（INFO 级别但表示真正失败）
                if "步骤失败:" in line or "步骤异常:" in line:
                    status = JobStatus.FAILED
                    error = line[:200]
                    # 失败时保留当前进度，不设为100%
                    progress = self._parse_progress_from_log(job.log_file)
                    break
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE web_jobs 
                SET status = ?, error = ?, progress = ?, finished_at = ?
                WHERE job_id = ?
            """, (status.value, error, progress, datetime.now(), job_id))
    
    def get_log_content(self, job_id: str, tail_lines: int = 100) -> List[str]:
        """获取任务日志（最后 N 行）"""
        job = self.get_job(job_id)
        if not job or not job.log_file:
            return []
        
        log_path = Path(job.log_file)
        if not log_path.exists():
            return []
        
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                return [line.rstrip() for line in lines[-tail_lines:]]
        except Exception as e:
            return [f"读取日志失败: {e}"]
    
    def delete_job(self, job_id: str) -> bool:
        """删除任务记录（仅删除已完成/失败/取消的任务）"""
        job = self.get_job(job_id)
        if not job:
            return False
        
        # 不允许删除运行中的任务
        if job.status == JobStatus.RUNNING:
            return False
        
        # 删除日志文件
        if job.log_file and Path(job.log_file).exists():
            try:
                Path(job.log_file).unlink()
            except Exception:
                pass  # 忽略删除日志失败
        
        # 从数据库删除
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM web_jobs WHERE job_id = ?", (job_id,))
        
        return True
    
    def _row_to_job(self, row) -> WebJob:
        """将数据库行转换为 WebJob 对象"""
        return WebJob(
            job_id=row['job_id'],
            video_ids=json.loads(row['video_ids']),
            steps=json.loads(row['steps']),
            gpu_device=row['gpu_device'] or 'auto',
            force=bool(row['force']),
            status=JobStatus(row['status']),
            pid=row['pid'],
            log_file=row['log_file'],
            progress=row['progress'] or 0.0,
            error=row['error'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            finished_at=datetime.fromisoformat(row['finished_at']) if row['finished_at'] else None,
        )
