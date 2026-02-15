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
    PARTIAL_COMPLETED = "partial_completed"  # 部分视频失败，其余成功
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
    upload_cron: Optional[str] = None
    
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
            "upload_cron": self.upload_cron,
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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
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
            # 增量迁移：添加 upload_cron 列（已有则忽略）
            try:
                cursor.execute("ALTER TABLE web_jobs ADD COLUMN upload_cron TEXT")
            except Exception:
                pass  # 列已存在
    
    def submit_job(
        self,
        video_ids: List[str],
        steps: List[str],
        gpu_device: str = "auto",
        force: bool = False,
        concurrency: int = 1,
        playlist_id: Optional[str] = None,
        upload_cron: Optional[str] = None
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
                (job_id, video_ids, steps, gpu_device, force, status, log_file, created_at, upload_cron)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                json.dumps(video_ids),
                json.dumps(steps),
                gpu_device,
                1 if force else 0,
                JobStatus.PENDING.value,
                log_file,
                now,
                upload_cron
            ))
        
        # 启动子进程
        self._start_job_process(job_id, video_ids, steps, gpu_device, force, log_file, concurrency, playlist_id, upload_cron)
        
        logger.info(f"任务已提交: {job_id}, 视频数: {len(video_ids)}, 步骤: {steps}")
        return job_id
    
    def _start_job_process(
        self,
        job_id: str,
        video_ids: List[str],
        steps: List[str],
        gpu_device: str,
        force: bool,
        log_file: str,
        concurrency: int = 1,
        playlist_id: Optional[str] = None,
        upload_cron: Optional[str] = None
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
        
        if playlist_id:
            cmd.extend(["-p", playlist_id])
        
        if concurrency > 1:
            cmd.extend(["-c", str(concurrency)])
        
        if upload_cron:
            cmd.extend(["--upload-cron", upload_cron])
        
        # 打开日志文件
        log_fd = open(log_file, "w", buffering=1)  # 行缓冲
        
        # 启动子进程（PYTHONUNBUFFERED=1 确保日志实时写入文件，不被缓冲）
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        process = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # 独立进程组，不受父进程影响
            cwd=str(Path(__file__).parent.parent.parent),  # VAT 项目根目录
            env=env,
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
        
        # 进程已结束，先清理该 job 关联视频中残留的 running 状态 task
        self._cleanup_orphaned_running_tasks(job)
        
        # 基于数据库中视频任务的实际状态判定 job 状态
        status, error, progress = self._determine_job_result(job)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE web_jobs 
                SET status = ?, error = ?, progress = ?, finished_at = ?
                WHERE job_id = ?
            """, (status.value, error, progress, datetime.now(), job_id))
    
    def _cleanup_orphaned_running_tasks(self, job: WebJob):
        """清理 job 关联视频中残留的 running 状态 task
        
        当 job 进程已结束但 tasks 表中仍有 running 状态记录时，
        将这些记录标记为 failed（进程异常终止）。
        """
        video_ids = job.video_ids
        requested_steps = job.steps
        if not video_ids or not requested_steps:
            return
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for vid in video_ids:
                    placeholders = ','.join('?' * len(requested_steps))
                    # 查找该视频在请求步骤中仍为 running 的最新 task 记录
                    cursor.execute(f"""
                        SELECT id, step FROM (
                            SELECT id, step, status,
                                   ROW_NUMBER() OVER (PARTITION BY step ORDER BY id DESC) as rn
                            FROM tasks
                            WHERE video_id = ? AND step IN ({placeholders})
                        ) WHERE rn = 1 AND status = 'running'
                    """, [vid] + requested_steps)
                    
                    orphaned = cursor.fetchall()
                    for row in orphaned:
                        cursor.execute("""
                            UPDATE tasks SET status = 'failed', 
                                   error_message = '进程异常终止（job 进程已退出）'
                            WHERE id = ?
                        """, (row['id'],))
                        logger.warning(
                            f"清理孤儿 task: video={vid} step={row['step']} "
                            f"task_id={row['id']} (job={job.job_id})"
                        )
        except Exception as e:
            logger.error(f"清理孤儿 running tasks 失败: {e}")
    
    def _determine_job_result(self, job: WebJob) -> tuple:
        """基于数据库中视频任务的实际状态判定 job 结果
        
        查询 tasks 表中每个视频的各步骤状态，而非扫描日志文件。
        
        Args:
            job: WebJob 对象
            
        Returns:
            (status: JobStatus, error: Optional[str], progress: float)
        """
        video_ids = job.video_ids
        requested_steps = job.steps  # 请求执行的步骤名称列表
        
        if not video_ids or not requested_steps:
            return JobStatus.COMPLETED, None, 1.0
        
        # 查询所有相关视频的任务状态
        # 对每个视频，检查请求步骤中是否有 failed 状态的任务
        failed_videos = []  # [(video_id, failed_step, error_message)]
        completed_videos = []
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for vid in video_ids:
                # 获取该视频各步骤的最新状态
                placeholders = ','.join('?' * len(requested_steps))
                cursor.execute(f"""
                    SELECT step, status, error_message,
                           ROW_NUMBER() OVER (PARTITION BY step ORDER BY id DESC) as rn
                    FROM tasks
                    WHERE video_id = ? AND step IN ({placeholders})
                """, [vid] + requested_steps)
                
                rows = cursor.fetchall()
                
                # 只取每个 step 的最新记录 (rn=1)
                step_statuses = {}
                for row in rows:
                    if row['rn'] == 1:
                        step_statuses[row['step']] = {
                            'status': row['status'],
                            'error': row['error_message']
                        }
                
                # 判断该视频的状态
                has_failed = False
                for step_name in requested_steps:
                    info = step_statuses.get(step_name, {})
                    if info.get('status') == 'failed':
                        has_failed = True
                        failed_videos.append((vid, step_name, info.get('error', '')))
                        break
                
                if not has_failed:
                    completed_videos.append(vid)
        
        total = len(video_ids)
        failed_count = len(failed_videos)
        completed_count = len(completed_videos)
        
        # 判定 job 状态
        if failed_count == 0:
            # 全部成功
            return JobStatus.COMPLETED, None, 1.0
        elif completed_count > 0:
            # 部分成功、部分失败
            progress = completed_count / total
            # 构建错误摘要
            error_parts = [f"{failed_count}/{total} 个视频处理失败:"]
            for vid, step, err in failed_videos[:5]:  # 最多显示 5 个
                short_err = err[:80] if err else '未知错误'
                error_parts.append(f"  {vid} [{step}]: {short_err}")
            if failed_count > 5:
                error_parts.append(f"  ... 还有 {failed_count - 5} 个失败")
            error_msg = "\n".join(error_parts)
            return JobStatus.PARTIAL_COMPLETED, error_msg, progress
        else:
            # 全部失败
            progress = self._parse_progress_from_log(job.log_file)
            error_parts = [f"全部 {total} 个视频处理失败:"]
            for vid, step, err in failed_videos[:5]:
                short_err = err[:80] if err else '未知错误'
                error_parts.append(f"  {vid} [{step}]: {short_err}")
            if failed_count > 5:
                error_parts.append(f"  ... 还有 {failed_count - 5} 个失败")
            error_msg = "\n".join(error_parts)
            return JobStatus.FAILED, error_msg, progress

    def cleanup_all_orphaned_running_tasks(self):
        """全局清理：将所有没有活跃 job 进程的 running tasks 标记为 failed
        
        适用于启动时或定期检查，清理因进程崩溃导致的孤儿 running 记录。
        """
        # 获取真正在运行的 job 的 video_ids
        active_video_ids = set()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM web_jobs WHERE status = 'running'")
            for row in cursor.fetchall():
                job = self._row_to_job(row)
                # 检查进程是否真正存活
                if job.pid:
                    try:
                        os.kill(job.pid, 0)
                        active_video_ids.update(job.video_ids)
                    except ProcessLookupError:
                        pass  # 进程已死，不加入活跃集合
        
        # 查找所有 running 状态的 tasks
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, video_id, step FROM tasks WHERE status = 'running'
            """)
            orphaned = []
            for row in cursor.fetchall():
                if row['video_id'] not in active_video_ids:
                    orphaned.append(row)
            
            if orphaned:
                for row in orphaned:
                    cursor.execute("""
                        UPDATE tasks SET status = 'failed',
                               error_message = '进程异常终止（清理孤儿记录）'
                        WHERE id = ?
                    """, (row['id'],))
                logger.warning(f"全局清理: 修复 {len(orphaned)} 条孤儿 running task 记录")
    
    def get_running_job_for_video(self, video_id: str) -> Optional[WebJob]:
        """查找正在处理指定视频的 running job（假设同一视频同时只有一个 running job）"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM web_jobs 
                WHERE status = 'running'
                ORDER BY created_at DESC
            """)
            for row in cursor.fetchall():
                job = self._row_to_job(row)
                if video_id in job.video_ids:
                    return job
        return None
    
    def get_running_video_ids(self) -> set:
        """获取所有正在 running job 中的 video_id 集合"""
        result = set()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT video_ids FROM web_jobs WHERE status = 'running'
            """)
            for row in cursor.fetchall():
                result.update(json.loads(row['video_ids']))
        return result
    
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
        # upload_cron 列可能不存在（旧数据库），安全读取
        try:
            upload_cron = row['upload_cron']
        except (IndexError, KeyError):
            upload_cron = None
        
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
            upload_cron=upload_cron,
        )
