"""
Web 任务持久化

任务通过子进程执行 CLI 命令，与 Web UI 完全解耦
"""
import os
import signal
import subprocess
import json
import sqlite3
import sys
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

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


# 支持的 tools 任务类型（非 process 类型）
TOOLS_TASK_TYPES = {
    'fix-violation',
    'sync-playlist',
    'refresh-playlist',
    'retranslate-playlist',
    'upload-sync',
    'update-info',
    'sync-db',
    'season-sync',
    'watch',
}


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
    concurrency: int = 1
    fail_fast: bool = False
    task_type: str = 'process'       # 'process' | tools 任务类型
    task_params: Optional[Dict] = None  # tools 任务的额外参数
    cancel_requested: bool = False
    
    @property
    def is_tools_task(self) -> bool:
        """是否为 tools 类型任务（非视频处理 pipeline）"""
        return self.task_type != 'process'
    
    def to_dict(self) -> Dict:
        d = {
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
            "concurrency": self.concurrency,
            "fail_fast": self.fail_fast,
            "task_type": self.task_type,
            "task_params": self.task_params or {},
            "cancel_requested": self.cancel_requested,
        }
        return d


class JobManager:
    """
    任务管理器
    
    任务通过子进程执行，与 Web 服务器生命周期解耦
    """
    
    def __init__(self, db_path: str, log_dir: str, config_path: Optional[str] = None):
        self.db_path = db_path
        self.log_dir = Path(log_dir)
        self.config_path = config_path
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_table()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接（上下文管理器，确保连接关闭）"""
        conn = sqlite3.connect(self.db_path, timeout=120)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=120000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_table(self):
        """初始化 web_jobs 表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # WAL 模式：持久化设置，只需设置一次
            conn.execute("PRAGMA journal_mode=WAL")
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
            # 增量迁移：添加新列（已有则忽略）
            for col_sql in [
                "ALTER TABLE web_jobs ADD COLUMN upload_cron TEXT",
                "ALTER TABLE web_jobs ADD COLUMN concurrency INTEGER DEFAULT 1",
                "ALTER TABLE web_jobs ADD COLUMN fail_fast INTEGER DEFAULT 0",
                "ALTER TABLE web_jobs ADD COLUMN task_type TEXT DEFAULT 'process'",
                "ALTER TABLE web_jobs ADD COLUMN task_params TEXT DEFAULT '{}'",
                "ALTER TABLE web_jobs ADD COLUMN cancel_requested INTEGER DEFAULT 0",
            ]:
                try:
                    cursor.execute(col_sql)
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
        upload_cron: Optional[str] = None,
        upload_batch_size: int = 1,
        upload_mode: str = 'cron',
        fail_fast: bool = False,
        delay_start: int = 0,
        task_type: str = 'process',
        task_params: Optional[Dict] = None
    ) -> str:
        """
        提交任务并立即启动子进程执行
        
        所有 task-specific 参数统一存储在 task_params JSON 字段中。
        对 process 类型，playlist_id / upload_batch_size / upload_mode
        会自动合并到 task_params；对 tools 类型，调用者直接传 task_params。
        
        Args:
            video_ids: 视频ID列表（process 类型必填；tools 类型可为空列表）
            steps: 处理步骤列表（process 类型为步骤名；tools 类型为 [task_type]）
            task_type: 任务类型，'process' 或 TOOLS_TASK_TYPES 中的值
            task_params: 任务特有参数（JSON 可序列化的 dict），会被持久化
            playlist_id, upload_batch_size, upload_mode: process 类型的便捷参数，
                内部自动合并到 task_params
        
        Returns:
            job_id
        """
        import uuid
        job_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        log_file = str(self.log_dir / f"job_{job_id}.log")
        
        # 统一：将 process 特有参数合并到 task_params
        merged_params = dict(task_params or {})
        if task_type == 'process':
            if playlist_id:
                merged_params['playlist_id'] = playlist_id
            if upload_batch_size != 1:
                merged_params['upload_batch_size'] = upload_batch_size
            if upload_mode and upload_mode != 'cron':
                merged_params['upload_mode'] = upload_mode
            if delay_start and delay_start > 0:
                merged_params['delay_start'] = delay_start
        
        # 写入数据库
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO web_jobs 
                (job_id, video_ids, steps, gpu_device, force, status, log_file, created_at,
                 upload_cron, concurrency, fail_fast, task_type, task_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                json.dumps(video_ids),
                json.dumps(steps),
                gpu_device,
                1 if force else 0,
                JobStatus.PENDING.value,
                log_file,
                now,
                upload_cron,
                concurrency,
                1 if fail_fast else 0,
                task_type,
                json.dumps(merged_params),
            ))
        
        try:
            # 启动子进程
            self._start_job_process(
                job_id, video_ids, steps, gpu_device, force, log_file,
                concurrency, upload_cron, fail_fast, task_type, merged_params
            )
        except Exception:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM web_jobs WHERE job_id = ?", (job_id,))
            raise
        
        logger.info(f"任务已提交: {job_id}, type={task_type}, 步骤: {steps}")
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
        upload_cron: Optional[str] = None,
        fail_fast: bool = False,
        task_type: str = 'process',
        task_params: Optional[Dict] = None
    ):
        """启动子进程执行任务
        
        task-specific 参数统一从 task_params 读取，不再散落在函数签名中。
        """
        params = task_params or {}
        if task_type != 'process':
            cmd = self._build_tools_command(task_type, params, self.config_path)
        else:
            cmd = self._build_process_command(
                video_ids, steps, gpu_device, force, concurrency,
                upload_cron, fail_fast, params, self.config_path
            )
        
        logger.info(f"启动命令: {' '.join(cmd)}")
        
        # 打开日志文件
        log_fd = open(log_file, "w", buffering=1)  # 行缓冲
        process = None
        try:
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
        except Exception:
            if process is not None:
                self._terminate_process_group(process.pid)
            raise
        finally:
            log_fd.close()
        
        logger.info(f"任务进程已启动: {job_id}, PID: {process.pid}")

    def _terminate_process_group(self, pid: int) -> None:
        """终止任务进程组；用于启动失败后的兜底清理。"""
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            logger.warning(f"任务启动未完成，已终止遗留子进程组: pid={pid}, pgid={pgid}")
        except ProcessLookupError:
            logger.warning(f"任务启动清理时未找到进程: pid={pid}")
    
    @staticmethod
    def _build_process_command(
        video_ids: List[str],
        steps: List[str],
        gpu_device: str,
        force: bool,
        concurrency: int,
        upload_cron: Optional[str],
        fail_fast: bool,
        task_params: Dict,
        config_path: Optional[str] = None,
    ) -> List[str]:
        """构建 vat process 命令
        
        通用参数从函数签名传入，task-specific 参数从 task_params 读取：
        - playlist_id: playlist 上下文
        - upload_batch_size: 每次上传批次大小
        - upload_mode: 定时上传模式 (cron/dtime)
        """
        cmd = [sys.executable, "-m", "vat"]
        if config_path:
            cmd.extend(["-c", config_path])
        cmd.append("process")
        
        for vid in video_ids:
            cmd.extend(["-v", vid])
        
        if steps:
            cmd.extend(["-s", ",".join(steps)])
        
        if gpu_device != "auto":
            cmd.extend(["-g", gpu_device])
        
        if force:
            cmd.append("-f")
        
        # task-specific: playlist_id
        playlist_id = task_params.get('playlist_id')
        if playlist_id:
            cmd.extend(["-p", str(playlist_id)])
        
        if concurrency > 1:
            cmd.extend(["-c", str(concurrency)])
        
        if upload_cron:
            cmd.extend(["--upload-cron", upload_cron])
        
        # task-specific: upload_batch_size
        upload_batch_size = task_params.get('upload_batch_size', 1)
        if upload_batch_size and int(upload_batch_size) > 1:
            cmd.extend(["--upload-batch-size", str(upload_batch_size)])
        
        # task-specific: upload_mode
        upload_mode = task_params.get('upload_mode', 'cron')
        if upload_mode and upload_mode != 'cron':
            cmd.extend(["--upload-mode", str(upload_mode)])
        
        if fail_fast:
            cmd.append("--fail-fast")
        
        # task-specific: delay_start
        delay_start = task_params.get('delay_start', 0)
        if delay_start and int(delay_start) > 0:
            cmd.extend(["--delay-start", str(delay_start)])
        
        return cmd
    
    @staticmethod
    def _build_tools_command(
        task_type: str,
        params: Dict,
        config_path: Optional[str] = None,
    ) -> List[str]:
        """根据 task_type 和 params 构建 vat tools <subcommand> 命令
        
        每个 task_type 对应一个 vat tools 子命令，参数从 params dict 映射到 CLI 选项。
        """
        cmd = [sys.executable, "-m", "vat"]
        if config_path:
            cmd.extend(["-c", config_path])
        cmd.extend(["tools", task_type])
        
        # 通用映射：params 中的 key 转换为 CLI 选项
        # 布尔值 True → --flag，False → 不加
        # 其他值 → --key value
        PARAM_MAP = {
            'fix-violation': {
                'aid': '--aid',
                'video_path': '--video-path',
                'margin': '--margin',
                'mask_text': '--mask-text',
                'dry_run': '--dry-run',
                'max_rounds': '--max-rounds',
                'wait_seconds': '--wait-seconds',
            },
            'sync-playlist': {
                'playlist_id': '--playlist',
                'url': '--url',
                'fetch_upload_dates': '--fetch-dates',
            },
            'refresh-playlist': {
                'playlist_id': '--playlist',
                'force_refetch': '--force-refetch',
                'force_retranslate': '--force-retranslate',
            },
            'retranslate-playlist': {
                'playlist_id': '--playlist',
            },
            'upload-sync': {
                'playlist_id': '--playlist',
                'retry_delay': '--retry-delay',
            },
            'update-info': {
                'playlist_id': '--playlist',
                'dry_run': '--dry-run',
            },
            'sync-db': {
                'season_id': '--season',
                'playlist_id': '--playlist',
                'dry_run': '--dry-run',
            },
            'season-sync': {
                'playlist_id': '--playlist',
            },
            'watch': {
                'playlist_ids': '--playlist',  # 特殊处理：多值参数
                'interval': '--interval',
                'once': '--once',
                'stages': '--stages',
                'gpu': '--gpu',
                'concurrency': '--concurrency',
                'force': '--force',
                'fail_fast': '--fail-fast',
            },
        }
        
        mapping = PARAM_MAP.get(task_type, {})
        for key, flag in mapping.items():
            value = params.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            elif isinstance(value, list):
                # 多值参数：每个值分别传递（如 --playlist A --playlist B）
                for item in value:
                    cmd.extend([flag, str(item)])
            else:
                cmd.extend([flag, str(value)])
        
        return cmd
    
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

    def find_latest_job(
        self,
        task_type: str,
        task_params_subset: Optional[Dict] = None,
        statuses: Optional[List[JobStatus]] = None,
        limit: Optional[int] = None,
    ) -> Optional[WebJob]:
        """
        按 task_type + task_params 子集查找最近的任务。

        用于 Web 路由按业务 scope（如 playlist_id）反查后台任务，
        逐步替代进程内状态字典。
        """
        expected_params = task_params_subset or {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM web_jobs WHERE task_type = ?"
            params = [task_type]

            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                query += f" AND status IN ({placeholders})"
                params.extend(status.value for status in statuses)

            query += " ORDER BY created_at DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            for row in cursor.fetchall():
                job = self._row_to_job(row)
                actual_params = job.task_params or {}
                if all(actual_params.get(key) == value for key, value in expected_params.items()):
                    return job
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """发起取消请求，不在调用线程中阻塞等待子进程退出。"""
        self.update_job_status(job_id)
        job = self.get_job(job_id)
        if not job or job.status != JobStatus.RUNNING or not job.pid:
            return False

        try:
            pgid = os.getpgid(job.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            # 进程已在退出/已退出，刷新状态后再决定能否视作取消完成。
            self.update_job_status(job_id)
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE web_jobs
                SET cancel_requested = 1
                WHERE job_id = ?
            """, (job_id,))

        logger.info(f"已发送取消请求: {job_id}, PID: {job.pid}, PGID: {pgid}")
        return True

    def _mark_job_cancelled(self, job_id: str, pid: int) -> bool:
        """将任务标记为已取消。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE web_jobs 
                SET status = ?, finished_at = ?
                WHERE job_id = ?
            """, (JobStatus.CANCELLED.value, datetime.now(), job_id))
        
        logger.info(f"任务已取消: {job_id}, PID: {pid}")
        return True

    def _is_process_alive(self, pid: int) -> bool:
        """检查进程是否仍在运行；僵尸进程视为已结束。"""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # 进程存在但当前进程无权限发送信号。
            return True
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                status_content = f.read()
                if "State:\tZ" in status_content:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
                    return False
        except FileNotFoundError:
            return False
        return True
    
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

        if self._is_process_alive(job.pid):
            # 进程仍在运行，更新实时进度
            progress = self._parse_progress_from_log(job.log_file)
            if progress > job.progress:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE web_jobs SET progress = ? WHERE job_id = ?
                    """, (progress, job_id))
            return
        
        # tools 任务没有 tasks 表记录，无需清理孤儿 task
        if not job.is_tools_task:
            self._cleanup_orphaned_running_tasks(job)

        if job.cancel_requested:
            status, error, progress = JobStatus.CANCELLED, None, self._parse_progress_from_log(job.log_file)
        else:
            # 判定 job 结果
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
        """判定 job 结果
        
        - process 类型：查询 tasks 表中每个视频的各步骤状态
        - tools 类型：检查日志中的 [SUCCESS] / [FAILED:msg] 标记
        
        Args:
            job: WebJob 对象
            
        Returns:
            (status: JobStatus, error: Optional[str], progress: float)
        """
        if job.is_tools_task:
            return self._determine_tools_job_result(job)
        
        video_ids = job.video_ids
        requested_steps = job.steps
        
        if not video_ids or not requested_steps:
            return JobStatus.COMPLETED, None, 1.0
        
        # 查询所有相关视频的任务状态
        # 对每个视频，检查请求步骤中是否存在 failed / missing / incomplete。
        failed_videos = []  # [(video_id, step, error_message)]
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
                
                if has_failed:
                    continue

                incomplete_steps = []
                for step_name in requested_steps:
                    info = step_statuses.get(step_name, {})
                    if info.get('status') != 'completed':
                        incomplete_steps.append(step_name)

                if incomplete_steps:
                    step_name = incomplete_steps[0]
                    failed_videos.append(
                        (vid, step_name, f"步骤未完成或缺少任务记录: {step_name}")
                    )
                else:
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

    def _determine_tools_job_result(self, job: WebJob) -> tuple:
        """基于日志标记判定 tools 任务结果
        
        tools 子命令在完成时输出:
        - [SUCCESS] 成功完成
        - [SUCCESS] 成功信息... 成功并带消息
        - [FAILED] 失败原因... 失败并带原因
        """
        import re
        
        if not job.log_file or not Path(job.log_file).exists():
            return JobStatus.FAILED, "日志文件不存在", 0.0
        
        try:
            log_content = Path(job.log_file).read_text()
            lines = log_content.strip().split("\n")
            
            # 从后往前搜索结果标记
            for line in reversed(lines):
                if '[SUCCESS]' in line:
                    # 提取 [SUCCESS] 后的消息
                    msg_match = re.search(r'\[SUCCESS\]\s*(.*)', line)
                    msg = msg_match.group(1).strip() if msg_match else ''
                    return JobStatus.COMPLETED, None, 1.0
                if '[FAILED]' in line:
                    msg_match = re.search(r'\[FAILED\]\s*(.*)', line)
                    error_msg = msg_match.group(1).strip() if msg_match else '未知错误'
                    progress = self._parse_progress_from_log(job.log_file)
                    return JobStatus.FAILED, error_msg, progress
            
            # 日志中没有明确标记，进程已结束但无标记——可能崩溃
            progress = self._parse_progress_from_log(job.log_file)
            # 检查最后几行是否有错误信息
            last_lines = lines[-5:] if len(lines) >= 5 else lines
            error_hint = '\n'.join(last_lines)[-200:]
            return JobStatus.FAILED, f"进程异常终止\n{error_hint}", progress
        except Exception as e:
            return JobStatus.FAILED, f"日志解析失败: {e}", 0.0
    
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
        """查找正在处理指定视频且该视频尚未完成的 running job
        
        只有视频在 job 中尚未完成所有请求步骤时才算 active。
        已成功完成所有步骤的视频不再关联到 running job。
        """
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
                    if not self._is_video_completed_in_job(cursor, video_id, job.steps):
                        return job
        return None
    
    def get_running_video_ids(self) -> set:
        """获取所有 running job 中尚未完成处理的 video_id 集合
        
        已在 job 中成功完成所有请求步骤的视频不再被阻塞，
        允许它们被提交到新任务中。
        """
        result = set()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT video_ids, steps FROM web_jobs WHERE status = 'running'
            """)
            for row in cursor.fetchall():
                video_ids = json.loads(row['video_ids'])
                steps = json.loads(row['steps'])
                if not video_ids or not steps:
                    continue
                completed_vids = self._get_completed_video_ids(
                    cursor, video_ids, steps
                )
                for vid in video_ids:
                    if vid not in completed_vids:
                        result.add(vid)
        return result

    def get_running_and_queued_video_ids(self) -> set:
        """获取首页/列表页应隐藏的视频 ID 集合。

        包含两部分：
        1. 现有 running process job 中尚未完成处理的视频；
        2. running job 的 task_params 中显式声明的待处理视频列表。
        """
        result = set(self.get_running_video_ids())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT task_params FROM web_jobs WHERE status = 'running'
            """)
            for row in cursor.fetchall():
                task_params = self._load_task_params(row['task_params'])
                result.update(self._extract_task_param_video_ids(task_params))

        return result

    @staticmethod
    def _load_task_params(task_params_raw) -> Dict:
        """安全解析 task_params JSON。"""
        if not task_params_raw:
            return {}

        if isinstance(task_params_raw, dict):
            return task_params_raw

        try:
            parsed = json.loads(task_params_raw)
        except Exception:
            return {}

        return parsed if isinstance(parsed, dict) else {}

    @classmethod
    def _extract_task_param_video_ids(cls, task_params: Optional[Dict]) -> set:
        """从 task_params 中提取待处理视频 ID 列表。"""
        if not isinstance(task_params, dict):
            return set()

        result = set()
        for key in (
            "todo_video_ids",
            "pending_video_ids",
            "queued_video_ids",
            "candidate_video_ids",
            "video_ids",
        ):
            result.update(cls._normalize_video_id_list(task_params.get(key)))

        todo_list = task_params.get("todo_list")
        if isinstance(todo_list, list):
            for item in todo_list:
                if isinstance(item, str) and item:
                    result.add(item)
                    continue
                if isinstance(item, dict):
                    video_id = item.get("video_id") or item.get("id")
                    if isinstance(video_id, str) and video_id:
                        result.add(video_id)

        return result

    @staticmethod
    def _normalize_video_id_list(raw_value) -> set:
        """将 JSON 字段中的视频 ID 列表标准化为 set[str]。"""
        if not isinstance(raw_value, list):
            return set()

        result = set()
        for item in raw_value:
            if isinstance(item, str) and item:
                result.add(item)
        return result
    
    @staticmethod
    def _is_video_completed_in_job(cursor, video_id: str, steps: List[str]) -> bool:
        """判断视频是否已在 job 中完成所有请求步骤
        
        Args:
            cursor: 已打开的 DB cursor（web_jobs 和 tasks 在同一库）
            video_id: 视频 ID
            steps: job 请求的步骤列表（如 ['download', 'whisper', ...]）
        
        Returns:
            True 表示该视频的所有请求步骤最新记录均为 satisfied（completed/skipped）
        """
        if not steps:
            return True
        placeholders = ','.join('?' * len(steps))
        cursor.execute(f"""
            SELECT COUNT(*) as completed_count FROM (
                SELECT step, status,
                       ROW_NUMBER() OVER (PARTITION BY step ORDER BY id DESC) as rn
                FROM tasks
                WHERE video_id = ? AND step IN ({placeholders})
            ) WHERE rn = 1 AND status IN ('completed', 'skipped')
        """, [video_id] + list(steps))
        row = cursor.fetchone()
        return row['completed_count'] >= len(steps)
    
    @staticmethod
    def _get_completed_video_ids(cursor, video_ids: List[str], steps: List[str]) -> set:
        """批量查询：哪些视频已完成 job 的所有请求步骤
        
        Args:
            cursor: 已打开的 DB cursor
            video_ids: 待检查的视频 ID 列表
            steps: job 请求的步骤列表
        
        Returns:
            已完成所有步骤的 video_id 集合
        """
        if not video_ids or not steps:
            return set()
        
        num_steps = len(steps)
        vid_ph = ','.join('?' * len(video_ids))
        step_ph = ','.join('?' * num_steps)
        
        cursor.execute(f"""
            SELECT video_id, COUNT(*) as completed_count FROM (
                SELECT video_id, step, status,
                       ROW_NUMBER() OVER (PARTITION BY video_id, step ORDER BY id DESC) as rn
                FROM tasks
                WHERE video_id IN ({vid_ph}) AND step IN ({step_ph})
            ) WHERE rn = 1 AND status IN ('completed', 'skipped')
            GROUP BY video_id
            HAVING completed_count >= ?
        """, list(video_ids) + list(steps) + [num_steps])
        
        return {row['video_id'] for row in cursor.fetchall()}
    
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
        
        # concurrency 列可能不存在（旧数据库），安全读取
        try:
            concurrency = row['concurrency'] or 1
        except (IndexError, KeyError):
            concurrency = 1
        
        # fail_fast 列可能不存在（旧数据库），安全读取
        try:
            fail_fast = bool(row['fail_fast'])
        except (IndexError, KeyError):
            fail_fast = False

        # cancel_requested 列可能不存在（旧数据库），安全读取
        try:
            cancel_requested = bool(row['cancel_requested'])
        except (IndexError, KeyError):
            cancel_requested = False
        
        # task_type 列可能不存在（旧数据库），安全读取
        try:
            task_type = row['task_type'] or 'process'
        except (IndexError, KeyError):
            task_type = 'process'
        
        # task_params 列可能不存在（旧数据库），安全读取
        try:
            task_params_raw = row['task_params'] or '{}'
            task_params = json.loads(task_params_raw) if isinstance(task_params_raw, str) else task_params_raw
        except (IndexError, KeyError):
            task_params = {}
        
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
            concurrency=concurrency,
            fail_fast=fail_fast,
            task_type=task_type,
            task_params=task_params,
            cancel_requested=cancel_requested,
        )
