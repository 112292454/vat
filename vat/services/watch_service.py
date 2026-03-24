"""
Watch 模式服务

持续监控指定的 YouTube Playlist，发现新视频后自动提交处理任务。

核心职责：
- 定期同步 playlist，发现新增视频
- 过滤已有 pending/running task 或 unavailable 的视频
- 检查上一轮失败的视频，纳入重试列表
- 通过子进程执行 `vat process` 处理新视频
- 在 watch_sessions / watch_rounds 表中追踪状态

不负责：
- 实际的视频处理（由 vat process 子进程执行）
- GPU 选择（由 vat process 内部的 auto 选择处理）
"""

import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set, Tuple

from ..config import Config
from ..database import Database
from ..models import TaskStatus, DEFAULT_STAGE_SEQUENCE, TaskStep
from ..services.playlist_service import PlaylistService
from ..services.process_job_submitter import build_process_job_submitter
from ..utils.logger import setup_logger

logger = setup_logger("watch_service")


class WatchService:
    """
    Watch 模式主服务
    
    Args:
        config: VAT 配置
        db: 数据库实例
        playlist_ids: 要监控的 playlist ID 列表
        interval_minutes: 轮询间隔（分钟）
        stages: 处理阶段（逗号分隔字符串，如 "all"）
        gpu_device: GPU 设备选择（如 "auto", "cuda:0"）
        concurrency: 并发处理数
        force: 是否强制重处理
        fail_fast: 失败时是否停止
        once: 是否仅运行一次
    """
    
    def __init__(
        self,
        config: Config,
        db: Database,
        playlist_ids: List[str],
        interval_minutes: int = 60,
        stages: str = "all",
        gpu_device: str = "auto",
        concurrency: int = 1,
        force: bool = False,
        fail_fast: bool = False,
        once: bool = False,
        process_job_submitter: Optional[Callable[..., Optional[str]]] = None,
    ):
        self.config = config
        self.db = db
        self.playlist_service = PlaylistService(db, config)
        self.playlist_ids = playlist_ids
        self.interval_minutes = interval_minutes
        self.stages = stages
        self.gpu_device = gpu_device
        self.concurrency = concurrency
        self.force = force
        self.fail_fast = fail_fast
        self.once = once
        
        self._session_id: Optional[str] = None
        self._round_number = 0
        self._stop_requested = False
        self._pid = os.getpid()
        
        # 追踪每个视频的重试次数（跨轮次累计）
        self._retry_counts: Dict[str, int] = {}
        self._max_retries = config.watch.max_retries
        self._max_new_per_round = config.watch.max_new_videos_per_round
        
        # 追踪本 session 已提交的视频 ID（用于限定 retry 范围）
        # retry 只重试本 session 提交过的视频，不扫描全量 playlist 历史失败
        self._session_submitted_ids: Set[str] = set()
        
        # 安全上限：当 max_new_per_round=0（不限制）时的硬上限
        # 防止首次 sync 或异常情况一次提交数千视频
        self._safety_cap = 50
        self._process_job_submitter = process_job_submitter or build_process_job_submitter(config)
    
    def run(self):
        """
        运行 Watch 主循环
        
        流程：
        1. 检测是否已有同 playlist 的 running session
        2. 创建 session 记录
        3. 循环：sync → filter → submit → sleep
        4. 退出时更新 session 状态
        """
        # 检测并处理已有 session
        self._check_existing_sessions()
        
        # 创建新 session
        self._session_id = str(uuid.uuid4())[:8]
        self._create_session()
        
        logger.info(
            f"Watch 启动 (session={self._session_id}, "
            f"playlists={self.playlist_ids}, "
            f"interval={self.interval_minutes}min, "
            f"stages={self.stages}, once={self.once})"
        )
        
        try:
            while not self._stop_requested:
                self._round_number += 1
                round_start = datetime.now()
                
                logger.info(f"=== Watch 第 {self._round_number} 轮 ===")
                
                total_new = 0
                total_jobs = 0
                
                for pl_id in self.playlist_ids:
                    try:
                        new_count, job_count = self._process_playlist(pl_id)
                        total_new += new_count
                        total_jobs += job_count
                    except Exception as e:
                        logger.error(f"处理 playlist {pl_id} 出错: {e}")
                        self._record_round_error(pl_id, str(e))
                
                # 更新 session 统计
                self._update_session_stats(total_new, total_jobs)
                
                round_elapsed = (datetime.now() - round_start).total_seconds()
                logger.info(
                    f"第 {self._round_number} 轮完成: "
                    f"发现 {total_new} 个新视频, 提交 {total_jobs} 个任务, "
                    f"耗时 {round_elapsed:.0f}s"
                )
                
                if self.once:
                    logger.info("单次模式，退出")
                    break
                
                # 计算下次检查时间并等待
                next_check = datetime.now() + timedelta(minutes=self.interval_minutes)
                self._update_session_next_check(next_check)
                
                logger.info(f"下次检查: {next_check.strftime('%H:%M:%S')}（{self.interval_minutes} 分钟后）")
                
                # 可中断的等待
                wait_seconds = self.interval_minutes * 60
                for _ in range(wait_seconds):
                    if self._stop_requested:
                        break
                    time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止 Watch")
        except Exception as e:
            logger.error(f"Watch 异常退出: {e}")
            self._update_session_error(str(e))
            raise
        finally:
            self._finalize_session()
    
    def stop(self):
        """请求停止 Watch（可从其他线程调用）"""
        self._stop_requested = True
    
    # ========== Playlist 处理 ==========
    
    def _process_playlist(self, playlist_id: str) -> Tuple[int, int]:
        """
        处理单个 playlist：同步、发现新视频、提交任务
        
        Returns:
            (new_videos_found, jobs_submitted)
        """
        round_started_at = datetime.now()
        
        # 1. 验证 playlist 存在
        pl = self.playlist_service.get_playlist(playlist_id)
        if not pl:
            logger.warning(f"Playlist 不存在: {playlist_id}，跳过")
            return 0, 0
        
        logger.info(f"同步 playlist: {pl.title} ({playlist_id})")
        
        # 2. 同步 playlist（增量获取新视频列表）
        try:
            sync_result = self.playlist_service.sync_playlist(
                playlist_url=pl.source_url,
                auto_add_videos=True,
                fetch_upload_dates=True,
                target_playlist_id=playlist_id,
            )
            new_video_ids = sync_result.new_videos if sync_result.new_videos else []
            logger.info(
                f"同步完成: 新增 {len(new_video_ids)} 个视频, "
                f"总计 {sync_result.total_videos} 个"
            )
        except Exception as e:
            logger.error(f"同步 playlist {playlist_id} 失败: {e}")
            self._record_round(
                playlist_id=playlist_id,
                started_at=round_started_at,
                error=str(e),
            )
            raise
        
        # 3. 从新增视频中筛选可处理的（排除 unavailable、running 等）
        processable = self._get_processable_videos(new_video_ids)
        
        # 4. 从本 session 之前提交过的视频中查找重试候选
        # 关键设计：只重试本 session 提交过且失败的视频，
        # 不扫描全量 playlist 历史，避免首轮/误触时重处理数千视频
        session_submitted_list = list(self._session_submitted_ids)
        retry_ids = self._get_retry_candidates(session_submitted_list) if session_submitted_list else []
        
        # 5. 合并：新视频 + 重试视频（去重）
        retry_id_set = set(retry_ids)
        all_candidates = list(dict.fromkeys(processable + retry_ids))
        
        new_count = len([v for v in all_candidates if v not in retry_id_set])
        retry_count = len(all_candidates) - new_count
        
        # 6. 应用数量限制
        effective_limit = self._max_new_per_round if self._max_new_per_round > 0 else self._safety_cap
        if len(all_candidates) > effective_limit:
            logger.warning(
                f"候选视频 {len(all_candidates)} 个超过限制 {effective_limit}，截断。"
                f"（新增={new_count}, 重试={retry_count}）"
            )
            all_candidates = all_candidates[:effective_limit]
            # 重新计算截断后的计数
            new_count = len([v for v in all_candidates if v not in retry_id_set])
            retry_count = len(all_candidates) - new_count
        
        if not all_candidates:
            logger.info(f"Playlist {playlist_id}: 无需处理的视频")
            self._record_round(
                playlist_id=playlist_id,
                started_at=round_started_at,
                new_videos_found=0,
            )
            return 0, 0
        
        logger.info(
            f"Playlist {playlist_id}: {new_count} 个新视频 + {retry_count} 个重试"
        )
        
        # 8. 提交处理任务
        job_id = self._submit_process_job(all_candidates, playlist_id)
        
        # 记录本次提交的视频到 session 级别追踪（用于下一轮 retry 范围限定）
        self._session_submitted_ids.update(all_candidates)
        
        # 更新重试计数
        for vid in retry_ids:
            if vid in all_candidates:
                self._retry_counts[vid] = self._retry_counts.get(vid, 0) + 1
        
        # 9. 记录本轮状态
        self._record_round(
            playlist_id=playlist_id,
            started_at=round_started_at,
            new_videos_found=new_count,
            jobs_submitted=1 if job_id else 0,
            submitted_video_ids=all_candidates,
            submitted_job_ids=[job_id] if job_id else [],
            retry_video_ids=[v for v in all_candidates if v in retry_ids],
        )
        
        return new_count, 1 if job_id else 0
    
    # ========== 视频筛选 ==========
    
    def _get_processable_videos(self, video_ids: List[str]) -> List[str]:
        """
        从视频列表中筛选出可以提交处理的视频
        
        排除条件：
        1. 视频有任何目标阶段处于 running 状态
        2. 视频 metadata 中标记为 unavailable
        3. 视频所有目标阶段都已 completed（除非 force=True）
        4. 视频有 failed task 且已超过最大重试次数
        """
        processable = []
        
        for vid in video_ids:
            video = self.db.get_video(vid)
            if not video:
                continue
            
            # 排除 unavailable 视频
            meta = video.metadata or {}
            if meta.get('unavailable', False):
                continue
            
            # 预约/首播视频（is_upcoming）：提交任务，download 阶段会自动阻塞等待
            # 不再在这里跳过，让 pipeline 处理
            
            # 检查任务状态
            has_running = False
            has_failed = False
            all_completed = True
            
            for step in DEFAULT_STAGE_SEQUENCE:
                task = self._get_latest_task(vid, step)
                if task and task['status'] == TaskStatus.RUNNING.value:
                    has_running = True
                    break
                if task and task['status'] == TaskStatus.FAILED.value:
                    has_failed = True
                if not task or task['status'] != TaskStatus.COMPLETED.value:
                    all_completed = False
            
            if has_running:
                continue
            
            if all_completed and not self.force:
                continue
            
            # 排除超过重试上限的失败视频
            if has_failed and self._retry_counts.get(vid, 0) >= self._max_retries:
                continue
            
            processable.append(vid)
        
        return processable
    
    def _get_retry_candidates(self, video_ids: List[str]) -> List[str]:
        """
        从视频列表中找出之前失败且可以重试的视频
        
        条件：
        - 视频有 failed task
        - 不在 running 状态
        - 未超过最大重试次数
        """
        candidates = []
        
        for vid in video_ids:
            # 检查重试次数限制
            if self._retry_counts.get(vid, 0) >= self._max_retries:
                continue
            
            video = self.db.get_video(vid)
            if not video:
                continue
            
            meta = video.metadata or {}
            if meta.get('unavailable', False):
                continue
            
            has_failed = False
            has_running = False
            all_completed = True
            
            for step in DEFAULT_STAGE_SEQUENCE:
                task = self._get_latest_task(vid, step)
                if task:
                    if task['status'] == TaskStatus.RUNNING.value:
                        has_running = True
                        break
                    if task['status'] == TaskStatus.FAILED.value:
                        has_failed = True
                    if task['status'] != TaskStatus.COMPLETED.value:
                        all_completed = False
                else:
                    all_completed = False
            
            if has_running or all_completed:
                continue
            
            if has_failed:
                candidates.append(vid)
        
        return candidates
    
    def _get_latest_task(self, video_id: str, step: TaskStep):
        """获取视频某个阶段的最新 task 记录"""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT status FROM tasks WHERE video_id = ? AND step = ? ORDER BY id DESC LIMIT 1",
                (video_id, step.value)
            ).fetchone()
            return row
    
    # ========== 任务提交 ==========
    
    def _submit_process_job(self, video_ids: List[str], playlist_id: str) -> Optional[str]:
        """
        通过注入的 submitter 提交 process job 处理视频。
        
        Returns:
            job_id 字符串（用于追踪），失败返回 None
        """
        if not video_ids:
            return None
        
        # stages 字符串转列表（JobManager 接口需要 List[str]）
        steps = self.stages.split(",") if self.stages else ["all"]
        
        try:
            job_id = self._process_job_submitter(
                video_ids=video_ids,
                steps=steps,
                gpu_device=self.gpu_device,
                force=self.force,
                concurrency=self.concurrency,
                playlist_id=playlist_id,
                fail_fast=self.fail_fast,
            )
            logger.info(
                f"通过 process submitter 提交处理任务: {job_id}, "
                f"{len(video_ids)} 个视频, stages={self.stages}, playlist={playlist_id}"
            )
            return job_id
            
        except Exception as e:
            logger.error(f"提交任务失败: {e}")
            return None
    
    # ========== Session 管理 ==========
    
    def _check_existing_sessions(self):
        """检查是否已有同 playlist 的 running session"""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT session_id, playlist_ids, pid FROM watch_sessions WHERE status = 'running'"
            ).fetchall()
        
        for row in rows:
            existing_playlists = json.loads(row['playlist_ids'])
            overlap = set(existing_playlists) & set(self.playlist_ids)
            
            if overlap:
                existing_pid = row['pid']
                session_id = row['session_id']
                
                # 检查 PID 是否存活
                if existing_pid and self._is_pid_alive(existing_pid):
                    if self.force:
                        logger.warning(
                            f"强制覆盖: 停止已有 session {session_id} "
                            f"(pid={existing_pid}, playlists={overlap})"
                        )
                        self._stop_existing_session(session_id, existing_pid)
                    else:
                        raise RuntimeError(
                            f"Playlist {overlap} 已被 session {session_id} "
                            f"(pid={existing_pid}) 监控中。"
                            f"使用 --force 强制覆盖，或先停止已有 session。"
                        )
                else:
                    # PID 已死，标记为 stopped
                    logger.info(
                        f"清理死亡 session {session_id} (pid={existing_pid})"
                    )
                    self._mark_session_stopped(session_id, error="进程已不存在")
    
    def _stop_existing_session(self, session_id: str, pid: int):
        """停止已有的 session（发送 SIGTERM）"""
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
            # 等待一小段时间让进程退出
            time.sleep(2)
        except (ProcessLookupError, PermissionError):
            pass
        self._mark_session_stopped(session_id, error="被新 session 强制替换")
    
    def _create_session(self):
        """在数据库中创建 watch session 记录"""
        now = datetime.now()
        config_json = json.dumps({
            'interval': self.interval_minutes,
            'stages': self.stages,
            'gpu': self.gpu_device,
            'concurrency': self.concurrency,
            'force': self.force,
            'fail_fast': self.fail_fast,
            'once': self.once,
        }, ensure_ascii=False)
        
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO watch_sessions 
                (session_id, playlist_ids, status, pid, config, started_at, next_check_at)
                VALUES (?, ?, 'running', ?, ?, ?, ?)
            """, (
                self._session_id,
                json.dumps(self.playlist_ids),
                self._pid,
                config_json,
                now.isoformat(),
                now.isoformat(),
            ))
    
    def _update_session_stats(self, new_found: int, jobs_submitted: int):
        """更新 session 的累计统计"""
        now = datetime.now()
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE watch_sessions SET
                    total_rounds = total_rounds + 1,
                    total_new_found = total_new_found + ?,
                    total_jobs_submitted = total_jobs_submitted + ?,
                    last_check_at = ?
                WHERE session_id = ?
            """, (new_found, jobs_submitted, now.isoformat(), self._session_id))
    
    def _update_session_next_check(self, next_check: datetime):
        """更新下次检查时间"""
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE watch_sessions SET next_check_at = ? WHERE session_id = ?",
                (next_check.isoformat(), self._session_id)
            )
    
    def _update_session_error(self, error: str):
        """更新 session 错误状态"""
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE watch_sessions SET status = 'error', error = ? WHERE session_id = ?",
                (error, self._session_id)
            )
    
    def _finalize_session(self):
        """Watch 退出时更新 session 状态"""
        if not self._session_id:
            return
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE watch_sessions SET 
                    status = CASE WHEN status = 'error' THEN status ELSE 'stopped' END,
                    stopped_at = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), self._session_id))
        logger.info(f"Watch session {self._session_id} 已结束")
    
    def _mark_session_stopped(self, session_id: str, error: Optional[str] = None):
        """将指定 session 标记为 stopped"""
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE watch_sessions SET status = 'stopped', stopped_at = ?, error = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), error, session_id))
    
    # ========== Round 记录 ==========
    
    def _record_round(
        self,
        playlist_id: str,
        started_at: datetime,
        new_videos_found: int = 0,
        jobs_submitted: int = 0,
        submitted_video_ids: Optional[List[str]] = None,
        submitted_job_ids: Optional[List[str]] = None,
        retry_video_ids: Optional[List[str]] = None,
        error: Optional[str] = None,
    ):
        """记录一轮检查的结果"""
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO watch_rounds
                (session_id, round_number, playlist_id, started_at, finished_at,
                 new_videos_found, jobs_submitted, submitted_video_ids, submitted_job_ids,
                 retry_video_ids, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self._session_id,
                self._round_number,
                playlist_id,
                started_at.isoformat(),
                datetime.now().isoformat(),
                new_videos_found,
                jobs_submitted,
                json.dumps(submitted_video_ids) if submitted_video_ids else None,
                json.dumps(submitted_job_ids) if submitted_job_ids else None,
                json.dumps(retry_video_ids) if retry_video_ids else None,
                error,
            ))
    
    def _record_round_error(self, playlist_id: str, error: str):
        """记录本轮错误"""
        self._record_round(
            playlist_id=playlist_id,
            started_at=datetime.now(),
            error=error,
        )
    
    # ========== 工具方法 ==========
    
    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """检查进程是否存活"""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
