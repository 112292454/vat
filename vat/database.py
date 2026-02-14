"""
SQLite数据库操作层

子阶段独立化设计：
- 每个细粒度阶段（WHISPER, SPLIT, OPTIMIZE, TRANSLATE 等）都有独立的任务记录
- 支持 Playlist 管理
- 支持数据库迁移
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from contextlib import contextmanager

from .models import (
    Video, Task, Playlist, SourceType, TaskStep, TaskStatus,
    DEFAULT_STAGE_SEQUENCE, STAGE_DEPENDENCIES
)
from .utils.logger import setup_logger

logger = setup_logger("database")

# 当前数据库版本
DB_VERSION = 3


class Database:
    """数据库管理类"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建版本表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS db_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            
            # 获取当前版本
            cursor.execute("SELECT version FROM db_version LIMIT 1")
            row = cursor.fetchone()
            current_version = row['version'] if row else 0
            
            # 创建videos表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    title TEXT,
                    output_dir TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    playlist_id TEXT,
                    playlist_index INTEGER,
                    FOREIGN KEY (playlist_id) REFERENCES playlists(id)
                )
            """)
            
            # 创建playlists表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS playlists (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    channel TEXT,
                    channel_id TEXT,
                    video_count INTEGER DEFAULT 0,
                    last_synced_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            """)
            
            # 创建tasks表（细粒度阶段）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    gpu_id INTEGER,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos(id)
                )
            """)
            
            # 执行迁移（先迁移再创建索引，确保列存在）
            if current_version < DB_VERSION:
                self._migrate_database(conn, current_version, DB_VERSION)
                
                # 更新版本
                cursor.execute("DELETE FROM db_version")
                cursor.execute("INSERT INTO db_version (version) VALUES (?)", (DB_VERSION,))
            
            # 创建索引（迁移后执行，确保所有列都存在）
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON tasks(video_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_step_status ON tasks(step, status)")
            
            # playlist_id 索引只在列存在时创建
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_playlist_id ON videos(playlist_id)")
            except sqlite3.OperationalError:
                pass  # 列不存在时跳过
    
    def _migrate_database(self, conn, from_version: int, to_version: int):
        """执行数据库迁移"""
        cursor = conn.cursor()
        
        if from_version < 2:
            # 迁移 v1 -> v2: 添加 playlist 相关字段
            logger.info("执行数据库迁移 v1 -> v2")
            
            # 添加 videos 表的 playlist 字段（如果不存在）
            try:
                cursor.execute("ALTER TABLE videos ADD COLUMN playlist_id TEXT")
            except sqlite3.OperationalError:
                pass  # 字段已存在
            
            try:
                cursor.execute("ALTER TABLE videos ADD COLUMN playlist_index INTEGER")
            except sqlite3.OperationalError:
                pass
            
            # 迁移旧的 ASR/TRANSLATE 任务到新的细粒度阶段
            # ASR -> WHISPER + SPLIT（复制记录并更新 step）
            cursor.execute("SELECT video_id, status, error_message, started_at, completed_at FROM tasks WHERE step = 'asr'")
            asr_tasks = cursor.fetchall()
            for row in asr_tasks:
                video_id, status, error_message, started_at, completed_at = row['video_id'], row['status'], row['error_message'], row['started_at'], row['completed_at']
                # 插入 WHISPER 记录
                cursor.execute("""
                    INSERT OR IGNORE INTO tasks (video_id, step, status, error_message, started_at, completed_at)
                    VALUES (?, 'whisper', ?, ?, ?, ?)
                """, (video_id, status, error_message, started_at, completed_at))
                # 插入 SPLIT 记录
                cursor.execute("""
                    INSERT OR IGNORE INTO tasks (video_id, step, status, error_message, started_at, completed_at)
                    VALUES (?, 'split', ?, ?, ?, ?)
                """, (video_id, status, error_message, started_at, completed_at))
            
            # TRANSLATE -> OPTIMIZE + TRANSLATE
            cursor.execute("SELECT video_id, status, error_message, started_at, completed_at FROM tasks WHERE step = 'translate'")
            translate_tasks = cursor.fetchall()
            for row in translate_tasks:
                video_id, status, error_message, started_at, completed_at = row['video_id'], row['status'], row['error_message'], row['started_at'], row['completed_at']
                # 插入 OPTIMIZE 记录
                cursor.execute("""
                    INSERT OR IGNORE INTO tasks (video_id, step, status, error_message, started_at, completed_at)
                    VALUES (?, 'optimize', ?, ?, ?, ?)
                """, (video_id, status, error_message, started_at, completed_at))
                # 更新原 translate 记录保持不变（已经是 translate）
            
            # 删除旧的 asr 记录（已拆分为 whisper + split）
            cursor.execute("DELETE FROM tasks WHERE step = 'asr'")
            
            logger.info("数据库迁移 v1 -> v2 完成")
        
        if from_version < 3:
            # 迁移 v2 -> v3: 添加 playlist_videos 关联表（多对多）
            logger.info("执行数据库迁移 v2 -> v3: 添加 playlist_videos 关联表")
            
            # 创建关联表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS playlist_videos (
                    playlist_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    playlist_index INTEGER,
                    upload_order_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (playlist_id, video_id),
                    FOREIGN KEY (playlist_id) REFERENCES playlists(id),
                    FOREIGN KEY (video_id) REFERENCES videos(id)
                )
            """)
            
            # 迁移现有数据：从 videos.playlist_id 迁移到 playlist_videos
            cursor.execute("""
                INSERT OR IGNORE INTO playlist_videos (playlist_id, video_id, playlist_index, upload_order_index)
                SELECT playlist_id, id, playlist_index, json_extract(metadata, '$.upload_order_index')
                FROM videos
                WHERE playlist_id IS NOT NULL
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pv_playlist ON playlist_videos(playlist_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pv_video ON playlist_videos(video_id)")
            
            logger.info("数据库迁移 v2 -> v3 完成")
    
    def add_video(self, video: Video) -> None:
        """添加视频记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO videos 
                (id, source_type, source_url, title, output_dir, metadata, created_at, updated_at, playlist_id, playlist_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video.id,
                video.source_type.value,
                video.source_url,
                video.title,
                video.output_dir,
                json.dumps(video.metadata, ensure_ascii=False),
                video.created_at,
                video.updated_at,
                video.playlist_id,
                video.playlist_index
            ))
    
    def get_video(self, video_id: str) -> Optional[Video]:
        """获取视频记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_video(row)
            return None
    
    def _row_to_video(self, row: sqlite3.Row) -> Video:
        """将数据库行转换为 Video 对象"""
        return Video(
            id=row['id'],
            source_type=SourceType(row['source_type']),
            source_url=row['source_url'],
            title=row['title'],
            output_dir=row['output_dir'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            playlist_id=row['playlist_id'] if 'playlist_id' in row.keys() else None,
            playlist_index=row['playlist_index'] if 'playlist_index' in row.keys() else None
        )
    
    def update_video(self, video_id: str, **kwargs) -> None:
        """更新视频记录"""
        allowed_fields = {'title', 'output_dir', 'metadata', 'playlist_id', 'playlist_index'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return
        
        updates['updated_at'] = datetime.now()
        
        # 处理metadata字段
        if 'metadata' in updates:
            updates['metadata'] = json.dumps(updates['metadata'], ensure_ascii=False)
        
        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [video_id]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE videos SET {set_clause} WHERE id = ?", values)
    
    def list_videos(self, source_type: Optional[SourceType] = None, playlist_id: Optional[str] = None) -> List[Video]:
        """列出视频
        
        Args:
            source_type: 按来源类型过滤
            playlist_id: 按 Playlist 过滤（使用关联表，支持多对多）
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if playlist_id:
                # 使用关联表查询（多对多）
                query = """
                    SELECT v.*, pv.playlist_index as pv_index, pv.upload_order_index as pv_order_index
                    FROM videos v
                    INNER JOIN playlist_videos pv ON v.id = pv.video_id
                    WHERE pv.playlist_id = ?
                """
                params = [playlist_id]
                if source_type:
                    query += " AND v.source_type = ?"
                    params.append(source_type.value)
                query += " ORDER BY pv.playlist_index ASC"
                cursor.execute(query, params)
                
                videos = []
                for row in cursor.fetchall():
                    video = self._row_to_video(row)
                    # 使用关联表中的索引覆盖
                    if row['pv_index'] is not None:
                        video.playlist_index = row['pv_index']
                    videos.append(video)
                return videos
            else:
                # 不按 playlist 过滤时，普通查询
                query = "SELECT * FROM videos WHERE 1=1"
                params = []
                if source_type:
                    query += " AND source_type = ?"
                    params.append(source_type.value)
                query += " ORDER BY created_at DESC"
                cursor.execute(query, params)
                return [self._row_to_video(row) for row in cursor.fetchall()]
    
    def list_videos_paginated(
        self,
        page: int = 1,
        per_page: int = 50,
        status: Optional[str] = None,
        search: Optional[str] = None,
        playlist_id: Optional[str] = None,
        stage_filters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """分页列出视频（SQL 层面过滤，避免全量加载）
        
        Args:
            page: 页码（1-based）
            per_page: 每页数量
            status: 状态过滤 ('completed', 'failed', 'running', None=全部)
            search: 搜索关键词（匹配标题/频道/ID）
            playlist_id: Playlist 过滤
            stage_filters: 阶段级过滤（AND 逻辑），如 {"download": "failed", "whisper": "pending"}
                           支持的状态值: 'pending'（就绪待运行）, 'completed', 'failed'
            
        Returns:
            {'videos': List[Video], 'total': int, 'page': int, 'total_pages': int}
        """
        total_steps = len(DEFAULT_STAGE_SEQUENCE)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 基础查询：根据是否有 playlist_id 决定 JOIN
            if playlist_id:
                base_from = """
                    FROM videos v
                    INNER JOIN playlist_videos pv ON v.id = pv.video_id AND pv.playlist_id = ?
                """
                base_params = [playlist_id]
                order_by = "ORDER BY pv.playlist_index ASC"
            else:
                base_from = "FROM videos v"
                base_params = []
                order_by = "ORDER BY v.created_at DESC"
            
            # 构建 WHERE 条件
            where_clauses = []
            where_params = []
            
            # 搜索过滤
            if search:
                where_clauses.append("""
                    (v.title LIKE ? OR v.id LIKE ? OR 
                     json_extract(v.metadata, '$.channel') LIKE ? OR
                     json_extract(v.metadata, '$.translated.title_translated') LIKE ?)
                """)
                pattern = f"%{search}%"
                where_params.extend([pattern, pattern, pattern, pattern])
            
            # 状态过滤（需要 JOIN tasks 表）
            if status:
                # 子查询：计算每个视频的状态
                status_subquery = f"""
                    v.id IN (
                        SELECT vs.video_id FROM (
                            SELECT 
                                lt.video_id,
                                SUM(CASE WHEN lt.status IN ('completed', 'skipped') THEN 1 ELSE 0 END) as completed_count,
                                SUM(CASE WHEN lt.status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                                SUM(CASE WHEN lt.status = 'running' THEN 1 ELSE 0 END) as running_count
                            FROM (
                                SELECT video_id, step, status,
                                       ROW_NUMBER() OVER (PARTITION BY video_id, step ORDER BY id DESC) as rn
                                FROM tasks
                            ) lt
                            WHERE lt.rn = 1
                            GROUP BY lt.video_id
                        ) vs
                        WHERE {
                            f"vs.completed_count >= {total_steps}" if status == 'completed' else
                            "vs.failed_count > 0" if status == 'failed' else
                            "vs.running_count > 0" if status == 'running' else "1=1"
                        }
                    )
                """
                where_clauses.append(status_subquery)
            
            # 阶段级过滤（AND 逻辑）
            if stage_filters:
                for step_name, step_status in stage_filters.items():
                    if step_status == 'completed':
                        # 该阶段最新任务为 completed/skipped
                        where_clauses.append("""
                            v.id IN (
                                SELECT t_sf.video_id FROM (
                                    SELECT video_id, status,
                                           ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY id DESC) as rn
                                    FROM tasks WHERE step = ?
                                ) t_sf WHERE t_sf.rn = 1 AND t_sf.status IN ('completed', 'skipped')
                            )
                        """)
                        where_params.append(step_name)
                    elif step_status == 'failed':
                        # 该阶段最新任务为 failed
                        where_clauses.append("""
                            v.id IN (
                                SELECT t_sf.video_id FROM (
                                    SELECT video_id, status,
                                           ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY id DESC) as rn
                                    FROM tasks WHERE step = ?
                                ) t_sf WHERE t_sf.rn = 1 AND t_sf.status = 'failed'
                            )
                        """)
                        where_params.append(step_name)
                    elif step_status == 'pending':
                        # "pending" = 该阶段未完成/未失败 AND 所有前置阶段已完成
                        # 1) 该阶段没有 completed/skipped/failed/running 的最新记录
                        where_clauses.append("""
                            v.id NOT IN (
                                SELECT t_sf.video_id FROM (
                                    SELECT video_id, status,
                                           ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY id DESC) as rn
                                    FROM tasks WHERE step = ?
                                ) t_sf WHERE t_sf.rn = 1
                                    AND t_sf.status IN ('completed', 'skipped', 'failed', 'running')
                            )
                        """)
                        where_params.append(step_name)
                        # 2) 所有前置阶段已完成（线性依赖链）
                        try:
                            step_enum = TaskStep(step_name)
                            prereqs = STAGE_DEPENDENCIES.get(step_enum, [])
                            for prereq in prereqs:
                                where_clauses.append("""
                                    v.id IN (
                                        SELECT t_sf.video_id FROM (
                                            SELECT video_id, status,
                                                   ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY id DESC) as rn
                                            FROM tasks WHERE step = ?
                                        ) t_sf WHERE t_sf.rn = 1
                                            AND t_sf.status IN ('completed', 'skipped')
                                    )
                                """)
                                where_params.append(prereq.value)
                        except ValueError:
                            pass  # 未知阶段名，忽略前置条件
            
            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)
            
            all_params = base_params + where_params
            
            # 计算总数
            count_sql = f"SELECT COUNT(*) as cnt {base_from} {where_sql}"
            cursor.execute(count_sql, all_params)
            total = cursor.fetchone()['cnt']
            
            # per_page=0 表示显示全部
            if per_page <= 0:
                per_page = total if total > 0 else 1
                total_pages = 1
                offset = 0
            else:
                total_pages = (total + per_page - 1) // per_page if total > 0 else 1
                offset = (page - 1) * per_page
            
            # 查询当前页数据
            if playlist_id:
                select_sql = f"""
                    SELECT v.*, pv.playlist_index as pv_index, pv.upload_order_index as pv_order_index
                    {base_from} {where_sql} {order_by}
                    LIMIT ? OFFSET ?
                """
            else:
                select_sql = f"""
                    SELECT v.*
                    {base_from} {where_sql} {order_by}
                    LIMIT ? OFFSET ?
                """
            
            cursor.execute(select_sql, all_params + [per_page, offset])
            
            videos = []
            for row in cursor.fetchall():
                video = self._row_to_video(row)
                if playlist_id and row.get('pv_index') is not None:
                    video.playlist_index = row['pv_index']
                videos.append(video)
            
            return {
                'videos': videos,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages
            }
    
    def add_task(self, task: Task) -> int:
        """添加任务记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tasks 
                (video_id, step, status, gpu_id, started_at, completed_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task.video_id,
                task.step.value,
                task.status.value,
                task.gpu_id,
                task.started_at,
                task.completed_at,
                task.error_message
            ))
            return cursor.lastrowid
    
    def get_task(self, video_id: str, step: TaskStep) -> Optional[Task]:
        """获取特定步骤的任务"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tasks 
                WHERE video_id = ? AND step = ?
                ORDER BY id DESC LIMIT 1
            """, (video_id, step.value))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_task(row)
            return None
    
    def get_tasks(self, video_id: str) -> List[Task]:
        """获取视频的所有任务"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tasks WHERE video_id = ? ORDER BY id
            """, (video_id,))
            
            return [self._row_to_task(row) for row in cursor.fetchall()]
    
    def update_task_status(self, video_id: str, step: TaskStep, 
                          status: TaskStatus, **kwargs) -> None:
        """更新任务状态
        
        Args:
            video_id: 视频ID
            step: 任务步骤（细粒度阶段）
            status: 任务状态
            **kwargs: 可选参数，包括:
                - error_message: 错误信息
                - gpu_id: GPU编号
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 获取最新的该步骤任务
            cursor.execute("""
                SELECT id FROM tasks 
                WHERE video_id = ? AND step = ?
                ORDER BY id DESC LIMIT 1
            """, (video_id, step.value))
            row = cursor.fetchone()
            
            if not row:
                # 如果任务不存在，创建新任务
                task = Task(
                    video_id=video_id,
                    step=step,
                    status=status,
                    **kwargs
                )
                self.add_task(task)
                return
            
            task_id = row['id']
            
            # 更新任务
            updates = {'status': status.value}
            if status == TaskStatus.RUNNING and 'started_at' not in kwargs:
                updates['started_at'] = datetime.now()
            if status == TaskStatus.COMPLETED and 'completed_at' not in kwargs:
                updates['completed_at'] = datetime.now()
            # 任务开始运行或成功完成时清除旧的错误信息
            if status in (TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.SKIPPED) and 'error_message' not in kwargs:
                updates['error_message'] = None
            
            updates.update(kwargs)
            
            set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [task_id]
            
            cursor.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", values)
    
    def get_videos_by_task_status(self, step: Optional[TaskStep] = None, 
                                  status: Optional[TaskStatus] = None) -> List[str]:
        """根据任务状态获取视频ID列表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT DISTINCT video_id FROM tasks WHERE 1=1"
            params = []
            
            if step:
                query += " AND step = ?"
                params.append(step.value)
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            cursor.execute(query, params)
            return [row['video_id'] for row in cursor.fetchall()]
    
    def get_pending_steps(self, video_id: str) -> List[TaskStep]:
        """获取视频的待处理步骤（细粒度阶段）"""
        completed_steps = set()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT step FROM tasks 
                WHERE video_id = ? AND status = ?
            """, (video_id, TaskStatus.COMPLETED.value))
            
            for row in cursor.fetchall():
                try:
                    completed_steps.add(TaskStep(row['step']))
                except ValueError:
                    pass  # 忽略无效的阶段名
        
        return [step for step in DEFAULT_STAGE_SEQUENCE if step not in completed_steps]
    
    def is_step_completed(self, video_id: str, step: TaskStep) -> bool:
        """检查步骤是否已完成"""
        task = self.get_task(video_id, step)
        return task is not None and task.status == TaskStatus.COMPLETED
    
    def delete_tasks_for_video(self, video_id: str) -> int:
        """
        删除视频的所有任务记录
        
        用于重新创建视频时清理旧任务，避免重复记录。
        
        Args:
            video_id: 视频ID
            
        Returns:
            删除的任务数量
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tasks WHERE video_id = ?", (video_id,))
            return cursor.rowcount
    
    def invalidate_downstream_tasks(self, video_id: str, from_step: TaskStep) -> int:
        """
        使某阶段之后的所有任务失效（设为待处理）
        
        当强制重新处理某个阶段时，后续阶段即使之前已完成也应该被重置，
        因为它们的输入依赖于被重新处理的阶段。
        
        Args:
            video_id: 视频ID
            from_step: 起始阶段（该阶段之后的所有阶段将被重置）
            
        Returns:
            被重置的任务数量
        """
        # 获取阶段顺序索引
        try:
            from_index = DEFAULT_STAGE_SEQUENCE.index(from_step)
        except ValueError:
            return 0
        
        # 获取需要重置的后续阶段
        downstream_steps = DEFAULT_STAGE_SEQUENCE[from_index + 1:]
        if not downstream_steps:
            return 0
        
        step_values = [s.value for s in downstream_steps]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 将后续已完成的任务重置为待处理
            placeholders = ','.join('?' * len(step_values))
            cursor.execute(f"""
                UPDATE tasks 
                SET status = 'pending', started_at = NULL, completed_at = NULL, error_message = NULL
                WHERE video_id = ? AND step IN ({placeholders}) AND status = 'completed'
            """, [video_id] + step_values)
            
            return cursor.rowcount
    
    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """将数据库行转换为 Task 对象"""
        return Task(
            id=row['id'],
            video_id=row['video_id'],
            step=TaskStep(row['step']),
            status=TaskStatus(row['status']),
            gpu_id=row['gpu_id'],
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            error_message=row['error_message']
        )
    
    def batch_get_video_progress(self, video_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        批量获取视频的任务进度（消除 N+1 查询）
        
        一次 SQL 查询获取所有视频的已完成阶段数，替代逐个调用
        get_tasks() + get_pending_steps() 的模式。
        
        Args:
            video_ids: 要查询的视频ID列表，None 表示查询全部
            
        Returns:
            {video_id: {"completed": int, "total": 7, "progress": int,
                        "task_status": {step: {"status": str, "error": str|None}}}}
        """
        total_steps = len(DEFAULT_STAGE_SEQUENCE)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if video_ids:
                placeholders = ','.join('?' * len(video_ids))
                cursor.execute(f"""
                    SELECT video_id, step, status, error_message,
                           ROW_NUMBER() OVER (PARTITION BY video_id, step ORDER BY id DESC) as rn
                    FROM tasks
                    WHERE video_id IN ({placeholders})
                """, video_ids)
            else:
                cursor.execute("""
                    SELECT video_id, step, status, error_message,
                           ROW_NUMBER() OVER (PARTITION BY video_id, step ORDER BY id DESC) as rn
                    FROM tasks
                """)
            
            # 按视频分组，每个 step 只取最新记录 (rn=1)
            video_tasks: Dict[str, Dict[str, Dict]] = {}
            for row in cursor.fetchall():
                if row['rn'] != 1:
                    continue
                vid = row['video_id']
                if vid not in video_tasks:
                    video_tasks[vid] = {}
                video_tasks[vid][row['step']] = {
                    "status": row['status'],
                    "error": row['error_message']
                }
            
            # 构建结果
            result = {}
            target_ids = video_ids if video_ids else list(video_tasks.keys())
            for vid in target_ids:
                tasks = video_tasks.get(vid, {})
                # 构建完整的 task_status（7个阶段）
                task_status = {}
                completed_count = 0
                has_failed = False
                has_running = False
                has_preceding_failure = False
                for step in DEFAULT_STAGE_SEQUENCE:
                    step_val = step.value
                    if step_val in tasks:
                        t = tasks[step_val]
                        task_status[step_val] = t
                        if t["status"] in ("completed", "skipped"):
                            completed_count += 1
                        if t["status"] == "failed":
                            has_failed = True
                            has_preceding_failure = True
                        if t["status"] == "running":
                            has_running = True
                    else:
                        # 前置阶段 failed → 后续未执行的阶段标记为 blocked 而非 pending
                        status = "blocked" if has_preceding_failure else "pending"
                        task_status[step_val] = {"status": status, "error": None}
                
                progress = int(completed_count / total_steps * 100)
                result[vid] = {
                    "completed": completed_count,
                    "total": total_steps,
                    "progress": progress,
                    "task_status": task_status,
                    "has_failed": has_failed,
                    "has_running": has_running,
                }
            
            return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息（包含视频级别的状态统计）
        
        Returns:
            dict with keys:
                total_videos: 视频总数
                completed_videos: 所有7步都完成的视频数
                failed_videos: 有任意步骤失败的视频数
                running_videos: 有任意步骤正在运行的视频数
                tasks_by_status: 各步骤状态统计
        """
        total_steps = len(DEFAULT_STAGE_SEQUENCE)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 视频总数
            cursor.execute("SELECT COUNT(*) as count FROM videos")
            total_videos = cursor.fetchone()['count']
            
            # 视频级别状态统计（基于 tasks 表，每个 video+step 取最新记录）
            # 使用子查询取每个 video_id+step 的最新状态
            cursor.execute(f"""
                WITH latest_tasks AS (
                    SELECT video_id, step, status,
                           ROW_NUMBER() OVER (PARTITION BY video_id, step ORDER BY id DESC) as rn
                    FROM tasks
                ),
                video_stats AS (
                    SELECT 
                        video_id,
                        SUM(CASE WHEN status IN ('completed', 'skipped') THEN 1 ELSE 0 END) as completed_count,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                        SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running_count
                    FROM latest_tasks
                    WHERE rn = 1
                    GROUP BY video_id
                )
                SELECT 
                    SUM(CASE WHEN completed_count >= {total_steps} THEN 1 ELSE 0 END) as completed_videos,
                    SUM(CASE WHEN failed_count > 0 THEN 1 ELSE 0 END) as failed_videos,
                    SUM(CASE WHEN running_count > 0 THEN 1 ELSE 0 END) as running_videos
                FROM video_stats
            """)
            row = cursor.fetchone()
            completed_videos = row['completed_videos'] or 0
            failed_videos = row['failed_videos'] or 0
            running_videos = row['running_videos'] or 0
            
            # 各步骤状态统计
            cursor.execute("""
                SELECT step, status, COUNT(*) as count 
                FROM tasks 
                GROUP BY step, status
            """)
            
            stats = {
                'total_videos': total_videos,
                'completed_videos': completed_videos,
                'failed_videos': failed_videos,
                'running_videos': running_videos,
                'tasks_by_status': {}
            }
            
            for row in cursor.fetchall():
                step = row['step']
                status = row['status']
                count = row['count']
                
                if step not in stats['tasks_by_status']:
                    stats['tasks_by_status'][step] = {}
                
                stats['tasks_by_status'][step][status] = count
            
            return stats
    
    # ==================== Playlist 相关方法 ====================
    
    def add_playlist(self, playlist: Playlist) -> None:
        """添加 Playlist 记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO playlists 
                (id, title, source_url, channel, channel_id, video_count, 
                 last_synced_at, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                playlist.id,
                playlist.title,
                playlist.source_url,
                playlist.channel,
                playlist.channel_id,
                playlist.video_count,
                playlist.last_synced_at,
                playlist.created_at,
                playlist.updated_at,
                json.dumps(playlist.metadata, ensure_ascii=False)
            ))
    
    def get_playlist(self, playlist_id: str) -> Optional[Playlist]:
        """获取 Playlist 记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM playlists WHERE id = ?", (playlist_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_playlist(row)
            return None
    
    def list_playlists(self) -> List[Playlist]:
        """列出所有 Playlist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM playlists ORDER BY updated_at DESC")
            return [self._row_to_playlist(row) for row in cursor.fetchall()]
    
    def update_playlist(self, playlist_id: str, **kwargs) -> None:
        """更新 Playlist 记录"""
        allowed_fields = {'title', 'video_count', 'last_synced_at', 'metadata'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return
        
        updates['updated_at'] = datetime.now()
        
        if 'metadata' in updates:
            updates['metadata'] = json.dumps(updates['metadata'], ensure_ascii=False)
        
        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [playlist_id]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE playlists SET {set_clause} WHERE id = ?", values)
    
    def delete_playlist(self, playlist_id: str) -> None:
        """删除 Playlist（不删除关联的视频）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 删除关联表记录
            cursor.execute("DELETE FROM playlist_videos WHERE playlist_id = ?", (playlist_id,))
            # 清除视频的 playlist 关联（向后兼容）
            cursor.execute("""
                UPDATE videos SET playlist_id = NULL, playlist_index = NULL 
                WHERE playlist_id = ?
            """, (playlist_id,))
            # 删除 playlist
            cursor.execute("DELETE FROM playlists WHERE id = ?", (playlist_id,))
    
    def delete_video(self, video_id: str) -> None:
        """删除视频及其相关任务记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 删除关联表记录（多对多）
            cursor.execute("DELETE FROM playlist_videos WHERE video_id = ?", (video_id,))
            # 删除相关的任务记录
            cursor.execute("DELETE FROM tasks WHERE video_id = ?", (video_id,))
            # 删除视频记录
            cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    
    def get_playlist_video_ids(self, playlist_id: str) -> Set[str]:
        """获取 Playlist 下所有视频的 ID 集合（用于增量同步）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 优先从关联表查询（多对多）
            cursor.execute(
                "SELECT video_id FROM playlist_videos WHERE playlist_id = ?", 
                (playlist_id,)
            )
            result = {row['video_id'] for row in cursor.fetchall()}
            if result:
                return result
            # 向后兼容：从 videos 表查询
            cursor.execute(
                "SELECT id FROM videos WHERE playlist_id = ?", 
                (playlist_id,)
            )
            return {row['id'] for row in cursor.fetchall()}
    
    def add_video_to_playlist(
        self,
        video_id: str,
        playlist_id: str,
        playlist_index: int = 0,
        upload_order_index: int = 0
    ) -> None:
        """添加视频到 Playlist（多对多关联）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO playlist_videos 
                (playlist_id, video_id, playlist_index, upload_order_index, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (playlist_id, video_id, playlist_index, upload_order_index, datetime.now()))
            # 同时更新 video 表的 playlist_id（向后兼容，保留第一个关联）
            cursor.execute("""
                UPDATE videos SET playlist_id = ?, playlist_index = ?, updated_at = ?
                WHERE id = ? AND (playlist_id IS NULL OR playlist_id = ?)
            """, (playlist_id, playlist_index, datetime.now(), video_id, playlist_id))
    
    def update_video_playlist_info(
        self,
        video_id: str,
        playlist_id: str,
        playlist_index: int
    ) -> None:
        """更新视频的 Playlist 关联信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 更新关联表
            cursor.execute("""
                INSERT OR REPLACE INTO playlist_videos 
                (playlist_id, video_id, playlist_index, created_at)
                VALUES (?, ?, ?, ?)
            """, (playlist_id, video_id, playlist_index, datetime.now()))
            # 向后兼容：更新 videos 表（仅当该视频未关联其他 playlist 时）
            cursor.execute("""
                UPDATE videos 
                SET playlist_id = ?, playlist_index = ?, updated_at = ?
                WHERE id = ? AND (playlist_id IS NULL OR playlist_id = ?)
            """, (playlist_id, playlist_index, datetime.now(), video_id, playlist_id))
    
    def get_video_playlists(self, video_id: str) -> List[str]:
        """获取视频所属的所有 Playlist ID 列表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT playlist_id FROM playlist_videos WHERE video_id = ?",
                (video_id,)
            )
            return [row['playlist_id'] for row in cursor.fetchall()]
    
    def get_playlist_video_info(self, playlist_id: str, video_id: str) -> Optional[Dict[str, Any]]:
        """获取视频在指定 Playlist 中的关联信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT playlist_index, upload_order_index, created_at 
                FROM playlist_videos 
                WHERE playlist_id = ? AND video_id = ?
            """, (playlist_id, video_id))
            row = cursor.fetchone()
            if row:
                return {
                    'playlist_index': row['playlist_index'],
                    'upload_order_index': row['upload_order_index'],
                    'created_at': row['created_at']
                }
            return None
    
    def update_playlist_video_order_index(
        self,
        playlist_id: str,
        video_id: str,
        upload_order_index: int
    ) -> None:
        """更新视频在 Playlist 中的时间顺序索引"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE playlist_videos 
                SET upload_order_index = ?
                WHERE playlist_id = ? AND video_id = ?
            """, (upload_order_index, playlist_id, video_id))
    
    def _row_to_playlist(self, row: sqlite3.Row) -> Playlist:
        """将数据库行转换为 Playlist 对象"""
        return Playlist(
            id=row['id'],
            title=row['title'],
            source_url=row['source_url'],
            channel=row['channel'],
            channel_id=row['channel_id'],
            video_count=row['video_count'] or 0,
            last_synced_at=datetime.fromisoformat(row['last_synced_at']) if row['last_synced_at'] else None,
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
