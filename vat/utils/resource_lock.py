"""
跨进程资源协调锁

基于 SQLite 实现的跨进程 permit 锁 + 冷却时间控制，用于限制 YouTube 下载和 B站上传的并发速率。

核心机制：
- permit 锁：同一时间某类资源最多允许 N 个持有者（默认 N=1）
- 冷却时间：锁释放后，需等待最小间隔才允许下一个进程获取
- 心跳：持有锁的进程定期更新活动时间，用于检测进程死亡
- 死锁恢复：通过 PID 存活检测 + 心跳超时自动清理死锁

使用方式：
    with ResourceLock(db_path, 'youtube_download', cooldown_seconds=10) as lock:
        do_download()
"""

import atexit
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

from vat.utils.logger import setup_logger

logger = setup_logger("resource_lock")

# 心跳间隔（秒）：持有锁的进程每隔此时间更新 last_activity_at
_HEARTBEAT_INTERVAL = 30

# 心跳超时（秒）：如果 last_activity_at 距今超过此值，认为持有者已死
_HEARTBEAT_TIMEOUT = _HEARTBEAT_INTERVAL * 3  # 90 秒

# 获取锁的轮询间隔范围（秒），指数退避
_POLL_MIN_INTERVAL = 1.0
_POLL_MAX_INTERVAL = 15.0

# SQLite 连接配置
_BUSY_TIMEOUT_MS = 30_000
_CONNECT_TIMEOUT_S = 30


# 全局追踪：当前进程持有的所有锁（用于 atexit 清理）
_held_locks: dict = {}  # holder_token -> ResourceLock instance
_held_locks_lock = threading.Lock()


def _cleanup_all_locks():
    """进程退出时清理所有持有的锁（atexit 回调）"""
    with _held_locks_lock:
        for holder_token, lock_inst in list(_held_locks.items()):
            try:
                lock_inst._release_lock_internal()
            except Exception as e:
                try:
                    logger.warning(f"atexit 清理锁 holder={holder_token} 失败: {e}")
                except Exception:
                    pass


atexit.register(_cleanup_all_locks)


class ResourceLock:
    """
    跨进程资源锁

    推荐作为上下文管理器使用：
        with ResourceLock(db_path, 'youtube_download', cooldown_seconds=10) as lock:
            do_download()

    Args:
        db_path: SQLite 数据库文件路径（通常与主数据库相同）
        resource_type: 资源类型标识（如 'youtube_download', 'bilibili_upload'）
        cooldown_seconds: 冷却时间（秒），锁释放后需等待此时间才允许下一次获取
        timeout_seconds: 获取锁的最大等待时间（秒），超时则抛出 TimeoutError
        lock_ttl_seconds: 锁的最大生存时间（秒），超时后其他进程可强制接管
        max_concurrent: 该资源允许同时持有的 permit 数（默认 1）
    """

    def __init__(
        self,
        db_path: str,
        resource_type: str,
        cooldown_seconds: float = 0,
        timeout_seconds: float = 600,
        lock_ttl_seconds: float = 1800,
        max_concurrent: int = 1,
    ):
        if int(max_concurrent) < 1:
            raise ValueError(f"max_concurrent 必须 >= 1，得到 {max_concurrent}")

        self.db_path = Path(db_path).expanduser()
        self.resource_type = resource_type
        self.cooldown_seconds = cooldown_seconds
        self.timeout_seconds = timeout_seconds
        self.lock_ttl_seconds = lock_ttl_seconds
        self.max_concurrent = int(max_concurrent)

        self._pid = os.getpid()
        self._holder_token = uuid4().hex
        self._slot_id: Optional[int] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        self._acquired = False

        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """获取 SQLite 连接"""
        conn = sqlite3.connect(str(self.db_path), timeout=_CONNECT_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return conn

    def _ensure_tables(self):
        """确保锁表存在并完成旧 schema 迁移。"""
        last_error = None
        for attempt in range(5):
            conn = self._get_connection()
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                self._ensure_resource_lock_schema(conn)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS resource_cooldowns (
                        resource_type TEXT PRIMARY KEY,
                        last_completed_at TIMESTAMP NOT NULL
                    )
                """)
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                last_error = e
                if "locked" not in str(e).lower() or attempt == 4:
                    raise
                time.sleep(0.1 * (attempt + 1))
            finally:
                conn.close()
        if last_error:
            raise last_error

    def _ensure_resource_lock_schema(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='resource_locks'"
        ).fetchone()
        if row is None:
            self._create_resource_locks_table(conn)
            return

        cols = {
            col_row[1] for col_row in conn.execute("PRAGMA table_info(resource_locks)").fetchall()
        }
        if {'slot_id', 'holder_token'}.issubset(cols):
            return

        legacy_table = "resource_locks_legacy_v1"
        conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
        conn.execute(f"ALTER TABLE resource_locks RENAME TO {legacy_table}")
        self._create_resource_locks_table(conn)
        conn.execute(f"""
            INSERT INTO resource_locks (
                resource_type, slot_id, holder_token, holder_pid,
                acquired_at, last_activity_at, expires_at
            )
            SELECT
                resource_type,
                0,
                'legacy:' || resource_type || ':' || holder_pid || ':' || acquired_at,
                holder_pid,
                acquired_at,
                last_activity_at,
                expires_at
            FROM {legacy_table}
        """)
        conn.execute(f"DROP TABLE {legacy_table}")

    def _create_resource_locks_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_locks (
                resource_type TEXT NOT NULL,
                slot_id INTEGER NOT NULL,
                holder_token TEXT NOT NULL UNIQUE,
                holder_pid INTEGER NOT NULL,
                acquired_at TIMESTAMP NOT NULL,
                last_activity_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                PRIMARY KEY (resource_type, slot_id)
            )
        """)

    def acquire(self) -> bool:
        """
        获取资源锁（阻塞直到获取成功或超时）

        Returns:
            True 如果成功获取

        Raises:
            TimeoutError: 超过 timeout_seconds 仍未获取到锁
        """
        if self._acquired:
            return True

        start_time = time.monotonic()
        poll_interval = _POLL_MIN_INTERVAL

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > self.timeout_seconds:
                raise TimeoutError(
                    f"获取资源锁 '{self.resource_type}' 超时（等待 {elapsed:.0f}s > {self.timeout_seconds}s）。"
                    f"可能有其他 VAT 进程正在使用该资源。"
                )

            if self._try_acquire():
                self._acquired = True
                self._start_heartbeat()
                self._register_global()
                return True

            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, _POLL_MAX_INTERVAL)

    def release(self):
        """释放资源锁"""
        if not self._acquired:
            return
        self._release_lock_internal()

    def _release_lock_internal(self):
        """内部释放逻辑（也被 atexit 调用）"""
        self._stop_heartbeat()

        conn = self._get_connection()
        try:
            now = datetime.now()
            cursor = conn.execute(
                "DELETE FROM resource_locks WHERE resource_type = ? AND holder_token = ?",
                (self.resource_type, self._holder_token)
            )

            if cursor.rowcount:
                conn.execute("""
                    INSERT OR REPLACE INTO resource_cooldowns (resource_type, last_completed_at)
                    VALUES (?, ?)
                """, (self.resource_type, now.isoformat()))

            conn.commit()
            logger.debug(
                f"释放资源锁 '{self.resource_type}' "
                f"(pid={self._pid}, slot={self._slot_id}, holder={self._holder_token})"
            )
        except Exception as e:
            logger.warning(f"释放锁 '{self.resource_type}' 时出错: {e}")
        finally:
            conn.close()

        self._acquired = False
        self._slot_id = None
        self._unregister_global()

    def _try_acquire(self) -> bool:
        """
        尝试获取锁（单次，非阻塞）

        Returns:
            True 如果成功获取
        """
        conn = self._get_connection()
        try:
            now = datetime.now()

            row = conn.execute(
                "SELECT last_completed_at FROM resource_cooldowns WHERE resource_type = ?",
                (self.resource_type,)
            ).fetchone()

            if row and self.cooldown_seconds > 0:
                last_completed = datetime.fromisoformat(row['last_completed_at'])
                cooldown_remaining = self.cooldown_seconds - (now - last_completed).total_seconds()
                if cooldown_remaining > 0:
                    logger.debug(
                        f"资源 '{self.resource_type}' 冷却中，剩余 {cooldown_remaining:.0f}s"
                    )
                    return False

            own_row = conn.execute(
                "SELECT slot_id FROM resource_locks WHERE resource_type = ? AND holder_token = ?",
                (self.resource_type, self._holder_token)
            ).fetchone()
            if own_row:
                self._slot_id = own_row['slot_id']
                return True

            lock_rows = conn.execute(
                "SELECT * FROM resource_locks WHERE resource_type = ? ORDER BY slot_id",
                (self.resource_type,)
            ).fetchall()

            occupied_slots = set()
            for lock_row in lock_rows:
                holder_pid = lock_row['holder_pid']
                expires_at = datetime.fromisoformat(lock_row['expires_at'])
                last_activity = datetime.fromisoformat(lock_row['last_activity_at'])

                holder_alive = _is_pid_alive(holder_pid)
                lock_expired = now > expires_at
                heartbeat_timeout = (now - last_activity).total_seconds() > _HEARTBEAT_TIMEOUT

                if not holder_alive:
                    logger.info(
                        f"资源锁 '{self.resource_type}' 的持有者 pid={holder_pid} 已不存在，清理死锁"
                    )
                elif lock_expired:
                    logger.info(
                        f"资源锁 '{self.resource_type}' 已过期"
                        f"（expires_at={expires_at.isoformat()}），清理"
                    )
                elif heartbeat_timeout:
                    logger.info(
                        f"资源锁 '{self.resource_type}' 心跳超时"
                        f"（last_activity={last_activity.isoformat()}，"
                        f"超时阈值={_HEARTBEAT_TIMEOUT}s），清理"
                    )
                else:
                    occupied_slots.add(lock_row['slot_id'])
                    continue

                conn.execute(
                    "DELETE FROM resource_locks WHERE resource_type = ? AND slot_id = ?",
                    (self.resource_type, lock_row['slot_id'])
                )

            if len(occupied_slots) >= self.max_concurrent:
                conn.commit()
                return False

            free_slot = None
            for slot_id in range(self.max_concurrent):
                if slot_id not in occupied_slots:
                    free_slot = slot_id
                    break

            if free_slot is None:
                conn.commit()
                return False

            expires_at = now + timedelta(seconds=self.lock_ttl_seconds)
            try:
                conn.execute("""
                    INSERT INTO resource_locks (
                        resource_type, slot_id, holder_token, holder_pid,
                        acquired_at, last_activity_at, expires_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.resource_type,
                    free_slot,
                    self._holder_token,
                    self._pid,
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                ))
                conn.commit()
                self._slot_id = free_slot
                logger.info(
                    f"获取资源锁 '{self.resource_type}' (pid={self._pid}, "
                    f"slot={free_slot + 1}/{self.max_concurrent}, "
                    f"ttl={self.lock_ttl_seconds}s, cooldown={self.cooldown_seconds}s)"
                )
                return True
            except sqlite3.IntegrityError:
                conn.rollback()
                return False
        except Exception as e:
            logger.warning(f"尝试获取锁 '{self.resource_type}' 时出错: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            conn.close()

    def _start_heartbeat(self):
        """启动心跳守护线程"""
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"resource-lock-heartbeat-{self.resource_type}",
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        """停止心跳线程"""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_stop.set()
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def _heartbeat_loop(self):
        """心跳线程主循环：定期更新 last_activity_at 和 expires_at"""
        while not self._heartbeat_stop.wait(timeout=_HEARTBEAT_INTERVAL):
            try:
                conn = self._get_connection()
                try:
                    now = datetime.now()
                    new_expires = now + timedelta(seconds=self.lock_ttl_seconds)
                    conn.execute("""
                        UPDATE resource_locks
                        SET last_activity_at = ?, expires_at = ?
                        WHERE resource_type = ? AND holder_token = ?
                    """, (
                        now.isoformat(),
                        new_expires.isoformat(),
                        self.resource_type,
                        self._holder_token,
                    ))
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                logger.debug(f"心跳更新失败 '{self.resource_type}': {e}")

    def _register_global(self):
        """注册到全局追踪（用于 atexit 清理）"""
        with _held_locks_lock:
            _held_locks[self._holder_token] = self

    def _unregister_global(self):
        """从全局追踪中移除"""
        with _held_locks_lock:
            _held_locks.pop(self._holder_token, None)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def _is_pid_alive(pid: int) -> bool:
    """
    检查进程是否存活

    使用 os.kill(pid, 0) 检测：
    - 进程存在 → 不发送信号，返回 True
    - 进程不存在 → 抛出 ProcessLookupError，返回 False
    - 无权限 → 抛出 PermissionError，返回 True（进程存在但无权发信号）
    """
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


@contextmanager
def resource_lock(
    db_path: str,
    resource_type: str,
    cooldown_seconds: float = 0,
    timeout_seconds: float = 600,
    lock_ttl_seconds: float = 1800,
    max_concurrent: int = 1,
):
    """
    资源锁的便捷上下文管理器

    用法：
        with resource_lock(db_path, 'youtube_download', cooldown_seconds=10):
            do_download()
    """
    lock = ResourceLock(
        db_path=db_path,
        resource_type=resource_type,
        cooldown_seconds=cooldown_seconds,
        timeout_seconds=timeout_seconds,
        lock_ttl_seconds=lock_ttl_seconds,
        max_concurrent=max_concurrent,
    )
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
