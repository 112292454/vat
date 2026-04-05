"""
资源锁单元测试

测试 vat.utils.resource_lock 的核心功能：
- 基本获取/释放
- 冷却时间控制
- 死锁检测（PID 不存活）
- 心跳超时检测
- 上下文管理器
- 并发获取（多线程模拟多进程）
"""

import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from vat.utils.resource_lock import (
    ResourceLock,
    resource_lock,
    _is_pid_alive,
    _HEARTBEAT_INTERVAL,
    _HEARTBEAT_TIMEOUT,
)


@pytest.fixture
def db_path(tmp_path):
    """创建临时数据库路径"""
    return str(tmp_path / "test_locks.db")


class TestResourceLockBasic:
    """基本锁操作测试"""
    
    def test_acquire_and_release(self, db_path):
        """测试基本的获取和释放"""
        lock = ResourceLock(db_path, 'test_resource', timeout_seconds=5)
        
        assert lock.acquire() is True
        assert lock._acquired is True
        
        # 验证数据库记录
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM resource_locks WHERE resource_type = 'test_resource'"
        ).fetchone()
        assert row is not None
        assert row['holder_pid'] == os.getpid()
        conn.close()
        
        lock.release()
        assert lock._acquired is False
        
        # 验证锁已释放
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT * FROM resource_locks WHERE resource_type = 'test_resource'"
        ).fetchone()
        assert row is None
        conn.close()
    
    def test_context_manager(self, db_path):
        """测试上下文管理器"""
        with ResourceLock(db_path, 'test_ctx', timeout_seconds=5) as lock:
            assert lock._acquired is True
            
            # 锁存在
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT * FROM resource_locks WHERE resource_type = 'test_ctx'"
            ).fetchone()
            assert row is not None
            conn.close()
        
        # 退出后锁已释放
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT * FROM resource_locks WHERE resource_type = 'test_ctx'"
        ).fetchone()
        assert row is None
        conn.close()
    
    def test_context_manager_function(self, db_path):
        """测试 resource_lock() 便捷函数"""
        with resource_lock(db_path, 'test_func', timeout_seconds=5):
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT * FROM resource_locks WHERE resource_type = 'test_func'"
            ).fetchone()
            assert row is not None
            conn.close()
    
    def test_context_manager_exception_cleanup(self, db_path):
        """测试异常时上下文管理器正确释放锁"""
        with pytest.raises(ValueError):
            with ResourceLock(db_path, 'test_exc', timeout_seconds=5):
                raise ValueError("test error")
        
        # 锁应该已被释放
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT * FROM resource_locks WHERE resource_type = 'test_exc'"
        ).fetchone()
        assert row is None
        conn.close()
    
    def test_same_process_second_instance_blocks_when_max_concurrent_is_one(self, db_path):
        """同进程不同实例也必须遵守全局上限，不能按 PID 重入绕过"""
        lock = ResourceLock(db_path, 'test_reentrant', timeout_seconds=5, max_concurrent=1)
        assert lock.acquire() is True
        
        lock2 = ResourceLock(db_path, 'test_reentrant', timeout_seconds=5, max_concurrent=1)
        assert lock2._try_acquire() is False
        
        lock.release()

    def test_max_concurrent_two_allows_two_holders(self, db_path):
        """配置 max_concurrent=2 时，应允许同时持有两个 permit"""
        lock1 = ResourceLock(db_path, 'test_multi', timeout_seconds=5, max_concurrent=2)
        lock2 = ResourceLock(db_path, 'test_multi', timeout_seconds=5, max_concurrent=2)
        lock3 = ResourceLock(db_path, 'test_multi', timeout_seconds=5, max_concurrent=2)

        assert lock1.acquire() is True
        assert lock2._try_acquire() is True
        assert lock3._try_acquire() is False

        conn = sqlite3.connect(db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM resource_locks WHERE resource_type = 'test_multi'"
        ).fetchone()[0]
        conn.close()
        assert count == 2

        lock1.release()
        lock2.release()
    
    def test_different_resource_types(self, db_path):
        """测试不同资源类型互不影响"""
        lock1 = ResourceLock(db_path, 'resource_a', timeout_seconds=5)
        lock2 = ResourceLock(db_path, 'resource_b', timeout_seconds=5)
        
        assert lock1.acquire() is True
        assert lock2.acquire() is True
        
        lock1.release()
        lock2.release()


class TestCooldown:
    """冷却时间测试"""
    
    def test_cooldown_blocks_immediate_reacquire(self, db_path):
        """测试冷却时间阻止立即重新获取"""
        lock = ResourceLock(db_path, 'test_cooldown', cooldown_seconds=2, timeout_seconds=5)
        lock.acquire()
        lock.release()
        
        # 立即尝试获取应被冷却阻止
        lock2 = ResourceLock(db_path, 'test_cooldown', cooldown_seconds=2, timeout_seconds=1)
        assert lock2._try_acquire() is False
    
    def test_cooldown_allows_after_wait(self, db_path):
        """测试冷却时间过后可以获取"""
        lock = ResourceLock(db_path, 'test_cooldown2', cooldown_seconds=1, timeout_seconds=5)
        lock.acquire()
        lock.release()
        
        time.sleep(1.1)
        
        lock2 = ResourceLock(db_path, 'test_cooldown2', cooldown_seconds=1, timeout_seconds=5)
        assert lock2._try_acquire() is True
        lock2.release()
    
    def test_no_cooldown(self, db_path):
        """测试无冷却时间时可立即重新获取"""
        lock = ResourceLock(db_path, 'test_no_cd', cooldown_seconds=0, timeout_seconds=5)
        lock.acquire()
        lock.release()
        
        lock2 = ResourceLock(db_path, 'test_no_cd', cooldown_seconds=0, timeout_seconds=5)
        assert lock2._try_acquire() is True
        lock2.release()


class TestDeadLockDetection:
    """死锁检测测试"""
    
    def test_detect_dead_holder_pid(self, db_path):
        """测试检测到持有者 PID 不存在时清理死锁"""
        # 手动插入一个"死进程"持有的锁
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_locks (
                resource_type TEXT PRIMARY KEY,
                holder_pid INTEGER NOT NULL,
                acquired_at TIMESTAMP NOT NULL,
                last_activity_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_cooldowns (
                resource_type TEXT PRIMARY KEY,
                last_completed_at TIMESTAMP NOT NULL
            )
        """)
        
        now = datetime.now()
        fake_pid = 99999999  # 不存在的 PID
        conn.execute("""
            INSERT INTO resource_locks (resource_type, holder_pid, acquired_at, last_activity_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'test_dead',
            fake_pid,
            now.isoformat(),
            now.isoformat(),
            (now + timedelta(hours=1)).isoformat(),
        ))
        conn.commit()
        conn.close()
        
        # 新锁应能检测到死进程并抢占
        lock = ResourceLock(db_path, 'test_dead', timeout_seconds=5)
        assert lock._try_acquire() is True
        lock.release()
    
    def test_detect_expired_lock(self, db_path):
        """测试检测到过期锁时清理"""
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_locks (
                resource_type TEXT PRIMARY KEY,
                holder_pid INTEGER NOT NULL,
                acquired_at TIMESTAMP NOT NULL,
                last_activity_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_cooldowns (
                resource_type TEXT PRIMARY KEY,
                last_completed_at TIMESTAMP NOT NULL
            )
        """)
        
        now = datetime.now()
        # 插入一个已过期的锁，holder 是当前 PID（所以 PID 存活）
        # 但 expires_at 已经过去了
        conn.execute("""
            INSERT INTO resource_locks (resource_type, holder_pid, acquired_at, last_activity_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'test_expired',
            os.getpid() + 1000000,  # 用一个大数模拟其他进程
            (now - timedelta(hours=2)).isoformat(),
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(minutes=5)).isoformat(),  # 过期时间
        ))
        conn.commit()
        conn.close()
        
        # mock _is_pid_alive 让它返回 True（模拟进程存活但锁过期）
        with patch('vat.utils.resource_lock._is_pid_alive', return_value=True):
            lock = ResourceLock(db_path, 'test_expired', timeout_seconds=5)
            assert lock._try_acquire() is True
            lock.release()
    
    def test_detect_heartbeat_timeout(self, db_path):
        """测试心跳超时检测"""
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_locks (
                resource_type TEXT PRIMARY KEY,
                holder_pid INTEGER NOT NULL,
                acquired_at TIMESTAMP NOT NULL,
                last_activity_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_cooldowns (
                resource_type TEXT PRIMARY KEY,
                last_completed_at TIMESTAMP NOT NULL
            )
        """)
        
        now = datetime.now()
        # 锁未过期，但心跳已超时
        conn.execute("""
            INSERT INTO resource_locks (resource_type, holder_pid, acquired_at, last_activity_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'test_hb_timeout',
            os.getpid() + 1000000,
            (now - timedelta(minutes=10)).isoformat(),
            (now - timedelta(seconds=_HEARTBEAT_TIMEOUT + 10)).isoformat(),  # 心跳超时
            (now + timedelta(hours=1)).isoformat(),  # 未过期
        ))
        conn.commit()
        conn.close()
        
        with patch('vat.utils.resource_lock._is_pid_alive', return_value=True):
            lock = ResourceLock(db_path, 'test_hb_timeout', timeout_seconds=5)
            assert lock._try_acquire() is True
            lock.release()


class TestTimeout:
    """超时测试"""
    
    def test_acquire_timeout(self, db_path):
        """测试获取超时"""
        # 先持有锁
        lock1 = ResourceLock(db_path, 'test_timeout', timeout_seconds=5)
        lock1.acquire()
        
        # 模拟另一个进程尝试获取
        # 由于同进程会重入成功，我们手动修改 holder_pid
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE resource_locks SET holder_pid = ? WHERE resource_type = 'test_timeout'",
            (os.getpid() + 1,)
        )
        conn.commit()
        conn.close()
        
        # 尝试获取应超时（mock PID 存活以防止死锁清理）
        with patch('vat.utils.resource_lock._is_pid_alive', return_value=True):
            lock2 = ResourceLock(db_path, 'test_timeout', timeout_seconds=2)
            with pytest.raises(TimeoutError):
                lock2.acquire()
        
        # 恢复原始 PID 以便清理
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE resource_locks SET holder_pid = ? WHERE resource_type = 'test_timeout'",
            (os.getpid(),)
        )
        conn.commit()
        conn.close()
        lock1.release()


class TestHeartbeat:
    """心跳测试"""
    
    def test_heartbeat_updates_activity(self, db_path):
        """测试心跳线程更新 last_activity_at"""
        # 使用较短的心跳间隔来加速测试
        lock = ResourceLock(db_path, 'test_hb', timeout_seconds=5, lock_ttl_seconds=60)
        lock.acquire()
        
        # 记录获取时的 last_activity_at
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT last_activity_at FROM resource_locks WHERE resource_type = 'test_hb'"
        ).fetchone()
        initial_activity = row['last_activity_at']
        conn.close()
        
        # 等待一个心跳周期
        time.sleep(_HEARTBEAT_INTERVAL + 2)
        
        # 检查 last_activity_at 是否已更新
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT last_activity_at FROM resource_locks WHERE resource_type = 'test_hb'"
        ).fetchone()
        updated_activity = row['last_activity_at']
        conn.close()
        
        assert updated_activity > initial_activity, (
            f"心跳未更新 last_activity_at: {initial_activity} -> {updated_activity}"
        )
        
        lock.release()


class TestConcurrency:
    """并发测试（使用线程模拟）"""
    
    def test_mutual_exclusion_threads(self, db_path):
        """测试多线程下的互斥性"""
        results = []
        errors = []
        lock_order = []
        
        def worker(worker_id):
            try:
                # 每个线程模拟不同的"进程"（通过不同的 ResourceLock 实例）
                with resource_lock(db_path, 'test_mutex', cooldown_seconds=0, timeout_seconds=10):
                    lock_order.append(f"acquired_{worker_id}")
                    time.sleep(0.5)  # 模拟工作
                    lock_order.append(f"released_{worker_id}")
                results.append(worker_id)
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        
        # 同进程线程不再按 PID 重入，但会在释放后依次获取
        assert len(results) == 3, f"部分线程失败: results={results}, errors={errors}"

    def test_max_concurrent_threads(self, db_path):
        """max_concurrent=2 时，最多只允许两个线程同时进入临界区"""
        active = 0
        peak_active = 0
        active_lock = threading.Lock()

        def worker():
            nonlocal active, peak_active
            with resource_lock(db_path, 'test_thread_multi', cooldown_seconds=0, timeout_seconds=10, max_concurrent=2):
                with active_lock:
                    active += 1
                    peak_active = max(peak_active, active)
                time.sleep(0.2)
                with active_lock:
                    active -= 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert peak_active == 2


class TestIsPidAlive:
    """PID 检测测试"""
    
    def test_current_pid_alive(self):
        """当前进程应该存活"""
        assert _is_pid_alive(os.getpid()) is True
    
    def test_invalid_pid(self):
        """无效 PID 应该不存活"""
        assert _is_pid_alive(0) is False
        assert _is_pid_alive(-1) is False
    
    def test_nonexistent_pid(self):
        """不存在的 PID 应该不存活"""
        # 使用一个很大的 PID，几乎不可能存在
        assert _is_pid_alive(99999999) is False


class TestTableCreation:
    """数据库表创建测试"""
    
    def test_tables_created(self, db_path):
        """确保表被正确创建"""
        _ = ResourceLock(db_path, 'test_tables')
        
        conn = sqlite3.connect(db_path)
        
        # 检查 resource_locks 表
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='resource_locks'"
        )
        assert cursor.fetchone() is not None
        
        # 检查 resource_cooldowns 表
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='resource_cooldowns'"
        )
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_idempotent_table_creation(self, db_path):
        """表创建应幂等"""
        _ = ResourceLock(db_path, 'test1')
        _ = ResourceLock(db_path, 'test2')
        # 不应抛异常
