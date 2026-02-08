"""
WebUI 共享依赖

所有路由模块统一从此处获取 Database 实例，避免每个路由各自创建连接。
"""
from typing import Optional

from vat.database import Database
from vat.config import load_config

_db: Optional[Database] = None


def get_db() -> Database:
    """获取全局 Database 单例"""
    global _db
    if _db is None:
        config = load_config()
        _db = Database(config.storage.database_path)
    return _db
