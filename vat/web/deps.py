"""
WebUI 共享依赖

所有路由模块统一从此处获取配置和 Database 实例，避免每个路由各自
load_config() 导致 CLI 传入的 --config 在 Web runtime 中失效。
"""
from typing import Optional

from vat.database import Database
from vat.config import Config, load_config

_db: Optional[Database] = None
_config: Optional[Config] = None
_config_path: Optional[str] = None


def reset_web_runtime_state() -> None:
    """重置 Web runtime 的全局单例缓存。"""
    global _db, _config
    _db = None
    _config = None


def set_web_config_path(config_path: Optional[str]) -> None:
    """设置 Web runtime 应使用的配置路径。"""
    global _config_path
    _config_path = config_path
    reset_web_runtime_state()

    try:
        from vat.web.routes import tasks as tasks_routes
        tasks_routes._job_manager = None
    except Exception:
        pass


def get_web_config() -> Config:
    """获取 Web runtime 的配置单例。"""
    global _config
    if _config is None:
        _config = load_config(_config_path)
    return _config


def get_web_config_path() -> Optional[str]:
    """获取 Web runtime 当前使用的配置路径。"""
    return _config_path


def get_db() -> Database:
    """获取全局 Database 单例"""
    global _db
    if _db is None:
        config = get_web_config()
        _db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    return _db
