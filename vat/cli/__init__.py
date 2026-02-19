"""
命令行接口模块
"""
from .commands import cli
from . import tools  # noqa: F401  注册 tools 子命令组

__all__ = ['cli']
