"""
日志工具模块（从 VideoCaptioner 适配）

使用 ContextVars 来管理 video_id 上下文：
- ContextVars 是线程安全的，每个线程有独立的上下文
- 在多进程环境下，每个进程有独立的 Python 解释器，ContextVars 不会跨进程传递
- 这是正确的行为：每个进程应该独立设置自己的 video_id
- 在 VideoProcessor.process() 开始时调用 set_video_id() 即可
"""
import logging
import logging.handlers
from pathlib import Path
from contextvars import ContextVar
from typing import Optional

# 默认配置
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DIR = Path.home() / ".vat" / "logs"

# 上下文变量，用于存储当前的 video_id
_video_id_ctx: ContextVar[Optional[str]] = ContextVar("video_id", default=None)


def set_video_id(video_id: Optional[str]):
    """设置当前上下文的 video_id"""
    _video_id_ctx.set(video_id)


def get_video_id() -> Optional[str]:
    """获取当前上下文的 video_id"""
    return _video_id_ctx.get()


class ContextFilter(logging.Filter):
    """
    日志过滤器，用于注入 video_id 到日志记录中
    """
    def filter(self, record):
        video_id = get_video_id()
        if video_id:
            record.video_id_str = f"[{video_id}]"
        else:
            record.video_id_str = "[GLOBAL]"
        return True

def setup_logger(
    name: str,
    level: int = DEFAULT_LOG_LEVEL,
    info_fmt: str = "%(asctime)s | %(levelname)-8s | %(video_id_str)s | %(name)-25s | %(message)s",
    default_fmt: str = "%(asctime)s | %(levelname)-8s | %(video_id_str)s | %(name)-25s | %(message)s",
    datefmt: str = "%H:%M:%S",
    log_file: str = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    创建并配置日志记录器。

    参数:
    - name: 日志记录器的名称
    - level: 日志级别
    - info_fmt: INFO级别的日志格式字符串
    - default_fmt: 其他级别的日志格式字符串
    - datefmt: 时间格式字符串
    - log_file: 日志文件路径
    - console_output: 是否输出到控制台
    """
    
    # 如果没有指定日志文件，使用默认路径
    if log_file is None:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = str(DEFAULT_LOG_DIR / f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 添加上下文过滤器
    context_filter = ContextFilter()
    logger.addFilter(context_filter)

    if not logger.handlers:
        # 创建级别特定的格式化器
        class LevelSpecificFormatter(logging.Formatter):
            def format(self, record):
                # 确保 record 有 video_id_str 属性（防止 filter 未执行的情况）
                if not hasattr(record, 'video_id_str'):
                    video_id = get_video_id()
                    record.video_id_str = f"[{video_id}]" if video_id else "[GLOBAL]"
                
                if record.levelno == logging.INFO:
                    self._style._fmt = info_fmt
                else:
                    self._style._fmt = default_fmt
                return super().format(record)

        level_formatter = LevelSpecificFormatter(default_fmt, datefmt=datefmt)

        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(level_formatter)
            # 也要给 handler 添加 filter，以防 logger 已经有 filter 但 handler 没有
            # 不过通常 logger 的 filter 就够了，除非 handler 是独立添加的
            # 这里为了保险起见，不给 handler 加，因为 logger 已经加了
            logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(level_formatter)
            logger.addHandler(file_handler)

    # 设置特定库的日志级别为ERROR以减少日志噪音
    error_loggers = [
        "urllib3",
        "requests",
        "openai",
        "httpx",
        "httpcore",
        "ssl",
        "certifi",
    ]
    for lib in error_loggers:
        logging.getLogger(lib).setLevel(logging.ERROR)

    return logger
