"""
日志配置
"""
import logging
import colorlog
from pathlib import Path


def setup_logger(
    name: str = 'vat',
    level: str = 'INFO',
    log_file: str = None,
    log_format: str = None
) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        log_format: 日志格式（可选）
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 默认日志格式
    if log_format is None:
        log_format = '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s'
    
    # 控制台处理器（彩色）
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    console_formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
