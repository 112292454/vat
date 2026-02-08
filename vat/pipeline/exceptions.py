"""
Pipeline 异常定义

子阶段独立化后，每个阶段（WHISPER, SPLIT, OPTIMIZE, TRANSLATE 等）
都是独立的 TaskStep，不再需要 sub_phase 参数。
"""
from typing import Optional


class PipelineError(Exception):
    """Pipeline 执行错误基类"""
    
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None
    ):
        """
        Args:
            message: 错误信息
            original_error: 原始异常
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error
    
    def __str__(self):
        return self.message


class ASRError(PipelineError):
    """ASR 阶段错误"""
    pass


class TranslateError(PipelineError):
    """翻译阶段错误"""
    pass


class EmbedError(PipelineError):
    """嵌入阶段错误"""
    pass


class DownloadError(PipelineError):
    """下载阶段错误"""
    pass


class UploadError(PipelineError):
    """上传阶段错误"""
    pass
