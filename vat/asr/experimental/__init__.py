"""
实验性功能模块

包含已搁置/实验中的功能：
- Pipeline模式（Transformers Pipeline + Diarization）
- 多说话人识别

警告：这些功能可能不稳定或效果不佳，不建议在生产环境使用。
"""

import warnings

def warn_experimental(feature_name: str):
    """发出实验性功能警告"""
    warnings.warn(
        f"[VAT] 正在使用实验性功能: {feature_name}。"
        f"此功能已搁置，可能不稳定或效果不佳。",
        UserWarning,
        stacklevel=3
    )
