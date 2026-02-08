"""
人声分离模块

使用 Mel-Band-Roformer 模型分离人声和背景音乐
适用于游戏直播、歌回直播等有 BGM 干扰的场景
"""
from .separator import (
    VocalSeparator,
    VocalSeparationResult,
    separate_vocals,
    is_vocal_separation_available,
    check_vocal_separation_requirements,
    get_vocal_separator,
)

# MelBandRoformer 延迟导入（需要额外依赖）
def get_mel_band_roformer():
    """延迟导入 MelBandRoformer 模型类"""
    from .mel_band_roformer import MelBandRoformer
    return MelBandRoformer

__all__ = [
    'VocalSeparator',
    'VocalSeparationResult',
    'separate_vocals',
    'is_vocal_separation_available',
    'check_vocal_separation_requirements',
    'get_vocal_separator',
    'get_mel_band_roformer',
]
