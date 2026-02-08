"""
[已搁置] 多说话人感知处理

此模块包含多说话人场景下的断句和翻译处理逻辑。

状态：已搁置
原因：多说话人识别功能效果不佳，暂时不使用。

保留此代码供未来可能的改进使用。
"""

from typing import Dict, List, Callable, Optional
from collections import defaultdict

from . import warn_experimental


def group_segments_by_speaker(segments: List) -> Dict[str, List]:
    """
    按说话人ID分组segments
    
    Args:
        segments: ASRDataSeg 列表
        
    Returns:
        说话人ID -> segments列表 的映射
    """
    warn_experimental("多说话人分组")
    
    groups = defaultdict(list)
    for seg in segments:
        speaker_id = getattr(seg, 'speaker_id', None) or "SPEAKER_UNKNOWN"
        groups[speaker_id].append(seg)
    
    return dict(groups)


def split_with_speaker_awareness(
    asr_data,
    split_func: Callable,
    realign_func: Callable,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> 'ASRData':
    """
    按说话人分组进行智能断句
    
    Args:
        asr_data: ASRData 对象
        split_func: 断句函数
        realign_func: 时间戳重分配函数
        progress_callback: 进度回调
        
    Returns:
        处理后的 ASRData
    """
    warn_experimental("多说话人断句")
    
    from vat.asr.asr_data import ASRData, ASRDataSeg
    
    has_speakers = any(getattr(seg, 'speaker_id', None) is not None for seg in asr_data.segments)
    
    if not has_speakers:
        # 无说话人信息，使用原有逻辑
        full_text = "".join(seg.text for seg in asr_data.segments)
        split_texts = split_func(full_text)
        return realign_func(asr_data, split_texts)
    
    # 有说话人信息，按说话人分组处理
    speaker_groups = group_segments_by_speaker(asr_data.segments)
    all_segments = []
    
    for speaker_id, segments in speaker_groups.items():
        if progress_callback:
            progress_callback(f"[实验性] 处理说话人 {speaker_id} 的断句...")
        
        speaker_text = "".join(seg.text for seg in segments)
        split_texts = split_func(speaker_text)
        
        speaker_asr = ASRData(segments)
        speaker_asr_split = realign_func(speaker_asr, split_texts)
        
        for seg in speaker_asr_split.segments:
            seg.speaker_id = speaker_id
        
        all_segments.extend(speaker_asr_split.segments)
    
    all_segments.sort(key=lambda x: x.start_time)
    return ASRData(all_segments)


def translate_with_speaker_awareness(
    asr_data,
    translator,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> 'ASRData':
    """
    按说话人分组进行翻译
    
    Args:
        asr_data: ASRData 对象
        translator: 翻译器实例
        progress_callback: 进度回调
        
    Returns:
        翻译后的 ASRData
    """
    warn_experimental("多说话人翻译")
    
    from vat.asr.asr_data import ASRData
    
    has_speakers = any(getattr(seg, 'speaker_id', None) is not None for seg in asr_data.segments)
    
    if not has_speakers:
        return translator.translate_subtitle(asr_data)
    
    speaker_groups = group_segments_by_speaker(asr_data.segments)
    all_segments = []
    
    for speaker_id, segments in speaker_groups.items():
        if progress_callback:
            progress_callback(f"[实验性] 正在翻译 {speaker_id} 的字幕...")
        
        speaker_asr = ASRData(segments)
        translated_speaker_asr = translator.translate_subtitle(speaker_asr)
        
        for seg in translated_speaker_asr.segments:
            seg.speaker_id = speaker_id
        
        all_segments.extend(translated_speaker_asr.segments)
    
    all_segments.sort(key=lambda x: x.start_time)
    return ASRData(all_segments)
