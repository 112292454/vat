"""
[已搁置] Transformers Pipeline 模式 ASR

此模块包含基于 Transformers Pipeline 的语音识别实现，
支持说话人分离（Diarization）功能。

状态：已搁置
原因：基于 kotoba-whisper 的多说话人识别效果不如 faster-whisper large-v3，
     且依赖的 diarizers 库有 bug。

保留此代码供未来可能的改进使用。
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Callable, Union, List

from . import warn_experimental


def load_pipeline_model(
    model_name: str,
    device: str,
    batch_size: int,
    hf_token: Optional[str] = None,
):
    """
    加载 Transformers Pipeline 模型
    
    Args:
        model_name: 模型名称（如 kotoba-tech/kotoba-whisper-v2.2）
        device: 设备（cuda/cpu）
        batch_size: 批次大小
        hf_token: HuggingFace token
        
    Returns:
        Pipeline 对象
    """
    warn_experimental("Pipeline ASR 模式")
    
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        raise ImportError(
            "Pipeline模式需要transformers库，请运行: pip install transformers accelerate"
        )
    
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_kwargs = {"attn_implementation": "sdpa"} if device == "cuda" else {}
    
    pipe = pipeline(
        model=model_name,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
        trust_remote_code=True,
        token=hf_token
    )
    
    return pipe


def transcribe_with_pipeline(
    pipe,
    audio_path: Union[str, Path],
    language: str,
    chunk_length_s: int = 30,
    enable_diarization: bool = False,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """
    使用 Pipeline 进行转录
    
    Args:
        pipe: Pipeline 对象
        audio_path: 音频文件路径
        language: 语言代码
        chunk_length_s: 分块长度（秒）
        enable_diarization: 是否启用说话人分离
        num_speakers: 固定说话人数量
        min_speakers: 最小说话人数量
        max_speakers: 最大说话人数量
        progress_callback: 进度回调
        
    Returns:
        转录结果列表
    """
    warn_experimental("Pipeline ASR 转录")
    
    if progress_callback:
        progress_callback(f"[实验性] 开始Pipeline转录: {audio_path}")
    
    generate_kwargs = {
        "language": language,
        "task": "transcribe",
    }
    
    call_kwargs = {
        "chunk_length_s": chunk_length_s,
        "return_timestamps": True,
        "generate_kwargs": generate_kwargs,
    }
    
    if num_speakers is not None:
        call_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        call_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        call_kwargs["max_speakers"] = max_speakers
    
    result = pipe(str(audio_path), **call_kwargs)
    
    chunks = result.get('chunks', [])
    segments = []
    
    for chunk in chunks:
        text = chunk.get('text', '').strip()
        if not text:
            continue
        
        # 过滤无用文本
        if any(x in text for x in ["・", "作詞", "編曲"]) or text.startswith(("【", "（")):
            continue
        
        segment = {
            'text': text,
            'start_time': int(chunk['timestamp'][0] * 1000),
            'end_time': int(chunk['timestamp'][1] * 1000),
        }
        
        if enable_diarization and 'speaker_id' in chunk:
            segment['speaker_id'] = chunk['speaker_id']
        
        segments.append(segment)
        
        if progress_callback:
            speaker_info = f" [{chunk.get('speaker_id')}]" if chunk.get('speaker_id') else ""
            progress_callback(f"[{chunk['timestamp'][0]:.2f}s]{speaker_info} {text}")
    
    return segments
