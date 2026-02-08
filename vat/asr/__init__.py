"""
语音识别模块（集成 ASRData、ChunkedASR 和智能断句）
"""
from .whisper_wrapper import WhisperASR, WhisperCPPASR, WhisperASRAdapter
from .asr_data import ASRData, ASRDataSeg
from .chunked_asr import ChunkedASR
from .chunk_merger import ChunkMerger
from .split import split_by_llm
from .subtitle_utils import (
    write_srt, parse_srt, write_ass,
    create_bilingual_srt, merge_srt_files
)
from .postprocessing import (
    ASRPostProcessor,
    HallucinationDetector,
    RepetitionCleaner,
    JapanesePostProcessor,
    postprocess_asr_text,
    is_hallucination,
)
from .vocal_separation import (
    VocalSeparator,
    VocalSeparationResult,
    separate_vocals,
    is_vocal_separation_available,
    check_vocal_separation_requirements,
    get_vocal_separator,
    get_mel_band_roformer,
)
from .dynamic_chunker import (
    DynamicChunker,
    AudioChunk,
    is_vad_available,
)

__all__ = [
    # Whisper 转录器
    'WhisperASR',
    'WhisperCPPASR',
    'WhisperASRAdapter',
    # ASRData 数据结构
    'ASRData',
    'ASRDataSeg',
    # 分块处理
    'ChunkedASR',
    'ChunkMerger',
    # 智能断句
    'split_by_llm',
    # 字幕工具
    'write_srt',
    'parse_srt',
    'write_ass',
    'create_bilingual_srt',
    'merge_srt_files',
    # 后处理
    'ASRPostProcessor',
    'HallucinationDetector',
    'RepetitionCleaner',
    'JapanesePostProcessor',
    'postprocess_asr_text',
    'is_hallucination',
    # 人声分离
    'VocalSeparator',
    'VocalSeparationResult',
    'separate_vocals',
    'is_vocal_separation_available',
    'check_vocal_separation_requirements',
    'get_vocal_separator',
    'get_mel_band_roformer',
    # 动态分块
    'DynamicChunker',
    'AudioChunk',
    'is_vad_available',
]
