"""
动态音频分块器：基于 VAD（语音活动检测）的智能分块

借鉴自 WhisperJAV 的 DynamicSceneDetector，适配 VAT 的分块 ASR 流程

分块策略：
- fixed: 固定时长分块（默认，简单可靠）
- vad: 基于 VAD 的语音边界分块（避免切断句子）
"""
import io
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger("dynamic_chunker")


@dataclass
class AudioChunk:
    """音频块信息"""
    start_ms: int           # 开始时间（毫秒）
    end_ms: int             # 结束时间（毫秒）
    audio_bytes: bytes      # 音频数据
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
    
    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000


class DynamicChunker:
    """
    动态音频分块器
    
    支持两种分块策略：
    1. fixed: 固定时长分块（传统方式）
    2. vad: 基于 VAD 的语音边界分块
    
    VAD 策略使用 Silero VAD 模型检测语音活动区域，
    在静音边界处分块，避免切断正在进行的句子。
    """
    
    def __init__(
        self,
        method: str = "fixed",
        # 通用参数
        chunk_length_sec: int = 600,
        chunk_overlap_sec: int = 10,
        min_chunk_sec: float = 30.0,
        max_chunk_sec: float = 900.0,
        # VAD 参数
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 200,
    ):
        """
        初始化动态分块器
        
        Args:
            method: 分块方法 ("fixed" | "vad")
            chunk_length_sec: 固定分块长度（秒）
            chunk_overlap_sec: 块间重叠时长（秒）
            min_chunk_sec: 最小块长度（秒）
            max_chunk_sec: 最大块长度（秒）
            vad_threshold: VAD 阈值（0-1，越高越严格）
            min_silence_duration_ms: 最小静音时长（毫秒）
            speech_pad_ms: 语音前后填充时长（毫秒）
        """
        self.method = method
        self.chunk_length_sec = chunk_length_sec
        self.chunk_overlap_sec = chunk_overlap_sec
        self.min_chunk_sec = min_chunk_sec
        self.max_chunk_sec = max_chunk_sec
        
        # VAD 参数
        self.vad_threshold = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        # VAD 模型（延迟加载）
        self._vad_model = None
        self._vad_utils = None
        
        logger.info(f"DynamicChunker 初始化: method={method}, "
                   f"chunk_length={chunk_length_sec}s, overlap={chunk_overlap_sec}s")
    
    def split_audio(
        self,
        audio_path: str,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[AudioChunk]:
        """
        分割音频文件
        
        Args:
            audio_path: 音频文件路径
            callback: 进度回调函数
            
        Returns:
            音频块列表
        """
        if self.method == "vad":
            return self._split_by_vad(audio_path, callback)
        else:
            return self._split_by_fixed(audio_path, callback)
    
    def _split_by_fixed(
        self,
        audio_path: str,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[AudioChunk]:
        """固定时长分块"""
        from pydub import AudioSegment
        
        audio = AudioSegment.from_file(audio_path)
        total_duration_ms = len(audio)
        
        chunk_length_ms = self.chunk_length_sec * 1000
        overlap_ms = self.chunk_overlap_sec * 1000
        
        logger.info(f"固定分块: 总时长 {total_duration_ms/1000:.1f}s, "
                   f"块长度 {self.chunk_length_sec}s, 重叠 {self.chunk_overlap_sec}s")
        
        chunks = []
        start_ms = 0
        
        while start_ms < total_duration_ms:
            end_ms = min(start_ms + chunk_length_ms, total_duration_ms)
            chunk = audio[start_ms:end_ms]
            
            buffer = io.BytesIO()
            chunk.export(buffer, format="mp3")
            
            chunks.append(AudioChunk(
                start_ms=start_ms,
                end_ms=end_ms,
                audio_bytes=buffer.getvalue(),
            ))
            
            if callback:
                progress = int((end_ms / total_duration_ms) * 50)
                callback(progress, f"分块中: {len(chunks)} 块")
            
            start_ms += chunk_length_ms - overlap_ms
            
            if end_ms >= total_duration_ms:
                break
        
        logger.info(f"固定分块完成: {len(chunks)} 块")
        return chunks
    
    def _split_by_vad(
        self,
        audio_path: str,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[AudioChunk]:
        """基于 VAD 的语音边界分块"""
        from pydub import AudioSegment
        
        # 加载 VAD 模型
        self._load_vad_model()
        
        # 加载音频
        audio = AudioSegment.from_file(audio_path)
        total_duration_ms = len(audio)
        
        logger.info(f"VAD 分块: 总时长 {total_duration_ms/1000:.1f}s")
        
        # 转换为 16kHz 单声道（VAD 要求）
        audio_16k = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio_16k.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0  # 归一化到 [-1, 1]
        
        # 获取语音时间戳
        speech_timestamps = self._get_speech_timestamps(samples, 16000)
        
        if not speech_timestamps:
            logger.warning("未检测到语音活动，使用固定分块")
            return self._split_by_fixed(audio_path, callback)
        
        # 根据语音边界分块
        chunks = self._create_chunks_from_timestamps(
            audio, speech_timestamps, total_duration_ms
        )
        
        logger.info(f"VAD 分块完成: {len(chunks)} 块")
        return chunks
    
    def _load_vad_model(self):
        """加载 Silero VAD 模型"""
        if self._vad_model is not None:
            return
        
        try:
            import torch
            
            logger.info("加载 Silero VAD 模型...")
            
            # 使用 torch.hub 加载 Silero VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            
            self._vad_model = model
            self._vad_utils = utils
            
            logger.info("Silero VAD 模型加载成功")
            
        except Exception as e:
            logger.error(f"VAD 模型加载失败: {e}")
            raise RuntimeError(
                f"无法加载 Silero VAD 模型: {e}\n"
                "请确保已安装 torch 并且网络可访问"
            )
    
    def _get_speech_timestamps(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> List[dict]:
        """获取语音时间戳"""
        import torch
        
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = self._vad_utils
        
        # 转换为 torch tensor
        audio_tensor = torch.from_numpy(samples)
        
        # 获取语音时间戳
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self._vad_model,
            threshold=self.vad_threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            sampling_rate=sample_rate,
        )
        
        return speech_timestamps
    
    def _create_chunks_from_timestamps(
        self,
        audio: 'AudioSegment',
        speech_timestamps: List[dict],
        total_duration_ms: int,
    ) -> List[AudioChunk]:
        """根据语音时间戳创建分块"""
        import io
        
        # 将样本索引转换为毫秒（假设 16kHz）
        sample_rate = 16000
        
        # 找出静音边界点
        silence_boundaries = []
        
        for i, ts in enumerate(speech_timestamps):
            start_ms = int(ts['start'] / sample_rate * 1000)
            end_ms = int(ts['end'] / sample_rate * 1000)
            
            if i > 0:
                prev_end = int(speech_timestamps[i-1]['end'] / sample_rate * 1000)
                # 静音区间
                if start_ms - prev_end > self.min_silence_duration_ms:
                    silence_boundaries.append((prev_end + start_ms) // 2)
        
        # 根据静音边界和最大长度创建分块
        chunks = []
        current_start = 0
        max_chunk_ms = int(self.max_chunk_sec * 1000)
        min_chunk_ms = int(self.min_chunk_sec * 1000)
        
        for boundary in silence_boundaries:
            chunk_length = boundary - current_start
            
            # 如果块太长，在边界处分割
            if chunk_length >= max_chunk_ms:
                chunk = audio[current_start:boundary]
                buffer = io.BytesIO()
                chunk.export(buffer, format="mp3")
                
                chunks.append(AudioChunk(
                    start_ms=current_start,
                    end_ms=boundary,
                    audio_bytes=buffer.getvalue(),
                ))
                current_start = boundary
            # 如果块太短，继续累积
            elif chunk_length < min_chunk_ms:
                continue
        
        # 处理最后一块
        if current_start < total_duration_ms:
            chunk = audio[current_start:total_duration_ms]
            buffer = io.BytesIO()
            chunk.export(buffer, format="mp3")
            
            chunks.append(AudioChunk(
                start_ms=current_start,
                end_ms=total_duration_ms,
                audio_bytes=buffer.getvalue(),
            ))
        
        # 如果分块数太少或分块失败，回退到固定分块
        if len(chunks) < 2:
            logger.warning("VAD 分块结果不理想，回退到固定分块")
            return self._split_by_fixed_internal(audio, total_duration_ms)
        
        return chunks
    
    def _split_by_fixed_internal(
        self,
        audio: 'AudioSegment',
        total_duration_ms: int,
    ) -> List[AudioChunk]:
        """内部固定分块方法"""
        import io
        
        chunk_length_ms = self.chunk_length_sec * 1000
        overlap_ms = self.chunk_overlap_sec * 1000
        
        chunks = []
        start_ms = 0
        
        while start_ms < total_duration_ms:
            end_ms = min(start_ms + chunk_length_ms, total_duration_ms)
            chunk = audio[start_ms:end_ms]
            
            buffer = io.BytesIO()
            chunk.export(buffer, format="mp3")
            
            chunks.append(AudioChunk(
                start_ms=start_ms,
                end_ms=end_ms,
                audio_bytes=buffer.getvalue(),
            ))
            
            start_ms += chunk_length_ms - overlap_ms
            
            if end_ms >= total_duration_ms:
                break
        
        return chunks


def is_vad_available() -> Tuple[bool, str]:
    """
    检查 VAD 功能是否可用
    
    Returns:
        (is_available, message)
    """
    issues = []
    
    try:
        import torch
    except ImportError:
        issues.append("缺少依赖: torch")
    
    try:
        from pydub import AudioSegment
    except ImportError:
        issues.append("缺少依赖: pydub (pip install pydub)")
    
    if issues:
        return False, "\n".join(issues)
    
    return True, "VAD 功能就绪"
