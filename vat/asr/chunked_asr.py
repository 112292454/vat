"""音频分块 ASR 装饰器

为任何 BaseASR 实现添加音频分块转录能力，适用于长音频处理。
使用装饰器模式实现关注点分离。

多 GPU 支持：
- 单 GPU：使用线程池（避免进程开销）
- 多 GPU：使用进程池，每个 chunk 分配到不同 GPU
"""

import io
import multiprocessing
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple, Dict, Any

# CUDA 多进程必须使用 spawn 方法（fork 会导致死锁）
# 在模块加载时设置，确保所有子进程使用 spawn
try:
    multiprocessing.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # 已经设置过了

from pydub import AudioSegment

from ..utils.logger import setup_logger
from .asr_data import ASRData, ASRDataSeg
from .chunk_merger import ChunkMerger

logger = setup_logger("chunked_asr")

# 常量定义
MS_PER_SECOND = 1000
DEFAULT_CHUNK_LENGTH_SEC = 60 * 10  # 10分钟
DEFAULT_CHUNK_OVERLAP_SEC = 10  # 10秒重叠
DEFAULT_CHUNK_CONCURRENCY = 3  # 3个并发


def _get_available_gpu_count() -> int:
    """获取可用 GPU 数量"""
    try:
        from ..utils.gpu import get_available_gpus
        gpus = get_available_gpus()
        return len(gpus)
    except Exception:
        return 0


def _worker_transcribe_chunk(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    子进程 worker 函数：转录单个音频块
    
    注意：此函数在独立进程中运行，需要重新加载模型。
    
    Args:
        args: 包含所有需要的参数的字典
            - chunk_idx: chunk 索引
            - chunk_file: 临时音频文件路径
            - gpu_id: 分配的 GPU ID（None 表示使用默认）
            - whisper_kwargs: WhisperASR 初始化参数
            - language: 语言
            
    Returns:
        包含结果的字典：
            - chunk_idx: chunk 索引
            - segments: 转录结果片段列表
            - error: 错误信息（如果有）
    """
    chunk_idx = args['chunk_idx']
    chunk_file = args['chunk_file']
    gpu_id = args['gpu_id']
    whisper_kwargs = args['whisper_kwargs']
    language = args['language']
    
    try:
        # 设置 GPU 环境变量（在模型加载前）
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Chunk {chunk_idx}: 使用 GPU {gpu_id}")
        
        # 延迟导入，避免在主进程中加载模型
        from .whisper_wrapper import WhisperASR
        
        # 创建 ASR 实例（会在此进程中加载模型）
        asr = WhisperASR(**whisper_kwargs)
        
        # 执行转录
        asr_data = asr.asr_audio(chunk_file, language=language)
        
        # 序列化结果（ASRData 不能直接跨进程传递）
        segments = [
            {'text': seg.text, 'start_time': seg.start_time, 'end_time': seg.end_time}
            for seg in asr_data.segments
        ]
        
        return {
            'chunk_idx': chunk_idx,
            'segments': segments,
            'error': None
        }
    except Exception as e:
        logger.error(f"Chunk {chunk_idx} 转录失败: {e}")
        return {
            'chunk_idx': chunk_idx,
            'segments': [],
            'error': str(e)
        }


class ChunkedASR:
    """音频分块 ASR 包装器

    为任何 BaseASR 子类添加音频分块能力。
    适用于长音频的分块转录，避免 API 超时或内存溢出。

    工作流程：
        1. 将长音频切割为多个重叠的块
        2. 为每个块创建独立的 ASR 实例并发转录
        3. 使用 ChunkMerger 合并结果，消除重叠区域的重复内容

    示例:
        >>> # 使用 ASR 类和参数创建分块转录器
        >>> chunked_asr = ChunkedASR(
        ...     asr_class=BcutASR,
        ...     audio_path="long_audio.mp3",
        ...     asr_kwargs={"need_word_time_stamp": True},
        ...     chunk_length=1200
        ... )
        >>> result = chunked_asr.run(callback)

    Args:
        asr_class: ASR 类（非实例），如 BcutASR, JianYingASR
        audio_path: 音频文件路径
        asr_kwargs: 传递给 ASR 构造函数的参数字典
        chunk_length: 每块长度（秒），默认 480 秒（8分钟）
        chunk_overlap: 块之间重叠时长（秒），默认 10 秒
        chunk_concurrency: 并发转录数量，默认 3
    """

    def __init__(
        self,
        asr_class: type,  # 移除 BaseASR 类型提示以避免导入问题
        audio_path: str,
        asr_kwargs: Optional[dict] = None,
        chunk_length: int = DEFAULT_CHUNK_LENGTH_SEC,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_SEC,
        chunk_concurrency: int = DEFAULT_CHUNK_CONCURRENCY,
    ):
        self.asr_class = asr_class
        self.audio_path = audio_path
        self.asr_kwargs = asr_kwargs or {}
        self.chunk_length_ms = chunk_length * MS_PER_SECOND
        self.chunk_overlap_ms = chunk_overlap * MS_PER_SECOND
        self.chunk_concurrency = chunk_concurrency

        # 读取完整音频文件（用于分块）
        with open(audio_path, "rb") as f:
            self.file_binary = f.read()

    def run(self, callback: Optional[Callable[[int, str], None]] = None) -> ASRData:
        """执行分块转录

        Args:
            callback: 进度回调函数(progress: int, message: str)

        Returns:
            ASRData: 合并后的转录结果
        """
        # 1. 分块音频
        chunks = self._split_audio()

        # 2. 如果只有一块，直接创建单个 ASR 实例转录
        if len(chunks) == 1:
            logger.info("音频短于分块长度，直接转录")
            single_asr = self.asr_class(self.audio_path, **self.asr_kwargs)
            return single_asr.run(callback)

        logger.info(f"音频分为 {len(chunks)} 块，开始并发转录")

        # 3. 并发转录所有块
        chunk_results = self._asr_chunks(chunks, callback)

        # 4. 合并结果
        merged_result = self._merge_results(chunk_results, chunks)

        logger.info(f"分块转录完成，共 {len(merged_result.segments)} 个片段")
        return merged_result

    def _split_audio(self) -> List[Tuple[bytes, int]]:
        """使用 pydub 将音频切割为重叠的块

        Returns:
            List[(chunk_bytes, offset_ms), ...]
            每个元素包含音频块的字节数据和时间偏移（毫秒）
        """
        # 从字节数据加载音频
        if self.file_binary is None:
            raise ValueError("file_binary is None, cannot split audio")

        audio = AudioSegment.from_file(io.BytesIO(self.file_binary))
        total_duration_ms = len(audio)

        logger.info(
            f"音频总时长: {total_duration_ms/1000:.1f}s, "
            f"分块长度: {self.chunk_length_ms/1000:.1f}s, "
            f"重叠: {self.chunk_overlap_ms/1000:.1f}s"
        )

        chunks = []
        start_ms = 0

        while start_ms < total_duration_ms:
            end_ms = min(start_ms + self.chunk_length_ms, total_duration_ms)
            chunk = audio[start_ms:end_ms]

            buffer = io.BytesIO()
            chunk.export(buffer, format="mp3")
            chunk_bytes = buffer.getvalue()

            chunks.append((chunk_bytes, start_ms))
            logger.debug(
                f"切割 chunk {len(chunks)}: "
                f"{start_ms/1000:.1f}s - {end_ms/1000:.1f}s ({len(chunk_bytes)} bytes)"
            )

            # 下一个块的起始位置（有重叠）
            start_ms += self.chunk_length_ms - self.chunk_overlap_ms

            # 如果已到末尾，停止
            if end_ms >= total_duration_ms:
                break

        # logger.info(f"音频切割完成，共 {len(chunks)} 个块")
        return chunks

    def _asr_chunks(
        self,
        chunks: List[Tuple[bytes, int]],
        callback: Optional[Callable[[int, str], None]],
    ) -> List[ASRData]:
        """并发转录多个音频块

        根据 GPU 数量自动选择执行策略：
        - 单 GPU：使用线程池（共享模型，避免重复加载）
        - 多 GPU：使用进程池（每个进程独立加载模型到不同 GPU）

        Args:
            chunks: 音频块列表 [(chunk_bytes, offset_ms), ...]
            callback: 进度回调

        Returns:
            List[ASRData]: 每个块的转录结果
        """
        total_chunks = len(chunks)
        gpu_count = _get_available_gpu_count()
        
        # 决定使用进程池还是线程池
        use_multiprocess = gpu_count > 1
        
        if use_multiprocess:
            logger.info(f"检测到 {gpu_count} 个 GPU，使用多进程模式")
            return self._asr_chunks_multiprocess(chunks, callback, gpu_count)
        else:
            logger.info(f"单 GPU 或无 GPU 模式，使用线程池")
            return self._asr_chunks_threaded(chunks, callback)
    
    def _asr_chunks_multiprocess(
        self,
        chunks: List[Tuple[bytes, int]],
        callback: Optional[Callable[[int, str], None]],
        gpu_count: int,
    ) -> List[ASRData]:
        """使用多进程转录（多 GPU 场景）
        
        每个 chunk 分配到不同的 GPU，轮询分配。
        """
        total_chunks = len(chunks)
        results: List[Optional[ASRData]] = [None] * total_chunks
        temp_files = []
        
        try:
            # 1. 将 chunk 写入临时文件（进程间通过文件传递）
            for i, (chunk_bytes, offset_ms) in enumerate(chunks):
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.mp3', delete=False, prefix=f'chunk_{i}_'
                )
                temp_file.write(chunk_bytes)
                temp_file.close()
                temp_files.append(temp_file.name)
            
            # 2. 准备 worker 参数
            # 从 asr_kwargs 中提取 whisper 初始化参数
            whisper_kwargs = self.asr_kwargs.get('whisper_asr', None)
            if whisper_kwargs is None:
                # 如果没有 whisper_asr 参数，尝试从 WhisperASRAdapter 的 kwargs 构建
                # 这种情况下需要获取完整的 Whisper 配置
                raise ValueError("多进程模式需要 whisper_asr 实例的配置参数")
            
            # 获取 WhisperASR 的初始化参数（需要从实例重建）
            whisper_instance = whisper_kwargs
            init_kwargs = self._extract_whisper_init_kwargs(whisper_instance)
            language = self.asr_kwargs.get('language', whisper_instance.language)
            
            worker_args = []
            for i, temp_file in enumerate(temp_files):
                gpu_id = i % gpu_count  # 轮询分配 GPU
                worker_args.append({
                    'chunk_idx': i,
                    'chunk_file': temp_file,
                    'gpu_id': gpu_id,
                    'whisper_kwargs': init_kwargs,
                    'language': language,
                })
            
            # 3. 使用进程池执行
            # 限制并发数为 GPU 数量（每个 GPU 同时只跑一个任务）
            max_workers = min(gpu_count, total_chunks)
            
            if callback:
                callback(0, f"开始多 GPU 并行转录 ({gpu_count} GPUs, {total_chunks} chunks)")
            
            # 必须使用 spawn 上下文，否则 CUDA 会死锁（fork 与 CUDA 不兼容）
            ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(_worker_transcribe_chunk, args): args['chunk_idx']
                    for args in worker_args
                }
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    idx = result['chunk_idx']
                    completed += 1
                    
                    if result['error']:
                        logger.error(f"Chunk {idx} 失败: {result['error']}")
                        results[idx] = ASRData([])
                    else:
                        # 反序列化结果
                        segments = [
                            ASRDataSeg(
                                text=seg['text'],
                                start_time=seg['start_time'],
                                end_time=seg['end_time']
                            )
                            for seg in result['segments']
                        ]
                        results[idx] = ASRData(segments)
                        logger.info(f"Chunk {idx+1}/{total_chunks} 完成 ({len(segments)} 片段)")
                    
                    if callback:
                        progress = int(completed / total_chunks * 100)
                        callback(progress, f"已完成 {completed}/{total_chunks} 块")
            
            logger.info(f"多进程转录完成，共 {total_chunks} 块")
            return [r for r in results if r is not None]
            
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
    
    def _extract_whisper_init_kwargs(self, whisper_instance) -> Dict[str, Any]:
        """从 WhisperASR 实例提取初始化参数"""
        return {
            'model_name': whisper_instance.model_name,
            'device': 'cuda',  # 子进程会通过 CUDA_VISIBLE_DEVICES 控制
            'compute_type': whisper_instance.compute_type,
            'language': whisper_instance.language,
            'vad_filter': whisper_instance.vad_filter,
            'beam_size': whisper_instance.beam_size,
            'download_root': whisper_instance.download_root,
            'word_timestamps': whisper_instance.word_timestamps,
            'condition_on_previous_text': whisper_instance.condition_on_previous_text,
            'temperature': whisper_instance.temperature,
            'compression_ratio_threshold': whisper_instance.compression_ratio_threshold,
            'log_prob_threshold': whisper_instance.log_prob_threshold,
            'no_speech_threshold': whisper_instance.no_speech_threshold,
            'initial_prompt': whisper_instance.initial_prompt,
            'repetition_penalty': whisper_instance.repetition_penalty,
            'hallucination_silence_threshold': whisper_instance.hallucination_silence_threshold,
            'vad_threshold': whisper_instance.vad_threshold,
            'vad_min_speech_duration_ms': whisper_instance.vad_min_speech_duration_ms,
            'vad_max_speech_duration_s': whisper_instance.vad_max_speech_duration_s,
            'vad_min_silence_duration_ms': whisper_instance.vad_min_silence_duration_ms,
            'vad_speech_pad_ms': whisper_instance.vad_speech_pad_ms,
            'enable_chunked': False,  # 子进程不再分块
            'chunk_length_sec': whisper_instance.chunk_length_sec,
            'chunk_overlap_sec': whisper_instance.chunk_overlap_sec,
            'chunk_concurrency': 1,
            'use_pipeline': whisper_instance.use_pipeline,
            'enable_diarization': whisper_instance.enable_diarization,
            'enable_punctuation': whisper_instance.enable_punctuation,
            'pipeline_batch_size': whisper_instance.pipeline_batch_size,
            'pipeline_chunk_length': whisper_instance.pipeline_chunk_length,
            'num_speakers': whisper_instance.num_speakers,
            'min_speakers': whisper_instance.min_speakers,
            'max_speakers': whisper_instance.max_speakers,
        }
    
    def _asr_chunks_threaded(
        self,
        chunks: List[Tuple[bytes, int]],
        callback: Optional[Callable[[int, str], None]],
    ) -> List[ASRData]:
        """使用线程池转录（单 GPU 场景，原有逻辑）"""
        results: List[Optional[ASRData]] = [None] * len(chunks)
        total_chunks = len(chunks)

        # 进度追踪
        chunk_progress = [0] * total_chunks
        last_overall = 0
        progress_lock = threading.Lock()

        def asr_single_chunk(
            idx: int, chunk_bytes: bytes, offset_ms: int
        ) -> Tuple[int, ASRData]:
            nonlocal last_overall
            logger.info(f"开始转录 chunk {idx+1}/{total_chunks} (offset={offset_ms}ms)")

            def chunk_callback(progress: int, message: str):
                nonlocal last_overall
                if not callback:
                    return
                with progress_lock:
                    chunk_progress[idx] = progress
                    overall = sum(chunk_progress) // total_chunks
                    if overall > last_overall:
                        last_overall = overall
                        callback(overall, f"{idx+1}/{total_chunks}: {message}")

            chunk_asr = self.asr_class(chunk_bytes, **self.asr_kwargs)
            asr_data = chunk_asr.run(chunk_callback)

            logger.info(
                f"Chunk {idx+1}/{total_chunks} 转录完成，"
                f"获得 {len(asr_data.segments)} 个片段"
            )
            return idx, asr_data

        with ThreadPoolExecutor(max_workers=self.chunk_concurrency) as executor:
            futures = {
                executor.submit(asr_single_chunk, i, chunk_bytes, offset): i
                for i, (chunk_bytes, offset) in enumerate(chunks)
            }

            for future in as_completed(futures):
                idx, asr_data = future.result()
                results[idx] = asr_data

        logger.info(f"所有 {total_chunks} 个块转录完成")
        return [r for r in results if r is not None]

    def _merge_results(
        self, chunk_results: List[ASRData], chunks: List[Tuple[bytes, int]]
    ) -> ASRData:
        """使用 ChunkMerger 合并转录结果

        Args:
            chunk_results: 每个块的 ASRData 结果
            chunks: 原始音频块信息（用于获取 offset）

        Returns:
            合并后的 ASRData
        """
        merger = ChunkMerger(min_match_count=2, fuzzy_threshold=0.7)

        # 提取每个 chunk 的时间偏移
        chunk_offsets = [offset for _, offset in chunks]

        # 合并
        merged = merger.merge_chunks(
            chunks=chunk_results,
            chunk_offsets=chunk_offsets,
            overlap_duration=self.chunk_overlap_ms,
        )
        return merged
