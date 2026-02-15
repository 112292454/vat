"""音频分块 ASR 装饰器

为任何 BaseASR 实现添加音频分块转录能力，适用于长音频处理。
使用装饰器模式实现关注点分离。

多 GPU 支持：
- 单 GPU：使用线程池（共享模型，避免重复加载）
- 多 GPU：per-GPU worker 进程模型
  - 每个 GPU 一个持久 worker 进程，模型只加载一次
  - 所有 chunk 放入共享队列，worker 自行取任务（天然负载均衡）
  - worker 加载模型前检查显存是否满足 min_free_memory_mb
  - OOM 时 chunk 重回队列重试，不丢失内容
"""

import io
import multiprocessing
import os
import queue
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# 多 GPU worker 常量
_MEMORY_CHECK_INTERVAL_SEC = 5   # 显存不足时的重试间隔（秒）
_MEMORY_CHECK_MAX_WAIT_SEC = 300  # 显存等待最长时间（秒）
_OOM_MAX_RETRIES = 3              # 单个 chunk OOM 最大重试次数
_OOM_COOLDOWN_SEC = 10            # OOM 后冷却等待时间（秒）
_WORKER_QUEUE_TIMEOUT_SEC = 2     # worker 从队列取任务的超时时间（秒）


def _get_available_gpu_count() -> int:
    """获取可用 GPU 数量"""
    try:
        from ..utils.gpu import get_available_gpus
        gpus = get_available_gpus()
        return len(gpus)
    except Exception:
        return 0


def _gpu_worker_loop(
    gpu_id: int,
    chunk_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    whisper_kwargs: Dict[str, Any],
    language: str,
    min_free_memory_mb: int,
):
    """
    Per-GPU 持久 worker 函数：在独立进程中运行
    
    每个 GPU 对应一个 worker 进程。worker 加载模型一次后循环处理 chunk。
    - 加载模型前检查显存是否满足 min_free_memory_mb
    - 显存不足时等待（有最大等待时间）
    - OOM 时清理 CUDA cache，将 chunk 重回队列重试
    - 收到 None 哨兵值时退出
    
    Args:
        gpu_id: 物理 GPU 索引
        chunk_queue: 共享 chunk 队列，元素为 (chunk_idx, chunk_file, retry_count) 或 None（哨兵）
        result_queue: 结果队列，元素为 dict(chunk_idx, segments, error)
        whisper_kwargs: WhisperASR 初始化参数
        language: 转录语言
        min_free_memory_mb: 加载模型所需的最低空闲显存 (MB)
    """
    # 设置 CUDA_VISIBLE_DEVICES：子进程隔离，不影响主进程
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 子进程中重新初始化 logger
    from ..utils.logger import setup_logger as _setup_logger
    worker_logger = _setup_logger(f"chunked_asr.gpu{gpu_id}")
    worker_logger.info(f"Worker GPU {gpu_id} 启动")
    
    asr_model = None  # 延迟加载
    
    while True:
        # 从共享队列取任务
        try:
            item = chunk_queue.get(timeout=_WORKER_QUEUE_TIMEOUT_SEC)
        except queue.Empty:
            continue
        
        # 哨兵值：退出
        if item is None:
            worker_logger.info(f"Worker GPU {gpu_id} 收到退出信号")
            break
        
        chunk_idx, chunk_file, retry_count = item
        worker_logger.info(
            f"GPU {gpu_id} 开始处理 Chunk {chunk_idx}"
            + (f" (重试 #{retry_count})" if retry_count > 0 else "")
        )
        
        try:
            # 延迟加载模型：首次处理 chunk 时加载
            if asr_model is None:
                asr_model = _load_model_with_memory_check(
                    gpu_id, whisper_kwargs, min_free_memory_mb, worker_logger
                )
                if asr_model is None:
                    # 等待超时仍无法加载，将 chunk 放回队列
                    worker_logger.error(
                        f"GPU {gpu_id} 显存等待超时，无法加载模型。"
                        f"Chunk {chunk_idx} 放回队列"
                    )
                    chunk_queue.put((chunk_idx, chunk_file, retry_count))
                    break  # 退出此 worker（显存长期不足，避免死循环）
            
            # 执行转录
            asr_data = asr_model.asr_audio(chunk_file, language=language)
            
            # 序列化结果（ASRData 不能直接跨进程传递）
            segments = [
                {'text': seg.text, 'start_time': seg.start_time, 'end_time': seg.end_time}
                for seg in asr_data.segments
            ]
            
            result_queue.put({
                'chunk_idx': chunk_idx,
                'segments': segments,
                'error': None,
            })
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if 'out of memory' in error_msg or 'cuda' in error_msg and 'memory' in error_msg:
                # OOM：清理 CUDA cache，卸载模型，chunk 重回队列
                worker_logger.warning(
                    f"GPU {gpu_id} Chunk {chunk_idx} OOM (重试 {retry_count}/{_OOM_MAX_RETRIES})"
                )
                _cleanup_cuda(asr_model, worker_logger)
                asr_model = None  # 下次循环会重新加载
                
                if retry_count < _OOM_MAX_RETRIES:
                    chunk_queue.put((chunk_idx, chunk_file, retry_count + 1))
                    worker_logger.info(
                        f"Chunk {chunk_idx} 已放回队列，等待 {_OOM_COOLDOWN_SEC}s 后继续"
                    )
                    time.sleep(_OOM_COOLDOWN_SEC)
                else:
                    worker_logger.error(
                        f"Chunk {chunk_idx} 达到最大重试次数 {_OOM_MAX_RETRIES}，标记为失败"
                    )
                    result_queue.put({
                        'chunk_idx': chunk_idx,
                        'segments': [],
                        'error': f"OOM after {_OOM_MAX_RETRIES} retries: {e}",
                    })
            else:
                # 其他 RuntimeError
                worker_logger.error(f"GPU {gpu_id} Chunk {chunk_idx} 转录失败: {e}")
                result_queue.put({
                    'chunk_idx': chunk_idx,
                    'segments': [],
                    'error': str(e),
                })
                
        except Exception as e:
            worker_logger.error(f"GPU {gpu_id} Chunk {chunk_idx} 转录异常: {e}")
            result_queue.put({
                'chunk_idx': chunk_idx,
                'segments': [],
                'error': str(e),
            })
    
    # 退出前清理
    _cleanup_cuda(asr_model, worker_logger)
    worker_logger.info(f"Worker GPU {gpu_id} 已退出")


def _load_model_with_memory_check(
    gpu_id: int,
    whisper_kwargs: Dict[str, Any],
    min_free_memory_mb: int,
    worker_logger,
):
    """
    在加载模型前检查 GPU 显存，显存不足时等待。
    
    Args:
        gpu_id: 物理 GPU 索引
        whisper_kwargs: WhisperASR 初始化参数
        min_free_memory_mb: 最低空闲显存要求
        worker_logger: worker 专属 logger
        
    Returns:
        加载成功的 WhisperASR 实例，或 None（等待超时）
    """
    from ..utils.gpu import check_gpu_free_memory
    
    waited = 0
    while waited < _MEMORY_CHECK_MAX_WAIT_SEC:
        if check_gpu_free_memory(gpu_id, min_free_memory_mb):
            break
        worker_logger.info(
            f"GPU {gpu_id} 显存不足 {min_free_memory_mb}MB，"
            f"等待 {_MEMORY_CHECK_INTERVAL_SEC}s... (已等待 {waited}s/{_MEMORY_CHECK_MAX_WAIT_SEC}s)"
        )
        time.sleep(_MEMORY_CHECK_INTERVAL_SEC)
        waited += _MEMORY_CHECK_INTERVAL_SEC
    else:
        # while 正常结束（未 break）= 等待超时
        return None
    
    # 显存满足，加载模型
    worker_logger.info(f"GPU {gpu_id} 显存充足，开始加载模型")
    from .whisper_wrapper import WhisperASR
    asr = WhisperASR(**whisper_kwargs)
    # 触发模型加载（_ensure_model_loaded 在 asr_audio 中调用，但这里提前加载以尽早发现问题）
    asr._ensure_model_loaded()
    worker_logger.info(f"GPU {gpu_id} 模型加载完成")
    return asr


def _cleanup_cuda(asr_model, worker_logger):
    """
    清理 CUDA 资源：删除模型引用并清空 CUDA cache
    
    Args:
        asr_model: WhisperASR 实例（可为 None）
        worker_logger: worker 专属 logger
    """
    try:
        if asr_model is not None:
            # 清除模型引用
            if hasattr(asr_model, 'model') and asr_model.model is not None:
                del asr_model.model
                asr_model.model = None
            del asr_model
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            worker_logger.debug("CUDA cache 已清理")
    except Exception as e:
        worker_logger.warning(f"CUDA 清理时出错: {e}")


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
        # 直接从文件路径加载音频（不使用 BytesIO）
        # pydub 需要文件扩展名来正确识别格式，BytesIO 无扩展名会导致大 WAV 文件时长误判
        audio = AudioSegment.from_file(self.audio_path)
        total_duration_ms = len(audio)

        logger.info(
            f"音频总时长: {total_duration_ms/1000:.1f}s, "
            f"分块长度: {self.chunk_length_ms/1000:.1f}s, "
            f"重叠: {self.chunk_overlap_ms/1000:.1f}s，开始分块音频"
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
        
        Per-GPU worker 模型：
        - 每个 GPU 启动一个持久 worker 进程，模型只加载一次
        - 所有 chunk 放入共享队列，worker 自行取任务（天然负载均衡）
        - worker 加载模型前检查 GPU 空闲显存是否满足 min_free_memory_mb
        - OOM 时 chunk 重回队列重试（最多 _OOM_MAX_RETRIES 次）
        """
        total_chunks = len(chunks)
        results: List[Optional[ASRData]] = [None] * total_chunks
        temp_files = []
        workers: List[multiprocessing.Process] = []
        
        try:
            # 1. 将 chunk 写入临时文件（进程间通过文件传递）
            for i, (chunk_bytes, offset_ms) in enumerate(chunks):
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.mp3', delete=False, prefix=f'chunk_{i}_'
                )
                temp_file.write(chunk_bytes)
                temp_file.close()
                temp_files.append(temp_file.name)
            
            # 2. 准备 whisper 初始化参数
            whisper_instance = self.asr_kwargs.get('whisper_asr', None)
            if whisper_instance is None:
                raise ValueError("多进程模式需要 whisper_asr 实例的配置参数")
            
            init_kwargs = self._extract_whisper_init_kwargs(whisper_instance)
            language = self.asr_kwargs.get('language', whisper_instance.language)
            
            # min_free_memory_mb: 使用 ASR 模型的显存要求（与 whisper_wrapper._resolve_device 一致）
            # TODO: 应从 config.gpu.min_free_memory_mb 统一传入，而非硬编码
            min_free_memory_mb = 8000
            
            # 3. 创建共享队列
            ctx = multiprocessing.get_context('spawn')
            chunk_queue = ctx.Queue()
            result_queue = ctx.Queue()
            
            # 将所有 chunk 放入队列（retry_count=0）
            for i, temp_file in enumerate(temp_files):
                chunk_queue.put((i, temp_file, 0))
            
            # 4. 为每个 GPU 启动一个 worker 进程
            active_gpu_count = min(gpu_count, total_chunks)
            
            if callback:
                callback(0, f"开始多 GPU 并行转录 ({active_gpu_count} GPUs, {total_chunks} chunks)")
            
            logger.info(
                f"启动 {active_gpu_count} 个 GPU worker，"
                f"处理 {total_chunks} 个 chunk，"
                f"min_free_memory_mb={min_free_memory_mb}"
            )
            
            for gpu_idx in range(active_gpu_count):
                p = ctx.Process(
                    target=_gpu_worker_loop,
                    args=(
                        gpu_idx,
                        chunk_queue,
                        result_queue,
                        init_kwargs,
                        language,
                        min_free_memory_mb,
                    ),
                    name=f"asr-worker-gpu{gpu_idx}",
                )
                p.start()
                workers.append(p)
            
            # 5. 收集结果
            completed = 0
            failed_chunks = []
            
            while completed < total_chunks:
                try:
                    result = result_queue.get(timeout=600)  # 10分钟超时（单个 chunk 最长处理时间）
                except queue.Empty:
                    # 检查是否所有 worker 都已退出
                    alive_workers = [w for w in workers if w.is_alive()]
                    if not alive_workers:
                        logger.error(
                            f"所有 worker 已退出，但只收到 {completed}/{total_chunks} 个结果"
                        )
                        break
                    logger.warning(
                        f"等待结果超时，已完成 {completed}/{total_chunks}，"
                        f"存活 worker: {len(alive_workers)}"
                    )
                    continue
                
                idx = result['chunk_idx']
                completed += 1
                
                if result['error']:
                    logger.error(f"Chunk {idx} 最终失败: {result['error']}")
                    results[idx] = ASRData([])
                    failed_chunks.append(idx)
                else:
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
            
            # 6. 发送哨兵值通知 worker 退出
            for _ in workers:
                chunk_queue.put(None)
            
            # 等待 worker 进程结束
            for w in workers:
                w.join(timeout=30)
                if w.is_alive():
                    logger.warning(f"Worker {w.name} 未在30s内退出，强制终止")
                    w.terminate()
                    w.join(timeout=5)
            
            # 7. 检查完整性：是否有 chunk 失败或丢失
            missing_chunks = [i for i, r in enumerate(results) if r is None]
            if missing_chunks:
                raise RuntimeError(
                    f"多 GPU 转录不完整: {len(missing_chunks)} 个 chunk 未收到结果 "
                    f"(chunk 索引: {missing_chunks})。"
                    f"可能原因: 所有 worker 显存等待超时退出"
                )
            if failed_chunks:
                raise RuntimeError(
                    f"多 GPU 转录失败: {len(failed_chunks)}/{total_chunks} 个 chunk 失败 "
                    f"(chunk 索引: {failed_chunks})"
                )
            
            logger.info(f"多进程转录完成，共 {total_chunks} 块")
            return [r for r in results if r is not None]
            
        finally:
            # 确保所有 worker 进程被清理
            for w in workers:
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=5)
            
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
