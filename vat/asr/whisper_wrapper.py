"""
Faster-Whisper语音识别封装
集成 ASRData 和 ChunkedASR 支持
"""
import os
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
import subprocess

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

from .asr_data import ASRData, ASRDataSeg
from .chunked_asr import ChunkedASR
from vat.utils.gpu import resolve_gpu_device, is_cuda_available
from vat.utils.logger import setup_logger

logger = setup_logger("whisper_asr")


class WhisperASR:
    """Faster-Whisper语音识别器，支持 ASRData 和分块处理"""
    
    # 类级别的锁，用于确保模型只加载一次
    _model_load_lock = threading.Lock()
    # 类级别的模型缓存，按 (model_name, device, compute_type, download_root) 缓存
    _model_cache = {}
    _model_cache_lock = threading.Lock()
    
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        language: str,
        vad_filter: bool,
        beam_size: int,
        download_root: Optional[str],
        # 高级参数
        word_timestamps: bool,
        condition_on_previous_text: bool,
        temperature: List[float],
        compression_ratio_threshold: float,
        log_prob_threshold: float,
        no_speech_threshold: float,
        initial_prompt: str,
        repetition_penalty: float,
        hallucination_silence_threshold: Optional[float],
        # VAD参数
        vad_threshold: float,
        vad_min_speech_duration_ms: int,
        vad_max_speech_duration_s: float,
        vad_min_silence_duration_ms: int,
        vad_speech_pad_ms: int,
        # ChunkedASR参数
        enable_chunked: bool,
        chunk_length_sec: int,
        chunk_overlap_sec: int,
        chunk_concurrency: int,
        # Pipeline模式配置
        use_pipeline: bool,
        enable_diarization: bool,
        enable_punctuation: bool,
        pipeline_batch_size: int,
        pipeline_chunk_length: int,
        num_speakers: Optional[int],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ):
        """
        初始化Whisper转录器
        
        注意：所有参数必须从 config 统一配置传入，禁止使用默认值。
        遵循项目"单一数据源原则"：配置有且只有一个来源（config 文件）。
        
        Args:
            model_name: 模型名称或路径
            device: 设备 (cuda/cpu)
            compute_type: 计算精度 (float16/float32/int8)
            language: 源语言代码
            vad_filter: 是否启用VAD过滤
            beam_size: beam搜索大小
            download_root: 模型下载根目录（可选，默认使用系统缓存目录）
            word_timestamps: 启用词级时间戳
            condition_on_previous_text: 基于前文的条件预测
            temperature: 温度回退列表
            compression_ratio_threshold: 压缩比阈值
            log_prob_threshold: 对数概率阈值
            no_speech_threshold: 无语音阈值
            initial_prompt: 初始提示词
            repetition_penalty: 重复惩罚
            hallucination_silence_threshold: 幻觉静默阈值
            vad_threshold: VAD激活阈值
            vad_min_speech_duration_ms: 最小语音段时长
            vad_max_speech_duration_s: 最大语音段时长
            vad_min_silence_duration_ms: 最小静音时长
            vad_speech_pad_ms: 语音前后填充时长
        """
        if not use_pipeline and not FASTER_WHISPER_AVAILABLE:
            raise ImportError("faster-whisper未安装，请运行: pip install faster-whisper")
        
        self.model_name = model_name
        self.compute_type = compute_type
        
        # GPU 设备解析：支持 "auto", "cpu", "cuda:N" 格式
        self._resolve_device(device)
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.download_root = download_root
        
        # 高级参数
        self.word_timestamps = word_timestamps
        self.condition_on_previous_text = condition_on_previous_text
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.initial_prompt = initial_prompt
        self.repetition_penalty = repetition_penalty
        self.hallucination_silence_threshold = hallucination_silence_threshold
        
        # VAD参数
        self.vad_threshold = vad_threshold
        self.vad_min_speech_duration_ms = vad_min_speech_duration_ms
        self.vad_max_speech_duration_s = vad_max_speech_duration_s
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms
        
        # ChunkedASR参数
        self.enable_chunked = enable_chunked
        self.chunk_length_sec = chunk_length_sec
        self.chunk_overlap_sec = chunk_overlap_sec
        self.chunk_concurrency = chunk_concurrency

        # Pipeline参数
        self.use_pipeline = use_pipeline
        self.enable_diarization = enable_diarization
        self.enable_punctuation = enable_punctuation
        self.pipeline_batch_size = pipeline_batch_size
        self.pipeline_chunk_length = pipeline_chunk_length
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # Hugging Face Token (用于访问受限模型)
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        # 初始化模型（延迟加载）
        self.model = None
    
    def _resolve_device(self, device: str) -> None:
        """
        解析并设置 GPU 设备
        
        支持格式：
        - "auto": 自动选择显存占用最低的 GPU
        - "cpu": 使用 CPU
        - "cuda": 使用默认 GPU (cuda:0)
        - "cuda:N": 使用指定 GPU
        
        Args:
            device: 设备标识符
        
        注意：
            此方法只解析设备并存储 gpu_id，不设置 CUDA_VISIBLE_DEVICES 环境变量。
            模型加载时通过 device_index 参数指定目标 GPU，避免环境变量污染。
            如果 CUDA_VISIBLE_DEVICES 已被外部设置（如 scheduler 子进程），
            表示当前进程已被限制到特定 GPU，直接使用 cuda 设备即可。
        """
        # 检查是否已有外部设置的 CUDA_VISIBLE_DEVICES（如 scheduler 子进程）
        # 此时当前进程只能看到被限定的 GPU，device_index 应为 0
        external_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if external_cuda_devices is not None and external_cuda_devices != "":
            self.device = "cuda"
            self.gpu_id = None  # 由 CUDA_VISIBLE_DEVICES 控制，device_index=0
            logger.info(f"ASR 将使用外部指定的 GPU (CUDA_VISIBLE_DEVICES={external_cuda_devices})")
            return
        
        # 兼容旧格式 "cuda" -> "auto"
        if device == "cuda":
            device = "auto"
        
        try:
            device_str, gpu_id = resolve_gpu_device(
                device,
                allow_cpu_fallback=False,  # 遵循 GPU 原则
                min_free_memory_mb=8000   # ASR模型需要约8GB显存
            )
            self.device = device_str
            self.gpu_id = gpu_id
            
            # 不再设置 CUDA_VISIBLE_DEVICES，避免多线程/多视频环境下的环境变量污染
            # 模型加载时通过 WhisperModel 的 device_index 参数指定目标 GPU
            if gpu_id is not None:
                logger.info(f"ASR 自动选择 GPU {gpu_id}")
            elif device_str == "cpu":
                logger.info("ASR 将使用 CPU 模式")
                
        except RuntimeError as e:
            logger.error(f"GPU 解析失败: {e}")
            raise
    
    def _ensure_model_loaded(self):
        """确保模型已加载（线程安全）"""
        if self.model is not None or (self.use_pipeline and hasattr(self, 'pipe')):
            return
        
        if self.use_pipeline:
            self._load_pipeline_model()
        else:
            self._load_faster_whisper_model()

    def _load_pipeline_model(self):
        """加载Transformers Pipeline模型 [已搁置 - 实验性功能]"""
        import warnings
        warnings.warn(
            "[VAT] Pipeline ASR 模式已搁置，效果不如 faster-whisper。"
            "此功能可能不稳定，建议使用默认的 faster-whisper 模式。",
            UserWarning,
            stacklevel=2
        )
        
        from .experimental.pipeline_asr import load_pipeline_model
        
        with self._model_load_lock:
            if hasattr(self, 'pipe'):
                return
            
            self.pipe = load_pipeline_model(
                model_name=self.model_name,
                device=self.device,
                batch_size=self.pipeline_batch_size,
                hf_token=self.hf_token
            )
            logger.info("Pipeline模型加载完成")

    def _load_faster_whisper_model(self):
        """加载faster-whisper模型（原有逻辑）"""
        # 生成缓存键（包含 gpu_id，因为不同 GPU 上的模型实例不能共享）
        device_index = self.gpu_id if self.gpu_id is not None else 0
        cache_key = (self.model_name, self.device, device_index, self.compute_type, self.download_root)
        
        # 先检查缓存
        with self._model_cache_lock:
            if cache_key in self._model_cache:
                self.model = self._model_cache[cache_key]
                return
        
        # 使用锁确保只有一个线程加载模型
        with self._model_load_lock:
            # 双重检查：可能其他线程已经加载了
            if self.model is not None:
                return
            
            # 再次检查缓存（可能在等待锁时其他线程已加载）
            with self._model_cache_lock:
                if cache_key in self._model_cache:
                    self.model = self._model_cache[cache_key]
                    return
            
            logger.info(f"正在加载Whisper模型: {self.model_name} ({self.device}, {self.compute_type})")
            if self.download_root:
                logger.info(f"模型下载目录: {self.download_root}")
            
            # 在模型下载时禁用 tqdm 以避免并发问题
            # 通过环境变量禁用 huggingface_hub 的 tqdm
            original_hf_hub_disable_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS")
            try:
                os.environ["HF_HUB_DISABLE_PROGRESS"] = "1"
                
                
                # 构建模型参数
                model_kwargs = {
                    "device": self.device,
                    "compute_type": self.compute_type,
                }
                if self.download_root:
                    model_kwargs["download_root"] = self.download_root
                
                # 通过 device_index 指定目标 GPU，不依赖 CUDA_VISIBLE_DEVICES
                # 当 CUDA_VISIBLE_DEVICES 已由外部设置时（如 scheduler 子进程），
                # gpu_id 为 None，device_index 默认为 0（即可见设备中的第一个）
                if self.device == "cuda":
                    model_kwargs["device_index"] = self.gpu_id if self.gpu_id is not None else 0
                
                # 加载模型
                model = WhisperModel(
                    self.model_name,
                    **model_kwargs
                )
                
                # 缓存模型
                with self._model_cache_lock:
                    self._model_cache[cache_key] = model
                    self.model = model
                
                logger.info("模型加载完成")
            finally:
                # 恢复环境变量
                if original_hf_hub_disable_progress is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS"] = original_hf_hub_disable_progress
    
    def asr_audio(
        self,
        audio_path: Union[Path, bytes],
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> ASRData:
        """
        转录音频文件（支持分块处理）
        
        Args:
            audio_path: 音频文件路径或字节数据
            language: 语言（可选，覆盖默认设置）
            progress_callback: 进度回调函数
            
        Returns:
            ASRData 对象包含所有转录片段
        """
        self._ensure_model_loaded()
        
        if self.use_pipeline:
            # return self._asr_with_pipeline(audio_path, language, progress_callback)
            raise NotImplementedError
        else:
            return self._asr_with_faster_whisper(audio_path, language, progress_callback)

    # def _asr_with_pipeline(
    #     self,
    #     audio_path: Union[Path, bytes],
    #     language: Optional[str] = None,
    #     progress_callback: Optional[Callable[[str], None]] = None
    # ) -> ASRData:
    #     """使用Pipeline进行转录 [已搁置 - 实验性功能]"""
    #     import warnings
    #     warnings.warn(
    #         "[VAT] Pipeline ASR 转录已搁置，效果不如 faster-whisper。",
    #         UserWarning,
    #         stacklevel=2
    #     )
    #     lang = language or self.language
        
    #     if progress_callback:
    #         progress_callback(f"开始Pipeline转录: {audio_path}")
        
    #     # 构建pipeline参数，复用Whisper的配置风格
    #     generate_kwargs = {
    #         "language": lang,
    #         "task": "transcribe",
    #         "num_beams": self.beam_size,
    #         "condition_on_prev_tokens": self.condition_on_previous_text,
    #         "compression_ratio_threshold": self.compression_ratio_threshold,
    #         "temperature": self.temperature,
    #         "logprob_threshold": self.log_prob_threshold,
    #         "no_speech_threshold": self.no_speech_threshold,
    #         "repetition_penalty": self.repetition_penalty,
    #         "initial_prompt": self.initial_prompt if self.initial_prompt else None,
    #     }
        
    #     # 移除 None 值
    #     generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
        
    #     # 执行转录参数
    #     call_kwargs = {
    #         "chunk_length_s": self.pipeline_chunk_length,
    #         "return_timestamps": True,
    #         "generate_kwargs": generate_kwargs,
    #         "add_punctuation": self.enable_punctuation,
    #         # "add_silence_end": 0.5, 
    #         # "add_silence_start": 0.5
    #     }
        
    #     # 说话人数量控制
    #     if self.num_speakers is not None:
    #         call_kwargs["num_speakers"] = self.num_speakers
    #     if self.min_speakers is not None:
    #         call_kwargs["min_speakers"] = self.min_speakers
    #     if self.max_speakers is not None:
    #         call_kwargs["max_speakers"] = self.max_speakers
        
    #     # 执行转录
    #     result = self.pipe(str(audio_path), **call_kwargs)
        
    #     return self._convert_pipeline_to_asr_data(result, progress_callback)

    # def _convert_pipeline_to_asr_data(
    #     self,
    #     result: dict,
    #     progress_callback: Optional[Callable[[str], None]] = None
    # ) -> ASRData:
    #     """将Pipeline结果转换为ASRData"""
    #     asr_segments = []
        
    #     if self.enable_diarization and 'chunks' in result:
    #         # 说话人分离模式
    #         for chunk in result['chunks']:
    #             text = chunk.get('text', '').strip()
    #             if not text:
    #                 continue
                
    #             # 过滤无用文本（保留原有逻辑）
    #             if any(x in text for x in ["・", "作詞", "編曲"]) or text.startswith(("【", "（")):
    #                 continue
                
    #             asr_segments.append(ASRDataSeg(
    #                 text=text,
    #                 start_time=int(chunk['timestamp'][0] * 1000),
    #                 end_time=int(chunk['timestamp'][1] * 1000),
    #                 speaker_id=chunk.get('speaker_id', None)
    #             ))
                
    #             if progress_callback:
    #                 speaker_info = f" [{chunk.get('speaker_id')}]" if chunk.get('speaker_id') else ""
    #                 progress_callback(f"[{chunk['timestamp'][0]:.2f}s]{speaker_info} {text}")
    #     else:
    #         # 标准模式（无说话人分离）
    #         chunks = result.get('chunks', [])
    #         for chunk in chunks:
    #             text = chunk.get('text', '').strip()
    #             if not text:
    #                 continue
                
    #             # 过滤无用文本
    #             if any(x in text for x in ["・", "作詞", "編曲"]) or text.startswith(("【", "（")):
    #                 continue
                
    #             asr_segments.append(ASRDataSeg(
    #                 text=text,
    #                 start_time=int(chunk['timestamp'][0] * 1000),
    #                 end_time=int(chunk['timestamp'][1] * 1000)
    #             ))
                
    #             if progress_callback:
    #                 progress_callback(f"[{chunk['timestamp'][0]:.2f}s -> {chunk['timestamp'][1]:.2f}s] {text}")
        
    #     return ASRData(asr_segments)

    def _asr_with_faster_whisper(
        self,
        audio_path: Union[Path, bytes],
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> ASRData:
        """使用faster-whisper进行转录"""
        assert audio_path is not None, "调用契约错误: audio_path 不能为空"
        if isinstance(audio_path, (str, Path)):
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        if progress_callback:
            progress_callback(f"开始转录: {audio_path}")
        
        # 使用的语言
        lang = language or self.language
        
        # 构建VAD参数
        vad_parameters = None
        if self.vad_filter:
            vad_parameters = {
                "threshold": self.vad_threshold,
                "min_speech_duration_ms": self.vad_min_speech_duration_ms,
                "max_speech_duration_s": self.vad_max_speech_duration_s,
                "min_silence_duration_ms": self.vad_min_silence_duration_ms,
                "speech_pad_ms": self.vad_speech_pad_ms,
            }
        
        # 执行转录
        segments, info = self.model.transcribe(
            str(audio_path),
            language=lang,
            vad_filter=self.vad_filter,
            vad_parameters=vad_parameters,
            beam_size=self.beam_size,
            word_timestamps=self.word_timestamps,
            condition_on_previous_text=self.condition_on_previous_text,
            temperature=self.temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            initial_prompt=self.initial_prompt if self.initial_prompt else None,
            repetition_penalty=self.repetition_penalty,
            hallucination_silence_threshold=self.hallucination_silence_threshold,
        )
        
        if info is None:
            raise RuntimeError("Whisper 转录未返回 info 对象")
        
        if progress_callback:
            progress_callback(f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")
        
        # 收集结果并转换为 ASRDataSeg
        asr_segments = []
        for segment in segments:
            text = segment.text.strip()
            if any(s in text for s in ["・", "作詞", "編曲", "ご視聴"]) or text.startswith(("【", "（")):
                continue
            if text:  # 跳过空文本
                assert segment.start < segment.end, f"逻辑错误: 无效的时间戳: {segment.start} -> {segment.end}"
                asr_segments.append(ASRDataSeg(
                    text=text,
                    start_time=int(segment.start * 1000),  # 转换为毫秒
                    end_time=int(segment.end * 1000)
                ))
            
            if progress_callback:
                progress_callback(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")
        
        return ASRData(asr_segments)
    
    def asr_video(
        self,
        video_path: Path,
        output_audio_path: Optional[Path] = None,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> ASRData:
        """
        转录视频文件（先提取音频，支持分块处理）
        
        Args:
            video_path: 视频文件路径
            output_audio_path: 输出音频路径（可选）
            language: 语言（可选）
            progress_callback: 进度回调函数
            
        Returns:
            ASRData 对象包含所有转录片段
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 确定音频输出路径
        if output_audio_path is None:
            output_audio_path = video_path.parent / f"{video_path.stem}.wav"
        else:
            output_audio_path = Path(output_audio_path)
        
        # 提取音频
        if progress_callback:
            progress_callback(f"正在提取音频: {video_path}")
        
        self._extract_audio(video_path, output_audio_path)
        
        # 转录音频（自动处理分块）
        asr_data = self._asr_with_chunking(output_audio_path, language, progress_callback)
        
        return asr_data
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """
        获取音频时长（秒）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            时长（秒）
        """
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _asr_with_chunking(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> ASRData:
        """
        转录音频，支持自动分块处理
        
        Args:
            audio_path: 音频文件路径
            language: 语言（可选）
            progress_callback: 进度回调
            
        Returns:
            ASRData 对象
        """
        # 检查音频时长
        duration_seconds = self._get_audio_duration(audio_path)
        
        # 检测多 GPU 环境
        from .chunked_asr import _get_available_gpu_count
        gpu_count = _get_available_gpu_count()
        use_multi_gpu = gpu_count > 1
        
        # 决定是否使用 ChunkedASR（worker 进程模式）：
        # - 多 GPU：必须走 worker 进程，避免主进程加载模型污染所有 GPU 显存
        #   （主进程 auto 选 GPU 会逐个累积模型到 class-level cache，最终占满所有 GPU）
        # - 单 GPU + 长音频（>10min）：分块并发处理
        should_use_chunked = self.enable_chunked and (use_multi_gpu or duration_seconds > 600)
        
        if should_use_chunked:
            if progress_callback:
                if use_multi_gpu:
                    progress_callback(
                        f"音频时长 {duration_seconds/60:.1f} 分钟，"
                        f"多 GPU 模式 ({gpu_count} GPUs)"
                    )
                else:
                    progress_callback(f"音频时长 {duration_seconds/60:.1f} 分钟，使用分块处理")
            
            # 创建 ChunkedASR 包装器
            chunked_asr = ChunkedASR(
                asr_class=WhisperASRAdapter,
                audio_path=str(audio_path),
                asr_kwargs={
                    'whisper_asr': self,
                    'language': language or self.language,
                },
                chunk_length=self.chunk_length_sec,
                chunk_overlap=self.chunk_overlap_sec,
                chunk_concurrency=self.chunk_concurrency,
            )
            
            # 执行分块转录
            return chunked_asr.run(lambda progress, msg: 
                progress_callback(msg) if progress_callback else None)
        else:
            # 单 GPU + 短音频：直接在主进程转录
            if progress_callback and duration_seconds > 0:
                progress_callback(f"音频时长 {duration_seconds/60:.1f} 分钟，使用普通模式")
            return self.asr_audio(audio_path, language, progress_callback)
    
    def _extract_audio(self, video_path: Path, audio_path: Path):
        """
        使用ffmpeg提取音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 输出音频路径
        """
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用ffmpeg提取音频为16kHz单声道WAV
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # 不处理视频
            '-acodec', 'pcm_s16le',  # 16位PCM编码
            '-ac', '1',  # 单声道
            '-ar', '16000',  # 16kHz采样率
            '-y',  # 覆盖输出文件
            str(audio_path)
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"音频提取失败: {e.stderr}")
    
    def save_results_as_json(self, asr_data: ASRData, output_path: Path):
        """
        保存结果为JSON格式
        
        Args:
            asr_data: ASRData 对象
            output_path: 输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为JSON格式
        data = []
        for seg in asr_data.segments:
            data.append({
                'start': seg.start_time / 1000.0,  # 转换为秒
                'end': seg.end_time / 1000.0,
                'message': seg.text
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def cleanup_audio(self, audio_path: Path):
        """
        清理临时音频文件
        
        Args:
            audio_path: 音频文件路径
        """
        if audio_path.exists():
            audio_path.unlink()


class WhisperCPPASR:
    """Whisper.cpp转录器（备用方案）"""
    
    def __init__(
        self,
        model_path: str,
        language: str = "ja",
        threads: int = 4
    ):
        """
        初始化Whisper.cpp转录器
        
        Args:
            model_path: GGML模型路径
            language: 源语言代码
            threads: 线程数
        """
        self.model_path = model_path
        self.language = language
        self.threads = threads
    
    def asr_audio(
        self,
        audio_path: Path,
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        使用whisper.cpp转录音频
        
        Args:
            audio_path: 音频文件路径
            output_path: 输出SRT路径（可选）
            
        Returns:
            转录结果列表
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}.srt"
        
        # 构建whisper.cpp命令
        cmd = [
            'whisper-cpp',
            '-m', self.model_path,
            '-l', self.language,
            '-t', str(self.threads),
            '-osrt',  # 输出SRT格式
            '-f', str(audio_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Whisper.cpp转录失败: {e.stderr}")
        
        # 解析SRT文件并转换为 ASRData
        from .subtitle_utils import parse_srt
        segments_list = parse_srt(output_path)
        
        # 转换为 ASRDataSeg
        asr_segments = [
            ASRDataSeg(
                text=seg['text'],
                start_time=int(seg['start'] * 1000),
                end_time=int(seg['end'] * 1000)
            )
            for seg in segments_list
        ]
        return ASRData(asr_segments)


class WhisperASRAdapter:
    """
    Whisper转录器适配器，用于 ChunkedASR
    实现 BaseASR 接口以供 ChunkedASR 使用
    """
    
    def __init__(self, audio_input: Union[str, bytes], whisper_asr: WhisperASR, language: str):
        """
        初始化适配器
        
        Args:
            audio_input: 音频文件路径或字节数据
            whisper_asr: WhisperASR 实例
            language: 转录语言
        """
        self.audio_input = audio_input
        self.whisper_asr = whisper_asr
        self.language = language
    
    def run(self, callback: Optional[Callable[[int, str], None]] = None) -> ASRData:
        """
        执行转录
        
        Args:
            callback: 进度回调函数(progress: int, message: str)
            
        Returns:
            ASRData 对象
        """
        # 如果是字节数据，写入临时文件
        if isinstance(self.audio_input, bytes):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(self.audio_input)
                tmp_path = tmp.name
            
            try:
                # 转录临时文件
                progress_callback = lambda msg: callback(50, msg) if callback else None
                asr_data = self.whisper_asr.asr_audio(
                    Path(tmp_path),
                    language=self.language,
                    progress_callback=progress_callback
                )
                if callback:
                    callback(100, "转录完成")
                return asr_data
            finally:
                # 清理临时文件
                Path(tmp_path).unlink(missing_ok=True)
        else:
            # 直接转录文件
            progress_callback = lambda msg: callback(50, msg) if callback else None
            asr_data = self.whisper_asr.asr_audio(
                Path(self.audio_input),
                language=self.language,
                progress_callback=progress_callback
            )
            if callback:
                callback(100, "转录完成")
            return asr_data
