"""
单视频处理执行器
"""
import os
import re
import json
import logging
import hashlib
import traceback
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime

from ..models import (
    Video, Task, TaskStep, TaskStatus, SourceType,
    STAGE_GROUPS, STAGE_DEPENDENCIES, DEFAULT_STAGE_SEQUENCE,
    expand_stage_group, get_required_stages
)
from ..database import Database
from ..config import Config
from ..downloaders import YouTubeDownloader
from ..asr import WhisperASR, ASRData, ASRDataSeg, write_srt, write_ass, split_by_llm, ASRPostProcessor
from ..asr.vocal_separation import VocalSeparator, VocalSeparationResult
from ..translator import LLMTranslator
from ..embedder import FFmpegWrapper
from ..utils.cache_metadata import CacheMetadata, WHISPER_KEY_CONFIGS, SPLIT_KEY_CONFIGS
from ..utils.cache import disable_cache, enable_cache
from ..utils.logger import setup_logger, set_video_id
from .exceptions import PipelineError, ASRError, TranslateError, EmbedError, DownloadError, UploadError
from .progress import ProgressTracker, ProgressEvent

class VideoProcessor:
    """单个视频的完整处理流程"""
    
    def __init__(
        self,
        video_id: str,
        config: Config,
        gpu_id: Optional[int] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
        video_index: int = 0,
        total_videos: int = 1
    ):
        """
        初始化视频处理器
        
        Args:
            video_id: 视频ID
            config: 配置对象
            gpu_id: GPU编号（用于并发处理）
            force: 是否强制重新处理（忽略所有缓存）
            progress_callback: 进度回调函数
            video_index: 当前视频在批次中的索引（0-based）
            total_videos: 批次总视频数
        """
        self.video_id = video_id
        self.config = config
        self.gpu_id = gpu_id
        self.force = force
        self.progress_callback = progress_callback or self._default_progress_callback
        self.video_index = video_index
        self.total_videos = total_videos
        
        # 初始化日志
        self.logger = setup_logger("pipeline.executor")
        
        # 初始化数据库
        self.db = Database(config.storage.database_path)
        
        # 初始化各个模块（延迟初始化以节省资源）
        self._downloader = None
        self._asr = None
        self._ffmpeg = None
        
        # 获取视频信息
        self.video = self.db.get_video(video_id)
        if not self.video:
            raise ValueError(f"视频不存在: {video_id}")
        
        # 设置输出目录
        self.output_dir = Path(self.video.output_dir) if self.video.output_dir else self._get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 更新数据库中的 output_dir（如果尚未设置）
        if not self.video.output_dir:
            self.db.update_video(video_id, output_dir=str(self.output_dir))
        
        # 设置GPU环境变量
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
        # 初始化缓存
        from vat.utils.cache import init_caches
        init_caches(config.storage.cache_dir)
        
        # 进度追踪器（在 process 方法中初始化）
        self._progress_tracker: Optional[ProgressTracker] = None
    
    def _default_progress_callback(self, message: str):
        """默认进度回调"""
        self.logger.info(message)
    
    def _progress_with_tracker(self, message: str, component_name: str = None):
        """带进度追踪的回调
        
        Args:
            message: 进度消息
            component_name: 组件名称，指定后日志来源显示为该组件而非 pipeline.executor
        """
        if self._progress_tracker:
            per_video_progress = self._progress_tracker.get_overall_progress()
            # 计算批次总进度：(已完成视频数 + 当前视频进度) / 总视频数
            total_progress = (self.video_index + per_video_progress) / self.total_videos
            # 格式化：[TOTAL:33%] [50%] message（TOTAL 为批次总进度，后者为单视频进度）
            prefix = f"[TOTAL:{total_progress:.0%}] [{per_video_progress:.0%}]"
            full_msg = f"{prefix} {message}" if message else prefix
        else:
            full_msg = message if message else None
        
        if full_msg:
            if component_name and (self.progress_callback is self._default_progress_callback):
                # 默认回调：用组件 logger 替代 pipeline.executor logger
                setup_logger(component_name).info(full_msg)
            else:
                self.progress_callback(full_msg)
    
    def _make_component_progress_callback(self, component_name: str) -> Callable[[str], None]:
        """
        创建组件专用进度回调，使日志来源显示为组件名而非 pipeline.executor
        
        Args:
            component_name: 组件名称（对应 setup_logger 的 name 参数，如 "subtitle_translator"）
        """
        component_logger = setup_logger(component_name)
        original_cb = self.progress_callback
        is_default = (original_cb is self._default_progress_callback)
        
        def callback(message: str):
            if is_default:
                # 默认回调：用组件 logger 替代 pipeline.executor logger
                component_logger.info(message)
            else:
                # 外部回调（如 web SSE）：直接透传
                original_cb(message)
        
        return callback
    
    def _get_output_dir(self) -> Path:
        """获取输出目录"""
        return Path(self.config.storage.output_dir) / self.video_id
    
    @property
    def downloader(self) -> YouTubeDownloader:
        """延迟初始化下载器"""
        if self._downloader is None:
            self._downloader = YouTubeDownloader(
                proxy=self.config.proxy.get_proxy(),
                video_format=self.config.downloader.youtube.format
            )
        return self._downloader
    
    @property
    def asr(self) -> WhisperASR:
        """延迟初始化转录器"""
        if self._asr is None:
            # 严格按照配置初始化，无兼容逻辑
            self._asr = WhisperASR(
                model_name=self.config.asr.model,
                device=self.config.asr.device,
                compute_type=self.config.asr.compute_type,
                language=self.config.asr.language,
                vad_filter=self.config.asr.vad_filter,
                beam_size=self.config.asr.beam_size,
                download_root=self.config.get_whisper_models_dir(),
                # 高级参数
                word_timestamps=self.config.asr.word_timestamps,
                condition_on_previous_text=self.config.asr.condition_on_previous_text,
                temperature=self.config.asr.temperature,
                compression_ratio_threshold=self.config.asr.compression_ratio_threshold,
                log_prob_threshold=self.config.asr.log_prob_threshold,
                no_speech_threshold=self.config.asr.no_speech_threshold,
                initial_prompt=self.config.asr.initial_prompt,
                repetition_penalty=self.config.asr.repetition_penalty,
                hallucination_silence_threshold=self.config.asr.hallucination_silence_threshold,
                # VAD参数
                vad_threshold=self.config.asr.vad_threshold,
                vad_min_speech_duration_ms=self.config.asr.vad_min_speech_duration_ms,
                vad_max_speech_duration_s=self.config.asr.vad_max_speech_duration_s,
                vad_min_silence_duration_ms=self.config.asr.vad_min_silence_duration_ms,
                vad_speech_pad_ms=self.config.asr.vad_speech_pad_ms,
                # ChunkedASR参数
                enable_chunked=self.config.asr.enable_chunked,
                chunk_length_sec=self.config.asr.chunk_length_sec,
                chunk_overlap_sec=self.config.asr.chunk_overlap_sec,
                chunk_concurrency=self.config.asr.chunk_concurrency,
                # Pipeline专属参数
                use_pipeline=self.config.asr.use_pipeline,
                enable_diarization=self.config.asr.enable_diarization,
                enable_punctuation=self.config.asr.enable_punctuation,
                pipeline_batch_size=self.config.asr.pipeline_batch_size,
                pipeline_chunk_length=self.config.asr.pipeline_chunk_length,
                num_speakers=self.config.asr.num_speakers,
                min_speakers=self.config.asr.min_speakers,
                max_speakers=self.config.asr.max_speakers,
            )
        return self._asr
    
    @property
    def ffmpeg(self) -> FFmpegWrapper:
        """延迟初始化FFmpeg"""
        if self._ffmpeg is None:
            self._ffmpeg = FFmpegWrapper()
        return self._ffmpeg
    
    def _resolve_stage_gaps(self, user_steps: List[str]) -> List[str]:
        """
        填充不连续阶段之间的直通阶段，并设置对应的 config 开关
        
        当用户选择不连续的阶段（如 whisper + embed）时，中间被跳过的阶段
        需要以"直通模式"执行。通过修改 config 中的开关，让各阶段内部
        自行处理直通逻辑（如复制文件），而非在 pipeline 层面手动处理。
        
        Args:
            user_steps: 用户指定的阶段列表
            
        Returns:
            填充后的完整阶段列表
        """
        if not user_steps:
            return []
        
        # 获取阶段在默认序列中的索引
        stage_order = {s.value: i for i, s in enumerate(DEFAULT_STAGE_SEQUENCE)}
        
        # 过滤出有效阶段并按顺序排序
        valid_steps = [s for s in user_steps if s in stage_order]
        if not valid_steps:
            return user_steps
        
        sorted_steps = sorted(valid_steps, key=lambda s: stage_order[s])
        
        # 找到用户指定阶段的范围
        first_idx = stage_order[sorted_steps[0]]
        last_idx = stage_order[sorted_steps[-1]]
        
        # 构建完整的阶段列表（包含中间需要直通的阶段）
        full_steps = []
        passthrough_steps = set()
        
        for i in range(first_idx, last_idx + 1):
            stage_name = DEFAULT_STAGE_SEQUENCE[i].value
            full_steps.append(stage_name)
            if stage_name not in user_steps:
                passthrough_steps.add(stage_name)
        
        # 记录直通阶段以便后续处理
        self._passthrough_stages = passthrough_steps
        
        if passthrough_steps:
            self.progress_callback(f"检测到不连续阶段，以下阶段将以直通模式执行: {', '.join(sorted(passthrough_steps, key=lambda s: stage_order[s]))}")
            # 设置对应的 config 开关，让各阶段内部自行处理直通
            self._set_passthrough_config(passthrough_steps)
        
        return full_steps
    
    def _set_passthrough_config(self, passthrough_steps: set):
        """
        设置直通阶段对应的 config 开关
        
        通过修改 config 中的开关，让各阶段内部自行处理直通逻辑。
        各阶段在检测到开关禁用时，会自动将输入复制到输出。
        
        注意：修改前先保存原始值到 _config_backup，process() 结束时恢复。
        
        Args:
            passthrough_steps: 需要直通的阶段集合
        """
        # 保存原始值以便恢复（防止多视频场景下 config 污染）
        self._config_backup = {
            'split_enable': self.config.asr.split.enable,
            'optimize_enable': self.config.translator.llm.optimize.enable,
            'skip_translate': self.config.translator.skip_translate,
        }
        
        for step in passthrough_steps:
            if step == "split":
                self.config.asr.split.enable = False
                self.progress_callback(f"  设置 asr.split.enable = False")
            elif step == "optimize":
                self.config.translator.llm.optimize.enable = False
                self.progress_callback(f"  设置 translator.llm.optimize.enable = False")
            elif step == "translate":
                self.config.translator.skip_translate = True
                self.progress_callback(f"  设置 translator.skip_translate = True")
    
    def _restore_passthrough_config(self):
        """恢复被 passthrough 修改的 config 值，防止影响后续视频"""
        if hasattr(self, '_config_backup') and self._config_backup:
            self.config.asr.split.enable = self._config_backup['split_enable']
            self.config.translator.llm.optimize.enable = self._config_backup['optimize_enable']
            self.config.translator.skip_translate = self._config_backup['skip_translate']
            self._config_backup = None
    
    def process(self, steps: Optional[List[str]] = None) -> bool:
        """
        执行处理流程
        
        Args:
            steps: 要执行的步骤列表，None表示执行所有未完成的步骤
            
        Returns:
            是否全部成功
            
        阶段跳跃处理:
            - 连续阶段（如 2,3,4）：缺少前置输出则报错
            - 不连续阶段（如 whisper + embed）：中间阶段自动以直通模式执行
            
        Note:
            force 参数统一在构造函数中设置，process() 不再接受。
        """
        # 设置当前上下文的 video_id
        set_video_id(self.video_id)
        
        # 检查视频是否为不可用（如会员限定），若是则直接标记所有阶段完成并跳过
        video_metadata = self.video.metadata or {}
        if video_metadata.get('unavailable', False):
            self.progress_callback(f"视频不可用（会员限定等），跳过处理，标记所有阶段为完成")
            for step in DEFAULT_STAGE_SEQUENCE:
                if not self.db.is_step_completed(self.video_id, step):
                    self.db.update_task_status(self.video_id, step, TaskStatus.COMPLETED)
            return True
        
        # 初始化直通阶段集合
        self._passthrough_stages = set()
        self._config_backup = None
        # 确定要执行的步骤
        if steps is None:
            steps = [step.value for step in self.db.get_pending_steps(self.video_id)]
        else:
            steps = [s if isinstance(s, str) else s.value for s in steps]
        
        # 展开阶段组名（如 'asr' → ['whisper', 'split'], 'translate' → ['optimize', 'translate']）
        expanded = []
        for s in steps:
            try:
                group = expand_stage_group(s)
                expanded.extend([step.value for step in group])
            except ValueError:
                expanded.append(s)  # 未知名称保留，后续 TaskStep() 会报错
        # 去重保序
        seen = set()
        steps = []
        for s in expanded:
            if s not in seen:
                seen.add(s)
                steps.append(s)
        
        if not steps:
            self.progress_callback("所有步骤已完成")
            return True
        
        # 填充不连续阶段之间的直通阶段
        original_steps = steps.copy()
        steps = self._resolve_stage_gaps(steps)
        
        self.progress_callback(f"待执行步骤: {', '.join(steps)}")
        
        # 初始化进度追踪器
        self._progress_tracker = ProgressTracker(stages=steps)
        
        # 执行每个步骤（try/finally 确保 config 恢复，即使发生未预期的异常）
        all_success = True
        try:
            for step_name in steps:
                try:
                    step = TaskStep(step_name)
                    is_passthrough = step_name in self._passthrough_stages
                    
                    # 检查是否已完成（force=True 时跳过检查，直通阶段总是执行）
                    if not self.force and not is_passthrough and self.db.is_step_completed(self.video_id, step):
                        self.progress_callback(f"跳过已完成步骤: {step.value}")
                        # 标记为已完成（用于进度计算）
                        self._progress_tracker.complete_stage(step.value)
                        continue
                    
                    # 开始阶段追踪
                    self._progress_tracker.start_stage(step.value)
                    
                    # 日志提示
                    if is_passthrough:
                        self._progress_with_tracker(f"直通模式执行步骤: {step.value}")
                    elif self.force and self.db.is_step_completed(self.video_id, step):
                        self._progress_with_tracker(f"强制重新执行步骤: {step.value}")
                    else:
                        self._progress_with_tracker(f"开始执行步骤: {step.value}")
                    
                    self.db.update_task_status(
                        self.video_id,
                        step,
                        TaskStatus.RUNNING,
                        gpu_id=self.gpu_id
                    )
                    
                    # 正常执行阶段（直通阶段已通过 config 开关控制，内部会自行处理）
                    success = self._execute_step(step)
                    
                    if success:
                        # 直通阶段标记为 SKIPPED，正常阶段标记为 COMPLETED
                        if is_passthrough:
                            self.db.update_task_status(
                                self.video_id,
                                step,
                                TaskStatus.SKIPPED
                            )
                            self._progress_tracker.complete_stage(step.value)
                            self._progress_with_tracker(f"步骤跳过（直通）: {step.value}")
                        else:
                            self.db.update_task_status(
                                self.video_id,
                                step,
                                TaskStatus.COMPLETED
                            )
                            self._progress_tracker.complete_stage(step.value)
                            self._progress_with_tracker(f"步骤完成: {step.value}")
                    else:
                        self.db.update_task_status(
                            self.video_id,
                            step,
                            TaskStatus.FAILED,
                            error_message="执行失败"
                        )
                        self.progress_callback(f"步骤失败: {step.value}")
                        all_success = False
                        break  # 失败后停止
                
                except PipelineError as e:
                    # 捕获 Pipeline 异常（子阶段已独立化，不再需要 sub_phase）
                    error_msg = f"{type(e).__name__}: {e.message}"
                    self.progress_callback(f"步骤异常: {step_name} - {error_msg}")
                    self.db.update_task_status(
                        self.video_id,
                        TaskStep(step_name),
                        TaskStatus.FAILED,
                        error_message=error_msg
                    )
                    all_success = False
                    self.logger.debug(traceback.format_exc())
                    break
                        
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    self.progress_callback(f"步骤异常: {step_name} - {error_msg}")
                    self.db.update_task_status(
                        self.video_id,
                        TaskStep(step_name),
                        TaskStatus.FAILED,
                        error_message=error_msg
                    )
                    all_success = False
                    self.logger.debug(traceback.format_exc())
                    break
        finally:
            # 恢复被 passthrough 修改的 config（防止影响后续视频）
            self._restore_passthrough_config()
        
        return all_success
    
    def _is_no_speech(self) -> bool:
        """检查视频是否被标记为无人声（PV/纯音乐视频），供下游阶段跳过"""
        video = self.db.get_video(self.video_id)
        return bool((video.metadata or {}).get('no_speech', False))
    
    def _execute_step(self, step: TaskStep) -> bool:
        """
        执行单个步骤（细粒度阶段）
        
        Args:
            step: 要执行的步骤（细粒度阶段）
            
        Note:
            force 通过 self.force 统一访问，不再作为参数传递。
        """
        # 细粒度阶段处理器映射
        handlers = {
            TaskStep.DOWNLOAD: lambda: self._run_download(),
            TaskStep.WHISPER: lambda: self._run_whisper(),
            TaskStep.SPLIT: lambda: self._run_split(),
            TaskStep.OPTIMIZE: lambda: self._run_optimize(),
            TaskStep.TRANSLATE: lambda: self._run_translate(),
            TaskStep.EMBED: lambda: self._run_embed(),
            TaskStep.UPLOAD: lambda: self._run_upload(),
        }
        
        handler = handlers.get(step)
        if handler is None:
            raise ValueError(f"未知步骤: {step}")
        
        return handler()
    
    def _run_download(self) -> bool:
        """下载视频"""
        if self.video.source_type == SourceType.LOCAL:
            self.progress_callback("本地文件，跳过下载")
            if not Path(self.video.source_url).exists():
                raise DownloadError(f"本地视频文件不存在: {self.video.source_url}")
            return True
        
        if self.video.source_type == SourceType.YOUTUBE:
            self._progress_with_tracker(f"开始下载YouTube视频: {self.video.source_url}")
            
            # 从配置获取字幕下载设置
            yt_config = self.config.downloader.youtube
            download_subs = yt_config.download_subtitles
            sub_langs = yt_config.subtitle_languages
            
            try:
                result = self.downloader.download(
                    self.video.source_url,
                    self.output_dir,
                    download_subs=download_subs,
                    sub_langs=sub_langs
                )
                
                if 'video_path' not in result:
                    raise DownloadError("下载器未返回视频路径")
                    
                video_path = Path(result['video_path'])
                if not video_path.exists():
                    raise DownloadError(f"下载的文件不存在: {video_path}")
                
                # 报告进度：视频下载完成 (60%)
                if self._progress_tracker:
                    self._progress_tracker.report_event(ProgressEvent.DOWNLOAD_VIDEO_DONE, "视频下载完成")
                
                # 更新视频信息（merge，保留 playlist sync 阶段写入的字段如 thumbnail）
                existing_metadata = self.video.metadata or {}
                metadata = {**existing_metadata, **result.get('metadata', {})}
                title = result.get('title', '')
                
                # 处理 YouTube 字幕（如果有）
                subtitles = result.get('subtitles', {})
                available_manual = metadata.get('available_subtitles', [])
                available_auto = metadata.get('available_auto_subtitles', [])
                
                if subtitles:
                    self.progress_callback(f"已获取 YouTube 字幕: {list(subtitles.keys())}")
                    # 存储字幕路径到 metadata
                    metadata['youtube_subtitles'] = {
                        lang: str(path) for lang, path in subtitles.items()
                    }
                
                # 记录可用字幕语言
                metadata['available_subtitles'] = available_manual
                metadata['available_auto_subtitles'] = available_auto
                
                # 检测人工字幕并决定字幕来源
                target_lang = self.config.asr.language or 'ja'
                has_manual_target = target_lang in available_manual
                has_auto_target = target_lang in available_auto
                
                # 检查是否下载到了目标语言的人工字幕
                manual_sub_path = subtitles.get(target_lang)
                if manual_sub_path and Path(manual_sub_path).exists() and has_manual_target:
                    metadata['subtitle_source'] = 'manual'
                    metadata['manual_subtitle_path'] = str(manual_sub_path)
                    self.progress_callback(f"✓ 检测到人工{target_lang}字幕，将跳过ASR")
                elif has_auto_target:
                    metadata['subtitle_source'] = 'auto'
                    self.progress_callback(f"检测到自动{target_lang}字幕，将使用ASR")
                else:
                    metadata['subtitle_source'] = 'asr'
                
                # 场景识别（新增）+ 存储完整的视频信息冗余
                if title:
                    self.progress_callback("正在识别视频场景...")
                    try:
                        from vat.llm.scene_identifier import SceneIdentifier
                        
                        si_cfg = self.config.downloader.scene_identify
                        identifier = SceneIdentifier(
                            model=si_cfg.model or self.config.llm.model,
                            api_key=si_cfg.api_key,
                            base_url=si_cfg.base_url,
                        )
                        description = metadata.get('description', '')
                        scene_info = identifier.detect_scene(title, description)
                        
                        # 存入 metadata（新增：完整的视频信息冗余存储）
                        metadata['scene'] = scene_info['scene_id']
                        metadata['scene_name'] = scene_info['scene_name']
                        metadata['scene_auto_detected'] = scene_info['auto_detected']
                        
                        # 增强 metadata：存储完整的视频信息副本以便后续使用
                        # （包括：视频ID、URL、作者、简介、时长、上传时间等）
                        metadata['_video_info'] = {
                            'video_id': metadata.get('video_id', ''),
                            'url': metadata.get('url', self.video.source_url),
                            'title': title,
                            'uploader': metadata.get('uploader', ''),
                            'description': description,
                            'duration': metadata.get('duration', 0),
                            'upload_date': metadata.get('upload_date', ''),
                            'thumbnail': metadata.get('thumbnail', ''),
                        }
                        
                        self.progress_callback(
                            f"场景识别: {scene_info['scene_name']} ({scene_info['scene_id']})"
                        )
                    except Exception as e:
                        self.logger.error(f"场景识别异常: {e}")
                        # 场景识别失败不影响下载流程，设置默认场景
                        metadata['scene'] = 'chatting'
                        metadata['scene_name'] = '闲聊直播'
                        metadata['scene_auto_detected'] = False
                        # 即使识别失败，也存储视频信息冗余
                        metadata['_video_info'] = {
                            'video_id': metadata.get('video_id', ''),
                            'url': metadata.get('url', self.video.source_url),
                            'title': title,
                            'uploader': metadata.get('uploader', ''),
                            'description': metadata.get('description', ''),
                            'duration': metadata.get('duration', 0),
                            'upload_date': metadata.get('upload_date', ''),
                            'thumbnail': metadata.get('thumbnail', ''),
                        }
                    
                    # 视频信息翻译（用于后续上传到B站）
                    # 检查是否已有翻译结果（可能在 sync 阶段异步翻译完成）
                    existing_translated = self.video.metadata.get('translated') if self.video.metadata else None
                    
                    if existing_translated and not self.force:
                        self.progress_callback("复用已有的视频信息翻译结果")
                        metadata['translated'] = existing_translated
                    elif self.config.llm.is_available():
                        self.progress_callback("正在翻译视频信息...")
                        try:
                            from vat.llm.video_info_translator import VideoInfoTranslator
                            
                            vit_cfg = self.config.downloader.video_info_translate
                            translator = VideoInfoTranslator(
                                model=vit_cfg.model or self.config.llm.model,
                                api_key=vit_cfg.api_key,
                                base_url=vit_cfg.base_url,
                            )
                            description = metadata.get('description', '')
                            tags = metadata.get('tags', [])
                            uploader = metadata.get('uploader', '')
                            
                            translated_info = translator.translate(
                                title=title,
                                description=description,
                                tags=tags,
                                uploader=uploader
                            )
                            
                            # 存储翻译结果到metadata
                            metadata['translated'] = translated_info.to_dict()
                            self._progress_with_tracker(
                                f"视频信息翻译完成，推荐分区: {translated_info.recommended_tid_name}"
                            )
                        except Exception as e:
                            self.logger.warning(f"视频信息翻译失败: {e}")
                            # 翻译失败不影响下载流程
                    else:
                        self.logger.info("LLM配置不可用，跳过视频信息翻译")
                    
                    # 报告进度：翻译完成 (20%)
                    if self._progress_tracker:
                        self._progress_tracker.report_event(ProgressEvent.DOWNLOAD_TRANSLATE_DONE)
                
                self.db.update_video(
                    self.video_id,
                    title=title,
                    metadata=metadata
                )
                
                self._progress_with_tracker(f"下载完成: {result['video_path']}")
                return True
                
            except Exception as e:
                error_msg = f"下载失败: {e}"
                self.progress_callback(error_msg)
                raise DownloadError(error_msg, original_error=e)
        
        raise DownloadError(f"不支持的视频来源类型: {self.video.source_type}")
    
    # ==================== 细粒度阶段实现 ====================
    
    def _run_whisper(self) -> bool:
        """
        Whisper 语音识别阶段
        输出：original_raw.srt（原始 ASR 输出）
        
        流程：
        1. 检查是否有人工字幕，有则跳过ASR
        2. 提取音频
        3. 如需人声分离，执行分离（输出 xxx_vocals.wav）
        4. 使用音频（或分离后的人声）进行 Whisper 识别
        """
        video_file = self._find_video_file()
        assert video_file and video_file.exists(), f"视频文件不存在: {video_file}"
        
        raw_srt = self.output_dir / "original_raw.srt"
        metadata = CacheMetadata.load(self.output_dir)
        metadata.video_id = self.video_id
        
        # 检查是否有人工字幕可以直接使用
        video_metadata = self.video.metadata or {}
        subtitle_source = video_metadata.get('subtitle_source', 'asr')
        manual_sub_path = video_metadata.get('manual_subtitle_path')
        
        if subtitle_source == 'manual' and manual_sub_path and Path(manual_sub_path).exists():
            self._progress_with_tracker(f"✓ 使用人工字幕，跳过ASR: {manual_sub_path}")
            try:
                # 将VTT转换为SRT格式并保存
                asr_data = ASRData.from_subtitle_file(manual_sub_path)
                assert len(asr_data) > 0, "人工字幕为空"
                asr_data.save(str(raw_srt))
                metadata.update_substep('whisper', {'source': 'manual_subtitle'}, "original_raw.srt")
                metadata.save(self.output_dir)
                self._progress_with_tracker(f"人工字幕加载完成，共 {len(asr_data)} 个片段")
                return True
            except Exception as e:
                self.logger.warning(f"人工字幕加载失败，回退到ASR: {e}")
                # 回退到ASR
        
        self._progress_with_tracker(f"开始 Whisper 语音识别: {video_file}")
        
        try:
            audio_file = self.output_dir / f"{video_file.stem}.wav"
            
            # ====== 人声分离处理 ======
            asr_audio_file = audio_file  # 默认使用原始音频
            vocal_sep_config = self.config.asr.vocal_separation
            use_vocal_separation = self._should_use_vocal_separation(video_metadata)
            
            if use_vocal_separation:
                vocals_file = self.output_dir / f"{video_file.stem}_vocals.wav"
                
                # 检查是否已有分离结果
                if vocals_file.exists():
                    self._progress_with_tracker(f"复用已有人声分离结果: {vocals_file.name}")
                    asr_audio_file = vocals_file
                else:
                    self._progress_with_tracker("开始人声分离（去除BGM）...")
                    
                    # 确保原始音频已提取
                    if not audio_file.exists():
                        self._progress_with_tracker("提取音频...")
                        self._extract_audio(video_file, audio_file)
                    
                    # 执行人声分离
                    try:
                        separator = VocalSeparator(
                            models_dir=self.config.storage.models_dir,
                            model_filename=vocal_sep_config.model_filename,
                            device="auto",
                        )
                        
                        result = separator.separate(
                            audio_path=audio_file,
                            output_dir=self.output_dir,
                            save_accompaniment=vocal_sep_config.save_accompaniment,
                        )
                        
                        if result.success and result.vocals_path and result.vocals_path.exists():
                            asr_audio_file = result.vocals_path
                            self._progress_with_tracker(
                                f"人声分离完成 ({result.processing_time_seconds:.1f}s): {result.vocals_path.name}"
                            )
                            # 更新 metadata
                            video_metadata['vocal_separation_used'] = True
                            video_metadata['vocals_file'] = str(result.vocals_path)
                            self.db.update_video(self.video_id, metadata=video_metadata)
                        else:
                            self.logger.warning(f"人声分离失败: {result.error_message}，使用原始音频")
                            self._progress_with_tracker(f"人声分离失败，使用原始音频: {result.error_message}")
                    except Exception as e:
                        self.logger.warning(f"人声分离异常: {e}，使用原始音频")
                        self._progress_with_tracker(f"人声分离异常，使用原始音频: {e}")
            
            # ====== Whisper ASR ======
            whisper_config = self._extract_whisper_config()
            # 将人声分离状态加入缓存 key
            whisper_config['vocal_separation'] = use_vocal_separation
            
            if self._should_use_cache('whisper', whisper_config, raw_srt):
                self._progress_with_tracker("复用 Whisper 缓存")
                asr_data = ASRData.from_subtitle_file(str(raw_srt))
            else:
                self._progress_with_tracker(f"运行 Whisper 语音识别... (音频: {asr_audio_file.name})")
                
                # Whisper 进度回调（解析 chunk 完成消息以更新进度条）
                def whisper_progress(msg):
                    # 解析 chunked_asr 的 "已完成 X/Y 块" 消息
                    m = re.search(r'已完成 (\d+)/(\d+) 块', msg)
                    if m and self._progress_tracker:
                        completed, total = int(m.group(1)), int(m.group(2))
                        stage_prog = self._progress_tracker._stage_progress.get('whisper')
                        if stage_prog:
                            stage_prog.total_items = total
                            stage_prog.completed_items = completed
                    self._progress_with_tracker(msg, component_name="whisper_asr")
                
                # 确保音频文件存在
                if not asr_audio_file.exists():
                    # 需要从视频提取音频
                    self._progress_with_tracker("提取音频...")
                    self._extract_audio(video_file, audio_file)
                    asr_audio_file = audio_file
                
                # 直接使用音频文件进行 ASR（支持分块处理）
                self.asr._ensure_model_loaded()
                asr_data = self.asr._asr_with_chunking(
                    asr_audio_file, 
                    progress_callback=whisper_progress
                )
                
                # ASR 输出为空：PV/纯音乐视频可能没有人声
                if len(asr_data) == 0:
                    self.progress_callback("⚠ ASR 输出为空（视频可能无人声，如PV/音乐视频）")
                    # 标记 metadata 供后续阶段检查
                    video_metadata = self.video.metadata or {}
                    video_metadata['no_speech'] = True
                    self.db.update_video(self.video_id, metadata=video_metadata)
                    # 保存空 SRT 文件（让后续阶段能正常检测到文件存在）
                    raw_srt.write_text("", encoding="utf-8")
                    return True
                
                # ====== ASR 后处理：过滤幻觉和重复 ======
                postproc_config = self.config.asr.postprocessing
                if postproc_config.enable_hallucination_detection or postproc_config.enable_repetition_cleaning:
                    self._progress_with_tracker("执行 ASR 后处理（过滤幻觉/重复）...")
                    postprocessor = ASRPostProcessor(
                        enable_hallucination_detection=postproc_config.enable_hallucination_detection,
                        enable_repetition_cleaning=postproc_config.enable_repetition_cleaning,
                        enable_japanese_processing=postproc_config.enable_japanese_processing,
                        custom_blacklist=postproc_config.custom_blacklist,
                    )
                    
                    # 转换为 segments 格式并处理
                    segments = [{'text': seg.text, 'start': seg.start_time, 'end': seg.end_time} 
                               for seg in asr_data.segments]
                    filtered_segments, stats = postprocessor.process_segments(segments)
                    
                    # 重建 ASRData
                    if stats.hallucinations_removed > 0 or stats.repetitions_cleaned > 0:
                        self._progress_with_tracker(
                            f"后处理完成: 移除 {stats.hallucinations_removed} 个幻觉, "
                            f"清理 {stats.repetitions_cleaned} 个重复"
                        )
                        # 从过滤后的 segments 重建 ASRData
                        new_segments = [
                            ASRDataSeg(text=s['text'], start_time=s['start'], end_time=s['end'])
                            for s in filtered_segments
                        ]
                        asr_data = ASRData(segments=new_segments)
                
                # ====== 崩溃检测与静默警告 ======
                from vat.utils.output_validator import validate_asr_segments
                # ASRDataSeg 的 start_time/end_time 是毫秒，validate_asr_segments 期望秒
                segments_for_validation = [
                    {'text': seg.text, 'start': seg.start_time / 1000.0, 'end': seg.end_time / 1000.0}
                    for seg in asr_data.segments
                ]
                validated_segments, warnings = validate_asr_segments(
                    segments_for_validation, remove_catastrophic=True
                )
                if warnings:
                    for w in warnings[:5]:  # 最多显示5条警告
                        self._progress_with_tracker(f"⚠️ {w}")
                    if len(warnings) > 5:
                        self._progress_with_tracker(f"⚠️ ...还有 {len(warnings) - 5} 条警告")
                # 如果有片段被移除，重建 ASRData
                if len(validated_segments) < len(asr_data.segments):
                    removed_count = len(asr_data.segments) - len(validated_segments)
                    self._progress_with_tracker(f"移除 {removed_count} 个崩溃片段")
                    # validated_segments 中时间戳是秒，转回毫秒给 ASRDataSeg
                    new_segments = [
                        ASRDataSeg(text=s['text'], start_time=int(s['start'] * 1000), end_time=int(s['end'] * 1000))
                        for s in validated_segments
                    ]
                    asr_data = ASRData(segments=new_segments)
                
                asr_data.save(str(raw_srt))
                metadata.update_substep('whisper', whisper_config, "original_raw.srt")
            
            metadata.save(self.output_dir)
            self._progress_with_tracker(f"Whisper 完成，共 {len(asr_data)} 个片段")
            return True
            
        except AssertionError as e:
            error_msg = f"Whisper 失败: {e}"
            self.progress_callback(error_msg)
            raise ASRError(error_msg, original_error=e)
        except Exception as e:
            error_msg = f"Whisper 失败: {e}"
            self.progress_callback(error_msg)
            self.logger.debug(traceback.format_exc())
            raise ASRError(error_msg, original_error=e)
    
    def _should_use_vocal_separation(self, video_metadata: dict) -> bool:
        """
        判断是否应该使用人声分离
        
        逻辑：
        1. 如果 vocal_separation.enable = True，始终启用
        2. 如果 vocal_separation.auto_detect_bgm = True，根据场景自动判断：
           - music_live（歌回）、gaming（游戏直播）场景启用
           - 其他场景不启用
        
        Args:
            video_metadata: 视频元数据（包含 scene 字段）
            
        Returns:
            是否使用人声分离
        """
        vocal_sep_config = self.config.asr.vocal_separation
        
        # 显式启用
        if vocal_sep_config.enable:
            self.logger.info("人声分离: 配置显式启用")
            return True
        
        # 自动检测
        if vocal_sep_config.auto_detect_bgm:
            scene = video_metadata.get('scene', '')
            # 这些场景通常有背景音乐，需要人声分离
            bgm_scenes = {'music_live', 'gaming', 'karaoke', 'cover', 'original_song'}
            
            if scene in bgm_scenes:
                self.logger.info(f"人声分离: 场景 '{scene}' 自动启用")
                self._progress_with_tracker(f"检测到 {scene} 场景，自动启用人声分离")
                return True
        
        return False
    
    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 输出音频路径
        """
        import subprocess
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
            str(audio_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"音频提取失败: {result.stderr}")
    
    def _run_split(self) -> bool:
        """
        智能断句阶段
        输入：original_raw.srt
        输出：original.srt（语义完整的原文字幕）
        """
        # 无人声视频跳过断句
        if self._is_no_speech():
            self.progress_callback("无人声视频，跳过断句")
            return True
        
        raw_srt = self.output_dir / "original_raw.srt"
        if not raw_srt.exists():
            raise ASRError(f"找不到 Whisper 输出: {raw_srt}")
        
        final_srt = self.output_dir / "original.srt"
        split_srt = self.output_dir / "original_split.srt"
        
        self._progress_with_tracker("开始智能断句")
        
        try:
            asr_data = ASRData.from_subtitle_file(str(raw_srt))
            
            # ASR 后处理：移除日语文本中的无意义空格（防止断句碎片化）
            asr_data.strip_cjk_spaces()
            
            metadata = CacheMetadata.load(self.output_dir)
            
            if not self.config.asr.split.enable:
                self._progress_with_tracker("智能断句已禁用，使用原始 ASR 输出")
                asr_data.save(str(final_srt))
                return True
            
            split_config = self._extract_split_config()
            if self._should_use_cache('split', split_config, split_srt):
                self.progress_callback("复用断句缓存")
                asr_data = ASRData.from_subtitle_file(str(split_srt))
            else:
                if not self.config.llm.is_available():
                    self.logger.warning("LLM 配置不完整，跳过智能断句")
                    self.progress_callback("警告: LLM 配置不完整，使用原始 ASR 输出")
                else:
                    # 获取场景特定的断句提示词
                    split_scene_prompt = ""
                    scene_id = self.video.metadata.get('scene', '')
                    if scene_id:
                        try:
                            from vat.llm.scene_identifier import SceneIdentifier
                            identifier = SceneIdentifier()
                            scene_prompts = identifier.get_scene_prompts(scene_id)
                            split_scene_prompt = scene_prompts.get('split', '')
                            if split_scene_prompt:
                                self.progress_callback(f"断句使用场景优化: {scene_id}")
                        except Exception as e:
                            self.logger.warning(f"加载断句场景提示词失败: {e}")
                    
                    # 执行断句
                    if (self.config.asr.split.enable_chunking and 
                        len(asr_data.segments) >= self.config.asr.split.chunk_min_threshold):
                        self.progress_callback(f"启用分块断句 (共 {len(asr_data.segments)} 个片段)")
                        from vat.asr.chunked_split import ChunkedSplitter
                        
                        split_creds = self.config.get_stage_llm_credentials("split")
                        splitter = ChunkedSplitter(
                            chunk_size_sentences=self.config.asr.split.chunk_size_sentences,
                            chunk_overlap_sentences=self.config.asr.split.chunk_overlap_sentences,
                            model=split_creds["model"],
                            max_word_count_cjk=self.config.asr.split.max_words_cjk,
                            max_word_count_english=self.config.asr.split.max_words_english,
                            min_word_count_cjk=self.config.asr.split.min_words_cjk,
                            min_word_count_english=self.config.asr.split.min_words_english,
                            recommend_word_count_cjk=self.config.asr.split.recommend_words_cjk,
                            recommend_word_count_english=self.config.asr.split.recommend_words_english,
                            scene_prompt=split_scene_prompt,
                            mode=self.config.asr.split.mode,
                            allow_model_upgrade=self.config.asr.split.allow_model_upgrade,
                            model_upgrade_chain=self.config.asr.split.model_upgrade_chain,
                            api_key=split_creds["api_key"],
                            base_url=split_creds["base_url"],
                        )
                        asr_data = splitter.split(asr_data, progress_callback=self._make_component_progress_callback("chunked_split"))
                    else:
                        if len(asr_data.segments) < self.config.asr.split.chunk_min_threshold:
                            self.progress_callback("片段数较少，使用全文断句")
                        asr_data = self._split_with_speaker_awareness(asr_data, split_scene_prompt)
                    
                    asr_data.optimize_timing()
                    asr_data.save(str(split_srt))
                    metadata.update_substep('split', split_config, "original_split.srt")
                    self.progress_callback(f"断句完成，共 {len(asr_data)} 句")
            
            # 保存最终输出
            asr_data.save(str(final_srt))
            original_json = self.output_dir / "original.json"
            self.asr.save_results_as_json(asr_data, original_json)
            metadata.save(self.output_dir)
            
            return True
            
        except Exception as e:
            error_msg = f"智能断句失败: {e}"
            self.progress_callback(error_msg)
            self.logger.debug(traceback.format_exc())
            raise ASRError(error_msg, original_error=e)
    
    def _get_scene_prompt(self, prompt_key: str, base_prompt: str = "") -> str:
        """
        获取场景提示词并与用户自定义 prompt 合并
        
        Args:
            prompt_key: 场景提示词类型（'optimize', 'translate', 'split'）
            base_prompt: 用户自定义的基础 prompt
            
        Returns:
            合并后的 prompt 字符串
        """
        scene_id = self.video.metadata.get('scene', '')
        if not scene_id:
            return base_prompt
        
        try:
            from vat.llm.scene_identifier import SceneIdentifier
            identifier = SceneIdentifier()
            scene_prompts = identifier.get_scene_prompts(scene_id)
            scene_prompt = scene_prompts.get(prompt_key, '')
            if scene_prompt:
                self.progress_callback(f"使用场景{prompt_key}提示词: {scene_id}")
                return f"{scene_prompt}\n\n{base_prompt}" if base_prompt else scene_prompt
        except Exception:
            pass
        
        return base_prompt
    
    def _create_translator(
        self,
        custom_translate_prompt: str = "",
        is_reflect: bool = False,
        enable_optimize: bool = False,
        custom_optimize_prompt: str = "",
        model_override: str = "",
        thread_num_override: int = 0,
        batch_num_override: int = 0,
        api_key_override: str = "",
        base_url_override: str = "",
        optimize_model_override: str = "",
        optimize_api_key_override: str = "",
        optimize_base_url_override: str = "",
    ) -> 'LLMTranslator':
        """
        创建 LLMTranslator 实例（工厂方法）
        
        公共参数从 self.config 读取，差异参数通过参数传入。
        *_override 参数优先于 config 中的值，留空/0 则使用 config 默认值。
        """
        from vat.translator.types import str_to_target_language
        target_lang = str_to_target_language(self.config.translator.target_language)
        
        # resolve translate 凭据
        trans_creds = self.config.get_stage_llm_credentials("translate")
        
        return LLMTranslator(
            thread_num=thread_num_override or self.config.translator.llm.thread_num,
            batch_num=batch_num_override or self.config.translator.llm.batch_size,
            target_language=target_lang,
            output_dir=str(self.output_dir),
            model=model_override or trans_creds["model"],
            custom_translate_prompt=custom_translate_prompt,
            is_reflect=is_reflect,
            enable_optimize=enable_optimize,
            custom_optimize_prompt=custom_optimize_prompt,
            enable_context=self.config.translator.llm.enable_context,
            api_key=api_key_override or trans_creds["api_key"],
            base_url=base_url_override or trans_creds["base_url"],
            optimize_model=optimize_model_override,
            optimize_api_key=optimize_api_key_override,
            optimize_base_url=optimize_base_url_override,
            progress_callback=self._make_component_progress_callback("subtitle_translator"),
        )
    
    def _run_optimize(self) -> bool:
        """
        字幕优化阶段
        输入：original.srt
        输出：optimized.srt
        """
        # 无人声视频跳过优化
        if self._is_no_speech():
            self.progress_callback("无人声视频，跳过字幕优化")
            return True
        
        original_srt = self.output_dir / "original.srt"
        if not original_srt.exists():
            raise TranslateError(f"找不到原文字幕: {original_srt}")
        
        optimized_srt = self.output_dir / "optimized.srt"
        
        # 检查是否启用优化
        if not self.config.translator.llm.optimize.enable:
            self.progress_callback("字幕优化已禁用，跳过")
            # 直接复制原文作为优化结果
            import shutil
            shutil.copy(original_srt, optimized_srt)
            return True
        
        self.progress_callback("开始字幕优化")
        
        if self.force:
            disable_cache()
            self.progress_callback("强制模式：已禁用优化缓存")
        
        try:
            asr_data = ASRData.from_subtitle_file(str(original_srt))
            
            optimize_prompt = self._get_scene_prompt(
                'optimize', self.config.translator.llm.optimize.custom_prompt or ""
            )
            
            # resolve optimize 阶段的有效配置（含继承链）
            opt_config = self.config.get_optimize_effective_config()
            
            translator = self._create_translator(
                custom_translate_prompt=self.config.translator.llm.custom_prompt or "",
                is_reflect=False,  # 优化阶段不需要反思翻译
                enable_optimize=True,
                custom_optimize_prompt=optimize_prompt,
                thread_num_override=opt_config["thread_num"],
                batch_num_override=opt_config["batch_size"],
                optimize_model_override=opt_config["model"],
                optimize_api_key_override=opt_config["api_key"],
                optimize_base_url_override=opt_config["base_url"],
            )
            
            # 调用内部优化方法（OPTIMIZE 阶段独立执行）
            optimized_data = translator._optimize_subtitle(asr_data)
            
            optimized_data.save(str(optimized_srt))
            self.progress_callback(f"字幕优化完成，共 {len(optimized_data)} 条")
            
            if self.force:
                enable_cache()
            
            return True
            
        except Exception as e:
            if self.force:
                enable_cache()
            error_msg = f"字幕优化失败: {e}"
            self.progress_callback(error_msg)
            self.logger.debug(traceback.format_exc())
            raise TranslateError(error_msg, original_error=e)
    
    def _run_translate(self) -> bool:
        """
        LLM 翻译阶段
        输入：optimized.srt（或 original.srt 如果优化被跳过）
        输出：translated.srt
        """
        # 无人声视频跳过翻译
        if self._is_no_speech():
            self.progress_callback("无人声视频，跳过翻译")
            return True
        # 优先使用优化后的字幕
        optimized_srt = self.output_dir / "optimized.srt"
        original_srt = self.output_dir / "original.srt"
        
        if optimized_srt.exists():
            input_srt = optimized_srt
        elif original_srt.exists():
            input_srt = original_srt
        else:
            raise TranslateError("找不到输入字幕文件")
        
        # 跳过翻译模式
        if self.config.translator.skip_translate:
            self.progress_callback("跳过翻译：直接使用原文（skip_translate=true）")
            translated_srt = self.output_dir / "translated.srt"
            import shutil
            shutil.copy(input_srt, translated_srt)
            return True
        
        self._progress_with_tracker("开始 LLM 翻译")
        
        if self.force:
            disable_cache()
            self._progress_with_tracker("强制模式：已禁用翻译缓存")
        
        try:
            asr_data = ASRData.from_subtitle_file(str(input_srt))
            
            translate_prompt = self._get_scene_prompt(
                'translate', self.config.translator.llm.custom_prompt or ""
            )
            
            translator = self._create_translator(
                custom_translate_prompt=translate_prompt,
                is_reflect=self.config.translator.llm.enable_reflect,
            )
            
            # 说话人感知翻译：有说话人信息时按说话人分组独立翻译
            has_speakers = any(seg.speaker_id is not None for seg in asr_data.segments)
            if has_speakers:
                speaker_groups = self._group_segments_by_speaker(asr_data)
                all_segments = []
                for speaker_id, segments in speaker_groups.items():
                    self.progress_callback(f"翻译 {speaker_id} 的字幕...")
                    speaker_asr = ASRData(segments)
                    translated_speaker = translator.translate_subtitle(speaker_asr)
                    for seg in translated_speaker.segments:
                        seg.speaker_id = speaker_id
                    all_segments.extend(translated_speaker.segments)
                all_segments.sort(key=lambda x: x.start_time)
                translated_data = ASRData(all_segments)
            else:
                translated_data = translator.translate_subtitle(asr_data)
            
            # 保存翻译结果
            translated_srt = self.output_dir / "translated.srt"
            translated_data.save(str(translated_srt))
            
            # 同时生成 ASS 格式
            from vat.asr.subtitle import get_subtitle_style
            style_str = get_subtitle_style(
                self.config.embedder.subtitle_style,
                style_dir=self.config.storage.subtitle_style_dir
            )
            translated_ass = self.output_dir / "translated.ass"
            translated_data.save(str(translated_ass), ass_style=style_str)
            
            self.progress_callback(f"翻译完成，共 {len(translated_data)} 条")
            
            if self.force:
                enable_cache()
            
            return True
            
        except Exception as e:
            if self.force:
                enable_cache()
            error_msg = f"翻译失败: {e}"
            self.progress_callback(error_msg)
            self.logger.debug(traceback.format_exc())
            raise TranslateError(error_msg, original_error=e)
    
    def _realign_timestamps(self, original_asr: ASRData, split_texts: List[str]) -> ASRData:
        """
        重新分配时间戳（基于字符串匹配的精确定位）
        
        核心逻辑：
        1. 构建原始文本的字符级时间戳映射
        2. 使用字符串匹配（而非简单字符计数）定位断句文本在原始文本中的位置
        3. 根据匹配位置分配精确的起止时间
        
        修复：LLM 断句可能改变空格/标点，导致字符数不匹配。
        使用 find() 匹配实际位置，避免时间戳错位。
        
        Args:
            original_asr: 原始 ASR 数据（词级或短句级）
            split_texts: 断句后的文本列表
            
        Returns:
            重新分配时间戳的 ASRData
        """
        if not split_texts:
            return original_asr
        
        # 构建原始文本（保留所有字符，包括空格）
        original_full_text = ""
        char_times = []  # [(char_start_time, char_end_time), ...]
        
        for seg in original_asr.segments:
            text = seg.text.strip()
            if not text:
                continue
            duration = seg.end_time - seg.start_time
            text_len = len(text)
            for i, char in enumerate(text):
                char_start = seg.start_time + duration * (i / text_len)
                char_end = seg.start_time + duration * ((i + 1) / text_len)
                original_full_text += char
                char_times.append((char_start, char_end))
        
        if not char_times:
            return original_asr
        
        # 为断句后的每个文本分配时间戳
        new_segments = []
        search_start = 0  # 在原始文本中的搜索起点
        
        for split_text in split_texts:
            text = split_text.strip()
            if not text:
                continue
            
            # 在原始文本中查找该断句文本的位置
            # 优先从上次匹配结束位置开始搜索（保证顺序性）
            pos = original_full_text.find(text, search_start)
            
            if pos == -1:
                # 未找到精确匹配，尝试移除空格后匹配
                text_no_space = text.replace(" ", "").replace("　", "")
                original_no_space = original_full_text[search_start:].replace(" ", "").replace("　", "")
                pos_no_space = original_no_space.find(text_no_space)
                
                if pos_no_space != -1:
                    # 找到了无空格匹配，需要映射回原始位置
                    actual_pos = search_start
                    no_space_count = 0
                    while no_space_count < pos_no_space and actual_pos < len(original_full_text):
                        if original_full_text[actual_pos] not in " 　":
                            no_space_count += 1
                        actual_pos += 1
                    pos = actual_pos
                else:
                    # 仍未找到，使用上次结束位置继续
                    self.logger.warning(f"断句文本未在原始文本中找到精确匹配: '{text[:20]}...'")
                    pos = search_start
            
            # 计算该文本在原始文本中覆盖的字符范围
            text_no_space = text.replace(" ", "").replace("　", "")
            end_pos = pos
            matched_chars = 0
            while matched_chars < len(text_no_space) and end_pos < len(original_full_text):
                if original_full_text[end_pos] not in " 　":
                    matched_chars += 1
                end_pos += 1
            
            # 分配时间戳
            if pos < len(char_times):
                start_time = char_times[pos][0]
            else:
                start_time = char_times[-1][0]
            
            if end_pos > 0 and end_pos <= len(char_times):
                end_time = char_times[end_pos - 1][1]
            elif end_pos > len(char_times):
                end_time = char_times[-1][1]
            else:
                end_time = start_time + 100  # fallback: 100ms
            
            new_segments.append(ASRDataSeg(
                text=text,
                start_time=int(start_time),
                end_time=int(end_time)
            ))
            
            # 更新搜索起点，避免重复匹配
            search_start = end_pos
        
        return ASRData(new_segments)

    def _split_with_speaker_awareness(self, asr_data: ASRData, scene_prompt: str = "") -> ASRData:
        """按说话人分组进行智能断句
        
        Args:
            asr_data: ASR 数据
            scene_prompt: 场景特定提示词（可选）
        """
        # 检查是否有说话人信息
        has_speakers = any(seg.speaker_id is not None for seg in asr_data.segments)
        
        split_creds = self.config.get_stage_llm_credentials("split")
        
        if not has_speakers:
            # 无说话人信息，使用原有逻辑
            full_text = "".join(seg.text for seg in asr_data.segments)
            split_texts = split_by_llm(
                full_text,
                model=split_creds["model"],
                max_word_count_cjk=self.config.asr.split.max_words_cjk,
                max_word_count_english=self.config.asr.split.max_words_english,
                min_word_count_cjk=self.config.asr.split.min_words_cjk,
                min_word_count_english=self.config.asr.split.min_words_english,
                recommend_word_count_cjk=self.config.asr.split.recommend_words_cjk,
                recommend_word_count_english=self.config.asr.split.recommend_words_english,
                scene_prompt=scene_prompt,
                mode=self.config.asr.split.mode,
                allow_model_upgrade=self.config.asr.split.allow_model_upgrade,
                model_upgrade_chain=self.config.asr.split.model_upgrade_chain,
                api_key=split_creds["api_key"],
                base_url=split_creds["base_url"],
            )
            return self._realign_timestamps(asr_data, split_texts)
        
        # 有说话人信息，按说话人分组处理
        speaker_groups = self._group_segments_by_speaker(asr_data)
        all_segments = []
        
        for speaker_id, segments in speaker_groups.items():
            # 合并该说话人的所有文本
            speaker_text = "".join(seg.text for seg in segments)
            
            # 对该说话人独立断句
            split_texts = split_by_llm(
                speaker_text,
                model=split_creds["model"],
                max_word_count_cjk=self.config.asr.split.max_words_cjk,
                max_word_count_english=self.config.asr.split.max_words_english,
                min_word_count_cjk=self.config.asr.split.min_words_cjk,
                min_word_count_english=self.config.asr.split.min_words_english,
                recommend_word_count_cjk=self.config.asr.split.recommend_words_cjk,
                recommend_word_count_english=self.config.asr.split.recommend_words_english,
                scene_prompt=scene_prompt,
                mode=self.config.asr.split.mode,
                allow_model_upgrade=self.config.asr.split.allow_model_upgrade,
                model_upgrade_chain=self.config.asr.split.model_upgrade_chain,
                api_key=split_creds["api_key"],
                base_url=split_creds["base_url"],
            )
            
            # 为该说话人的片段重新分配时间戳
            speaker_asr = ASRData(segments)
            speaker_asr_split = self._realign_timestamps(speaker_asr, split_texts)
            
            # 保留speaker_id信息
            for seg in speaker_asr_split.segments:
                seg.speaker_id = speaker_id
            
            all_segments.extend(speaker_asr_split.segments)
        
        # 按时间顺序重新排序
        all_segments.sort(key=lambda x: x.start_time)
        return ASRData(all_segments)

    def _group_segments_by_speaker(self, asr_data: ASRData) -> Dict[str, List[ASRDataSeg]]:
        """按说话人ID分组segments，保持时间顺序"""
        from collections import defaultdict
        groups = defaultdict(list)
        
        for seg in asr_data.segments:
            speaker_id = seg.speaker_id or "SPEAKER_UNKNOWN"
            groups[speaker_id].append(seg)
        
        return dict(groups)

    def _run_embed(self) -> bool:
        """嵌入字幕"""
        # 无人声视频跳过嵌入
        if self._is_no_speech():
            self.progress_callback("无人声视频，跳过字幕嵌入")
            # 直接复制原视频作为 final.mp4
            video_file = self._find_video_file()
            if video_file:
                import shutil
                final_video = self.output_dir / "final.mp4"
                if not final_video.exists():
                    shutil.copy2(video_file, final_video)
                    self.progress_callback(f"已复制原视频为 {final_video.name}")
            return True
        
        # 查找视频和字幕文件
        video_file = self._find_video_file()
        if not video_file:
            raise EmbedError("找不到视频文件")
        
        # 优先从 translated.srt 生成/缓存 translated.ass（用于统一样式）
        subtitle_srt = self.output_dir / "translated.srt"
        subtitle_ass = self.output_dir / "translated.ass"
        
        if subtitle_srt.exists():
            need_regenerate_ass = (
                self.force
                or (not subtitle_ass.exists())
                or subtitle_ass.stat().st_mtime < subtitle_srt.stat().st_mtime
            )
            if need_regenerate_ass:
                self.progress_callback("根据 translated.srt 生成 ASS 字幕...")
                from vat.asr import ASRData
                from vat.asr.subtitle import get_subtitle_style
                asr_data = ASRData.from_subtitle_file(str(subtitle_srt))
                assert len(asr_data) > 0, "translated.srt 为空，无法生成 ASS"
                style_str = get_subtitle_style(
                    self.config.embedder.subtitle_style,
                    style_dir=self.config.storage.subtitle_style_dir,
                )
                asr_data.save(
                    str(subtitle_ass),
                    ass_style=style_str,
                    style_name=self.config.embedder.subtitle_style,
                )
                self.progress_callback(f"ASS 已生成: {subtitle_ass}")
            subtitle_file = subtitle_ass
        else:
            subtitle_file = subtitle_ass
            if not subtitle_file.exists():
                raise EmbedError(f"找不到翻译字幕文件 (SRT/ASS) 在 {self.output_dir}")
            self.progress_callback("警告: 未找到 translated.srt，使用现有 ASS 直接嵌入")
        
        # 根据配置选择嵌入模式
        embed_mode = self.config.embedder.embed_mode
        container = self.config.embedder.output_container
        self.progress_callback(f"嵌入模式: {embed_mode} (容器: {container})")
        
        try:
            # 确定输出文件名
            final_video = self.output_dir / f"final.{container}"
            
            # 软字幕嵌入
            if embed_mode == "soft":
                self.progress_callback(f"使用软字幕 (快速，保持原画质)")
                success = self.ffmpeg.embed_subtitle_soft(
                    video_path=video_file,
                    subtitle_path=subtitle_file,
                    output_path=final_video,
                    subtitle_language='chi',
                    subtitle_title='中文'
                )
                if success:
                    self.progress_callback(f"软字幕嵌入完成: {final_video}")
                    return True
                else:
                    raise EmbedError("软字幕嵌入失败")
            
            # 硬字幕嵌入（统一使用 FFmpegWrapper）
            codec_display = "H.265 GPU" if self.config.embedder.use_gpu and self.config.embedder.video_codec in ['libx265', 'hevc'] else \
                           "AV1 GPU" if self.config.embedder.use_gpu and self.config.embedder.video_codec == 'av1' else \
                           "H.264 GPU" if self.config.embedder.use_gpu else \
                           self.config.embedder.video_codec
            
            self.progress_callback(f"使用硬字幕 (编码器: {codec_display})")
            
            # 统一使用 FFmpegWrapper，支持 ASS 和 SRT
            # 如果是 ASS 格式，传递样式参数以重新应用样式（根据视频分辨率缩放）
            subtitle_style = None
            style_dir = None
            if subtitle_file.suffix.lower() == '.ass':
                subtitle_style = self.config.embedder.subtitle_style
                style_dir = self.config.storage.subtitle_style_dir
            
            # 构建 gpu_device 参数
            gpu_device = "auto"
            if hasattr(self, 'gpu_id') and self.gpu_id is not None:
                gpu_device = f"cuda:{self.gpu_id}"
            
            def embed_progress_callback(progress_str, message):
                """嵌入进度回调，集成到进度追踪器"""
                # progress_str 格式: "5%" 或 "5% | 耗时: 00:01 | 预计剩余: 00:10"
                try:
                    percent_str = progress_str.split('%')[0].strip()
                    percent = float(percent_str)
                    if self._progress_tracker:
                        self._progress_tracker.report_embed_progress(percent)
                except (ValueError, IndexError):
                    pass
                self._progress_with_tracker(f"{message}: {progress_str}", component_name="ffmpeg_wrapper")
            
            success = self.ffmpeg.embed_subtitle_hard(
                video_path=video_file,
                subtitle_path=subtitle_file,
                output_path=final_video,
                video_codec=self.config.embedder.video_codec,
                audio_codec=self.config.embedder.audio_codec,
                crf=self.config.embedder.crf,
                preset=self.config.embedder.preset,
                gpu_device=gpu_device,
                progress_callback=embed_progress_callback,
                fonts_dir=self.config.storage.fonts_dir,
                subtitle_style=subtitle_style,
                style_dir=style_dir
            )
            
            if success:
                if not final_video.exists():
                    raise EmbedError("硬字幕生成成功但找不到输出文件")
                self.progress_callback(f"硬字幕嵌入完成: {final_video}")
                return True
            else:
                raise EmbedError("硬字幕嵌入失败")
                
        except Exception as e:
            error_msg = f"字幕嵌入失败: {e}"
            self.progress_callback(error_msg)
            raise EmbedError(error_msg, original_error=e)
    
    def _run_upload(self) -> bool:
        """
        上传视频到B站
        
        使用模板系统渲染标题/描述，支持通过 config/upload.yaml 自定义
        需要配置B站cookie文件
        """
        
        # 查找最终视频文件
        final_video = self.output_dir / "final.mp4"
        if not final_video.exists():
            # 尝试查找其他格式
            for ext in ['mp4', 'mkv', 'webm']:
                candidate = self.output_dir / f"final.{ext}"
                if candidate.exists():
                    final_video = candidate
                    break
        
        if not final_video.exists():
            raise UploadError("找不到最终视频文件")
        
        # 检查cookie文件
        cookie_file = Path(self.config.uploader.bilibili.cookies_file).expanduser()
        if not cookie_file.exists():
            raise UploadError(
                f"B站cookie文件不存在: {cookie_file}，请运行 vat bilibili login 获取cookie"
            )
        
        # 上传配置统一从 self.config.uploader.bilibili 获取
        # （Config 加载阶段已合并 default.yaml 连接设置 + upload.yaml 内容设置）
        from vat.uploaders.template import render_upload_metadata
        
        bilibili_config = self.config.uploader.bilibili
        
        # 构建模板配置
        templates = {
            'title': bilibili_config.templates.title,
            'description': bilibili_config.templates.description,
            'custom_vars': bilibili_config.templates.custom_vars,
        }
        
        # 获取播放列表信息（如果有）
        playlist_info = None
        if self.video.playlist_id:
            playlist = self.db.get_playlist(self.video.playlist_id)
            if playlist:
                pl_metadata = playlist.metadata or {}
                pl_upload_config = pl_metadata.get('upload_config', {})
                
                # 使用同步时持久化存储的 upload_order_index（只读，不会变）
                # 如果没有，回退到原始 playlist_index
                video_metadata = self.video.metadata or {}
                upload_order_index = video_metadata.get('upload_order_index', self.video.playlist_index or 0)
                
                playlist_info = {
                    'name': playlist.title,
                    'id': playlist.id,
                    'index': upload_order_index,
                    'uploader_name': pl_upload_config.get('uploader_name', ''),
                }
        
        # 使用模板渲染上传元数据
        rendered = render_upload_metadata(self.video, templates, playlist_info)
        title = rendered['title']
        description = rendered['description']
        
        # 获取翻译信息中的标签和分区
        metadata = self.video.metadata or {}
        translated = metadata.get('translated', {})
        
        # 合并标签：翻译标签 + 生成标签 + 全局默认标签（去重）
        tags = translated.get('tags_translated', []) + translated.get('tags_generated', [])
        # 追加全局默认标签
        default_tags = bilibili_config.default_tags or []
        for tag in default_tags:
            if tag and tag not in tags:
                tags.append(tag)
        tags = tags[:12] if tags else []  # B站最多12个标签
        
        tid = translated.get('recommended_tid', bilibili_config.default_tid)
        copyright_type = bilibili_config.copyright
        
        # 转载来源
        source_url = ''
        if copyright_type == 2:  # 转载
            source_url = self.video.source_url or metadata.get('url', '')
        
        self.progress_callback(f"准备上传到B站...")
        self.progress_callback(f"标题: {title[:50]}...")
        self.progress_callback(f"分区: {tid}, 类型: {'自制' if copyright_type == 1 else '转载'}")
        self.progress_callback(f"标签: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
        
        # 处理封面
        cover_path = None
        temp_cover_file = None  # 用于清理临时下载的封面
        
        if bilibili_config.auto_cover:
            # 1. 先查找本地封面文件
            if self.video.output_dir:
                for cover_name in ['thumbnail.jpg', 'thumbnail.png', 'cover.jpg', 'cover.png']:
                    potential = Path(self.video.output_dir) / cover_name
                    if potential.exists():
                        cover_path = potential
                        self.progress_callback(f"使用本地封面: {cover_name}")
                        break
            
            # 2. 如果没有本地封面，尝试下载 thumbnail URL
            if not cover_path:
                thumbnail_url = metadata.get('thumbnail', '')
                if thumbnail_url:
                    try:
                        import requests
                        import tempfile
                        
                        self.progress_callback("下载封面图片...")
                        resp = requests.get(thumbnail_url, timeout=30)
                        resp.raise_for_status()
                        
                        # 根据 content-type 确定扩展名
                        content_type = resp.headers.get('content-type', '')
                        ext = '.jpg'
                        if 'png' in content_type:
                            ext = '.png'
                        elif 'webp' in content_type:
                            ext = '.webp'
                        
                        # 保存到临时文件
                        temp_cover_file = tempfile.NamedTemporaryFile(
                            suffix=ext, delete=False, prefix='cover_'
                        )
                        temp_cover_file.write(resp.content)
                        temp_cover_file.close()
                        cover_path = Path(temp_cover_file.name)
                        self.progress_callback(f"封面下载成功")
                    except Exception as e:
                        self.logger.warning(f"下载封面失败，将使用B站自动识别: {e}")
        
        try:
            from vat.uploaders.bilibili import BilibiliUploader, BILIUP_AVAILABLE
            
            if not BILIUP_AVAILABLE:
                raise UploadError("biliup 库不可用，请安装: pip install biliup")
            
            uploader = BilibiliUploader(
                cookies_file=str(cookie_file),
                line=self.config.uploader.bilibili.line,
                threads=self.config.uploader.bilibili.threads
            )
            
            result = uploader.upload(
                video_path=final_video,
                title=title[:80],  # B站标题限制80字符
                description=description[:2000],  # B站简介限制2000字符
                tid=tid,
                tags=tags,
                copyright=copyright_type,
                source=source_url,
                cover_path=cover_path,
            )
            
            # 清理临时封面文件
            if temp_cover_file:
                try:
                    import os
                    os.unlink(temp_cover_file.name)
                except Exception:
                    pass
            
            if not result.success:
                raise UploadError(f"上传失败: {result.error}")
            
            self.progress_callback(f"上传成功! BV号: {result.bvid}")
            
            # 更新视频元数据（合并到现有 metadata）
            updated_metadata = dict(metadata)  # 复制现有 metadata
            updated_metadata['bilibili_bvid'] = result.bvid
            updated_metadata['bilibili_url'] = f"https://www.bilibili.com/video/{result.bvid}"
            updated_metadata['uploaded_at'] = datetime.now().isoformat()
            self.db.update_video(self.video.id, metadata=updated_metadata)
            
            # 添加到合集（如果配置了）
            if bilibili_config.season_id:
                self.progress_callback(f"添加到合集 {bilibili_config.season_id}...")
                try:
                    aid = uploader.bvid_to_aid(result.bvid)
                    if aid and uploader.add_to_season(aid, bilibili_config.season_id):
                        self.progress_callback("✓ 已添加到合集")
                    else:
                        self.progress_callback("⚠ 添加到合集失败")
                except Exception as e:
                    self.progress_callback(f"⚠ 添加到合集异常: {e}")
            
            return True
                
        except UploadError:
            raise  # 直接向上传播
        except Exception as e:
            error_msg = f"上传异常: {e}"
            self.progress_callback(error_msg)
            self.logger.debug(traceback.format_exc())
            raise UploadError(error_msg, original_error=e)
    
    def _should_use_cache(self, substep_name: str, current_config: Dict, output_file: Path) -> bool:
        """
        判断是否使用缓存
        
        Args:
            substep_name: 子步骤名称 ('whisper', 'split', 'optimize')
            current_config: 当前配置快照
            output_file: 输出文件路径
            
        Returns:
            是否使用缓存
        """
        # 1. 如果 force=True，跳过所有缓存
        if self.force:
            return False
        
        # 2. 检查输出文件是否存在且非空
        if not output_file.exists() or output_file.stat().st_size == 0:
            return False
        
        # 3. 检查 metadata 中的配置快照
        metadata = CacheMetadata.load(self.output_dir)
        if not metadata.is_substep_valid(substep_name, current_config):
            self.progress_callback(f"检测到 {substep_name} 配置变更，缓存失效")
            return False
        
        return True
    
    def _extract_whisper_config(self) -> Dict[str, Any]:
        """提取 Whisper 关键配置"""
        from vat.utils.cache_metadata import extract_key_config, WHISPER_KEY_CONFIGS
        return extract_key_config(self.config.asr, WHISPER_KEY_CONFIGS)
    
    def _extract_split_config(self) -> Dict[str, Any]:
        """提取 Split 关键配置"""
        config_dict = {
            'enable': self.config.asr.split.enable,
            'mode': self.config.asr.split.mode,
            'max_words_cjk': self.config.asr.split.max_words_cjk,
            'max_words_english': self.config.asr.split.max_words_english,
            'model': self.config.asr.split.model,
        }
        return config_dict
    
    def _find_video_file(self) -> Optional[Path]:
        """查找视频文件"""
        # 如果是本地文件，直接返回源路径
        if self.video.source_type == SourceType.LOCAL:
            source_path = Path(self.video.source_url)
            if source_path.exists():
                return source_path
        
        # 在输出目录中查找
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov']
        
        # 首先查找original.*
        for ext in video_extensions:
            video_file = self.output_dir / f"original{ext}"
            if video_file.exists():
                return video_file
        
        # 查找任何视频文件（排除final.*）
        for ext in video_extensions:
            for video_file in self.output_dir.glob(f"*{ext}"):
                if not video_file.name.startswith("final."):
                    return video_file
        
        return None


def create_video_from_url(
    url: str,
    db: Database,
    source_type: SourceType = SourceType.YOUTUBE
) -> str:
    """
    从URL创建视频记录
    
    Args:
        url: 视频URL
        db: 数据库实例
        source_type: 来源类型
        
    Returns:
        视频ID
    """
    # 生成视频ID
    video_id = hashlib.md5(url.encode()).hexdigest()[:16]
    
    # 检查视频是否已存在，清理旧任务记录避免重复
    existing = db.get_video(video_id)
    if existing:
        deleted = db.delete_tasks_for_video(video_id)
        if deleted > 0:
            setup_logger("pipeline.executor").info(
                f"视频 {video_id} 已存在，已清理 {deleted} 条旧任务记录"
            )
    
    # 创建/替换视频记录
    video = Video(
        id=video_id,
        source_type=source_type,
        source_url=url
    )
    
    db.add_video(video)
    
    # 创建初始任务（使用细粒度阶段）
    for step in DEFAULT_STAGE_SEQUENCE:
        task = Task(
            video_id=video_id,
            step=step,
            status=TaskStatus.PENDING
        )
        db.add_task(task)
    
    return video_id
