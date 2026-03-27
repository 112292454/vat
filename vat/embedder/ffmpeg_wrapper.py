"""
FFmpeg视频处理封装
"""
import re
import time
import tempfile
import subprocess
import shutil
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union

from vat.media import extract_audio_ffmpeg, probe_media_info
from vat.utils.gpu import resolve_gpu_device, get_available_gpus, is_cuda_available
from vat.utils.logger import setup_logger

logger = setup_logger("ffmpeg_wrapper")


class _NvencSessionManager:
    """NVENC 编码会话管理器
    
    NVIDIA 消费级显卡（如 RTX 4090）限制每张卡同时最多 N 个 NVENC 会话。
    此管理器通过 per-GPU 信号量控制并发会话数，并实现均衡 GPU 分配。
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
        # gpu_id -> Semaphore
        self._semaphores: Dict[int, threading.Semaphore] = {}
        # gpu_id -> 当前活跃会话数（用于选择最空闲 GPU）
        self._active_sessions: Dict[int, int] = {}
        self._max_per_gpu = 5
    
    def init(self, max_per_gpu: int = 5) -> None:
        """初始化（幂等，首次调用时探测可用 GPU）
        
        Args:
            max_per_gpu: 每张 GPU 最大并发 NVENC 会话数（RTX 消费级默认 5）
        """
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._max_per_gpu = max_per_gpu
            try:
                gpus = get_available_gpus()
                for gpu in gpus:
                    self._semaphores[gpu.index] = threading.Semaphore(max_per_gpu)
                    self._active_sessions[gpu.index] = 0
                logger.info(
                    f"NVENC 会话管理器已初始化: {len(gpus)} 张 GPU, "
                    f"每卡最大 {max_per_gpu} 会话"
                )
            except Exception as e:
                logger.warning(f"无法探测 GPU 列表，NVENC 会话管理器降级: {e}")
            self._initialized = True
    
    def _ensure_gpu(self, gpu_id: int) -> None:
        """确保指定 GPU 有对应的信号量（动态扩展）"""
        if gpu_id not in self._semaphores:
            with self._lock:
                if gpu_id not in self._semaphores:
                    self._semaphores[gpu_id] = threading.Semaphore(self._max_per_gpu)
                    self._active_sessions[gpu_id] = 0
    
    def select_gpu(self) -> int:
        """选择当前活跃会话最少的 GPU（均衡分配）
        
        Returns:
            gpu_id: 选中的 GPU 索引
            
        Raises:
            RuntimeError: 没有可用 GPU
        """
        if not self._semaphores:
            # 尚未初始化或无 GPU，fallback 到 resolve_gpu_device
            from vat.utils.gpu import select_best_gpu
            gpu_id = select_best_gpu(min_free_memory_mb=1000)
            if gpu_id is None:
                raise RuntimeError("没有可用 GPU")
            self._ensure_gpu(gpu_id)
            return gpu_id
        
        with self._lock:
            # 选择活跃会话最少的 GPU
            best_gpu = min(self._active_sessions, key=self._active_sessions.get)
            return best_gpu
    
    def acquire(self, gpu_id: int, timeout: float = 600) -> bool:
        """获取指定 GPU 的 NVENC 会话槽位
        
        Args:
            gpu_id: GPU 索引
            timeout: 最大等待秒数（默认 10 分钟）
            
        Returns:
            是否成功获取
        """
        self._ensure_gpu(gpu_id)
        acquired = self._semaphores[gpu_id].acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._active_sessions[gpu_id] = self._active_sessions.get(gpu_id, 0) + 1
            logger.debug(
                f"NVENC 会话已获取: GPU {gpu_id} "
                f"(活跃: {self._active_sessions[gpu_id]}/{self._max_per_gpu})"
            )
        else:
            logger.warning(
                f"NVENC 会话获取超时 ({timeout}s): GPU {gpu_id}, "
                f"当前活跃: {self._active_sessions.get(gpu_id, '?')}/{self._max_per_gpu}"
            )
        return acquired
    
    def release(self, gpu_id: int) -> None:
        """释放指定 GPU 的 NVENC 会话槽位"""
        if gpu_id in self._semaphores:
            with self._lock:
                self._active_sessions[gpu_id] = max(0, self._active_sessions.get(gpu_id, 1) - 1)
            self._semaphores[gpu_id].release()
            logger.debug(
                f"NVENC 会话已释放: GPU {gpu_id} "
                f"(活跃: {self._active_sessions[gpu_id]}/{self._max_per_gpu})"
            )


# 模块级单例
_nvenc_manager = _NvencSessionManager()


def _format_time(seconds: float) -> str:
    """格式化时间为 MM:SS 或 HH:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class FFmpegWrapper:
    """FFmpeg操作封装类"""
    
    def __init__(self):
        """初始化FFmpeg封装器"""
        # 检查ffmpeg是否可用
        if not shutil.which('ffmpeg'):
            raise RuntimeError("ffmpeg未安装或不在PATH中")
    
    def extract_audio(
        self,
        video_path: Path,
        audio_path: Path,
        sample_rate: int = 16000,
        channels: int = 1,
        codec: str = 'pcm_s16le'
    ) -> bool:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 输出音频路径
            sample_rate: 采样率
            channels: 声道数
            codec: 音频编码器
            
        Returns:
            是否成功
        """
        try:
            extract_audio_ffmpeg(
                video_path,
                audio_path,
                sample_rate=sample_rate,
                channels=channels,
                codec=codec,
            )
            return True
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(str(e))
            return False
    
    def embed_subtitle_soft(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        subtitle_language: str = 'chi',
        subtitle_title: str = '中文'
    ) -> bool:
        """
        软字幕嵌入（作为独立字幕流，不重新编码视频）
        
        优势：
        - 极快（几秒钟完成）
        - 文件大小几乎不变
        - 保持原始视频质量和编码格式
        - 用户可以选择开关字幕
        
        Args:
            video_path: 输入视频路径
            subtitle_path: 字幕文件路径（支持SRT/ASS）
            output_path: 输出视频路径
            subtitle_language: 字幕语言代码（chi/zh/zho）
            subtitle_title: 字幕标题（显示在播放器中）
            
        Returns:
            是否成功
        """
        if not video_path.exists():
            print(f"错误: 输入视频文件不存在: {video_path}")
            return False
        if not subtitle_path.exists():
            print(f"错误: 字幕文件不存在: {subtitle_path}")
            return False
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查字幕格式
        subtitle_ext = subtitle_path.suffix.lower()
        
        # MKV容器支持更多字幕格式，MP4需要转换
        output_ext = output_path.suffix.lower()
        
        if output_ext == '.mkv':
            # MKV支持原生ASS字幕
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-i', str(subtitle_path),
                '-c:v', 'copy',  # 复制视频流（不重新编码）
                '-c:a', 'copy',  # 复制音频流
                '-c:s', 'copy' if subtitle_ext == '.ass' else 'srt',  # ASS可直接复制
                '-metadata:s:s:0', f'language={subtitle_language}',
                '-metadata:s:s:0', f'title={subtitle_title}',
                '-disposition:s:0', 'default',  # 设为默认字幕
                '-y',
                str(output_path)
            ]
        else:
            # MP4容器，字幕需要转换为mov_text格式
            # 注意：MP4不支持ASS样式，会丢失样式信息
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-i', str(subtitle_path),
                '-c:v', 'copy',  # 复制视频流
                '-c:a', 'copy',  # 复制音频流
                '-c:s', 'mov_text',  # MP4字幕格式
                '-metadata:s:s:0', f'language={subtitle_language}',
                '-metadata:s:s:0', f'title={subtitle_title}',
                '-y',
                str(output_path)
            ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if not output_path.exists():
                print(f"错误: 软字幕嵌入完成但未生成文件: {output_path}")
                return False
            return True
        except subprocess.CalledProcessError as e:
            print(f"软字幕嵌入失败: {e.stderr}")
            # 如果是MP4+ASS失败，提示用户
            if output_ext == '.mp4' and subtitle_ext == '.ass':
                print("提示: MP4容器不完全支持ASS字幕样式，建议使用MKV容器或硬字幕")
            return False
    
    def _get_video_resolution(self, video_path: Path) -> tuple[int, int]:
        """获取视频分辨率"""
        result = subprocess.run(
            ["ffmpeg", "-i", str(video_path)],
            capture_output=True,
            text=True,
        )
        
        # 从 ffmpeg 输出中解析分辨率
        pattern = r"(\d{2,5})x(\d{2,5})"
        match = re.search(pattern, result.stderr)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 1920, 1080  # 默认返回 1080P
    
    def _scale_ass_style(
        self, style_str: str, scale_factor: float,
        video_width: int = 0, video_height: int = 0,
    ) -> str:
        """缩放 ASS 样式中的数值参数
        
        竖屏额外调整：
        - MarginV：在缩放基础上再翻倍（模板值 * scale * 2）
        - MarginL/R：至少保证 5% 视频宽度，防止文字贴边
        
        Args:
            style_str: ASS 样式字符串
            scale_factor: 字体等通用缩放因子
            video_width: 视频宽度（用于判断横竖屏，0=不做竖屏特殊处理）
            video_height: 视频高度
        """
        if scale_factor == 1.0 and not (video_height > video_width > 0):
            return style_str
        
        is_portrait = video_height > video_width > 0
        
        lines = style_str.split("\n")
        scaled_lines = []
        
        for line in lines:
            if line.startswith("Style:"):
                parts = line.split(",")
                if len(parts) >= 23:
                    # parts[2]: Fontsize
                    parts[2] = str(int(float(parts[2]) * scale_factor))
                    # parts[13]: Spacing
                    parts[13] = str(float(parts[13]) * scale_factor)
                    # parts[16]: Outline
                    parts[16] = str(float(parts[16]) * scale_factor)
                    # parts[19]: MarginL
                    parts[19] = str(int(float(parts[19]) * scale_factor))
                    # parts[20]: MarginR
                    parts[20] = str(int(float(parts[20]) * scale_factor))
                    # parts[21]: MarginV
                    scaled_mv = int(float(parts[21]) * scale_factor)
                    
                    if is_portrait:
                        # 竖屏：MarginV 在缩放基础上再翻倍
                        scaled_mv *= 2
                        # 竖屏：MarginL/R 至少 5% 视频宽度，防止文字贴边
                        min_margin = int(video_width * 0.05)
                        parts[19] = str(max(int(parts[19]), min_margin))
                        parts[20] = str(max(int(parts[20]), min_margin))
                    
                    parts[21] = str(scaled_mv)
                    
                    line = ",".join(parts)
            scaled_lines.append(line)
        
        return "\n".join(scaled_lines)
    
    def embed_subtitle_hard(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        video_codec: str = 'hevc',
        audio_codec: str = 'copy',
        crf: int = 28,
        preset: str = 'p4',
        gpu_device: str = "auto",  # "auto", "cpu", "cuda:N"
        progress_callback: Optional[Callable[[str, str], None]] = None,
        fonts_dir: Optional[str] = None,
        subtitle_style: Optional[str] = None,
        style_dir: Optional[str] = None,
        reference_height: int = 720,
        max_nvenc_sessions: int = 5
    ) -> bool:
        """
        硬字幕嵌入（烧录到视频画面）
        
        Args:
            video_path: 输入视频路径
            subtitle_path: 字幕文件路径 (SRT/ASS)
            output_path: 输出视频路径
            video_codec: 视频编码器 (libx264, libx265, hevc, av1)
            audio_codec: 音频编码器 (aac, copy等)
            crf: 视频质量 (0-51, 越小质量越好)
            preset: 编码预设
            gpu_device: GPU 设备标识符 ("auto", "cpu", "cuda:N")
            progress_callback: 进度回调函数 (progress_str, message) -> None
            fonts_dir: 字体目录路径（仅ASS格式需要）
            subtitle_style: 字幕样式模板名称（仅ASS格式需要）
            style_dir: 样式文件目录（仅ASS格式需要）
            reference_height: 参考高度，用于样式缩放（默认720）
            max_nvenc_sessions: 每张 GPU 最大并发 NVENC 会话数（默认 5）
            
        Returns:
            是否成功
        """
        if not self._prepare_hard_embed_preflight(
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_path=output_path,
            gpu_device=gpu_device,
            max_nvenc_sessions=max_nvenc_sessions,
        ):
            return False

        subtitle_ext, processed_subtitle, temp_files_to_cleanup, vf = self._plan_hard_embed_subtitle_inputs(
            video_path=video_path,
            subtitle_path=subtitle_path,
            subtitle_style=subtitle_style,
            style_dir=style_dir,
            fonts_dir=fonts_dir,
            reference_height=reference_height,
        )

        # ========== 阶段 2: 获取 NVENC 会话 + 构建 ffmpeg 命令 ==========
        # 预处理完成后才获取 GPU session，最大化 session 利用率。
        # ================================================================
        gpu_id, cmd = self._plan_hard_embed_execution(
            video_path=video_path,
            output_path=output_path,
            vf=vf,
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
            preset=preset,
            gpu_device=gpu_device,
            max_nvenc_sessions=max_nvenc_sessions,
        )

        return self._run_hard_embed_runtime_stage(
            gpu_id=gpu_id,
            cmd=cmd,
            output_path=output_path,
            progress_callback=progress_callback,
            temp_files_to_cleanup=temp_files_to_cleanup,
        )

    def _prepare_hard_embed_preflight(
        self,
        *,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        gpu_device: str,
        max_nvenc_sessions: int,
    ) -> bool:
        """执行硬字幕合成前的输入校验与基础环境准备。"""
        if not video_path.exists():
            logger.error(f"输入视频文件不存在: {video_path}")
            return False
        if not subtitle_path.exists():
            logger.error(f"字幕文件不存在: {subtitle_path}")
            return False

        _nvenc_manager.init(max_per_gpu=max_nvenc_sessions)

        if gpu_device not in ("auto",) and not gpu_device.startswith("cuda:"):
            error_msg = "Embed 阶段需要 GPU，按 GPU 原则禁止 CPU 回退"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return True

    def _finalize_hard_embed_resources(
        self,
        *,
        gpu_id: int,
        temp_files_to_cleanup: list[str],
    ) -> None:
        """释放硬字幕合成阶段占用的资源。"""
        _nvenc_manager.release(gpu_id)
        for temp_file in temp_files_to_cleanup:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass

    def _resolve_hard_embed_gpu_device(self, gpu_device: str) -> int:
        """解析硬字幕合成使用的 GPU 设备。"""
        if gpu_device == "auto":
            gpu_id = _nvenc_manager.select_gpu()
            logger.info(f"NVENC 会话均衡分配: 选择 GPU {gpu_id}")
            return gpu_id

        try:
            return int(gpu_device.split(":")[1])
        except (IndexError, ValueError):
            raise ValueError(f"无效的 GPU 设备格式: {gpu_device}")

    def _plan_hard_embed_subtitle_inputs(
        self,
        *,
        video_path: Path,
        subtitle_path: Path,
        subtitle_style: Optional[str],
        style_dir: Optional[str],
        fonts_dir: Optional[str],
        reference_height: int,
    ) -> tuple[str, Path, list[str], str]:
        """规划硬字幕合成使用的字幕输入与滤镜。"""
        subtitle_ext = subtitle_path.suffix.lower()
        processed_subtitle = subtitle_path
        temp_files_to_cleanup: list[str] = []

        if subtitle_ext == '.ass' and subtitle_style:
            processed_subtitle, temp_files_to_cleanup = self._prepare_hard_embed_ass_subtitle(
                video_path=video_path,
                subtitle_path=subtitle_path,
                subtitle_style=subtitle_style,
                style_dir=style_dir,
                fonts_dir=fonts_dir,
                reference_height=reference_height,
            )

        vf = self._build_hard_embed_subtitle_filter(
            subtitle_ext=subtitle_ext,
            processed_subtitle=processed_subtitle,
            fonts_dir=fonts_dir,
        )
        return subtitle_ext, processed_subtitle, temp_files_to_cleanup, vf

    def _plan_hard_embed_execution(
        self,
        *,
        video_path: Path,
        output_path: Path,
        vf: str,
        video_codec: str,
        audio_codec: str,
        crf: int,
        preset: str,
        gpu_device: str,
        max_nvenc_sessions: int,
    ) -> tuple[int, List[str]]:
        """规划硬字幕合成执行阶段所需 GPU 与 ffmpeg 命令。"""
        gpu_id = self._resolve_hard_embed_gpu_device(gpu_device)
        self._prepare_hard_embed_nvenc_session(
            gpu_id=gpu_id,
            max_nvenc_sessions=max_nvenc_sessions,
        )
        original_bitrate = self._probe_hard_embed_original_bitrate(video_path)
        cmd = self._build_hard_embed_ffmpeg_command(
            video_path=video_path,
            output_path=output_path,
            vf=vf,
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
            preset=preset,
            gpu_id=gpu_id,
            original_bitrate=original_bitrate,
        )
        return gpu_id, cmd

    def _run_hard_embed_runtime_stage(
        self,
        *,
        gpu_id: int,
        cmd: List[str],
        output_path: Path,
        progress_callback: Optional[Callable[[str, str], None]],
        temp_files_to_cleanup: list[str],
    ) -> bool:
        """执行硬字幕合成运行阶段并确保资源清理。"""
        try:
            return self._run_ffmpeg_embed_process(
                cmd=cmd,
                output_path=output_path,
                progress_callback=progress_callback,
            )
        finally:
            self._finalize_hard_embed_resources(
                gpu_id=gpu_id,
                temp_files_to_cleanup=temp_files_to_cleanup,
            )

    def _prepare_hard_embed_ass_subtitle(
        self,
        *,
        video_path: Path,
        subtitle_path: Path,
        subtitle_style: str,
        style_dir: Optional[str],
        fonts_dir: Optional[str],
        reference_height: int,
    ) -> tuple[Path, list[str]]:
        """为硬字幕合成预处理 ASS 字幕输入。"""
        processed_subtitle = subtitle_path
        temp_files_to_cleanup: list[str] = []

        try:
            from vat.asr import ASRData
            from vat.asr.subtitle import get_subtitle_style, auto_wrap_ass_file, compute_subtitle_scale_factor

            width, height = self._get_video_resolution(video_path)

            style_str = get_subtitle_style(subtitle_style, style_dir=style_dir)
            if not style_str:
                logger.warning(f"无法加载样式 '{subtitle_style}'，使用默认样式")
                style_str = get_subtitle_style("default", style_dir=style_dir) or ""

            scale_factor = compute_subtitle_scale_factor(width, height, reference_height)
            style_str = self._scale_ass_style(
                style_str,
                scale_factor,
                video_width=width,
                video_height=height,
            )

            asr_data = ASRData.from_subtitle_file(str(subtitle_path))

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".ass", delete=False, encoding="utf-8"
            ) as temp_file:
                ass_content = asr_data.to_ass(
                    style_str=style_str,
                    video_width=width,
                    video_height=height,
                )
                temp_file.write(ass_content)
                temp_ass_path = temp_file.name
                temp_files_to_cleanup.append(temp_ass_path)

            processed_subtitle_path = auto_wrap_ass_file(temp_ass_path, fonts_dir=fonts_dir)
            processed_subtitle = Path(processed_subtitle_path)
            if processed_subtitle_path != temp_ass_path:
                temp_files_to_cleanup.append(processed_subtitle_path)
        except Exception as e:
            logger.warning(f"ASS 预处理失败，使用原始文件: {e}")
            processed_subtitle = subtitle_path

        return processed_subtitle, temp_files_to_cleanup

    def _build_hard_embed_subtitle_filter(
        self,
        *,
        subtitle_ext: str,
        processed_subtitle: Path,
        fonts_dir: Optional[str],
    ) -> str:
        """构建硬字幕合成使用的字幕滤镜。"""
        subtitle_path_escaped = Path(processed_subtitle).as_posix().replace(":", r"\:")

        if subtitle_ext == '.ass':
            vf = f"ass='{subtitle_path_escaped}'"
            if fonts_dir:
                fonts_dir_escaped = Path(fonts_dir).as_posix().replace(":", r"\:")
                vf += f":fontsdir='{fonts_dir_escaped}'"
            return vf

        return f"subtitles='{subtitle_path_escaped}'"

    def _probe_hard_embed_original_bitrate(self, video_path: Path) -> int:
        """探测硬字幕合成使用的原视频码率。"""
        video_info = self.get_video_info(video_path)
        return video_info.get('bit_rate', 0) if video_info else 0

    def _prepare_hard_embed_nvenc_session(self, *, gpu_id: int, max_nvenc_sessions: int) -> None:
        """为硬字幕合成获取 NVENC 会话并校验支持情况。"""
        if not _nvenc_manager.acquire(gpu_id, timeout=600):
            raise RuntimeError(
                f"NVENC 会话获取超时: GPU {gpu_id}，"
                f"所有 {max_nvenc_sessions} 个槽位已满且 10 分钟内未释放"
            )

        if not self._check_nvenc_support():
            _nvenc_manager.release(gpu_id)
            error_msg = "当前环境不支持 NVENC，按 GPU 原则禁止 CPU 回退"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _build_hard_embed_ffmpeg_command(
        self,
        *,
        video_path: Path,
        output_path: Path,
        vf: str,
        video_codec: str,
        audio_codec: str,
        crf: int,
        preset: str,
        gpu_id: int,
        original_bitrate: int,
    ) -> List[str]:
        """构建硬字幕合成使用的 FFmpeg 命令。"""
        if video_codec in ['libx265', 'hevc']:
            actual_codec = 'hevc_nvenc'
        elif video_codec == 'av1':
            if self._check_encoder_support('av1_nvenc'):
                actual_codec = 'av1_nvenc'
            else:
                logger.warning("当前环境不支持 av1_nvenc，回退到 hevc_nvenc")
                actual_codec = 'hevc_nvenc'
        else:
            if video_codec != 'h264':
                logger.info(f"未知编码器 {video_codec}，使用 h264_nvenc")
            actual_codec = 'h264_nvenc'

        if original_bitrate > 0:
            target_bitrate = int(original_bitrate * 1.1)
            max_bitrate = int(original_bitrate * 1.5)
            codec_params = [
                '-rc', 'vbr',
                '-cq', str(crf),
                '-b:v', str(target_bitrate),
                '-maxrate', str(max_bitrate),
                '-bufsize', str(max_bitrate * 2),
                '-preset', preset if preset.startswith('p') else 'p4',
            ]
        else:
            codec_params = [
                '-rc', 'constqp',
                '-qp', str(crf),
                '-preset', preset if preset.startswith('p') else 'p4',
            ]

        codec_params.extend([
            '-gpu', str(gpu_id),
            '-spatial_aq', '1',
            '-temporal_aq', '1',
        ])

        return [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-i', str(video_path),
            '-vf', vf,
            '-c:v', actual_codec,
            *codec_params,
            '-c:a', audio_codec,
            '-movflags', '+faststart',
            '-y',
            str(output_path),
        ]

    def _run_ffmpeg_embed_process(
        self,
        *,
        cmd: List[str],
        output_path: Path,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> bool:
        """执行 FFmpeg 硬字幕合成并处理进度与日志。"""
        try:
            log_path = output_path.parent / "ffmpeg_embed.log"
            ffmpeg_log_lines = []

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            total_duration = None
            current_time = 0
            last_progress = -1
            start_time = time.time()

            while True:
                output_line = process.stderr.readline()
                if not output_line or (process.poll() is not None):
                    break

                ffmpeg_log_lines.append(output_line)

                if total_duration is None:
                    duration_match = re.search(
                        r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", output_line
                    )
                    if duration_match:
                        h, m, s = map(float, duration_match.groups())
                        total_duration = h * 3600 + m * 60 + s

                time_match = re.search(
                    r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})", output_line
                )
                if time_match:
                    h, m, s = map(float, time_match.groups())
                    current_time = h * 3600 + m * 60 + s

                if total_duration:
                    progress = (current_time / total_duration) * 100
                    current_progress_int = int(progress)

                    if (
                        current_progress_int >= last_progress + 5
                        or (last_progress == -1 and current_progress_int == 0)
                        or current_progress_int == 100
                    ):
                        elapsed = time.time() - start_time
                        info_str = f"{current_progress_int}%"

                        if current_progress_int > 0:
                            total_estimated = elapsed / (progress / 100)
                            remaining = total_estimated - elapsed
                            info_str += f" | 耗时: {_format_time(elapsed)} | 预计剩余: {_format_time(remaining)}"

                        if progress_callback:
                            progress_callback(info_str, "正在合成")
                        last_progress = current_progress_int
                time.sleep(0.1)

            if progress_callback:
                progress_callback("100", "合成完成")

            return_code = process.wait()

            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("=== FFmpeg 命令 ===\n")
                    f.write(" ".join(str(c) for c in cmd) + "\n\n")
                    f.write("=== FFmpeg 输出 ===\n")
                    f.writelines(ffmpeg_log_lines)
                    if return_code != 0:
                        remaining_output = process.stderr.read()
                        if remaining_output:
                            f.write(remaining_output)
            except Exception as e:
                logger.warning(f"无法保存 FFmpeg 日志: {e}")

            if return_code != 0:
                error_lines = [l.strip() for l in ffmpeg_log_lines if 'error' in l.lower() or 'failed' in l.lower()]
                error_summary = '; '.join(error_lines[-3:]) if error_lines else '(无详细错误，见日志)'
                logger.error(f"硬字幕嵌入失败: {error_summary}")
                logger.info(f"完整日志已保存至: {log_path}")
                return False

            if not output_path.exists():
                logger.error(f"硬字幕嵌入完成但未生成文件: {output_path}")
                return False
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"硬字幕嵌入失败: {e.stderr}")
            return False

    def _check_encoder_support(self, encoder_name: str) -> bool:
        """检查是否支持特定编码器"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                check=True
            )
            return encoder_name in result.stdout
        except:
            return False
    
    def _check_nvenc_support(self) -> bool:
        """检查是否支持 NVENC 硬件编码 (H.264)"""
        return self._check_encoder_support('h264_nvenc')
    
    def get_video_info(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        return probe_media_info(video_path)
    
    def convert_video(
        self,
        input_path: Path,
        output_path: Path,
        video_codec: str = 'libx264',
        audio_codec: str = 'aac',
        crf: int = 23,
        preset: str = 'medium'
    ) -> bool:
        """
        转换视频格式
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            video_codec: 视频编码器
            audio_codec: 音频编码器
            crf: 视频质量
            preset: 编码预设
            
        Returns:
            是否成功
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', video_codec,
            '-crf', str(crf),
            '-preset', preset,
            '-c:a', audio_codec,
            '-y',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"视频转换失败: {e.stderr}")
            return False
    
    def extract_thumbnail(
        self,
        video_path: Path,
        output_path: Path,
        time_position: str = '00:00:01'
    ) -> bool:
        """
        提取视频缩略图
        
        Args:
            video_path: 视频文件路径
            output_path: 输出图片路径
            time_position: 时间位置 (HH:MM:SS)
            
        Returns:
            是否成功
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-ss', time_position,
            '-i', str(video_path),
            '-vframes', '1',
            '-q:v', '2',
            '-y',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"缩略图提取失败: {e.stderr}")
            return False

    @staticmethod
    def _find_cjk_font() -> Optional[str]:
        """查找系统中可用的 CJK（中日韩）字体文件路径
        
        Returns:
            字体文件绝对路径，未找到返回 None
        """
        # 优先级：Noto Sans CJK SC > WenQuanYi > 任何 CJK 字体
        preferred = [
            'Noto Sans CJK SC',
            'Noto Sans CJK',
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Source Han Sans SC',
            'Source Han Sans CN',
        ]
        try:
            for font_name in preferred:
                result = subprocess.run(
                    ['fc-match', font_name, '--format=%{file}'],
                    capture_output=True, text=True, timeout=5
                )
                path = result.stdout.strip()
                if path and Path(path).exists():
                    # 验证确实是 CJK 字体（非 fallback 到拉丁字体）
                    if 'CJK' in path or 'WenQuanYi' in path or 'SourceHan' in path or 'Noto' in path:
                        return path
            
            # 兜底：直接搜索常见路径
            for candidate in [
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
            ]:
                if Path(candidate).exists():
                    return candidate
        except Exception as e:
            logger.warning(f"查找 CJK 字体失败: {e}")
        
        return None

    def _prepare_mask_violation_context(
        self,
        *,
        video_path: Path,
        output_path: Path,
        violation_ranges: List[tuple],
        margin_sec: float,
    ) -> Optional[tuple[Dict[str, Any], int, int, List[tuple]]]:
        """准备违规遮罩前置上下文。"""
        if not video_path.exists():
            logger.error(f"输入视频不存在: {video_path}")
            return None

        if not violation_ranges:
            logger.warning("无违规时间段，无需处理")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)

        video_info = self.get_video_info(video_path)
        if not video_info:
            logger.error("无法获取视频信息")
            return None

        duration = video_info.get('duration', 0)
        width = video_info.get('video', {}).get('width', 1920) if video_info.get('video') else 1920
        height = video_info.get('video', {}).get('height', 1080) if video_info.get('video') else 1080
        merged = self._merge_ranges(violation_ranges, margin_sec, duration)
        return video_info, width, height, merged

    def _plan_mask_violation_filters(
        self,
        *,
        width: int,
        height: int,
        merged: List[tuple],
        mask_text: str,
    ) -> tuple[str, str]:
        """规划违规遮罩所需的音视频滤镜。"""
        vf_parts = []
        af_parts = []

        escaped_text = mask_text.replace("'", "\\'").replace(":", "\\:")
        fontsize = max(24, height // 30)
        cjk_font = self._find_cjk_font()
        if cjk_font:
            escaped_font = cjk_font.replace(":", "\\:").replace("'", "\\'")
            font_param = f":fontfile='{escaped_font}'"
            logger.info(f"使用 CJK 字体: {cjk_font}")
        else:
            logger.warning("未找到 CJK 字体，中文文字可能无法正常显示")
            font_param = ""

        for start, end in merged:
            vf_parts.append(
                f"drawbox=x=0:y=0:w={width}:h={height}:color=black:t=fill"
                f":enable='between(t,{start},{end})'"
            )
            vf_parts.append(
                f"drawtext=text='{escaped_text}'"
                f"{font_param}"
                f":fontsize={fontsize}:fontcolor=white"
                f":x=(w-text_w)/2:y=(h-text_h)/2"
                f":enable='between(t,{start},{end})'"
            )
            af_parts.append(
                f"volume=enable='between(t,{start},{end})':volume=0"
            )

        vf = ",".join(vf_parts)
        af = ",".join(af_parts) if af_parts else "anull"
        return vf, af

    def _plan_mask_violation_execution(
        self,
        *,
        video_path: Path,
        output_path: Path,
        vf: str,
        af: str,
        video_info: Dict[str, Any],
        gpu_device: str,
    ) -> Optional[tuple[int, List[str]]]:
        """规划违规遮罩执行阶段所需 GPU 与 ffmpeg 命令。"""
        _nvenc_manager.init()

        if gpu_device == "auto":
            gpu_id = _nvenc_manager.select_gpu()
        else:
            try:
                gpu_id = int(gpu_device.split(":")[1])
            except (IndexError, ValueError):
                gpu_id = 0

        if not _nvenc_manager.acquire(gpu_id, timeout=300):
            logger.error(f"NVENC 会话获取超时: GPU {gpu_id}")
            return None

        original_bitrate = video_info.get('bit_rate', 0)
        if original_bitrate > 0:
            codec_params = [
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', str(int(original_bitrate * 1.1)),
                '-maxrate', str(int(original_bitrate * 1.5)),
            ]
        else:
            codec_params = ['-rc', 'constqp', '-qp', '23']

        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-i', str(video_path),
            '-vf', vf,
            '-af', af,
            '-c:v', 'hevc_nvenc',
            '-gpu', str(gpu_id),
            *codec_params,
            '-preset', 'p4',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ]
        return gpu_id, cmd

    def _run_mask_violation_runtime_stage(
        self,
        *,
        video_path: Path,
        output_path: Path,
        gpu_id: int,
        cmd: List[str],
    ) -> bool:
        """执行违规遮罩运行阶段并确保释放 NVENC 会话。"""
        try:
            logger.info(f"开始遮罩处理 (GPU {gpu_id})...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1小时超时
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg 遮罩处理失败: {result.stderr[-500:]}")
                return False

            if not output_path.exists():
                logger.error("遮罩处理完成但未生成文件")
                return False

            in_size = video_path.stat().st_size
            out_size = output_path.stat().st_size
            ratio = out_size / in_size if in_size > 0 else 0
            logger.info(
                f"遮罩处理完成: {output_path.name} "
                f"({out_size / 1024 / 1024:.1f}MB, 相对原文件 {ratio:.1%})"
            )
            return True

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg 遮罩处理超时 (>1小时)")
            return False
        except Exception as e:
            logger.error(f"遮罩处理异常: {e}")
            return False
        finally:
            _nvenc_manager.release(gpu_id)

    def mask_violation_segments(
        self,
        video_path: Path,
        output_path: Path,
        violation_ranges: List[tuple],
        mask_text: str = "此处内容因平台合规要求已被遮罩",
        gpu_device: str = "auto",
        margin_sec: float = 1.0,
    ) -> bool:
        """
        遮罩视频中的违规时间段：用黑屏+说明文字替换违规片段，音频静音。
        
        使用 GPU 加速编码（NVENC），通过 ffmpeg drawtext + colorkey 滤镜实现。
        不裁剪视频（保持总时长不变），仅将违规区间替换为黑底+白字说明。
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            violation_ranges: 违规时间段列表 [(start_sec, end_sec), ...]
            mask_text: 遮罩区域显示的说明文字
            gpu_device: GPU 设备 ("auto" / "cuda:N")
            margin_sec: 每段前后额外扩展的安全边距（秒）
            
        Returns:
            是否成功
        """
        context = self._prepare_mask_violation_context(
            video_path=video_path,
            output_path=output_path,
            violation_ranges=violation_ranges,
            margin_sec=margin_sec,
        )
        if context is None:
            return False

        video_info, width, height, merged = context

        logger.info(f"遮罩 {len(merged)} 个违规片段（含 {margin_sec}s 安全边距）:")
        for start, end in merged:
            logger.info(f"  {_format_time(start)} - {_format_time(end)} ({end - start:.1f}s)")

        vf, af = self._plan_mask_violation_filters(
            width=width,
            height=height,
            merged=merged,
            mask_text=mask_text,
        )

        planned_execution = self._plan_mask_violation_execution(
            video_path=video_path,
            output_path=output_path,
            vf=vf,
            af=af,
            video_info=video_info,
            gpu_device=gpu_device,
        )
        if planned_execution is None:
            return False

        gpu_id, cmd = planned_execution
        return self._run_mask_violation_runtime_stage(
            video_path=video_path,
            output_path=output_path,
            gpu_id=gpu_id,
            cmd=cmd,
        )

    @staticmethod
    def _merge_ranges(
        ranges: List[tuple], margin: float, max_duration: float
    ) -> List[tuple]:
        """
        合并重叠/相邻的时间区间，并添加安全边距。
        
        Args:
            ranges: [(start, end), ...]
            margin: 前后扩展的安全边距（秒）
            max_duration: 视频总时长（用于 clamp）
            
        Returns:
            合并后的区间列表，按起始时间排序
        """
        if not ranges:
            return []
        
        # 添加安全边距并 clamp
        expanded = []
        for start, end in ranges:
            s = max(0, start - margin)
            e = min(max_duration, end + margin)
            expanded.append((s, e))
        
        # 按起始时间排序
        expanded.sort(key=lambda x: x[0])
        
        # 合并重叠区间
        merged = [expanded[0]]
        for s, e in expanded[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        
        return merged
