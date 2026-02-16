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
        if not video_path.exists():
            print(f"错误: 输入视频文件不存在: {video_path}")
            return False
            
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # 不处理视频
            '-acodec', codec,
            '-ac', str(channels),
            '-ar', str(sample_rate),
            '-y',  # 覆盖输出
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            if not audio_path.exists():
                print(f"错误: 音频提取完成但未生成文件: {audio_path}")
                return False
            return True
        except subprocess.CalledProcessError as e:
            print(f"音频提取失败: {e.stderr}")
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
    
    def _scale_ass_style(self, style_str: str, scale_factor: float) -> str:
        """缩放 ASS 样式中的数值参数"""
        if scale_factor == 1.0:
            return style_str
        
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
                    # parts[21]: MarginV (垂直间距)
                    parts[21] = str(int(float(parts[21]) * scale_factor))
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
        if not video_path.exists():
            logger.error(f"输入视频文件不存在: {video_path}")
            return False
        if not subtitle_path.exists():
            logger.error(f"字幕文件不存在: {subtitle_path}")
            return False
        
        # 初始化 NVENC 会话管理器（幂等）
        _nvenc_manager.init(max_per_gpu=max_nvenc_sessions)
        
        # GPU 设备校验（仅校验格式，不占用 session）
        if gpu_device not in ("auto",) and not gpu_device.startswith("cuda:"):
            error_msg = "Embed 阶段需要 GPU，按 GPU 原则禁止 CPU 回退"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        subtitle_ext = subtitle_path.suffix.lower()
        processed_subtitle = subtitle_path
        temp_files_to_cleanup = []  # 记录需要清理的临时文件
        
        # ========== 阶段 1: ASS 字幕预处理（CPU 密集，不占用 GPU session） ==========
        # 在获取 NVENC 会话之前完成所有 CPU 预处理工作，
        # 避免在字体渲染/自动换行期间空耗 GPU 会话槽位。
        # ============================================================================
        if subtitle_ext == '.ass' and subtitle_style:
            try:
                from vat.asr import ASRData
                from vat.asr.subtitle import get_subtitle_style, auto_wrap_ass_file
                
                # Step 1: 获取视频分辨率，用于样式缩放
                width, height = self._get_video_resolution(video_path)
                
                # Step 2: 加载并缩放样式
                style_str = get_subtitle_style(subtitle_style, style_dir=style_dir)
                if not style_str:
                    logger.warning(f"无法加载样式 '{subtitle_style}'，使用默认样式")
                    style_str = get_subtitle_style("default", style_dir=style_dir) or ""
                
                scale_factor = height / reference_height
                style_str = self._scale_ass_style(style_str, scale_factor)
                
                # Step 3: 加载字幕数据并重新生成 ASS
                asr_data = ASRData.from_subtitle_file(str(subtitle_path))
                
                # 生成临时 ASS 文件（布局固定：原文在上，译文在下）
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
                
                # Step 4: 自动换行处理（CPU 密集：逐行字体渲染测量宽度）
                processed_subtitle_path = auto_wrap_ass_file(temp_ass_path, fonts_dir=fonts_dir)
                processed_subtitle = Path(processed_subtitle_path)
                # 如果换行处理生成了新文件，也需要清理
                if processed_subtitle_path != temp_ass_path:
                    temp_files_to_cleanup.append(processed_subtitle_path)
                
            except Exception as e:
                logger.warning(f"ASS 预处理失败，使用原始文件: {e}")
                processed_subtitle = subtitle_path
        
        # 转义字幕路径（Windows路径处理）
        subtitle_path_escaped = Path(processed_subtitle).as_posix().replace(":", r"\:")
        
        # 根据字幕格式选择滤镜
        if subtitle_ext == '.ass':
            vf = f"ass='{subtitle_path_escaped}'"
            # 添加字体目录支持
            if fonts_dir:
                fonts_dir_escaped = Path(fonts_dir).as_posix().replace(":", r"\:")
                vf += f":fontsdir='{fonts_dir_escaped}'"
        else:
            vf = f"subtitles='{subtitle_path_escaped}'"
        
        # ========== 阶段 2: 获取 NVENC 会话 + 构建 ffmpeg 命令 ==========
        # 预处理完成后才获取 GPU session，最大化 session 利用率。
        # ================================================================
        
        # GPU 选择：通过 session manager 均衡分配
        if gpu_device == "auto":
            gpu_id = _nvenc_manager.select_gpu()
            logger.info(f"NVENC 会话均衡分配: 选择 GPU {gpu_id}")
        else:
            # gpu_device = "cuda:N"
            try:
                gpu_id = int(gpu_device.split(":")[1])
            except (IndexError, ValueError):
                raise ValueError(f"无效的 GPU 设备格式: {gpu_device}")
        
        # 获取 NVENC 会话槽位（阻塞等待，最多 10 分钟）
        if not _nvenc_manager.acquire(gpu_id, timeout=600):
            raise RuntimeError(
                f"NVENC 会话获取超时: GPU {gpu_id}，"
                f"所有 {max_nvenc_sessions} 个槽位已满且 10 分钟内未释放"
            )
        
        # 检查是否支持 NVENC
        if not self._check_nvenc_support():
            _nvenc_manager.release(gpu_id)
            error_msg = "当前环境不支持 NVENC，按 GPU 原则禁止 CPU 回退"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 自动选择硬件编码器
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

        # 优化：获取原视频码率以控制输出体积
        video_info = self.get_video_info(video_path)
        original_bitrate = video_info.get('bit_rate', 0) if video_info else 0

        if original_bitrate > 0:
            # 使用受限码率模式 (VBR)，目标码率设为原视频的 1.1 倍，最大 1.5 倍
            target_bitrate = int(original_bitrate * 1.1)
            max_bitrate = int(original_bitrate * 1.5)
            codec_params = [
                '-rc', 'vbr',              # 变码率模式
                '-cq', str(crf),           # 目标质量
                '-b:v', str(target_bitrate),
                '-maxrate', str(max_bitrate),
                '-bufsize', str(max_bitrate * 2),
                '-preset', preset if preset.startswith('p') else 'p4',
            ]
        else:
            # 如果获取不到码率，回退到质量优先模式
            codec_params = [
                '-rc', 'constqp',
                '-qp', str(crf),
                '-preset', preset if preset.startswith('p') else 'p4',
            ]

        codec_params.extend([
            '-gpu', str(gpu_id),
            '-spatial_aq', '1',
            '-temporal_aq', '1'
        ])
        
        # 构建 ffmpeg 命令
        # 注意：-vf ass/subtitles 是 CPU 滤镜，帧流为:
        #   hwaccel 解码(GPU N) → 下载到 CPU → 字幕渲染 → 上传到 GPU N → NVENC 编码
        # 使用 -hwaccel_device 确保解码与编码在同一张 GPU，避免所有进程挤占 GPU 0。
        cmd = [
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
            str(output_path)
        ]
        
        try:
            # 保存 FFmpeg 输出日志到视频文件夹
            log_path = output_path.parent / "ffmpeg_embed.log"
            ffmpeg_log_lines = []
            
            # 使用 Popen 实时读取进度
            process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                
            # 实时读取输出并调用回调
            total_duration = None
            current_time = 0
            last_progress = -1
            start_time = time.time()
            
            while True:
                output_line = process.stderr.readline()
                if not output_line or (process.poll() is not None):
                    break
                
                # 保存所有输出行到日志
                ffmpeg_log_lines.append(output_line)
                
                # 解析总时长
                if total_duration is None:
                    duration_match = re.search(
                        r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", output_line
                    )
                    if duration_match:
                        h, m, s = map(float, duration_match.groups())
                        total_duration = h * 3600 + m * 60 + s
                
                # 解析当前处理时间
                time_match = re.search(
                    r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})", output_line
                )
                if time_match:
                    h, m, s = map(float, time_match.groups())
                    current_time = h * 3600 + m * 60 + s
                
                # 计算进度百分比
                if total_duration:
                    progress = (current_time / total_duration) * 100
                    current_progress_int = int(progress)
                    
                    # 只有当进度增加至少 5% 时才打印，或者刚开始/结束时
                    if current_progress_int >= last_progress + 5 or (last_progress == -1 and current_progress_int == 0) or current_progress_int == 100:
                        elapsed = time.time() - start_time
                        info_str = f"{current_progress_int}%"
                        
                        if current_progress_int > 0:
                            total_estimated = elapsed / (progress / 100)
                            remaining = total_estimated - elapsed
                            info_str += f" | 耗时: {_format_time(elapsed)} | 预计剩余: {_format_time(remaining)}"
                        
                        progress_callback(info_str, "正在合成")
                        last_progress = current_progress_int
                time.sleep(0.1)
            
            if progress_callback:
                progress_callback("100", "合成完成")
            
            # 检查返回码
            return_code = process.wait()
            
            # 写入完整日志
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
                # 从已收集的日志行中提取错误信息（stderr 已被 readline 循环消费）
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
        finally:
            # 释放 NVENC 会话槽位
            _nvenc_manager.release(gpu_id)
            # 清理临时文件
            for temp_file in temp_files_to_cleanup:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception:
                    pass
    
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
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            info = json.loads(result.stdout)
            
            # 提取关键信息
            video_stream = None
            audio_stream = None
            
            for stream in info.get('streams', []):
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            format_info = info.get('format', {})
            
            return {
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'bit_rate': int(format_info.get('bit_rate', 0)),
                'video': {
                    'codec': video_stream.get('codec_name', '') if video_stream else '',
                    'width': video_stream.get('width', 0) if video_stream else 0,
                    'height': video_stream.get('height', 0) if video_stream else 0,
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else 0,
                } if video_stream else None,
                'audio': {
                    'codec': audio_stream.get('codec_name', '') if audio_stream else '',
                    'sample_rate': audio_stream.get('sample_rate', 0) if audio_stream else 0,
                    'channels': audio_stream.get('channels', 0) if audio_stream else 0,
                } if audio_stream else None,
            }
        except Exception as e:
            print(f"获取视频信息失败: {e}")
            return None
    
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
