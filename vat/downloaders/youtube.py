"""
YouTube下载器实现
"""
import logging
import re
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
from yt_dlp import YoutubeDL

from .base import PlatformDownloader
from vat.utils.logger import setup_logger

logger = setup_logger("downloader.youtube")

# ==================== 网络错误分类与重试 ====================

# 可重试的瞬态网络错误关键词（VPN/proxy 故障、临时网络中断）
# 这类错误等一会通常能自行恢复
_RETRYABLE_ERROR_PATTERNS = [
    r'Connection reset by peer',
    r'Connection refused',
    r'Connection aborted',
    r'Unable to connect to proxy',
    r'Failed to establish a new connection',
    r'ProxyError',
    r'NewConnectionError',
    r'Remote end closed connection',
    r'Network is unreachable',
    r'No route to host',
    r'HTTP Error 503',
    r'HTTP Error 502',
    r'503.*Service',
    r'502.*Bad Gateway',
    r'TimeoutError',
    r'Read timed out',
    r'Connection timed out',
    r'SSLError',
    r'EOF occurred',
]

# 不可重试的错误关键词（YouTube 限制、需要用户操作）
# 等待无法解决，需要更换 IP/cookie 等
_NON_RETRYABLE_ERROR_PATTERNS = [
    r'Sign in to confirm',
    r'rate[\-\s]?limit',
    r'Video unavailable',
    r'This video is private',
    r'copyright',
    r'removed by',
    r'not available.*try again later',
    r'confirm you.re not a bot',
    r'This content isn.t available',
    r'been terminated',
    r'This video has been removed',
]

# 视频永久不可用的错误模式（视频本身的问题，非环境/网络因素）
# 区别于 rate limit / sign in / bot detection 等临时性、环境性问题
_VIDEO_PERMANENTLY_UNAVAILABLE_PATTERNS = [
    r'Video unavailable',
    r'This video is private',
    r'copyright',
    r'removed by',
    r'This content isn.t available',
    r'been terminated',
    r'This video has been removed',
    r'members-only',
    r'Join this channel',
    r'This video is no longer available',
    r'This video requires payment',
]

# 重试参数
_RETRY_INITIAL_WAIT_SEC = 30     # 首次重试等待（秒）
_RETRY_MAX_WAIT_SEC = 300        # 单次最大等待（秒）
_RETRY_BACKOFF_FACTOR = 2        # 退避倍数
_RETRY_MAX_TOTAL_SEC = 1800      # 最大总等待时间（30分钟）

# 预约/首播视频错误模式（yt-dlp 在 extract_info 阶段直接抛异常）
# 这类视频尚无可下载内容，需要等待直播开始
_UPCOMING_EVENT_PATTERNS = [
    r'This live event will begin in',
    r'Premieres in',
    r'This video is not yet available',
    r'Watch this video .* it premieres',
]

# 直播等待参数
_LIVE_POLL_INTERVAL_SEC = 600    # 轮询直播状态的间隔（秒）
_LIVE_FROM_START_MAX_RETRIES = 3 # live_from_start 因 fragment 错误失败时的最大重试次数
_LIVE_FROM_START_RETRY_WAIT = 30 # live_from_start 重试间隔（秒）
_LIVE_FRAGMENT_ERROR_PATTERNS = [
    r'fragment \d+ not found',
    r'unable to continue',
]


def is_video_permanently_unavailable(error_msg: str) -> bool:
    """判断错误是否表示视频本身永久不可用（已删除/私有/会员限定等）
    
    区别于 rate limit / sign in / bot detection 等临时性环境问题。
    用于决定是否在 DB 中标记 unavailable=True。
    
    Args:
        error_msg: 错误信息字符串
        
    Returns:
        True = 视频本身不可用（永久性），False = 其他原因（临时/未知）
    """
    for pattern in _VIDEO_PERMANENTLY_UNAVAILABLE_PATTERNS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return True
    return False


@dataclass
class VideoInfoResult:
    """get_video_info 的结构化返回值
    
    清晰区分三种结果，避免调用方用 None 猜测原因：
    - status='ok': 成功获取，info 中有完整视频信息
    - status='unavailable': 视频永久不可用（已删除/私有/会员限定），不应再尝试处理
    - status='error': 临时性获取失败（网络/限流/超时），视频本身可能正常，下次 sync 可能恢复
    """
    status: str                              # 'ok' | 'unavailable' | 'error'
    info: Optional[Dict[str, Any]] = None    # 视频信息（status='ok' 时有值）
    error_message: Optional[str] = None      # 错误详情（非 ok 时有值）
    
    @property
    def ok(self) -> bool:
        return self.status == 'ok'
    
    @property
    def is_unavailable(self) -> bool:
        return self.status == 'unavailable'
    
    @property
    def upload_date(self) -> Optional[str]:
        """便捷取 upload_date，失败时返回 None"""
        if self.info:
            return self.info.get('upload_date') or None
        return None


def is_upcoming_event_error(error_msg: str) -> bool:
    """判断错误是否表示视频为预约/首播状态（尚未开始）
    
    yt-dlp 对 upcoming 直播/首播视频在 extract_info 阶段直接抛异常，
    此函数用于从异常信息中识别这类情况。
    
    Args:
        error_msg: 错误信息字符串
        
    Returns:
        True = 视频为预约/首播，尚未开始
    """
    for pattern in _UPCOMING_EVENT_PATTERNS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return True
    return False


class LiveStreamError(RuntimeError):
    """视频正在直播中，无法下载。
    
    YouTube 直播由 _download_live_stream 内部处理（live_from_start → 等待结束 → VOD），
    不会抛出此异常。此类保留用于非 YouTube 平台的直播场景。
    调用方应捕获此异常并将视频标记为"待重试"而非"失败"。
    """
    pass


class VideoUpcomingError(RuntimeError):
    """视频为预约/首播状态（is_upcoming），尚无实际内容可下载。
    
    YouTube 预约直播或首播（premiere）在开始前 live_status='is_upcoming'，
    此时没有可下载的视频流。
    调用方应捕获此异常并将视频标记为不可用（可后续重试）。
    """
    pass


def _is_retryable_network_error(error_msg: str) -> bool:
    """判断错误是否为可重试的瞬态网络问题
    
    优先检查不可重试模式（YouTube 限制），再检查可重试模式（网络瞬态故障）。
    未匹配任何模式时返回 False（不重试，正常报错）。
    
    Args:
        error_msg: 错误信息字符串
        
    Returns:
        True = 可重试（VPN/proxy 故障等），False = 不可重试（立即失败）
    """
    # 先排除不可重试的错误（优先级更高）
    for pattern in _NON_RETRYABLE_ERROR_PATTERNS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return False
    
    # 再匹配可重试的网络错误
    for pattern in _RETRYABLE_ERROR_PATTERNS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return True
    
    return False


class YtDlpLogger:
    """yt-dlp 日志适配器
    
    根据 logger 的实际级别动态决定输出：
    - DEBUG 级别：输出所有 yt-dlp 日志（含内部调试、下载进度等）
    - INFO 级别：只输出关键信息（下载目标、完成、错误），过滤进度刷屏
    """
    # 需要降级为 debug 的 warning 关键词（非关键性 warning）
    _WARNING_TO_DEBUG_KEYWORDS = [
        'No supported JavaScript runtime',
        'JavaScript runtime',
        'SABR streaming',
        'Some web_safari client https formats have been skipped',
        'Some web client https formats have been skipped',
        'formats have been skipped',
        'has been deprecated',
    ]
    
    # info 级别下值得保留的关键信息模式
    _INFO_KEEP_PREFIXES = [
        '[download] Destination:',
        '[download] 100%',
        '[merger]',
        '[ffmpeg]',
    ]
    
    def debug(self, msg):
        # DEBUG 级别：输出所有（含 [debug] 前缀的 yt-dlp 内部调试）
        # INFO 级别：全部静默
        logger.debug(msg)

    def info(self, msg):
        # DEBUG 级别：全量输出
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[yt-dlp] {msg}")
            return
        # INFO 级别：只保留关键信息
        for prefix in self._INFO_KEEP_PREFIXES:
            if msg.startswith(prefix):
                logger.info(msg)
                return
        # 其余静默（进度百分比、格式选择等）

    def warning(self, msg):
        # 将常见的非关键 warning 降级为 debug
        msg_lower = msg.lower()
        for keyword in self._WARNING_TO_DEBUG_KEYWORDS:
            if keyword.lower() in msg_lower:
                logger.debug(msg)
                return
        # 其他 warning 正常输出
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)

class YouTubeDownloader(PlatformDownloader):
    """YouTube视频下载器"""
    
    @property
    def guaranteed_fields(self) -> set:
        return {'title', 'duration', 'description', 'uploader', 'thumbnail'}
    
    def __init__(self, proxy: str = None, video_format: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
                 cookies_file: str = "", remote_components: List[str] = None):
        """
        初始化YouTube下载器
        
        Args:
            proxy: 代理地址（可选，由调用方从 config.get_stage_proxy("downloader") 传入）
            video_format: 视频格式选择
            cookies_file: cookie 文件路径（Netscape 格式），解决 YouTube bot 检测
            remote_components: yt-dlp 远程组件列表，如 ["ejs:github"]，解决 JS challenge
        """
        self.proxy = proxy or ""
        self.video_format = video_format
        self.cookies_file = cookies_file or ""
        self.remote_components = remote_components or []
        
        # 编译URL正则表达式
        self.video_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
        )
        self.playlist_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)'
        )
        self.channel_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?youtube\.com/(?:c/|channel/|user/|@)([a-zA-Z0-9_-]+)'
        )
    
    def _get_ydl_opts(
        self, 
        output_dir: Path, 
        extract_info_only: bool = False,
        download_subs: bool = False,
        sub_langs: List[str] = None
    ) -> Dict[str, Any]:
        """获取yt-dlp配置"""
        opts = {
            'format': self.video_format,
            'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': extract_info_only,
            'logger': YtDlpLogger(),  # 使用自定义日志记录器
            'progress_hooks': [], # 可以添加进度回调
        }
        
        if self.proxy:
            opts['proxy'] = self.proxy
        
        # Cookie 认证（解决 YouTube bot 检测 / 限流）
        if self.cookies_file:
            cookie_path = Path(self.cookies_file)
            if cookie_path.exists():
                opts['cookiefile'] = str(cookie_path)
            else:
                logger.warning(f"配置的 cookie 文件不存在: {self.cookies_file}")
        
        # 远程组件（解决 YouTube JS challenge，如 n 参数解密）
        if self.remote_components:
            opts['remote_components'] = self.remote_components
        
        # 字幕下载配置
        if download_subs:
            opts['writeautomaticsub'] = True  # 下载自动生成字幕
            opts['writesubtitles'] = True     # 下载手动上传字幕
            opts['subtitleslangs'] = sub_langs or ['ja', 'zh', 'en']  # 配置的语言列表
            opts['subtitlesformat'] = 'vtt'   # VTT 格式
            # 字幕下载失败时不中止整个流程（避免 429 等临时错误阻断视频下载）
            opts['ignoreerrors'] = True
        
        return opts
    
    def download(
        self, 
        url: str, 
        output_dir: Path,
        download_subs: bool = True,
        sub_langs: List[str] = None
    ) -> Dict[str, Any]:
        """
        下载YouTube视频
        
        Args:
            url: YouTube视频URL
            output_dir: 输出目录
            download_subs: 是否下载字幕（默认True）
            sub_langs: 字幕语言列表（默认 ['ja', 'ja-orig', 'en']）
            
        Returns:
            下载信息字典，包含 video_path, title, metadata, subtitles
        """
        assert url and isinstance(url, str), "调用契约错误: url 必须是非空字符串"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ydl_opts = self._get_ydl_opts(output_dir, download_subs=download_subs, sub_langs=sub_langs)
        
        # ====== Phase 1: 提取视频信息（带网络重试） ======
        # 预约/首播视频：yt-dlp 在 extract_info 就会抛 VideoUpcomingError，
        # 此时阻塞轮询等待视频开始，类似直播等待机制
        upcoming_poll_count = 0
        while True:
            try:
                info = self._extract_info_with_retry(url, ydl_opts)
                break  # 成功获取 info，跳出等待循环
            except VideoUpcomingError as e:
                upcoming_poll_count += 1
                logger.info(
                    f"[预约] 视频尚未开始: {e}. "
                    f"第 {upcoming_poll_count} 次等待，"
                    f"{_LIVE_POLL_INTERVAL_SEC}s 后重试..."
                )
                time.sleep(_LIVE_POLL_INTERVAL_SEC)
                # 不设上限：预约视频可能在数小时后开始，持续等待直到可用
        
        video_id = info.get('id', '')
        if not video_id:
            raise RuntimeError(f"视频信息中缺少ID: {url}")
        
        title = info.get('title', '')
        if not title:
            raise RuntimeError(f"yt-dlp 未返回视频标题: {url}")
        description = info.get('description', '')
        duration = info.get('duration', 0)
        uploader = info.get('uploader', '')
        if not uploader:
            logger.warning(f"yt-dlp 未返回 uploader: {url}，翻译质量可能受影响")
        upload_date = info.get('upload_date', '')
        
        if upcoming_poll_count > 0:
            logger.info(f"[预约] 视频已开始！等待了 {upcoming_poll_count} 次轮询，继续下载流程")
        
        # 记录可用字幕信息
        available_subs = list(info.get('subtitles', {}).keys())
        available_auto_subs = list(info.get('automatic_captions', {}).keys())
        
        # ====== Phase 2: 下载视频和字幕 ======
        live_status = info.get('live_status')
        is_live = (live_status == 'is_live' or info.get('is_live'))
        
        if is_live:
            # 直播中：先尝试 live_from_start，失败则等待直播结束后下载
            logger.info(f"检测到直播中（live_status={live_status}），尝试从开头下载...")
            self._download_live_stream(url, output_dir, ydl_opts, video_id, download_subs, sub_langs)
        else:
            logger.info(f"开始下载视频: {title}")
            if download_subs and (available_subs or available_auto_subs):
                logger.info(f"同时下载字幕 - 手动: {available_subs[:5]}, 自动: {available_auto_subs[:5]}...")
            self._download_with_retry(url, ydl_opts, video_id)
        
        # 查找下载的视频文件
        video_path = None
        for ext in ['mp4', 'webm', 'mkv']:
            potential_path = output_dir / f"{video_id}.{ext}"
            if potential_path.exists():
                video_path = potential_path
                break
        
        if video_path is None:
            raise FileNotFoundError(
                f"下载完成但找不到视频文件: {video_id} 在 {output_dir}，"
                f"可能是输出格式不匹配（当前查找: mp4/webm/mkv）"
            )
        
        if video_path.stat().st_size == 0:
            raise RuntimeError(f"下载的视频文件大小为0: {video_path}")
        
        # 查找下载的字幕文件
        subtitles = {}
        if download_subs:
            for sub_file in output_dir.glob(f"{video_id}.*.vtt"):
                # 文件名格式: {video_id}.{lang}.vtt
                lang = sub_file.stem.replace(f"{video_id}.", "")
                subtitles[lang] = sub_file
                logger.info(f"已下载字幕: {lang} -> {sub_file.name}")
        
        return {
            'video_path': video_path,
            'title': title,
            'subtitles': subtitles,  # {lang: Path}
            'metadata': {
                'video_id': video_id,
                'description': description,
                'duration': duration,
                'uploader': uploader,
                'upload_date': upload_date,
                'thumbnail': info.get('thumbnail', ''),
                'url': url,
                'available_subtitles': available_subs,
                'available_auto_subtitles': available_auto_subs,
            }
        }
    
    def _extract_info_with_retry(self, url: str, ydl_opts: dict) -> dict:
        """提取视频信息，遇到瞬态网络错误时自动等待重试
        
        对于 VPN/proxy 故障等可重试错误，在当前线程内等待并重试，
        而不是立即失败让调度器跳到下一个视频（因为下一个也会失败）。
        
        对于 YouTube 限流/风控等不可重试错误，立即抛出异常。
        
        Args:
            url: 视频 URL
            ydl_opts: yt-dlp 配置
            
        Returns:
            视频信息字典
            
        Raises:
            RuntimeError: 不可重试错误或重试耗尽
        """
        total_waited = 0
        wait_sec = _RETRY_INITIAL_WAIT_SEC
        
        while True:
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    logger.info(f"正在提取视频信息: {url}")
                    info = ydl.extract_info(url, download=False)
                
                if info is None:
                    raise RuntimeError(f"无法获取视频信息: {url}")
                
                return info
                
            except Exception as e:
                error_msg = str(e)
                
                # 调试：输出实际异常类型和消息
                logger.debug(f"捕获异常: type={type(e).__name__}, msg={error_msg[:200]}")
                
                # 预约/首播视频：yt-dlp 在 extract_info 就会抛异常，
                # 必须在网络重试判断之前检测，否则会被包装为通用 RuntimeError
                if is_upcoming_event_error(error_msg):
                    raise VideoUpcomingError(
                        f"视频为预约/首播状态，尚未开始: {error_msg}"
                    ) from e
                
                # 调试：如果不是 upcoming，输出为什么没匹配
                if "live event will begin" in error_msg.lower() or "premieres in" in error_msg.lower():
                    logger.warning(f"Upcoming 模式匹配失败，实际消息: {error_msg}")
                
                if not _is_retryable_network_error(error_msg):
                    # 不可重试（YouTube 限制等），立即失败
                    raise RuntimeError(f"无法获取视频信息: {url}") from e
                
                # 可重试的网络错误
                if total_waited >= _RETRY_MAX_TOTAL_SEC:
                    raise RuntimeError(
                        f"网络错误持续 {total_waited // 60} 分钟未恢复，放弃重试。"
                        f"最后错误: {error_msg}"
                    ) from e
                
                logger.warning(
                    f"[网络瞬态错误] {error_msg[:120]}... "
                    f"等待 {wait_sec}s 后重试（已等待 {total_waited}s/{_RETRY_MAX_TOTAL_SEC}s）"
                )
                time.sleep(wait_sec)
                total_waited += wait_sec
                wait_sec = min(wait_sec * _RETRY_BACKOFF_FACTOR, _RETRY_MAX_WAIT_SEC)
    
    def _download_with_retry(self, url: str, ydl_opts: dict, video_id: str) -> None:
        """下载视频文件，遇到瞬态网络错误时自动等待重试
        
        逻辑与 _extract_info_with_retry 相同：可重试错误等待，不可重试错误立即失败。
        
        Args:
            url: 视频 URL
            ydl_opts: yt-dlp 配置
            video_id: 视频 ID（用于日志）
            
        Raises:
            RuntimeError: 不可重试错误或重试耗尽
        """
        total_waited = 0
        wait_sec = _RETRY_INITIAL_WAIT_SEC
        
        while True:
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ret_code = ydl.download([url])
                
                if ret_code != 0:
                    # yt-dlp 返回非零码，检查是否有可重试的错误
                    # 非零码可能是字幕下载失败（ignoreerrors=True 时不致命）
                    # 检查视频文件是否已存在来判断是否真的失败
                    # 这里先 return，让调用方检查文件是否存在
                    logger.warning(f"yt-dlp 返回码 {ret_code}，检查文件是否已下载")
                return
                
            except Exception as e:
                error_msg = str(e)
                
                if not _is_retryable_network_error(error_msg):
                    raise RuntimeError(
                        f"视频下载失败: {error_msg}"
                    ) from e
                
                if total_waited >= _RETRY_MAX_TOTAL_SEC:
                    raise RuntimeError(
                        f"下载网络错误持续 {total_waited // 60} 分钟未恢复，放弃重试。"
                        f"视频: {video_id}，最后错误: {error_msg}"
                    ) from e
                
                logger.warning(
                    f"[下载网络错误] {error_msg[:120]}... "
                    f"等待 {wait_sec}s 后重试（已等待 {total_waited}s/{_RETRY_MAX_TOTAL_SEC}s）"
                )
                time.sleep(wait_sec)
                total_waited += wait_sec
                wait_sec = min(wait_sec * _RETRY_BACKOFF_FACTOR, _RETRY_MAX_WAIT_SEC)
    
    def _download_live_stream(
        self, url: str, output_dir: Path, ydl_opts: dict,
        video_id: str, download_subs: bool, sub_langs: List[str]
    ) -> None:
        """处理直播视频的下载策略
        
        三阶段策略（在 download 步骤内阻塞，对上层透明）：
        1. 尝试 live_from_start（带重试）：fragment 错误可能是 SSL/网络瞬态问题导致
           yt-dlp 误判分片不存在，因此需要重试而非一次失败就放弃
        2. 重试耗尽仍失败：阻塞轮询等待直播结束
        3. 直播结束后：作为普通 VOD 下载完整视频
        
        Args:
            url: 视频 URL
            output_dir: 输出目录
            ydl_opts: 基础 yt-dlp 配置（不含 live_from_start）
            video_id: 视频 ID
            download_subs: 是否下载字幕
            sub_langs: 字幕语言列表
        """
        # === 阶段 1：尝试 live_from_start（带重试） ===
        # fragment 错误常由 SSL/网络瞬态问题导致（如 EOF occurred in violation of protocol），
        # yt-dlp 误判为分片不存在。重试通常能成功。
        live_opts = {**ydl_opts, 'live_from_start': True}
        last_error = None
        
        for attempt in range(1, _LIVE_FROM_START_MAX_RETRIES + 1):
            # 每次重试前清理上次的残留文件
            if attempt > 1:
                self._clean_partial_downloads(output_dir, video_id)
                logger.info(
                    f"[直播] live_from_start 第 {attempt}/{_LIVE_FROM_START_MAX_RETRIES} 次重试，"
                    f"等待 {_LIVE_FROM_START_RETRY_WAIT}s..."
                )
                time.sleep(_LIVE_FROM_START_RETRY_WAIT)
            
            try:
                logger.info(f"[直播] 尝试 live_from_start 从开头下载: {video_id} (第 {attempt} 次)")
                with YoutubeDL(live_opts) as ydl:
                    ret_code = ydl.download([url])
                if ret_code == 0:
                    logger.info(f"[直播] live_from_start 成功完成: {video_id}")
                    return
                # 非零返回码但没有异常，检查文件是否已存在
                for ext in ['mp4', 'webm', 'mkv']:
                    if (output_dir / f"{video_id}.{ext}").exists():
                        logger.info(f"[直播] live_from_start 返回码 {ret_code}，但视频文件已存在")
                        return
                logger.warning(f"[直播] live_from_start 返回码 {ret_code}，视频文件不存在")
                last_error = f"yt-dlp 返回码 {ret_code}"
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                if self._is_live_fragment_error(error_msg):
                    logger.warning(
                        f"[直播] live_from_start 第 {attempt} 次失败（fragment 错误，"
                        f"可能是网络瞬态问题）: {error_msg[:120]}"
                    )
                    continue  # 重试
                else:
                    # 非 fragment 错误（认证/永久性错误），不重试
                    raise RuntimeError(f"直播下载失败: {error_msg}") from e
        
        # 所有重试耗尽，进入等待模式
        logger.warning(
            f"[直播] live_from_start {_LIVE_FROM_START_MAX_RETRIES} 次重试均失败，"
            f"最后错误: {last_error[:120]}。将等待直播结束后下载完整 VOD。"
        )
        
        # 清理残留文件
        self._clean_partial_downloads(output_dir, video_id)
        
        # === 阶段 2：阻塞等待直播结束 ===
        self._wait_for_stream_end(url, video_id)
        
        # === 阶段 3：直播结束，作为普通 VOD 下载 ===
        logger.info(f"[直播] 直播已结束，开始下载完整 VOD: {video_id}")
        vod_opts = self._get_ydl_opts(output_dir, download_subs=download_subs, sub_langs=sub_langs)
        self._download_with_retry(url, vod_opts, video_id)
    
    def _wait_for_stream_end(self, url: str, video_id: str) -> None:
        """阻塞轮询等待直播结束
        
        每 _LIVE_POLL_INTERVAL_SEC 秒检查一次直播状态。
        直播结束（live_status 不再是 'is_live'）时返回。
        
        Args:
            url: 视频 URL
            video_id: 视频 ID（用于日志）
        """
        poll_count = 0
        while True:
            poll_count += 1
            logger.info(
                f"[直播] 等待直播结束... (第 {poll_count} 次轮询，"
                f"间隔 {_LIVE_POLL_INTERVAL_SEC}s)"
            )
            time.sleep(_LIVE_POLL_INTERVAL_SEC)
            
            # 重新提取视频信息检查直播状态
            try:
                check_opts = {'quiet': True, 'logger': YtDlpLogger()}
                if self.proxy:
                    check_opts['proxy'] = self.proxy
                if self.cookies_file:
                    cookie_path = Path(self.cookies_file)
                    if cookie_path.exists():
                        check_opts['cookiefile'] = str(cookie_path)
                if self.remote_components:
                    check_opts['remote_components'] = self.remote_components
                
                with YoutubeDL(check_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                
                if info is None:
                    logger.warning(f"[直播] 轮询返回 None，继续等待...")
                    continue
                
                live_status = info.get('live_status')
                is_live = (live_status == 'is_live' or info.get('is_live'))
                
                if not is_live:
                    logger.info(f"[直播] 直播已结束（live_status={live_status}）")
                    return
                
                duration = info.get('duration', 0)
                logger.info(
                    f"[直播] 仍在直播中（live_status={live_status}，"
                    f"当前时长≈{duration}s）"
                )
                
            except Exception as e:
                # 轮询失败不致命，继续等待
                logger.warning(f"[直播] 轮询状态失败: {e}，继续等待...")
    
    @staticmethod
    def _is_live_fragment_error(error_msg: str) -> bool:
        """判断错误是否为直播分片不可用（早期分片被 CDN 清除）"""
        for pattern in _LIVE_FRAGMENT_ERROR_PATTERNS:
            if re.search(pattern, error_msg, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def _clean_partial_downloads(output_dir: Path, video_id: str) -> None:
        """清理下载失败留下的残留文件（.part, .ytdl 等）"""
        for pattern in [f"{video_id}.*", f"{video_id}.*.part"]:
            for f in output_dir.glob(pattern):
                try:
                    f.unlink()
                    logger.debug(f"清理残留文件: {f.name}")
                except OSError:
                    pass
    
    def get_playlist_urls(self, playlist_url: str) -> List[str]:
        """
        获取播放列表中的所有视频URL
        
        Args:
            playlist_url: YouTube播放列表URL
            
        Returns:
            视频URL列表
        """
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'logger': YtDlpLogger(),
        }
        
        if self.proxy:
            ydl_opts['proxy'] = self.proxy
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            
            if info is None:
                return []
            
            # 处理播放列表
            if 'entries' in info:
                urls = []
                for entry in info['entries']:
                    if entry and 'id' in entry:
                        video_id = entry['id']
                        urls.append(f"https://www.youtube.com/watch?v={video_id}")
                return urls
            
            # 如果是频道，返回所有视频
            elif 'url' in info:
                return [info['url']]
            
            return []
    
    def validate_source(self, source: str) -> bool:
        """验证源是否为有效的YouTube URL"""
        return bool(
            self.video_pattern.match(source) or 
            self.playlist_pattern.match(source) or
            self.channel_pattern.match(source)
        )
    
    def validate_url(self, url: str) -> bool:
        """validate_source 的兼容别名"""
        return self.validate_source(url)
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        从URL中提取视频ID
        
        Args:
            url: YouTube视频URL
            
        Returns:
            视频ID，如果无法提取则返回None
        """
        match = self.video_pattern.match(url)
        if match:
            return match.group(1)
        
        # 尝试使用yt-dlp提取
        try:
            ydl_opts = {'quiet': True, 'extract_flat': True, 'logger': YtDlpLogger()}
            if self.proxy:
                ydl_opts['proxy'] = self.proxy
                
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info and 'id' in info:
                    return info['id']
        except:
            pass
        
        return None
    
    def get_video_info(self, url: str) -> 'VideoInfoResult':
        """
        获取视频信息（不下载）
        
        返回结构化结果，清晰区分三种情况：
        - ok: 成功获取到视频信息
        - unavailable: 视频永久不可用（已删除/私有/会员限定等）
        - error: 临时性获取失败（网络/限流/超时，视频本身可能正常）
        
        Args:
            url: YouTube视频URL
            
        Returns:
            VideoInfoResult
        """
        ydl_opts = {'quiet': True, 'logger': YtDlpLogger()}
        if self.proxy:
            ydl_opts['proxy'] = self.proxy
        # cookies 和 remote_components 与 download 路径保持一致，
        # 否则 YouTube bot 检测会拦截请求，导致 playlist sync 获取 upload_date 失败
        if self.cookies_file:
            cookie_path = Path(self.cookies_file)
            if cookie_path.exists():
                ydl_opts['cookiefile'] = str(cookie_path)
        if self.remote_components:
            ydl_opts['remote_components'] = self.remote_components
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info is None:
                    return VideoInfoResult(
                        status='error',
                        error_message='yt-dlp extract_info 返回 None（原因未知）',
                    )
                
                return VideoInfoResult(
                    status='ok',
                    info={
                        'video_id': info.get('id', ''),
                        'title': info.get('title', ''),
                        'description': info.get('description', ''),
                        'duration': info.get('duration', 0),
                        'uploader': info.get('uploader', ''),
                        'upload_date': info.get('upload_date', ''),
                        'thumbnail': info.get('thumbnail', ''),
                        'live_status': info.get('live_status'),
                        'url': url,
                    },
                )
        except Exception as e:
            error_msg = str(e)
            # 预约/首播视频：临时性错误，视频开始后会变为可用
            # 返回 error 状态，不要伪造成 ok（upcoming 会变化，不是永久失败）
            if is_upcoming_event_error(error_msg):
                logger.info(f"视频为预约/首播状态（临时）: {url} — {error_msg}")
                return VideoInfoResult(
                    status='error',
                    error_message=f"upcoming: {error_msg}",
                )
            if is_video_permanently_unavailable(error_msg):
                logger.info(f"视频永久不可用: {url} — {error_msg}")
                return VideoInfoResult(
                    status='unavailable',
                    error_message=error_msg,
                )
            else:
                logger.warning(f"获取视频信息失败（临时性）: {url} — {error_msg}")
                return VideoInfoResult(
                    status='error',
                    error_message=error_msg,
                )
    
    def check_manual_subtitles(self, url: str, target_lang: str = "ja") -> Dict[str, Any]:
        """
        检查视频是否有人工字幕（非自动生成）
        
        Args:
            url: YouTube视频URL
            target_lang: 目标语言代码（默认日语）
            
        Returns:
            字典包含:
            - has_manual_sub: bool - 是否有目标语言的人工字幕
            - manual_langs: list - 所有人工字幕语言
            - auto_langs: list - 所有自动字幕语言
            - recommended_source: str - 推荐的字幕来源 ("manual", "auto", "asr")
        """
        ydl_opts = {'quiet': True, 'logger': YtDlpLogger()}
        if self.proxy:
            ydl_opts['proxy'] = self.proxy
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info is None:
                    return {
                        'has_manual_sub': False,
                        'manual_langs': [],
                        'auto_langs': [],
                        'recommended_source': 'asr'
                    }
                
                # 人工上传的字幕
                manual_subs = info.get('subtitles', {})
                manual_langs = list(manual_subs.keys())
                
                # 自动生成的字幕
                auto_subs = info.get('automatic_captions', {})
                auto_langs = list(auto_subs.keys())
                
                # 判断是否有目标语言的人工字幕
                has_manual_target = target_lang in manual_langs
                has_auto_target = target_lang in auto_langs
                
                # 推荐来源：优先人工 > 自动 > ASR
                if has_manual_target:
                    recommended = 'manual'
                elif has_auto_target:
                    recommended = 'auto'
                else:
                    recommended = 'asr'
                
                return {
                    'has_manual_sub': has_manual_target,
                    'has_auto_sub': has_auto_target,
                    'manual_langs': manual_langs,
                    'auto_langs': auto_langs,
                    'recommended_source': recommended
                }
                
        except Exception as e:
            logger.error(f"检查字幕失败: {e}")
            return {
                'has_manual_sub': False,
                'manual_langs': [],
                'auto_langs': [],
                'recommended_source': 'asr'
            }
    
    def get_playlist_info(self, playlist_url: str) -> Optional[Dict[str, Any]]:
        """
        获取 Playlist 完整信息（用于增量同步）
        
        Args:
            playlist_url: YouTube Playlist URL
            
        Returns:
            Playlist 信息字典，包含:
            - id: Playlist ID
            - title: Playlist 标题
            - uploader: 频道名称
            - uploader_id: 频道 ID
            - entries: 视频列表（包含基本信息）
        """
        ydl_opts = {
            'extract_flat': 'in_playlist',  # 只提取 Playlist 结构，不递归
            'quiet': True,
            'logger': YtDlpLogger(),
        }
        
        if self.proxy:
            ydl_opts['proxy'] = self.proxy
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                # process=False: 跳过 yt-dlp 的 __process_playlist 内部遍历
                # 这样只做 API 分页获取（~30 次请求），不会对每个条目执行处理循环
                # 也不会产生 "Downloading item X of Y" 日志
                info = ydl.extract_info(playlist_url, download=False, process=False)
                
                if info is None:
                    logger.error(f"无法获取 Playlist 信息: {playlist_url}")
                    return None
                
                # process=False 返回的 entries 是懒迭代器（generator）
                # 迭代时触发 API 分页，每页 ~100 条目
                entries = []
                for entry in info.get('entries', []):
                    if entry is None:
                        continue
                    entries.append({
                        'id': entry.get('id', ''),
                        'title': entry.get('title', ''),
                        'duration': entry.get('duration', 0),
                        'uploader': entry.get('uploader', info.get('uploader', '')),
                        'thumbnail': entry.get('thumbnail', ''),
                        'upload_date': entry.get('upload_date', ''),
                        'live_status': entry.get('live_status'),
                    })
                
                logger.info(f"Playlist 列表获取完成: {len(entries)} 个条目")
                
                return {
                    'id': info.get('id', ''),
                    'title': info.get('title', ''),
                    'uploader': info.get('uploader', ''),
                    'uploader_id': info.get('uploader_id', ''),
                    'channel_url': info.get('channel_url', ''),
                    'entries': entries,
                    'playlist_count': len(entries),
                }
        except Exception as e:
            logger.error(f"获取 Playlist 信息失败: {e}")
            return None
    
    def extract_playlist_id(self, url: str) -> Optional[str]:
        """
        从 URL 中提取 Playlist ID
        
        Args:
            url: YouTube Playlist URL
            
        Returns:
            Playlist ID，如果无法提取则返回 None
        """
        match = self.playlist_pattern.match(url)
        if match:
            return match.group(1)
        
        # 尝试从 URL 参数中提取
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        if 'list' in params:
            return params['list'][0]
        
        return None
    
    def is_playlist_url(self, url: str) -> bool:
        """检查 URL 是否为 Playlist URL"""
        return bool(self.playlist_pattern.match(url)) or 'list=' in url
    
    @staticmethod
    def generate_video_id_from_url(url: str) -> str:
        """
        从URL生成唯一的视频ID（使用哈希）
        用于无法直接提取ID的情况
        
        Args:
            url: 视频URL
            
        Returns:
            生成的ID
        """
        return hashlib.md5(url.encode()).hexdigest()[:16]
