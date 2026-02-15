"""
YouTube下载器实现
"""
import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any
from yt_dlp import YoutubeDL

from .base import BaseDownloader
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

# 重试参数
_RETRY_INITIAL_WAIT_SEC = 30     # 首次重试等待（秒）
_RETRY_MAX_WAIT_SEC = 300        # 单次最大等待（秒）
_RETRY_BACKOFF_FACTOR = 2        # 退避倍数
_RETRY_MAX_TOTAL_SEC = 1800      # 最大总等待时间（30分钟）


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
    """yt-dlp 日志适配器"""
        # 需要降级为 debug 的 warning 关键词
    WARNING_TO_DEBUG_KEYWORDS = [
        'No supported JavaScript runtime',
        'JavaScript runtime',
        'SABR streaming',
        'Some web_safari client https formats have been skipped',
        'Some web client https formats have been skipped',
        'formats have been skipped',
        'has been deprecated',
    ]
    def debug(self, msg):
        # 忽略一些无用的调试信息
        if msg.startswith('[debug] '):
            return
        logger.debug(msg)

    def info(self, msg):
        # 将 yt-dlp 的 info 降级为 debug，避免刷屏
        # 除非是关键信息
        if msg.startswith('[download] Destination:'):
            logger.info(msg)
        elif msg.startswith('[download] 100%'):
            logger.info(msg)
        else:
            logger.debug(msg)

    def warning(self, msg):
        # 将常见的非关键 warning 降级为 debug
        msg_lower = msg.lower()
        for keyword in self.WARNING_TO_DEBUG_KEYWORDS:
            if keyword.lower() in msg_lower:
                logger.debug(msg)
                return
        # 其他 warning 正常输出
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)

class YouTubeDownloader(BaseDownloader):
    """YouTube视频下载器"""
    
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
        info = self._extract_info_with_retry(url, ydl_opts)
        
        video_id = info.get('id', '')
        if not video_id:
            raise RuntimeError(f"视频信息中缺少ID: {url}")
        
        title = info.get('title', 'Unknown')
        description = info.get('description', '')
        duration = info.get('duration', 0)
        uploader = info.get('uploader', '')
        upload_date = info.get('upload_date', '')
        
        # 记录可用字幕信息
        available_subs = list(info.get('subtitles', {}).keys())
        available_auto_subs = list(info.get('automatic_captions', {}).keys())
        
        # ====== Phase 2: 下载视频和字幕（带网络重试） ======
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
    
    def validate_url(self, url: str) -> bool:
        """
        验证URL是否为有效的YouTube URL
        
        Args:
            url: 要验证的URL
            
        Returns:
            是否为有效的YouTube URL
        """
        return bool(
            self.video_pattern.match(url) or 
            self.playlist_pattern.match(url) or
            self.channel_pattern.match(url)
        )
    
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
    
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        获取视频信息（不下载）
        
        Args:
            url: YouTube视频URL
            
        Returns:
            视频信息字典
        """
        ydl_opts = {'quiet': True, 'logger': YtDlpLogger()}
        if self.proxy:
            ydl_opts['proxy'] = self.proxy
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info is None:
                    return None
                
                return {
                    'video_id': info.get('id', ''),
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'url': url,
                }
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return None
    
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
                info = ydl.extract_info(playlist_url, download=False)
                
                if info is None:
                    logger.error(f"无法获取 Playlist 信息: {playlist_url}")
                    return None
                
                # 提取 entries 中的关键信息
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
                        # upload_date 在 flat 模式下可能不可用，但尝试获取
                        'upload_date': entry.get('upload_date', ''),
                    })
                
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
