"""
Playlist 管理服务

提供 Playlist 的增量同步、视频排序等功能。
"""
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set, Callable
from dataclasses import dataclass, field

from vat.models import Video, Playlist, SourceType
from vat.database import Database
from vat.downloaders import YouTubeDownloader, VideoInfoResult
from vat.utils.logger import setup_logger

# 避免循环导入，Config 仅用于类型标注
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vat.config import Config

logger = setup_logger("playlist_service")

# YouTube channel tab URL 中 tab 路径到 playlist ID 后缀的映射
# yt-dlp 对 channel tab URL 返回裸 channel ID（如 UCxxx），
# 需要追加后缀（如 -shorts）以在 DB 中区分不同 tab 的 playlist。
_CHANNEL_TAB_SUFFIX_MAP = {
    'shorts': '-shorts',
    'videos': '-videos',
    'streams': '-streams',
}

# 匹配 YouTube channel tab URL 中的 tab 部分
# 支持 /@handle/tab 和 /channel/UCxxx/tab 两种格式
_CHANNEL_TAB_RE = re.compile(
    r'youtube\.com/(?:@[^/]+|channel/[^/]+)/('
    + '|'.join(_CHANNEL_TAB_SUFFIX_MAP.keys())
    + r')(?:[/?#]|$)',
    re.IGNORECASE,
)


def resolve_playlist_id(url: str, yt_playlist_id: str) -> str:
    """从 YouTube URL 和 yt-dlp 返回的 playlist ID 推断正确的 DB playlist ID
    
    YouTube channel tab URL（如 /@channel/shorts）的 yt-dlp 返回裸 channel ID，
    需要追加对应 tab 后缀（-shorts/-videos/-streams）。
    普通 playlist URL（/playlist?list=PLxxx）不需要后缀。
    
    Args:
        url: 原始 YouTube URL
        yt_playlist_id: yt-dlp extract_info 返回的 playlist ID
        
    Returns:
        带 tab 后缀的 playlist ID（如果是 channel tab URL），否则原样返回
    """
    m = _CHANNEL_TAB_RE.search(url)
    if not m:
        return yt_playlist_id
    
    tab = m.group(1).lower()
    suffix = _CHANNEL_TAB_SUFFIX_MAP.get(tab, '')
    if suffix and not yt_playlist_id.endswith(suffix):
        return yt_playlist_id + suffix
    return yt_playlist_id


# 全局翻译线程池（限制并发避免 LLM API 过载）
_translate_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="translate_")


@dataclass
class SyncResult:
    """同步结果"""
    playlist_id: str
    new_videos: List[str]  # 新增视频 ID 列表
    existing_videos: List[str]  # 已存在视频 ID 列表
    total_videos: int
    
    @property
    def new_count(self) -> int:
        return len(self.new_videos)
    
    @property
    def existing_count(self) -> int:
        return len(self.existing_videos)


class PlaylistService:
    """Playlist 管理服务"""
    
    def __init__(self, db: Database, config: Optional['Config'] = None):
        """
        初始化 PlaylistService
        
        Args:
            db: 数据库实例
            config: VAT 配置（可选）。需要 sync/refresh 等下载操作时必须提供，
                    纯 DB 查询（get_playlist, get_playlist_videos 等）无需 config。
                    若未提供且需要下载器，会自动通过 load_config() 加载。
        """
        self.db = db
        self._config = config
        self._downloader: Optional[YouTubeDownloader] = None
    
    @property
    def downloader(self) -> YouTubeDownloader:
        """从 config 懒创建 YouTubeDownloader
        
        首次访问时根据 config 中的 proxy/cookies/remote_components 创建完整配置的下载器。
        若构造时未传 config，自动 load_config()。
        """
        if self._downloader is None:
            if self._config is None:
                from vat.config import load_config
                logger.warning("PlaylistService 未传入 config，自动 load_config()")
                self._config = load_config()
            self._downloader = YouTubeDownloader(
                proxy=self._config.get_stage_proxy("downloader"),
                video_format=self._config.downloader.youtube.format,
                cookies_file=self._config.downloader.youtube.cookies_file,
                remote_components=self._config.downloader.youtube.remote_components,
            )
        return self._downloader
    
    def sync_playlist(
        self,
        playlist_url: str,
        auto_add_videos: bool = True,
        fetch_upload_dates: bool = True,  # 默认获取，用于按时间排序
        rate_limit_delay: float = 0.0,
        progress_callback: Optional[callable] = None,
        target_playlist_id: Optional[str] = None
    ) -> SyncResult:
        """
        同步 Playlist（增量）
        
        只添加新视频，不删除已存在的视频。
        增量同步时只对新增视频获取 upload_date。
        
        Args:
            playlist_url: Playlist URL（用于从 YouTube 获取数据）
            auto_add_videos: 是否自动添加新视频到数据库
            fetch_upload_dates: 是否为每个视频单独获取 upload_date（默认开启，用于按时间排序）
            rate_limit_delay: 获取 upload_date 时的速率限制延迟（秒）
            progress_callback: 进度回调
            target_playlist_id: 显式指定 DB 中的 playlist ID。
                当 yt-dlp 返回的 playlist_id 与 DB 中的 ID 不一致时使用
                （例如 channel /videos 和 /streams tab 返回相同的 channel ID，
                但在 DB 中用后缀区分：-videos / -streams）。
                若为 None，则使用 yt-dlp 返回的 ID。
            
        Returns:
            SyncResult 同步结果
        """
        callback = progress_callback or (lambda msg: logger.info(msg))
        
        callback(f"开始同步 Playlist: {playlist_url}")
        
        # 获取 Playlist 信息
        playlist_info = self.downloader.get_playlist_info(playlist_url)
        if not playlist_info:
            raise ValueError(f"无法获取 Playlist 信息: {playlist_url}")
        
        yt_playlist_id = playlist_info['id']
        playlist_title = playlist_info.get('title', 'Unknown Playlist')
        channel = playlist_info.get('uploader', '')
        channel_id = playlist_info.get('uploader_id', '')
        
        # 使用显式指定的 playlist_id，或从 URL 推断 tab 后缀后回退到 yt-dlp ID
        # （防止 channel tab URL 创建裸 channel ID 的 playlist）
        playlist_id = target_playlist_id or resolve_playlist_id(playlist_url, yt_playlist_id)
        
        callback(f"Playlist: {playlist_title} (yt_id={yt_playlist_id}, db_id={playlist_id})")
        
        # 获取或创建 Playlist 记录
        existing_playlist = self.db.get_playlist(playlist_id)
        if existing_playlist:
            callback(f"更新已存在的 Playlist")
        else:
            callback(f"创建新 Playlist")
            existing_playlist = Playlist(
                id=playlist_id,
                title=playlist_title,
                source_url=playlist_url,
                channel=channel,
                channel_id=channel_id
            )
            self.db.add_playlist(existing_playlist)
        
        # 获取已存在的视频 ID
        existing_video_ids = self.db.get_playlist_video_ids(playlist_id)
        callback(f"已有 {len(existing_video_ids)} 个视频")
        
        # 获取 Playlist 中的所有视频
        # 注意：不在代码中按视频特征过滤。shorts/videos/streams 的区分
        # 完全依赖 YouTube tab URL（/@channel/shorts, /videos, /streams），
        # 各 tab 天然互斥且完整覆盖全部上传。
        entries = playlist_info.get('entries', [])
        total_videos = len(entries)
        callback(f"Playlist 共 {total_videos} 个视频")
        
        new_videos = []
        existing_videos = []
        new_video_candidates: Dict[str, Dict[str, Any]] = {}
        existing_playlist_updates: List[tuple[str, int]] = []
        
        for index, entry in enumerate(entries, start=1):
            if entry is None:
                continue
            
            video_id = entry.get('id', '')
            if not video_id:
                continue
            
            if video_id in existing_video_ids:
                existing_videos.append(video_id)
                # 延迟到写入阶段再更新，避免 sync 中途被打断留下半状态
                existing_playlist_updates.append((video_id, index))
            else:
                new_videos.append(video_id)
                new_video_candidates[video_id] = {
                    'entry': entry,
                    'playlist_index': index,
                    'existing_video': self.db.get_video(video_id) if auto_add_videos else None,
                }
        
        # 收集需要补抓 metadata 的已存在视频：
        # - upload_date 缺失
        # - upload_date 仍为插值值
        # - thumbnail 缺失
        # - live_status 仍停留在 is_upcoming / is_live，说明视频状态尚未稳定
        videos_needing_refresh = []
        stale_zero_index_existing_videos: Set[str] = set()
        if fetch_upload_dates and existing_videos:
            for vid in existing_videos:
                video = self.db.get_video(vid)
                if video is None:
                    continue
                if self._should_refresh_existing_video_info(video):
                    videos_needing_refresh.append(vid)
                    pv_info = self.db.get_playlist_video_info(playlist_id, vid)
                    current_index = (pv_info.get('upload_order_index') or 0) if pv_info else 0
                    if current_index == 0:
                        stale_zero_index_existing_videos.add(vid)
            if videos_needing_refresh:
                callback(f"发现 {len(videos_needing_refresh)} 个已存在视频需要补抓元信息，将一并获取")
        
        # 合并需要获取信息的视频：新视频 + 需补抓元信息的已存在视频
        videos_to_fetch = new_videos + videos_needing_refresh
        
        # 如果需要获取 upload_date，并行获取详细信息
        if fetch_upload_dates and videos_to_fetch:
            max_workers = 10  # 并行获取数量（测试可用 5-10 个）
            callback(f"开始并行获取视频信息（共 {len(videos_to_fetch)} 个，{max_workers} 个并发）...")
            
            # 收集所有视频的获取结果，用于后续插值和状态更新
            fetch_results = []  # [(video_id, VideoInfoResult)]
            completed_count = 0
            results_lock = threading.Lock()
            
            def fetch_video_info(vid: str) -> tuple:
                """获取单个视频的信息，返回结构化结果"""
                nonlocal completed_count
                
                result = self.downloader.get_video_info(f"https://www.youtube.com/watch?v={vid}")
                
                with results_lock:
                    completed_count += 1
                    if completed_count % 5 == 0 or completed_count == len(videos_to_fetch):
                        callback(f"已获取 {completed_count}/{len(videos_to_fetch)} 个视频信息...")
                
                return (vid, result)
            
            # 使用线程池并行获取
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch_video_info, vid): vid for vid in videos_to_fetch}
                for future in futures:
                    try:
                        vid, result = future.result(timeout=60)
                        fetch_results.append((vid, result))
                    except Exception as e:
                        vid = futures[future]
                        logger.warning(f"获取视频 {vid} 超时或异常: {e}")
                        fetch_results.append((vid, VideoInfoResult(
                            status='error', error_message=str(e)
                        )))
            
            pruned_new_videos, pruned_stale_existing_videos = self._classify_pruned_unavailable_videos(
                new_video_ids=set(new_videos),
                stale_zero_index_existing_videos=stale_zero_index_existing_videos,
                fetch_results=fetch_results,
                callback=callback,
            )
            pruned_videos = pruned_new_videos | pruned_stale_existing_videos
            if pruned_videos:
                new_videos = [vid for vid in new_videos if vid not in pruned_new_videos]
                existing_videos = [vid for vid in existing_videos if vid not in pruned_stale_existing_videos]
                fetch_results = [
                    (vid, result)
                    for vid, result in fetch_results
                    if vid not in pruned_videos
                ]
                existing_playlist_updates = [
                    (vid, index)
                    for vid, index in existing_playlist_updates
                    if vid not in pruned_stale_existing_videos
                ]
                for vid in pruned_new_videos:
                    new_video_candidates.pop(vid, None)

        # 到这里才真正落库：先判定可用性，再写入新增视频，避免中断留下半成品
        for vid, index in existing_playlist_updates:
            self.db.update_video_playlist_info(vid, playlist_id, index)

        if auto_add_videos:
            for vid in new_videos:
                candidate = new_video_candidates[vid]
                existing_video = candidate['existing_video']
                entry = candidate['entry']
                index = candidate['playlist_index']

                if existing_video:
                    self.db.add_video_to_playlist(vid, playlist_id, index)
                    callback(f"[{index}/{total_videos}] 关联已有视频: {existing_video.title[:40]}...")
                else:
                    video = Video(
                        id=vid,
                        source_type=SourceType.YOUTUBE,
                        source_url=f"https://www.youtube.com/watch?v={vid}",
                        title=entry.get('title', ''),
                        playlist_id=playlist_id,
                        playlist_index=index,
                        metadata=self._build_entry_metadata(entry, channel),
                    )
                    self.db.add_video(video)
                    self.db.add_video_to_playlist(vid, playlist_id, index)
                    callback(f"[{index}/{total_videos}] 新增: {video.title[:50]}...")

        if fetch_upload_dates and videos_to_fetch:
            for vid in pruned_stale_existing_videos:
                self._remove_stale_unavailable_video(playlist_id, vid, callback)

            for vid, result in fetch_results:
                if result.ok and result.upload_date:
                    self._apply_video_info_to_db(vid, result.info)
                    self._submit_translate_task(vid, result.info)
            
            # 对未成功获取日期的视频进行插值 + 标记永久不可用
            callback("处理无法获取日期的视频...")
            self._process_failed_fetches(playlist_id, fetch_results, callback)
            
            callback("发布日期获取完成")
        
        # 为新视频分配 upload_order_index（增量式，不全量重排）
        # 只处理 index=0 的新视频，从 max(existing)+1 开始
        callback("分配时间顺序索引...")
        self._assign_indices_to_new_videos(playlist_id, callback)

        # 更新 Playlist 信息
        # video_count 使用关联表实际数量，而非 yt-dlp 本次返回的 entries 数量
        # （增量同步时 entries 只包含本次获取到的视频，不代表全量）
        actual_video_count = len(self.db.get_playlist_video_ids(playlist_id))
        self.db.update_playlist(
            playlist_id,
            title=playlist_title,
            video_count=actual_video_count,
            last_synced_at=datetime.now()
        )

        callback(f"同步完成: 新增 {len(new_videos)} 个, 已存在 {len(existing_videos)} 个")
        
        return SyncResult(
            playlist_id=playlist_id,
            new_videos=new_videos,
            existing_videos=existing_videos,
            total_videos=total_videos
        )

    def _build_entry_metadata(self, entry: Dict[str, Any], channel: str) -> Dict[str, Any]:
        """将 playlist flat entry 转为视频初始 metadata"""
        thumb = entry.get('thumbnail') or ''
        if not thumb:
            thumbs_list = entry.get('thumbnails')
            if thumbs_list and isinstance(thumbs_list, list):
                thumb = thumbs_list[-1].get('url', '') if thumbs_list[-1] else ''

        entry_meta = {
            'duration': entry.get('duration') or 0,
            # flat extract 的 entry 中 uploader key 存在但值为 None，
            # dict.get('uploader', channel) 不会 fallback，必须用 or
            'uploader': entry.get('uploader') or channel,
            'thumbnail': thumb,
            'upload_date': entry.get('upload_date') or '',
        }
        if entry.get('description'):
            entry_meta['description'] = entry['description']
        if entry.get('live_status'):
            entry_meta['live_status'] = entry['live_status']
        if entry.get('release_timestamp'):
            entry_meta['release_timestamp'] = entry['release_timestamp']
        return entry_meta

    @staticmethod
    def _should_refresh_existing_video_info(video: Video) -> bool:
        """判断已存在视频是否需要在 sync 阶段重新补抓元信息。

        关键点：第一次失败后会写入插值日期，因此不能只看 upload_date 是否为空。
        """
        metadata = video.metadata or {}
        if not metadata.get('upload_date'):
            return True
        if metadata.get('upload_date_interpolated'):
            return True
        if not metadata.get('thumbnail'):
            return True
        if metadata.get('live_status') in {'is_upcoming', 'is_live'}:
            return True
        return False

    def _apply_video_info_to_db(self, video_id: str, info: Dict[str, Any]) -> None:
        """将 get_video_info 的成功结果回写到数据库"""
        video = self.db.get_video(video_id)
        if not video:
            return

        metadata = video.metadata or {}
        metadata['upload_date'] = info.get('upload_date') or ''
        metadata['duration'] = info.get('duration') or 0
        metadata['thumbnail'] = info.get('thumbnail') or ''
        new_uploader = info.get('uploader') or ''
        if new_uploader:
            metadata['uploader'] = new_uploader
        new_live_status = info.get('live_status')
        if new_live_status:
            metadata['live_status'] = new_live_status
        metadata['view_count'] = info.get('view_count') or 0
        metadata['like_count'] = info.get('like_count') or 0
        metadata.pop('unavailable', None)
        metadata.pop('unavailable_reason', None)
        metadata.pop('upload_date_interpolated', None)
        self.db.update_video(video_id, metadata=metadata)

    def _classify_pruned_unavailable_videos(
        self,
        new_video_ids: Set[str],
        stale_zero_index_existing_videos: Set[str],
        fetch_results: List[tuple],
        callback: Callable[[str], None],
    ) -> tuple[Set[str], Set[str]]:
        """识别本轮 sync 里应剔除的永久不可用视频

        返回两类集合：
        - pruned_new_videos: 本轮新发现，但已判定永久不可用，禁止入库
        - pruned_stale_existing_videos: 上轮中断残留的 zero-index 旧记录，本轮应清理
        """
        pruned_new_videos: Set[str] = set()
        pruned_stale_existing_videos: Set[str] = set()

        for vid, result in fetch_results:
            if not result.is_unavailable:
                continue

            if vid in new_video_ids:
                pruned_new_videos.add(vid)
            if vid in stale_zero_index_existing_videos:
                pruned_stale_existing_videos.add(vid)

        if pruned_new_videos:
            callback(f"  过滤永久不可用的新视频: {len(pruned_new_videos)} 个")
        if pruned_stale_existing_videos:
            callback(f"  清理上次中断遗留的永久不可用零索引视频: {len(pruned_stale_existing_videos)} 个")

        return pruned_new_videos, pruned_stale_existing_videos

    def _remove_stale_unavailable_video(
        self,
        playlist_id: str,
        video_id: str,
        callback: Callable[[str], None],
    ) -> None:
        """清理上次同步中断遗留的永久不可用零索引视频"""
        self.db.remove_video_from_playlist(video_id, playlist_id)
        remaining_playlists = self.db.get_video_playlists(video_id)
        if remaining_playlists:
            callback(f"  视频 {video_id}: 清理当前 playlist 残留关联，保留其他 playlist 关联")
            return

        self.db.delete_video(video_id)
        callback(f"  视频 {video_id}: 清理上次中断遗留的无效记录")
    
    def _process_failed_fetches(
        self,
        playlist_id: str,
        fetch_results: List[tuple],
        callback: callable
    ) -> None:
        """
        处理未成功获取信息的视频：日期插值 + 状态标记
        
        根据 VideoInfoResult.status 区分处理：
        - ok: 已在 fetch_video_info 中更新，此处跳过
        - unavailable: 标记 unavailable=True + 日期插值
        - error: 仅日期插值，不标记 unavailable（下次 sync 可能恢复）
        
        Args:
            playlist_id: Playlist ID（用于查询已有视频日期上下文）
            fetch_results: [(video_id, VideoInfoResult), ...]
            callback: 进度回调
        """
        # 构建完整的日期上下文：playlist 中所有已知的（非插值的）日期
        known_dates = {}
        all_playlist_videos = self.db.list_videos(playlist_id=playlist_id)
        for v in all_playlist_videos:
            if v.metadata:
                date = v.metadata.get('upload_date', '')
                if date and not v.metadata.get('upload_date_interpolated'):
                    known_dates[v.id] = date
        
        # 加入本轮成功获取的日期
        for vid, result in fetch_results:
            if result.upload_date:
                known_dates[vid] = result.upload_date
        
        # 构建 playlist_index -> upload_date 映射（用于插值）
        index_date_map = {}
        for v in all_playlist_videos:
            idx = v.playlist_index or 0
            if v.id in known_dates:
                index_date_map[idx] = known_dates[v.id]
        
        # 处理每个非 ok 的结果
        unavailable_count = 0
        error_count = 0
        for vid, result in fetch_results:
            if result.ok and result.upload_date:
                continue  # 已在 fetch_video_info 中更新
            
            video = self.db.get_video(vid)
            if not video:
                continue
            
            my_index = video.playlist_index or 0
            interpolated_date = self._calc_interpolated_date(my_index, index_date_map)
            
            metadata = video.metadata or {}
            metadata['upload_date'] = interpolated_date
            metadata['upload_date_interpolated'] = True
            
            if result.is_unavailable:
                # 视频永久不可用：标记 unavailable
                metadata['unavailable'] = True
                metadata['unavailable_reason'] = result.error_message
                self.db.update_video(vid, metadata=metadata)
                callback(f"  视频 {vid}: 永久不可用 ({result.error_message})")
                unavailable_count += 1
            else:
                # 临时性获取失败：仅插值日期，不标记 unavailable
                self.db.update_video(vid, metadata=metadata)
                callback(f"  视频 {vid}: 获取失败，使用插值日期 {interpolated_date}（下次sync可重试）")
                error_count += 1
        
        if unavailable_count or error_count:
            callback(f"  汇总: {unavailable_count} 个永久不可用, {error_count} 个临时获取失败")
    
    def _calc_interpolated_date(self, target_index: int, index_date_map: dict) -> str:
        """
        基于 playlist_index 位置计算插值日期
        
        playlist_index 规则：1=最新视频（最晚日期），越大越旧（越早日期）
        
        Args:
            target_index: 需要插值的视频的 playlist_index
            index_date_map: {playlist_index: upload_date} 已知日期映射
        """
        if not index_date_map:
            return datetime.now().strftime('%Y%m%d')
        
        # 找最近的较新视频（index 更小 → 日期更晚）和较旧视频（index 更大 → 日期更早）
        newer_date = None  # 来自 index < target 的视频（日期晚于目标）
        older_date = None  # 来自 index > target 的视频（日期早于目标）
        
        for idx, date_str in sorted(index_date_map.items()):
            if idx < target_index:
                newer_date = date_str  # 不断更新，取最近的（index 最大的那个）
            elif idx > target_index:
                if older_date is None:
                    older_date = date_str  # 取最近的（index 最小的那个）
                break
        
        try:
            if newer_date and older_date:
                d_newer = datetime.strptime(newer_date, '%Y%m%d')
                d_older = datetime.strptime(older_date, '%Y%m%d')
                mid = d_older + (d_newer - d_older) / 2
                return mid.strftime('%Y%m%d')
            elif older_date:
                # 目标是最新视频（没有比它更新的已知日期）
                # 刚出现在 playlist 说明是近期发布，偏向今天
                d_older = datetime.strptime(older_date, '%Y%m%d')
                d_today = datetime.now()
                gap_days = (d_today - d_older).days
                offset = min(7, max(1, gap_days // 10))
                return (d_today - timedelta(days=offset)).strftime('%Y%m%d')
            elif newer_date:
                # 目标是最旧视频（没有比它更旧的已知日期）
                d_newer = datetime.strptime(newer_date, '%Y%m%d')
                return (d_newer - timedelta(days=1)).strftime('%Y%m%d')
            else:
                return datetime.now().strftime('%Y%m%d')
        except Exception:
            return datetime.now().strftime('%Y%m%d')
    
    def _assign_indices_to_new_videos(
        self,
        playlist_id: str,
        callback: Callable[[str], None]
    ) -> None:
        """
        为新视频分配 upload_order_index（增量式，不做全量重排）
        
        只处理 upload_order_index=0 的新视频，从当前最大索引+1 开始，
        按 upload_date 排序分配。已有索引的视频不变。
        
        注意：YouTube 的 playlist_index 是 1=最新（每次 sync 都变），
        而 upload_order_index 是 1=最旧（稳定的时间顺序，只增不改）。
        两者语义相反，不可混用。
        
        Args:
            playlist_id: Playlist ID
            callback: 进度回调
        """
        all_videos = self.get_playlist_videos(playlist_id, order_by="upload_date")
        if not all_videos:
            return
        
        # 找出需要分配索引的新视频和当前最大索引
        max_index = 0
        new_videos = []
        for video in all_videos:
            pv_info = self.db.get_playlist_video_info(playlist_id, video.id)
            current_index = (pv_info.get('upload_order_index') or 0) if pv_info else 0
            if current_index > 0:
                max_index = max(max_index, current_index)
            else:
                new_videos.append(video)
        
        if not new_videos:
            return
        
        # 新视频按 upload_date 排序后，从 max_index+1 开始分配
        def get_upload_date(v):
            date_str = v.metadata.get('upload_date', '') if v.metadata else ''
            return date_str if date_str else '99999999'
        new_videos.sort(key=get_upload_date)
        
        for i, video in enumerate(new_videos, max_index + 1):
            self.db.update_playlist_video_order_index(playlist_id, video.id, i)
        
        callback(f"为 {len(new_videos)} 个新视频分配索引 ({max_index + 1}-{max_index + len(new_videos)})")
        logger.info(
            f"Playlist {playlist_id}: 分配 upload_order_index 给 {len(new_videos)} 个新视频, "
            f"范围 {max_index + 1}-{max_index + len(new_videos)}"
        )
    
    def _submit_translate_task(self, video_id: str, video_info: Dict[str, Any], force: bool = False) -> None:
        """
        提交异步翻译任务
        
        在获取到 video_info 后，异步调用 LLM 翻译 title/description。
        翻译结果存入 video.metadata['translated']。
        
        Args:
            video_id: 视频 ID
            video_info: 视频信息
            force: 强制重新翻译（即使已有翻译结果）
        """
        def translate_video_info():
            try:
                from vat.config import load_config
                config = load_config()
                
                if not config.llm.is_available():
                    logger.debug(f"LLM 配置不可用，跳过视频 {video_id} 的翻译")
                    return
                
                # 检查是否已有翻译结果
                video = self.db.get_video(video_id)
                if not video:
                    return
                
                metadata = video.metadata or {}
                if 'translated' in metadata and not force:
                    logger.debug(f"视频 {video_id} 已有翻译结果，跳过")
                    return
                
                # 执行翻译
                from vat.llm.video_info_translator import VideoInfoTranslator
                vit_cfg = config.downloader.video_info_translate
                translator = VideoInfoTranslator(
                    model=vit_cfg.model or config.llm.model,
                    api_key=vit_cfg.api_key,
                    base_url=vit_cfg.base_url,
                    proxy=config.get_stage_proxy("video_info_translate") or "",
                )
                
                title = video_info.get('title')
                if not title:
                    raise ValueError(f"视频 {video_id} 的 video_info 中 title 缺失，无法翻译")
                uploader = video_info.get('uploader')
                if not uploader:
                    logger.warning(f"视频 {video_id} 的 video_info 中 uploader 缺失，翻译质量可能下降")
                description = video_info.get('description', '')
                tags = video_info.get('tags', [])
                
                translated_info = translator.translate(
                    title=title,
                    description=description,
                    tags=tags,
                    uploader=uploader
                )
                
                # 更新 metadata
                metadata['translated'] = translated_info.to_dict()
                
                # 同时存储完整视频信息冗余
                metadata['_video_info'] = {
                    'video_id': video_info.get('id', video_id),
                    'url': video_info.get('webpage_url', f"https://www.youtube.com/watch?v={video_id}"),
                    'title': title,
                    'uploader': video_info.get('uploader', ''),
                    'description': description,
                    'duration': video_info.get('duration', 0),
                    'upload_date': video_info.get('upload_date', ''),
                    'thumbnail': video_info.get('thumbnail', ''),
                    'tags': tags,
                    'width': video_info.get('width', 0),
                    'height': video_info.get('height', 0),
                }
                
                self.db.update_video(video_id, title=title, metadata=metadata)
                logger.info(f"视频 {video_id} 翻译完成")
                
            except Exception as e:
                logger.warning(f"视频 {video_id} 翻译失败: {e}")
        
        # 提交到线程池异步执行
        _translate_executor.submit(translate_video_info)
    
    def get_playlist_videos(
        self,
        playlist_id: str,
        order_by: str = "upload_date"
    ) -> List[Video]:
        """
        获取 Playlist 下的所有视频
        
        Args:
            playlist_id: Playlist ID
            order_by: 排序方式
                - "upload_date": 按发布日期（默认，最早在前，支持增量更新）
                - "playlist_index": 按 Playlist 中的顺序
                - "created_at": 按添加时间
                
        Returns:
            视频列表
        """
        videos = self.db.list_videos(playlist_id=playlist_id)
        
        if order_by == "upload_date":
            # 按发布日期排序（最早在前），无日期的排最后
            def get_upload_date(v):
                date_str = v.metadata.get('upload_date', '') if v.metadata else ''
                return date_str if date_str else '99999999'  # 无日期排最后
            videos.sort(key=get_upload_date)
        elif order_by == "playlist_index":
            videos.sort(key=lambda v: v.playlist_index or 0)
        elif order_by == "created_at":
            videos.sort(key=lambda v: v.created_at or datetime.min)
        
        return videos
    
    def get_pending_videos(
        self,
        playlist_id: str,
        target_step: Optional[str] = None
    ) -> List[Video]:
        """
        获取 Playlist 中待处理的视频
        
        Args:
            playlist_id: Playlist ID
            target_step: 目标阶段（可选），如果指定则只返回该阶段未完成的视频
            
        Returns:
            待处理视频列表（按 playlist_index 排序）
        """
        from vat.models import TaskStep, TaskStatus
        
        videos = self.get_playlist_videos(playlist_id)
        pending_videos = []
        
        for video in videos:
            pending_steps = self.db.get_pending_steps(video.id)
            
            if target_step:
                # 检查特定阶段是否未完成
                try:
                    step = TaskStep(target_step.lower())
                    if step in pending_steps:
                        pending_videos.append(video)
                except ValueError:
                    pass
            else:
                # 任何阶段未完成都算待处理
                if pending_steps:
                    pending_videos.append(video)
        
        return pending_videos
    
    def get_completed_videos(self, playlist_id: str) -> List[Video]:
        """
        获取 Playlist 中已完成所有阶段的视频
        
        Args:
            playlist_id: Playlist ID
            
        Returns:
            已完成视频列表
        """
        videos = self.get_playlist_videos(playlist_id)
        completed_videos = []
        
        for video in videos:
            pending_steps = self.db.get_pending_steps(video.id)
            if not pending_steps:
                completed_videos.append(video)
        
        return completed_videos
    
    def list_playlists(self) -> List[Playlist]:
        """列出所有 Playlist"""
        return self.db.list_playlists()
    
    def get_playlist(self, playlist_id: str) -> Optional[Playlist]:
        """获取 Playlist 信息"""
        return self.db.get_playlist(playlist_id)
    
    def delete_playlist(self, playlist_id: str, delete_videos: bool = False) -> Dict[str, Any]:
        """
        删除 Playlist
        
        Args:
            playlist_id: Playlist ID
            delete_videos: 是否同时删除关联的视频（默认 False，只解除关联）
            
        Returns:
            {"deleted_videos": N} 如果 delete_videos=True
        """
        result = {"deleted_videos": 0}
        
        if delete_videos:
            from pathlib import Path
            from vat.utils.file_ops import delete_processed_files
            
            # 获取 Playlist 关联的所有视频
            videos = self.get_playlist_videos(playlist_id)
            deleted_count = 0
            for video in videos:
                try:
                    # 删除处理产物文件（保留原始下载文件）
                    if video.output_dir:
                        output_dir = Path(video.output_dir)
                        if output_dir.exists():
                            delete_processed_files(output_dir)
                    # 删除数据库记录
                    self.db.delete_video(video.id)
                    deleted_count += 1
                    logger.info(f"已删除视频: {video.id} ({video.title})")
                except Exception as e:
                    logger.warning(f"删除视频失败: {video.id} - {e}")
            result["deleted_videos"] = deleted_count
            logger.info(f"共删除 {deleted_count} 个视频")
        
        self.db.delete_playlist(playlist_id)
        logger.info(f"已删除 Playlist: {playlist_id}")
        return result
    
    def backfill_upload_order_index(
        self,
        playlist_id: str,
        callback: Callable[[str], None] = lambda x: None
    ) -> Dict[str, Any]:
        """
        全量重分配 upload_order_index（手动修复工具，日常 sync 不调用）
        
        按 upload_date 排序所有视频，分配连续的 1（最旧）~ N（最新）。
        会覆盖已有的索引。
        
        ⚠️ 如果已有视频的索引被改变，且该视频已上传 B站，
        需要额外同步 B站标题中的 #N（本函数不自动处理）。
        
        Args:
            playlist_id: Playlist ID
            callback: 进度回调
            
        Returns:
            {'total': N, 'updated': N, 'changed_videos': [(video_id, old_idx, new_idx), ...]}
        """
        videos = self.get_playlist_videos(playlist_id, order_by="upload_date")
        callback(f"全量重分配 {len(videos)} 个视频的时间顺序索引...")
        
        updated = 0
        changed_videos = []
        for i, video in enumerate(videos, 1):
            # 从 playlist_videos 关联表读取当前值（per-playlist）
            pv_info = self.db.get_playlist_video_info(playlist_id, video.id)
            current_index = pv_info.get('upload_order_index', 0) if pv_info else 0
            if current_index != i:
                self.db.update_playlist_video_order_index(playlist_id, video.id, i)
                changed_videos.append((video.id, current_index, i))
                updated += 1
        
        callback(f"重分配完成: 更新 {updated}/{len(videos)} 个")
        logger.info(f"Playlist {playlist_id}: backfill upload_order_index - updated={updated}/{len(videos)}")
        return {'total': len(videos), 'updated': updated, 'changed_videos': changed_videos}
    
    def retranslate_videos(
        self,
        playlist_id: str,
        callback: Callable[[str], None] = lambda x: None
    ) -> Dict[str, Any]:
        """
        重新翻译 Playlist 中所有视频的标题/简介
        
        用于在更新翻译逻辑或提示词后，批量更新已有视频的翻译结果。
        
        Args:
            playlist_id: Playlist ID
            callback: 进度回调函数
            
        Returns:
            {'submitted': N, 'skipped': N}
        """
        videos = self.get_playlist_videos(playlist_id)
        submitted = 0
        skipped = 0
        
        callback(f"开始重新翻译 {len(videos)} 个视频...")
        
        for i, video in enumerate(videos, 1):
            metadata = video.metadata or {}
            
            # 跳过不可用视频
            if metadata.get('unavailable', False):
                skipped += 1
                continue
            
            # 构建 video_info
            video_info = metadata.get('_video_info', {})
            if not video_info:
                # 没有缓存的 video_info，从 metadata 构建
                video_info = {
                    'title': video.title or '',
                    'description': metadata.get('description', ''),
                    'tags': metadata.get('tags', []),
                    'uploader': metadata.get('uploader', ''),
                }
            
            if video_info.get('title') or video_info.get('description'):
                self._submit_translate_task(video.id, video_info, force=True)
                submitted += 1
            else:
                skipped += 1
            
            if i % 10 == 0:
                callback(f"已提交 {i}/{len(videos)} 个视频...")
        
        callback(f"重新翻译任务已提交: {submitted} 个, 跳过 {skipped} 个")
        return {'submitted': submitted, 'skipped': skipped}
    
    def refresh_videos(
        self,
        playlist_id: str,
        force_refetch: bool = False,
        force_retranslate: bool = False,
        callback: Callable[[str], None] = lambda x: None
    ) -> Dict[str, Any]:
        """
        刷新 Playlist 中视频的元信息（封面、时长、日期等）
        
        默认 merge 模式：仅补全缺失字段，不破坏已有数据（尤其是翻译结果）。
        
        Args:
            playlist_id: Playlist ID
            force_refetch: 强制重新获取所有字段（覆盖已有值，但默认保留 translated）
            force_retranslate: 强制重新翻译（仅在 force_refetch 时有意义）
            callback: 进度回调
            
        Returns:
            {'refreshed': N, 'skipped': N, 'failed': N}
        """
        videos = self.get_playlist_videos(playlist_id)
        if not videos:
            callback("没有视频需要刷新")
            return {'refreshed': 0, 'skipped': 0, 'failed': 0}
        
        # 筛选需要刷新的视频
        _METADATA_FIELDS = ['thumbnail', 'duration', 'upload_date', 'uploader', 'view_count', 'like_count']
        
        videos_to_refresh = []
        for v in videos:
            metadata = v.metadata or {}
            if force_refetch:
                # 强制模式：所有非 unavailable 的视频
                if not metadata.get('unavailable', False):
                    videos_to_refresh.append(v)
            else:
                # merge 模式：只刷新有缺失字段的视频
                missing = []
                for field in _METADATA_FIELDS:
                    val = metadata.get(field)
                    if val is None or val == '' or val == 0:
                        missing.append(field)
                # 也检查 title
                if not v.title:
                    missing.append('title')
                if missing and not metadata.get('unavailable', False):
                    videos_to_refresh.append(v)
        
        if not videos_to_refresh:
            callback("所有视频信息已完整，无需刷新")
            return {'refreshed': 0, 'skipped': len(videos), 'failed': 0}
        
        mode_label = "强制重新获取" if force_refetch else "补全缺失信息"
        callback(f"开始刷新 ({mode_label}): {len(videos_to_refresh)}/{len(videos)} 个视频")
        
        refreshed = 0
        failed = 0
        completed_count = 0
        results_lock = threading.Lock()
        
        def refresh_single_video(video: 'Video') -> bool:
            """刷新单个视频的元信息"""
            nonlocal completed_count
            try:
                video_info = self.downloader.get_video_info(
                    f"https://www.youtube.com/watch?v={video.id}"
                )
                if not video_info:
                    logger.warning(f"视频 {video.id} 信息获取失败（返回空）")
                    return False
                
                metadata = video.metadata or {}
                new_title = video_info.get('title', '')
                
                if force_refetch:
                    # 强制模式：覆盖所有字段
                    metadata['upload_date'] = video_info.get('upload_date', '') or metadata.get('upload_date', '')
                    metadata['duration'] = video_info.get('duration', 0) or metadata.get('duration', 0)
                    metadata['thumbnail'] = video_info.get('thumbnail', '') or metadata.get('thumbnail', '')
                    metadata['uploader'] = video_info.get('uploader', '') or metadata.get('uploader', '')
                    metadata['view_count'] = video_info.get('view_count', 0)
                    metadata['like_count'] = video_info.get('like_count', 0)
                    # 清除 interpolated 标记（如果获取到了真实日期）
                    if video_info.get('upload_date'):
                        metadata.pop('upload_date_interpolated', None)
                    # 更新缓存的 _video_info
                    metadata['_video_info'] = {
                        'video_id': video_info.get('id', video.id),
                        'url': video_info.get('webpage_url', f"https://www.youtube.com/watch?v={video.id}"),
                        'title': new_title,
                        'uploader': video_info.get('uploader', ''),
                        'description': video_info.get('description', ''),
                        'duration': video_info.get('duration', 0),
                        'upload_date': video_info.get('upload_date', ''),
                        'thumbnail': video_info.get('thumbnail', ''),
                        'tags': video_info.get('tags', []),
                        'width': video_info.get('width', 0),
                        'height': video_info.get('height', 0),
                    }
                    # 保留 translated，除非 force_retranslate
                    if force_retranslate:
                        metadata.pop('translated', None)
                    
                    title_to_save = new_title or video.title
                    self.db.update_video(video.id, title=title_to_save, metadata=metadata)
                    
                    # 重新翻译
                    if force_retranslate:
                        self._submit_translate_task(video.id, video_info, force=True)
                    elif 'translated' not in metadata:
                        # 没有翻译结果，自动翻译
                        self._submit_translate_task(video.id, video_info, force=False)
                else:
                    # merge 模式：仅填充缺失字段
                    changed = False
                    if not metadata.get('upload_date') and video_info.get('upload_date'):
                        metadata['upload_date'] = video_info['upload_date']
                        metadata.pop('upload_date_interpolated', None)
                        changed = True
                    if not metadata.get('duration') and video_info.get('duration'):
                        metadata['duration'] = video_info['duration']
                        changed = True
                    if not metadata.get('thumbnail') and video_info.get('thumbnail'):
                        metadata['thumbnail'] = video_info['thumbnail']
                        changed = True
                    if not metadata.get('uploader') and video_info.get('uploader'):
                        metadata['uploader'] = video_info['uploader']
                        changed = True
                    if metadata.get('view_count') is None:
                        metadata['view_count'] = video_info.get('view_count', 0)
                        changed = True
                    if metadata.get('like_count') is None:
                        metadata['like_count'] = video_info.get('like_count', 0)
                        changed = True
                    # 补全 _video_info 缓存
                    if '_video_info' not in metadata:
                        metadata['_video_info'] = {
                            'video_id': video_info.get('id', video.id),
                            'url': video_info.get('webpage_url', f"https://www.youtube.com/watch?v={video.id}"),
                            'title': new_title or video.title or '',
                            'uploader': video_info.get('uploader', ''),
                            'description': video_info.get('description', ''),
                            'duration': video_info.get('duration', 0),
                            'upload_date': video_info.get('upload_date', ''),
                            'thumbnail': video_info.get('thumbnail', ''),
                            'tags': video_info.get('tags', []),
                            'width': video_info.get('width', 0),
                            'height': video_info.get('height', 0),
                        }
                        changed = True
                    
                    title_to_save = video.title or new_title or None
                    if changed or (not video.title and new_title):
                        self.db.update_video(video.id, title=title_to_save, metadata=metadata)
                    
                    # 如果没有翻译结果且有 video_info，自动翻译
                    if 'translated' not in metadata:
                        self._submit_translate_task(video.id, video_info, force=False)
                
                return True
            except Exception as e:
                logger.warning(f"刷新视频 {video.id} 失败: {e}")
                return False
            finally:
                with results_lock:
                    completed_count += 1
                    if completed_count % 5 == 0 or completed_count == len(videos_to_refresh):
                        callback(f"已处理 {completed_count}/{len(videos_to_refresh)} 个视频...")
        
        # 并行获取
        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(refresh_single_video, v): v for v in videos_to_refresh}
            for future in futures:
                try:
                    success = future.result(timeout=60)
                    if success:
                        refreshed += 1
                    else:
                        failed += 1
                except Exception as e:
                    vid = futures[future].id
                    logger.warning(f"刷新视频 {vid} 超时或异常: {e}")
                    failed += 1
        
        skipped = len(videos) - len(videos_to_refresh)
        callback(f"刷新完成: 成功 {refreshed}, 失败 {failed}, 跳过 {skipped}")
        return {'refreshed': refreshed, 'skipped': skipped, 'failed': failed}
    
    def get_playlist_progress(self, playlist_id: str) -> Dict[str, Any]:
        """
        获取 Playlist 处理进度统计
        
        Returns:
            {
                'total': 总视频数,
                'completed': 全部完成数,
                'partial_completed': 部分完成数（有已完成阶段但未全部完成，且无失败）,
                'pending': 完全未处理数,
                'failed': 有失败阶段的视频数,
                'unavailable': 不可用视频数,
                'by_step': {step: {'completed': N, 'pending': N, 'failed': N}}
            }
        """
        from vat.models import TaskStep, TaskStatus, DEFAULT_STAGE_SEQUENCE
        
        videos = self.get_playlist_videos(playlist_id)
        total = len(videos)
        completed = 0
        partial_completed = 0
        failed = 0
        unavailable = 0
        by_step = {}
        
        # 初始化每个阶段的统计
        for step in DEFAULT_STAGE_SEQUENCE:
            by_step[step.value] = {'completed': 0, 'pending': 0, 'failed': 0}
        
        for video in videos:
            metadata = video.metadata or {}
            
            # 检查是否为不可用视频
            if metadata.get('unavailable', False):
                unavailable += 1
                continue  # 不可用视频不计入阶段统计
            
            tasks = self.db.get_tasks(video.id)
            task_by_step = {t.step: t for t in tasks}
            
            # 检查是否全部完成
            pending_steps = self.db.get_pending_steps(video.id)
            has_completed_step = any(t.status == TaskStatus.COMPLETED for t in tasks)
            has_failed_step = any(t.status == TaskStatus.FAILED for t in tasks)
            
            if not pending_steps:
                completed += 1
            elif has_failed_step:
                failed += 1
            elif has_completed_step:
                partial_completed += 1
            
            # 统计每个阶段
            for step in DEFAULT_STAGE_SEQUENCE:
                task = task_by_step.get(step)
                if task:
                    if task.status == TaskStatus.COMPLETED:
                        by_step[step.value]['completed'] += 1
                    elif task.status == TaskStatus.FAILED:
                        by_step[step.value]['failed'] += 1
                    else:
                        by_step[step.value]['pending'] += 1
                else:
                    by_step[step.value]['pending'] += 1
        
        # 可处理视频数 = 总数 - 不可用
        processable = total - unavailable
        
        return {
            'total': total,
            'completed': completed,
            'partial_completed': partial_completed,
            'pending': processable - completed - partial_completed - failed,
            'failed': failed,
            'unavailable': unavailable,
            'by_step': by_step
        }
