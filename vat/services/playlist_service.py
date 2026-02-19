"""
Playlist 管理服务

提供 Playlist 的增量同步、视频排序等功能。
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set, Callable
from dataclasses import dataclass, field

from vat.models import Video, Playlist, SourceType
from vat.database import Database
from vat.downloaders import YouTubeDownloader
from vat.utils.logger import setup_logger

logger = setup_logger("playlist_service")

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
    
    def __init__(self, db: Database, downloader: Optional[YouTubeDownloader] = None):
        """
        初始化 PlaylistService
        
        Args:
            db: 数据库实例
            downloader: YouTube 下载器（用于获取 Playlist 信息）
        """
        self.db = db
        self._downloader = downloader
    
    @property
    def downloader(self) -> YouTubeDownloader:
        """延迟初始化下载器
        
        注意：裸初始化不带 cookies/remote_components，仅用于 Playlist URL 解析等轻量操作。
        完整功能（下载视频等）应由调用方传入已配置的 downloader。
        """
        if self._downloader is None:
            logger.warning("PlaylistService 使用裸初始化的 YouTubeDownloader（无 cookies/remote_components）")
            self._downloader = YouTubeDownloader()
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
        
        # 使用显式指定的 playlist_id，或回退到 yt-dlp 返回的 ID
        playlist_id = target_playlist_id or yt_playlist_id
        
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
        
        # 读取 sync_live_filter：用于从混合 playlist（如 UU uploads）中按 live_status 过滤
        # "not_live": 只保留非直播视频（排除 was_live / is_live）
        # None: 不过滤（默认）
        pl_metadata = existing_playlist.metadata or {}
        sync_live_filter = pl_metadata.get('sync_live_filter')
        
        # 获取已存在的视频 ID
        existing_video_ids = self.db.get_playlist_video_ids(playlist_id)
        callback(f"已有 {len(existing_video_ids)} 个视频")
        
        # 获取 Playlist 中的所有视频
        entries = playlist_info.get('entries', [])
        
        # 按 sync_live_filter 过滤 entries
        if sync_live_filter == 'not_live':
            # 排除直播录像（was_live / is_live），只保留普通上传视频
            original_count = len(entries)
            entries = [
                e for e in entries
                if e and e.get('live_status') not in ('was_live', 'is_live')
            ]
            callback(f"live_status 过滤: {original_count} → {len(entries)} (排除直播)")
        
        total_videos = len(entries)
        callback(f"Playlist 共 {total_videos} 个视频")
        
        new_videos = []
        existing_videos = []
        
        for index, entry in enumerate(entries, start=1):
            if entry is None:
                continue
            
            video_id = entry.get('id', '')
            if not video_id:
                continue
            
            if video_id in existing_video_ids:
                existing_videos.append(video_id)
                # 更新索引（可能顺序变化）- 使用关联表
                self.db.update_video_playlist_info(video_id, playlist_id, index)
            else:
                new_videos.append(video_id)
                
                if auto_add_videos:
                    # 检查视频是否已存在（可能在其他 playlist 中）
                    existing_video = self.db.get_video(video_id)
                    if existing_video:
                        # 视频已存在，只添加 playlist 关联
                        self.db.add_video_to_playlist(video_id, playlist_id, index)
                        callback(f"[{index}/{total_videos}] 关联已有视频: {existing_video.title[:40]}...")
                    else:
                        # 创建新视频记录
                        video = Video(
                            id=video_id,
                            source_type=SourceType.YOUTUBE,
                            source_url=f"https://www.youtube.com/watch?v={video_id}",
                            title=entry.get('title', ''),
                            playlist_id=playlist_id,
                            playlist_index=index,
                            metadata={
                                'duration': entry.get('duration', 0),
                                'uploader': entry.get('uploader', channel),
                                'thumbnail': entry.get('thumbnail', ''),
                                'upload_date': entry.get('upload_date', ''),  # YYYYMMDD 格式
                            }
                        )
                        self.db.add_video(video)
                        # 添加到关联表
                        self.db.add_video_to_playlist(video_id, playlist_id, index)
                        callback(f"[{index}/{total_videos}] 新增: {video.title[:50]}...")
        
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
        
        # 收集缺少日期的已存在视频（可能是之前同步被中断或删除playlist后重建）
        videos_missing_date = []
        if fetch_upload_dates and existing_videos:
            for vid in existing_videos:
                video = self.db.get_video(vid)
                if video and video.metadata:
                    upload_date = video.metadata.get('upload_date', '')
                    if not upload_date:
                        videos_missing_date.append(vid)
            if videos_missing_date:
                callback(f"发现 {len(videos_missing_date)} 个已存在视频缺少日期，将一并获取")
        
        # 合并需要获取日期的视频：新视频 + 缺少日期的已存在视频
        videos_to_fetch = new_videos + videos_missing_date
        
        # 如果需要获取 upload_date，并行获取详细信息
        if fetch_upload_dates and videos_to_fetch:
            max_workers = 10  # 并行获取数量（测试可用 5-10 个）
            callback(f"开始并行获取视频信息（共 {len(videos_to_fetch)} 个，{max_workers} 个并发）...")
            
            # 收集所有视频的日期信息，用于后续插值我
            date_results = []  # [(video_id, upload_date or None, is_unavailable)]
            completed_count = 0
            results_lock = threading.Lock()
            
            def fetch_video_info(vid: str) -> tuple:
                """获取单个视频的信息"""
                nonlocal completed_count
                upload_date = None
                is_unavailable = False
                video_info = None
                
                try:
                    video_info = self.downloader.get_video_info(f"https://www.youtube.com/watch?v={vid}")
                    if video_info and video_info.get('upload_date'):
                        upload_date = video_info['upload_date']
                        
                        # 立即更新基本信息到 metadata（duration、thumbnail 等）
                        video = self.db.get_video(vid)
                        if video:
                            metadata = video.metadata or {}
                            metadata['upload_date'] = upload_date
                            metadata['duration'] = video_info.get('duration', 0)
                            metadata['thumbnail'] = video_info.get('thumbnail', '')
                            metadata['uploader'] = video_info.get('uploader', '')
                            metadata['view_count'] = video_info.get('view_count', 0)
                            metadata['like_count'] = video_info.get('like_count', 0)
                            self.db.update_video(vid, metadata=metadata)
                        
                        # 异步发起 LLM 翻译（第二层异步）
                        self._submit_translate_task(vid, video_info)
                    else:
                        is_unavailable = True
                except Exception as e:
                    is_unavailable = True
                    logger.warning(f"获取视频 {vid} 信息失败: {e}")
                
                with results_lock:
                    completed_count += 1
                    if completed_count % 5 == 0 or completed_count == len(videos_to_fetch):
                        callback(f"已获取 {completed_count}/{len(videos_to_fetch)} 个视频信息...")
                
                return (vid, upload_date, is_unavailable, video_info)
            
            # 使用线程池并行获取
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch_video_info, vid): vid for vid in videos_to_fetch}
                for future in futures:
                    try:
                        vid, upload_date, is_unavailable, video_info = future.result(timeout=60)
                        date_results.append((vid, upload_date, is_unavailable))
                    except Exception as e:
                        vid = futures[future]
                        logger.warning(f"获取视频 {vid} 超时或异常: {e}")
                        date_results.append((vid, None, True))
            
            # 对获取失败的视频进行时间插值（传入 playlist_id 以利用已有视频日期作为上下文）
            callback("处理无法获取日期的视频...")
            self._interpolate_missing_dates(playlist_id, date_results, callback)
            
            callback("发布日期获取完成")
        
        # 全量重分配 upload_order_index（每次 sync 都执行，确保与时间顺序一致）
        # 放在 fetch_upload_dates 块外面：即使没有新视频，也修正已有的错误索引
        callback("分配时间顺序索引...")
        self._reassign_upload_order_indices(playlist_id, callback)
        
        return SyncResult(
            playlist_id=playlist_id,
            new_videos=new_videos,
            existing_videos=existing_videos,
            total_videos=total_videos
        )
    
    def _interpolate_missing_dates(
        self,
        playlist_id: str,
        date_results: List[tuple],
        callback: callable
    ) -> None:
        """
        对获取失败的视频进行时间插值
        
        利用 playlist 中已有视频的日期作为上下文：
        - YouTube playlist 中 playlist_index=1 是最新视频，越大越旧
        - 基于 playlist_index 顺序找到最近的已知日期进行插值
        
        Args:
            playlist_id: Playlist ID（用于查询已有视频日期上下文）
            date_results: [(video_id, upload_date or None, is_unavailable), ...]
            callback: 进度回调
        """
        # 构建完整的日期上下文：playlist 中所有已知的（非插值的）日期
        # key: video_id, value: upload_date
        known_dates = {}
        all_playlist_videos = self.db.list_videos(playlist_id=playlist_id)
        for v in all_playlist_videos:
            if v.metadata:
                date = v.metadata.get('upload_date', '')
                # 只使用非插值的可靠日期
                if date and not v.metadata.get('upload_date_interpolated'):
                    known_dates[v.id] = date
        
        # 加入本轮成功获取的日期（这些是新鲜可靠的）
        for vid, upload_date, is_unavailable in date_results:
            if upload_date:
                known_dates[vid] = upload_date
        
        # 构建 playlist_index -> upload_date 的映射（用于基于位置的插值）
        index_date_map = {}  # playlist_index -> upload_date
        for v in all_playlist_videos:
            idx = v.playlist_index or 0
            if v.id in known_dates:
                index_date_map[idx] = known_dates[v.id]
        
        for vid, upload_date, is_unavailable in date_results:
            if upload_date:
                # 已在 fetch_video_info 中更新，跳过
                continue
            if not is_unavailable:
                continue
            
            video = self.db.get_video(vid)
            if not video:
                continue
            
            my_index = video.playlist_index or 0
            interpolated_date = self._calc_interpolated_date(my_index, index_date_map)
            
            metadata = video.metadata or {}
            metadata['upload_date'] = interpolated_date
            metadata['upload_date_interpolated'] = True
            metadata['unavailable'] = True
            self.db.update_video(vid, metadata=metadata)
            callback(f"  视频 {vid}: 使用插值日期 {interpolated_date}（不可用）")
    
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
    
    def _reassign_upload_order_indices(
        self,
        playlist_id: str,
        callback: Callable[[str], None]
    ) -> None:
        """
        全量重分配 upload_order_index
        
        按 upload_date 排序所有视频，分配 1（最旧）~ N（最新）。
        每次 sync 都调用，确保索引始终与时间顺序一致。
        
        注意：YouTube 的 playlist_index 是 1=最新（每次 sync 都变），
        而 upload_order_index 是 1=最旧（稳定的时间顺序）。
        两者语义相反，不可混用。
        
        Args:
            playlist_id: Playlist ID
            callback: 进度回调
        """
        # 获取所有视频，按 upload_date 排序（最早在前）
        all_videos = self.get_playlist_videos(playlist_id, order_by="upload_date")
        if not all_videos:
            return
        
        updated_count = 0
        for i, video in enumerate(all_videos, 1):
            expected_index = i
            
            # 检查当前值是否已正确
            metadata = video.metadata or {}
            current_index = metadata.get('upload_order_index', 0)
            
            if current_index != expected_index:
                # 更新关联表
                self.db.update_playlist_video_order_index(playlist_id, video.id, expected_index)
                # 同时更新 video.metadata（上传时从这里读取）
                metadata['upload_order_index'] = expected_index
                self.db.update_video(video.id, metadata=metadata)
                updated_count += 1
        
        if updated_count > 0:
            callback(f"更新 {updated_count}/{len(all_videos)} 个视频的时间顺序索引")
            logger.info(
                f"Playlist {playlist_id}: 重分配 upload_order_index, "
                f"更新 {updated_count}/{len(all_videos)} 个"
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
        全量重分配 upload_order_index
        
        按 upload_date 排序所有视频，分配 1（最旧）~ N（最新）。
        会覆盖已有的错误索引。
        
        Args:
            playlist_id: Playlist ID
            callback: 进度回调
            
        Returns:
            {'total': N, 'updated': N}
        """
        videos = self.get_playlist_videos(playlist_id, order_by="upload_date")
        callback(f"全量重分配 {len(videos)} 个视频的时间顺序索引...")
        
        updated = 0
        for i, video in enumerate(videos, 1):
            metadata = video.metadata or {}
            if metadata.get('upload_order_index') != i:
                self.db.update_playlist_video_order_index(playlist_id, video.id, i)
                metadata['upload_order_index'] = i
                self.db.update_video(video.id, metadata=metadata)
                updated += 1
        
        callback(f"重分配完成: 更新 {updated}/{len(videos)} 个")
        logger.info(f"Playlist {playlist_id}: backfill upload_order_index - updated={updated}/{len(videos)}")
        return {'total': len(videos), 'updated': updated}
    
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
