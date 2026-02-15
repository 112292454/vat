"""
Playlist 管理 API
"""
import threading
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from vat.database import Database
from vat.models import Playlist
from vat.services import PlaylistService
from vat.utils.logger import setup_logger
from vat.web.deps import get_db

logger = setup_logger("playlist_api")

router = APIRouter(prefix="/api/playlists", tags=["playlists"])


class PlaylistResponse(BaseModel):
    """Playlist 响应"""
    id: str
    title: Optional[str]
    source_url: str
    channel: Optional[str]
    channel_id: Optional[str]
    video_count: int
    last_synced_at: Optional[str]


class AddPlaylistRequest(BaseModel):
    """添加 Playlist 请求"""
    url: str
    auto_sync: bool = True
    fetch_upload_dates: bool = True  # 默认获取发布日期（用于按时间排序）

class SyncPlaylistRequest(BaseModel):
    """同步 Playlist 请求"""
    fetch_upload_dates: bool = True  # 默认获取发布日期

class SyncResponse(BaseModel):
    """同步响应（后台任务启动）"""
    playlist_id: str
    message: str
    syncing: bool = True

class SyncResultResponse(BaseModel):
    """同步结果响应"""
    playlist_id: str
    new_videos: int
    existing_videos: int
    total_videos: int

# 全局同步状态存储
_sync_status = {}  # playlist_id -> {status, message, result}


def get_playlist_service(db: Database = Depends(get_db)) -> PlaylistService:
    from vat.downloaders import YouTubeDownloader
    from vat.config import load_config
    config = load_config()
    downloader = YouTubeDownloader(
        proxy=config.proxy.get_proxy(),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
    )
    return PlaylistService(db, downloader)


@router.get("")
async def list_playlists(db: Database = Depends(get_db)):
    """列出所有 Playlist"""
    playlists = db.list_playlists()
    
    return [
        PlaylistResponse(
            id=pl.id,
            title=pl.title,
            source_url=pl.source_url,
            channel=pl.channel,
            channel_id=pl.channel_id,
            video_count=pl.video_count or 0,
            last_synced_at=pl.last_synced_at.isoformat() if pl.last_synced_at else None
        )
        for pl in playlists
    ]


@router.get("/{playlist_id}")
async def get_playlist(
    playlist_id: str,
    service: PlaylistService = Depends(get_playlist_service)
):
    """获取 Playlist 详情及视频列表"""
    pl = service.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    videos = service.get_playlist_videos(playlist_id)
    progress = service.get_playlist_progress(playlist_id)
    
    return {
        "id": pl.id,
        "title": pl.title,
        "source_url": pl.source_url,
        "channel": pl.channel,
        "video_count": pl.video_count or 0,
        "last_synced_at": pl.last_synced_at.isoformat() if pl.last_synced_at else None,
        "metadata": pl.metadata,  # 包含 upload_config 等配置
        "progress": progress,
        "videos": [
            {
                "id": v.id,
                "title": v.title,
                "playlist_index": v.playlist_index,
            }
            for v in videos
        ]
    }


def _run_sync_in_background(
    playlist_url: str,
    playlist_id: str,
    auto_add_videos: bool,
    fetch_upload_dates: bool
):
    """后台执行同步任务"""
    from vat.config import load_config
    from vat.downloaders import YouTubeDownloader
    
    config = load_config()
    db = Database(config.storage.database_path)
    downloader = YouTubeDownloader(
        proxy=config.proxy.get_proxy(),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
    )
    service = PlaylistService(db, downloader)
    
    _sync_status[playlist_id] = {"status": "syncing", "message": "正在同步..."}
    
    try:
        result = service.sync_playlist(
            playlist_url,
            auto_add_videos=auto_add_videos,
            fetch_upload_dates=fetch_upload_dates,
            progress_callback=lambda msg: _update_sync_message(playlist_id, msg)
        )
        _sync_status[playlist_id] = {
            "status": "completed",
            "message": f"同步完成: 新增 {result.new_count} 个视频",
            "result": {
                "new_videos": result.new_count,
                "existing_videos": result.existing_count,
                "total_videos": result.total_videos
            }
        }
    except Exception as e:
        logger.error(f"同步 Playlist {playlist_id} 失败: {e}")
        _sync_status[playlist_id] = {"status": "failed", "message": str(e)}


def _update_sync_message(playlist_id: str, message: str):
    """更新同步进度消息"""
    if playlist_id in _sync_status:
        _sync_status[playlist_id]["message"] = message


@router.get("/{playlist_id}/sync-status")
async def get_sync_status(playlist_id: str):
    """获取同步状态"""
    status = _sync_status.get(playlist_id, {"status": "idle", "message": ""})
    return status


@router.post("", response_model=SyncResponse)
async def add_playlist(
    request: AddPlaylistRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db)
):
    """添加 Playlist（URL），后台执行同步"""
    # 先快速获取 playlist 基本信息
    from vat.downloaders import YouTubeDownloader
    from vat.config import load_config
    
    config = load_config()
    downloader = YouTubeDownloader(
        proxy=config.proxy.get_proxy(),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
    )
    
    try:
        # 快速获取 playlist 信息（不获取视频详情）
        playlist_info = downloader.get_playlist_info(request.url)
        if not playlist_info:
            raise HTTPException(400, "无法获取 Playlist 信息")
        
        playlist_id = playlist_info['id']
        
        # 检查是否已在同步中
        if playlist_id in _sync_status and _sync_status[playlist_id].get("status") == "syncing":
            return SyncResponse(
                playlist_id=playlist_id,
                message="同步已在进行中",
                syncing=True
            )
        
        # 启动后台同步任务
        background_tasks.add_task(
            _run_sync_in_background,
            request.url,
            playlist_id,
            request.auto_sync,
            request.fetch_upload_dates
        )
        
        return SyncResponse(
            playlist_id=playlist_id,
            message="已启动后台同步",
            syncing=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/{playlist_id}/sync", response_model=SyncResponse)
async def sync_playlist(
    playlist_id: str,
    request: SyncPlaylistRequest = None,
    background_tasks: BackgroundTasks = None,
    db: Database = Depends(get_db)
):
    """同步 Playlist（增量更新），后台执行"""
    pl = db.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    # 检查是否已在同步中
    if playlist_id in _sync_status and _sync_status[playlist_id].get("status") == "syncing":
        return SyncResponse(
            playlist_id=playlist_id,
            message="同步已在进行中",
            syncing=True
        )
    
    fetch_dates = request.fetch_upload_dates if request else True
    
    # 启动后台同步任务
    background_tasks.add_task(
        _run_sync_in_background,
        pl.source_url,
        playlist_id,
        True,
        fetch_dates
    )
    
    return SyncResponse(
        playlist_id=playlist_id,
        message="已启动后台同步",
        syncing=True
    )


class PlaylistPromptRequest(BaseModel):
    """Playlist Prompt 配置请求"""
    translate_prompt: str = ""
    optimize_prompt: str = ""


@router.put("/{playlist_id}/prompt")
async def update_playlist_prompt(
    playlist_id: str,
    request: PlaylistPromptRequest,
    db: Database = Depends(get_db)
):
    """更新 Playlist 的 Custom Prompt 配置"""
    pl = db.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    metadata = pl.metadata or {}
    metadata['custom_prompt_translate'] = request.translate_prompt
    metadata['custom_prompt_optimize'] = request.optimize_prompt
    
    db.update_playlist(playlist_id, metadata=metadata)
    return {"status": "updated", "playlist_id": playlist_id}


class RefreshPlaylistRequest(BaseModel):
    """刷新 Playlist 视频信息请求"""
    force_refetch: bool = False  # 强制重新获取所有字段
    force_retranslate: bool = False  # 强制重新翻译


# 全局刷新状态存储（与 sync 独立）
_refresh_status = {}  # playlist_id -> {status, message, result}


def _update_refresh_message(playlist_id: str, message: str):
    """更新刷新进度消息"""
    if playlist_id in _refresh_status:
        _refresh_status[playlist_id]["message"] = message


def _run_refresh_in_background(
    playlist_id: str,
    force_refetch: bool,
    force_retranslate: bool
):
    """后台执行刷新任务"""
    from vat.config import load_config
    from vat.downloaders import YouTubeDownloader
    
    config = load_config()
    db = Database(config.storage.database_path)
    downloader = YouTubeDownloader(
        proxy=config.proxy.get_proxy(),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
    )
    service = PlaylistService(db, downloader)
    
    _refresh_status[playlist_id] = {"status": "refreshing", "message": "正在刷新..."}
    
    try:
        result = service.refresh_videos(
            playlist_id,
            force_refetch=force_refetch,
            force_retranslate=force_retranslate,
            callback=lambda msg: _update_refresh_message(playlist_id, msg)
        )
        _refresh_status[playlist_id] = {
            "status": "completed",
            "message": f"刷新完成: 成功 {result['refreshed']}, 失败 {result['failed']}, 跳过 {result['skipped']}",
            "result": result
        }
    except Exception as e:
        logger.error(f"刷新 Playlist {playlist_id} 失败: {e}")
        _refresh_status[playlist_id] = {"status": "failed", "message": str(e)}


@router.post("/{playlist_id}/refresh")
async def refresh_playlist_videos(
    playlist_id: str,
    request: RefreshPlaylistRequest = None,
    background_tasks: BackgroundTasks = None,
    db: Database = Depends(get_db)
):
    """
    刷新 Playlist 视频信息（补全缺失的封面、时长、日期等）
    
    默认 merge 模式：仅补全缺失字段，不破坏已有翻译结果。
    """
    pl = db.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    # 检查是否已在刷新中
    if playlist_id in _refresh_status and _refresh_status[playlist_id].get("status") == "refreshing":
        return {"status": "refreshing", "message": "刷新已在进行中"}
    
    force_refetch = request.force_refetch if request else False
    force_retranslate = request.force_retranslate if request else False
    
    background_tasks.add_task(
        _run_refresh_in_background,
        playlist_id,
        force_refetch,
        force_retranslate
    )
    
    return {"status": "started", "message": "已启动后台刷新"}


@router.get("/{playlist_id}/refresh-status")
async def get_refresh_status(playlist_id: str):
    """获取刷新状态"""
    status = _refresh_status.get(playlist_id, {"status": "idle", "message": ""})
    return status


@router.post("/{playlist_id}/retranslate")
async def retranslate_playlist_videos(
    playlist_id: str,
    background_tasks: BackgroundTasks,
    service: PlaylistService = Depends(get_playlist_service)
):
    """
    重新翻译 Playlist 中所有视频的标题/简介
    
    用于在更新翻译逻辑或提示词后，批量更新已有视频的翻译结果。
    """
    pl = service.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    def run_retranslate():
        service.retranslate_videos(playlist_id)
    
    background_tasks.add_task(run_retranslate)
    return {"status": "started", "message": "重新翻译任务已启动"}


@router.post("/{playlist_id}/backfill-index")
async def backfill_upload_order_index(
    playlist_id: str,
    service: PlaylistService = Depends(get_playlist_service)
):
    """
    为现有 Playlist 补充 upload_order_index
    
    按 upload_date 排序，为缺少索引的视频分配时间顺序索引。
    """
    pl = service.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    result = service.backfill_upload_order_index(playlist_id)
    return {"status": "completed", **result}


@router.put("/{playlist_id}/metadata")
async def update_playlist_metadata(
    playlist_id: str,
    request: Dict[str, Any],
    db: Database = Depends(get_db)
):
    """更新 Playlist 的 metadata（通用接口）"""
    pl = db.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    metadata = pl.metadata or {}
    # 合并新的 metadata 字段
    for key, value in request.items():
        if value is None:
            # 删除该字段
            metadata.pop(key, None)
        else:
            metadata[key] = value
    
    db.update_playlist(playlist_id, metadata=metadata)
    return {"status": "updated", "playlist_id": playlist_id, "metadata": metadata}


@router.delete("/{playlist_id}")
async def delete_playlist(
    playlist_id: str,
    delete_videos: bool = False,
    service: PlaylistService = Depends(get_playlist_service)
):
    """
    删除 Playlist
    
    Args:
        playlist_id: Playlist ID
        delete_videos: 是否同时删除关联的视频（默认 False）
    """
    pl = service.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    result = service.delete_playlist(playlist_id, delete_videos=delete_videos)
    return {"status": "deleted", "playlist_id": playlist_id, **result}
