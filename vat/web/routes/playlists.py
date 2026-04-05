"""
Playlist 管理 API
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from vat.database import Database
from vat.models import Playlist
from vat.services import PlaylistService
from vat.services.playlist_service import resolve_playlist_id
from vat.utils.logger import setup_logger
from vat.web.deps import get_db, get_web_config

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
    mode: str = "youtube"  # youtube / manual
    url: Optional[str] = None
    title: Optional[str] = None
    description: str = ""
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


class PlaylistVideoMembershipRequest(BaseModel):
    """playlist 成员增删请求。"""
    video_id: str

# 全局同步状态存储
# 兼容保留：历史上用内存字典记录最近 job_id。
# 当前路由已优先通过 web_jobs 反查任务，后续可彻底删除。
_sync_status = {}  # playlist_id -> {status, message, result}


def get_playlist_service(db: Database = Depends(get_db)) -> PlaylistService:
    config = get_web_config()
    return PlaylistService(db, config)


def _ensure_syncable_playlist(playlist: Playlist) -> None:
    """阻止对手动列表执行平台同步/刷新。"""
    if PlaylistService.is_manual_playlist(playlist):
        raise HTTPException(400, "手动列表不支持同步")


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


def _get_job_manager():
    """获取 JobManager 实例"""
    from vat.web.routes.tasks import get_job_manager
    return get_job_manager()


def _find_latest_playlist_job(task_type: str, playlist_id: str):
    """按 playlist_id 反查最近的后台任务。"""
    jm = _get_job_manager()
    job = jm.find_latest_job(
        task_type=task_type,
        task_params_subset={"playlist_id": playlist_id},
    )
    if job:
        return job

    # 兼容旧的进程内状态字典（过渡期保留）
    legacy_status_maps = (_sync_status, _refresh_status, _retranslate_status)
    for status_map in legacy_status_maps:
        entry = status_map.get(playlist_id)
        if entry and 'job_id' in entry:
            return jm.get_job(entry['job_id'])
    return None


def _query_job_status(task_type: str, playlist_id: str) -> dict:
    """通用的 job 状态查询（优先通过 web_jobs 反查）。"""
    jm = _get_job_manager()
    job = _find_latest_playlist_job(task_type, playlist_id)
    if not job:
        return {"status": "idle", "message": ""}

    job_id = job.job_id
    jm.update_job_status(job_id)
    job = jm.get_job(job_id)
    
    if not job:
        return {"status": "idle", "message": ""}
    
    status_map = {
        'pending': 'syncing',
        'running': 'syncing',
        'completed': 'completed',
        'failed': 'failed',
        'cancelled': 'cancelled',
    }
    
    log_lines = jm.get_log_content(job_id, tail_lines=3)
    last_msg = log_lines[-1] if log_lines else ''
    
    return {
        "status": status_map.get(job.status.value, job.status.value),
        "message": job.error or last_msg,
        "job_id": job_id,
    }


@router.get("/{playlist_id}/sync-status")
async def get_sync_status(playlist_id: str):
    """获取同步状态（通过 JobManager 查询）"""
    return _query_job_status("sync-playlist", playlist_id)


@router.post("", response_model=SyncResponse)
async def add_playlist(
    request: AddPlaylistRequest,
    db: Database = Depends(get_db),
    service: PlaylistService = Depends(get_playlist_service),
):
    """添加 Playlist（URL），通过 JobManager 后台执行同步"""
    if request.mode == "manual":
        title = (request.title or "").strip()
        if not title:
            raise HTTPException(400, "手动列表标题不能为空")
        playlist = service.create_manual_playlist(
            title=title,
            description=request.description or "",
        )
        return SyncResponse(
            playlist_id=playlist.id,
            message="已创建手动列表",
            syncing=False,
        )

    from vat.downloaders import YouTubeDownloader

    config = get_web_config()
    downloader = YouTubeDownloader(
        proxy=config.get_stage_proxy("downloader"),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
        lock_db_path=config.storage.database_path,
        download_cooldown=config.downloader.youtube.download_delay,
    )
    
    try:
        if not request.url:
            raise HTTPException(400, "YouTube Playlist URL 不能为空")
        playlist_info = downloader.get_playlist_info(request.url)
        if not playlist_info:
            raise HTTPException(400, "无法获取 Playlist 信息")
        
        # yt-dlp 对 channel tab URL（/@channel/shorts 等）返回裸 channel ID，
        # 需要根据 URL 中的 tab 追加后缀（-shorts/-videos/-streams）以区分不同 tab
        playlist_id = resolve_playlist_id(request.url, playlist_info['id'])
        
        # 检查是否已有 running job
        existing_job = _find_latest_playlist_job("sync-playlist", playlist_id)
        if existing_job:
            jm = _get_job_manager()
            jm.update_job_status(existing_job.job_id)
            ej = jm.get_job(existing_job.job_id)
            if ej and ej.status.value == 'running':
                return SyncResponse(
                    playlist_id=playlist_id,
                    message="同步已在进行中",
                    syncing=True
                )
        
        # 通过 JobManager 提交 sync-playlist 任务
        jm = _get_job_manager()
        job_id = jm.submit_job(
            video_ids=[],
            steps=['sync-playlist'],
            task_type='sync-playlist',
            task_params={'playlist_id': playlist_id, 'url': request.url},
        )
        _sync_status[playlist_id] = {'job_id': job_id}
        
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
    db: Database = Depends(get_db)
):
    """同步 Playlist（增量更新），通过 JobManager 后台执行"""
    pl = db.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    _ensure_syncable_playlist(pl)
    
    # 检查是否已有 running job
    existing_job = _find_latest_playlist_job("sync-playlist", playlist_id)
    if existing_job:
        jm = _get_job_manager()
        jm.update_job_status(existing_job.job_id)
        ej = jm.get_job(existing_job.job_id)
        if ej and ej.status.value == 'running':
            return SyncResponse(
                playlist_id=playlist_id,
                message="同步已在进行中",
                syncing=True
            )
    
    jm = _get_job_manager()
    job_id = jm.submit_job(
        video_ids=[],
        steps=['sync-playlist'],
        task_type='sync-playlist',
        task_params={'playlist_id': playlist_id},
    )
    _sync_status[playlist_id] = {'job_id': job_id}
    
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
_refresh_status = {}  # playlist_id -> {job_id: str}

# 全局重新翻译状态存储
_retranslate_status = {}  # playlist_id -> {job_id: str}


@router.post("/{playlist_id}/refresh")
async def refresh_playlist_videos(
    playlist_id: str,
    request: RefreshPlaylistRequest = None,
    db: Database = Depends(get_db)
):
    """
    刷新 Playlist 视频信息（通过 JobManager 后台执行）
    
    默认 merge 模式：仅补全缺失字段，不破坏已有翻译结果。
    """
    pl = db.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    _ensure_syncable_playlist(pl)
    
    # 检查是否已有 running job
    existing_job = _find_latest_playlist_job("refresh-playlist", playlist_id)
    if existing_job:
        jm = _get_job_manager()
        jm.update_job_status(existing_job.job_id)
        ej = jm.get_job(existing_job.job_id)
        if ej and ej.status.value == 'running':
            return {"status": "refreshing", "message": "刷新已在进行中"}
    
    force_refetch = request.force_refetch if request else False
    force_retranslate = request.force_retranslate if request else False
    
    jm = _get_job_manager()
    job_id = jm.submit_job(
        video_ids=[],
        steps=['refresh-playlist'],
        task_type='refresh-playlist',
        task_params={
            'playlist_id': playlist_id,
            'force_refetch': force_refetch,
            'force_retranslate': force_retranslate,
        },
    )
    _refresh_status[playlist_id] = {'job_id': job_id}
    
    return {"status": "started", "message": "已启动后台刷新", "job_id": job_id}


@router.get("/{playlist_id}/refresh-status")
async def get_refresh_status(playlist_id: str):
    """获取刷新状态（通过 JobManager 查询）"""
    result = _query_job_status("refresh-playlist", playlist_id)
    # 兼容原有的 refreshing 状态名
    if result.get('status') == 'syncing':
        result['status'] = 'refreshing'
    return result


@router.post("/{playlist_id}/retranslate")
async def retranslate_playlist_videos(
    playlist_id: str,
    service: PlaylistService = Depends(get_playlist_service)
):
    """
    重新翻译 Playlist 中所有视频的标题/简介（通过 JobManager 后台执行）
    """
    pl = service.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    
    jm = _get_job_manager()
    job_id = jm.submit_job(
        video_ids=[],
        steps=['retranslate-playlist'],
        task_type='retranslate-playlist',
        task_params={'playlist_id': playlist_id},
    )
    _retranslate_status[playlist_id] = {'job_id': job_id}
    
    return {"status": "started", "message": "重新翻译任务已启动", "job_id": job_id}


@router.post("/{playlist_id}/backfill-index")
async def backfill_upload_order_index(
    playlist_id: str,
    service: PlaylistService = Depends(get_playlist_service)
):
    """
    全量重分配 upload_order_index
    
    按 upload_date 排序所有视频，分配 1（最旧）~ N（最新）。
    会覆盖已有的错误索引。
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


@router.get("/{playlist_id}/available-videos")
async def list_attachable_videos(
    playlist_id: str,
    q: str = Query("", description="标题/ID/source_url 模糊搜索"),
    limit: int = Query(50, ge=1, le=200),
    service: PlaylistService = Depends(get_playlist_service),
):
    """列出当前未加入该 playlist、可手动加入的视频。"""
    pl = service.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    try:
        videos = service.list_attachable_videos(playlist_id, query=q, limit=limit)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"videos": videos}


@router.post("/{playlist_id}/videos")
async def add_existing_video_to_playlist(
    playlist_id: str,
    request: PlaylistVideoMembershipRequest,
    service: PlaylistService = Depends(get_playlist_service),
):
    """把已有视频添加到指定 playlist。"""
    pl = service.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    try:
        return service.attach_video_to_playlist(request.video_id, playlist_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/{playlist_id}/videos/{video_id}")
async def remove_video_from_playlist(
    playlist_id: str,
    video_id: str,
    service: PlaylistService = Depends(get_playlist_service),
):
    """仅移除视频与当前 playlist 的关联。"""
    pl = service.get_playlist(playlist_id)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    try:
        return service.remove_video_from_playlist(video_id, playlist_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


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
