"""
视频管理 API
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from vat.database import Database
from vat.models import Video, SourceType
from vat.web.deps import get_db

router = APIRouter(prefix="/api/videos", tags=["videos"])


class VideoResponse(BaseModel):
    """视频响应"""
    id: str
    title: Optional[str]
    source_type: str
    source_url: str
    output_dir: Optional[str]
    playlist_id: Optional[str]
    playlist_index: Optional[int]
    metadata: Optional[dict]
    created_at: Optional[str]
    tasks: List[dict]
    progress: float


class VideoListResponse(BaseModel):
    """视频列表响应"""
    videos: List[VideoResponse]
    total: int
    page: int
    per_page: int


class AddVideoRequest(BaseModel):
    """添加视频请求"""
    url: str                          # 视频源（URL 或本地路径）
    source_type: str = "auto"         # auto = 自动检测
    title: Optional[str] = None       # 可选手动标题



@router.get("", response_model=VideoListResponse)
async def list_videos(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    playlist_id: Optional[str] = None,
    status: Optional[str] = None,
    db: Database = Depends(get_db)
):
    """列出所有视频（支持分页、过滤）"""
    videos = db.list_videos(playlist_id=playlist_id)
    progress_map = db.batch_get_video_progress([video.id for video in videos]) if videos else {}
    
    # 构建响应
    video_list = []
    for video in videos:
        tasks = db.get_tasks(video.id)
        task_list = [
            {
                "step": t.step.value,
                "status": t.status.value,
                "error_message": t.error_message,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in tasks
        ]
        
        # 统一复用 database 的批量进度聚合，保持与页面主链口径一致
        progress_info = progress_map.get(video.id, {"progress": 0})
        progress = (progress_info["progress"] or 0) / 100.0
        
        video_list.append(VideoResponse(
            id=video.id,
            title=video.title,
            source_type=video.source_type.value,
            source_url=video.source_url,
            output_dir=video.output_dir,
            playlist_id=(db.get_video_playlists(video.id) or [None])[0],
            playlist_index=video.playlist_index,
            metadata=video.metadata,
            created_at=video.created_at.isoformat() if video.created_at else None,
            tasks=task_list,
            progress=progress
        ))
    
    # 按状态过滤
    if status:
        if status == "completed":
            video_list = [v for v in video_list if v.progress >= 1.0]
        elif status == "pending":
            video_list = [v for v in video_list if v.progress < 1.0]
        elif status == "failed":
            video_list = [v for v in video_list 
                         if any(t["status"] == "failed" for t in v.tasks)]
    
    # 分页
    total = len(video_list)
    start = (page - 1) * per_page
    end = start + per_page
    
    return VideoListResponse(
        videos=video_list[start:end],
        total=total,
        page=page,
        per_page=per_page
    )


@router.get("/{video_id}")
async def get_video(video_id: str, db: Database = Depends(get_db)):
    """获取视频详情"""
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    
    tasks = db.get_tasks(video_id)
    pending_steps = db.get_pending_steps(video_id)
    
    return {
        "id": video.id,
        "title": video.title,
        "source_type": video.source_type.value,
        "source_url": video.source_url,
        "output_dir": video.output_dir,
        "playlist_id": (db.get_video_playlists(video.id) or [None])[0],
        "playlist_index": video.playlist_index,
        "metadata": video.metadata,
        "created_at": video.created_at.isoformat() if video.created_at else None,
        "tasks": [
            {
                "step": t.step.value,
                "status": t.status.value,
                "error_message": t.error_message,
                "gpu_id": t.gpu_id,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in tasks
        ],
        "pending_steps": [s.value for s in pending_steps],
    }


@router.post("")
async def add_video(request: AddVideoRequest, db: Database = Depends(get_db)):
    """添加视频（URL/本地路径，支持自动检测源类型）"""
    from vat.pipeline import create_video_from_source, detect_source_type
    
    try:
        if request.source_type == "auto":
            source_type = detect_source_type(request.url)
        else:
            source_type = SourceType(request.source_type)
    except ValueError as e:
        raise HTTPException(400, f"无效的 source_type 或无法识别的视频源: {e}")
    
    try:
        video_id = create_video_from_source(
            request.url, db, source_type, title=request.title or ""
        )
        return {"video_id": video_id, "source_type": source_type.value, "status": "created"}
    except Exception as e:
        raise HTTPException(400, str(e))



@router.delete("/{video_id}")
async def delete_video(
    video_id: str, 
    delete_all_files: bool = False,
    db: Database = Depends(get_db)
):
    """
    删除视频记录及其相关任务
    
    Args:
        delete_all_files: 是否删除所有文件（包括原始下载文件）
            - False（默认）：只删除处理产物，保留原始视频/音频/字幕
            - True：删除整个输出目录
    """
    from pathlib import Path
    import shutil
    from vat.utils.file_ops import delete_processed_files
    
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    
    deleted_files = []
    
    # 处理文件删除
    if video.output_dir:
        output_dir = Path(video.output_dir)
        if output_dir.exists():
            if delete_all_files:
                # 删除整个目录
                shutil.rmtree(output_dir)
                deleted_files.append(str(output_dir))
            else:
                # 只删除处理产物，保留原始下载文件
                # 安全策略：只有明确识别为处理产物的才删除，其他一律保留
                deleted_files = delete_processed_files(output_dir)
    
    # 删除数据库记录
    db.delete_video(video_id)
    
    return {
        "status": "deleted", 
        "video_id": video_id,
        "delete_all_files": delete_all_files,
        "deleted_files": deleted_files
    }


@router.post("/{video_id}/retranslate-info")
async def retranslate_video_info(video_id: str, db: Database = Depends(get_db)):
    """重新翻译视频的标题/简介信息"""
    from vat.services.playlist_service import PlaylistService
    
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    
    metadata = video.metadata or {}
    
    # 构建 video_info：优先用缓存的 _video_info，否则从 metadata 组装
    video_info = metadata.get('_video_info', {})
    if not video_info:
        video_info = {
            'title': video.title or '',
            'description': metadata.get('description', ''),
            'tags': metadata.get('tags', []),
            'uploader': metadata.get('uploader') or '',
        }
    
    # uploader fallback：如果视频 metadata 中缺失，从所属 playlist 的 channel 获取
    if not video_info.get('uploader'):
        playlists = db.get_video_playlists(video_id)
        for pl_id in playlists:
            pl = db.get_playlist(pl_id)
            if pl and pl.channel:
                video_info['uploader'] = pl.channel
                break
    
    if not video_info.get('title'):
        raise HTTPException(400, "视频缺少标题信息，无法翻译")
    
    # 提交异步翻译任务（force=True 强制覆盖已有翻译）
    service = PlaylistService(db)
    service._submit_translate_task(video_id, video_info, force=True)
    
    return {"status": "submitted", "video_id": video_id, "message": "翻译任务已提交，刷新页面查看结果"}


@router.get("/{video_id}/files")
async def get_video_files(video_id: str, db: Database = Depends(get_db)):
    """获取视频相关文件列表"""
    from pathlib import Path
    
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    
    if not video.output_dir:
        return {"files": []}
    
    output_dir = Path(video.output_dir)
    if not output_dir.exists():
        return {"files": []}
    
    files = []
    for f in output_dir.iterdir():
        if f.is_file():
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "type": f.suffix[1:] if f.suffix else "unknown",
                "modified": f.stat().st_mtime
            })
    
    return {"files": files}
