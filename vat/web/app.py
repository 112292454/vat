"""
VAT Web UI - FastAPI 应用

简单的视频管理界面，用于查看视频列表和任务状态
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from vat.database import Database
from vat.config import load_config
from vat.models import TaskStep, TaskStatus, DEFAULT_STAGE_SEQUENCE
from vat.web.deps import get_db
from vat.web.jobs import JobStatus

# 导入路由
from vat.web.routes import videos_router, playlists_router, tasks_router, files_router, prompts_router, bilibili_router

app = FastAPI(title="VAT Manager", description="视频处理任务管理界面")

# CORS 配置（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(videos_router)
app.include_router(playlists_router)
app.include_router(tasks_router)
app.include_router(files_router)
app.include_router(prompts_router)
app.include_router(bilibili_router)

# 模板目录
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def format_duration(seconds: float) -> str:
    """格式化时长"""
    if not seconds:
        return "-"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_datetime(dt: Optional[datetime]) -> str:
    """格式化日期时间"""
    if not dt:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M")


# 注册模板过滤器
templates.env.filters["format_duration"] = format_duration
templates.env.filters["format_datetime"] = format_datetime


# ==================== 页面路由 ====================

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    q: Optional[str] = None,  # 搜索关键词
    playlist_id: Optional[str] = None  # Playlist 过滤
):
    """首页 - 视频列表（SQL 层面分页+过滤，避免全量加载）"""
    db = get_db()
    
    # SQL 层面分页+过滤
    result = db.list_videos_paginated(
        page=page,
        per_page=per_page,
        status=status,
        search=q,
        playlist_id=playlist_id
    )
    
    page_videos = result['videos']
    total = result['total']
    total_pages = result['total_pages']
    
    # 获取所有 playlist 供过滤选择
    playlists = db.list_playlists()
    
    # 仅对当前页视频查询进度（性能关键优化）
    page_video_ids = [v.id for v in page_videos]
    progress_map = db.batch_get_video_progress(page_video_ids) if page_video_ids else {}
    
    # 构建视频列表（仅当前页）
    video_list = []
    for video in page_videos:
        vp = progress_map.get(video.id, {
            "progress": 0, "task_status": {s.value: {"status": "pending", "error": None} for s in DEFAULT_STAGE_SEQUENCE},
            "has_failed": False, "has_running": False
        })
        
        # 提取翻译后的标题 (字段名为 'translated')
        translated_title = None
        if video.metadata and "translated" in video.metadata:
            ti = video.metadata["translated"]
            translated_title = ti.get("title_translated")
        
        video_list.append({
            "id": video.id,
            "title": video.title or "未知标题",
            "translated_title": translated_title,
            "source_type": video.source_type.value,
            "source_url": video.source_url,
            "thumbnail": video.metadata.get("thumbnail") if video.metadata else None,
            "duration": video.metadata.get("duration") if video.metadata else None,
            "channel": video.metadata.get("channel") if video.metadata else None,
            "created_at": video.created_at,
            "task_status": vp["task_status"],
            "progress": vp["progress"]
        })
    
    # 统计信息（SQL 层面计算，不依赖当前页数据）
    stats = db.get_statistics()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "videos": video_list,
        "stats": stats,
        "current_page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "status_filter": status,
        "search_query": q or "",
        "playlist_filter": playlist_id or "",
        "playlists": [{"id": p.id, "title": p.title} for p in playlists]
    })


@app.get("/video/{video_id}", response_class=HTMLResponse)
async def video_detail(request: Request, video_id: str, from_playlist: Optional[str] = None):
    """视频详情页"""
    db = get_db()
    
    video = db.get_video(video_id)
    if not video:
        return HTMLResponse("<h1>视频不存在</h1>", status_code=404)
    
    tasks = db.get_tasks(video_id)
    
    # 解析翻译信息 (字段名为 'translated')
    # 注意：翻译标题不包含主播名前缀，前缀由上传模板系统添加
    translated_info = None
    if video.metadata and "translated" in video.metadata:
        translated_info = dict(video.metadata["translated"])
    
    # 构建任务时间线（使用细粒度阶段）
    step_names = {
        "download": "下载",
        "whisper": "语音识别",
        "split": "句子分割",
        "optimize": "提示词优化",
        "translate": "翻译",
        "embed": "嵌入字幕",
        "upload": "上传"
    }
    
    task_timeline = []
    for step in DEFAULT_STAGE_SEQUENCE:
        # 获取该阶段最新的任务（优先已完成的）
        step_tasks = [t for t in tasks if t.step == step]
        task = None
        if step_tasks:
            completed = [t for t in step_tasks if t.status == TaskStatus.COMPLETED]
            task = completed[-1] if completed else step_tasks[-1]
        
        task_timeline.append({
            "step": step.value,
            "step_name": step_names.get(step.value, step.value),
            "status": task.status.value if task else "pending",
            "started_at": task.started_at.isoformat() if task and task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task and task.completed_at else None,
            "error_message": task.error_message if task else None,
        })
    
    # 获取相关文件列表
    files_list = []
    if video.output_dir:
        from pathlib import Path
        output_path = Path(video.output_dir)
        if output_path.exists():
            for f in output_path.iterdir():
                if f.is_file():
                    files_list.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "ext": f.suffix.lower(),
                        "path": str(f)
                    })
            files_list.sort(key=lambda x: x["name"])
    
    return templates.TemplateResponse("video_detail.html", {
        "request": request,
        "video": video,
        "translated_info": translated_info,
        "task_timeline": task_timeline,
        "metadata": video.metadata,
        "files": files_list,
        "playlist_id": from_playlist or video.playlist_id,
        "from_playlist": bool(from_playlist)
    })


# ==================== Playlist 页面路由 ====================

@app.get("/playlists", response_class=HTMLResponse)
async def playlists_page(request: Request):
    """Playlist 列表页"""
    db = get_db()
    from vat.services import PlaylistService
    
    playlist_service = PlaylistService(db)
    playlists = db.list_playlists()
    
    # 添加进度信息
    playlist_list = []
    for pl in playlists:
        progress = playlist_service.get_playlist_progress(pl.id)
        playlist_list.append({
            "id": pl.id,
            "title": pl.title,
            "channel": pl.channel,
            "video_count": pl.video_count or 0,
            "completed": progress.get('completed', 0),
            "failed": progress.get('failed', 0),
            "progress_percent": int(progress.get('completed', 0) / max(progress.get('total', 1), 1) * 100),
            "last_synced_at": pl.last_synced_at
        })
    
    return templates.TemplateResponse("playlists.html", {
        "request": request,
        "playlists": playlist_list
    })


@app.get("/playlists/{playlist_id}", response_class=HTMLResponse)
async def playlist_detail_page(
    request: Request, 
    playlist_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(100, ge=1, le=500)
):
    """Playlist 详情页（分页）"""
    db = get_db()
    from vat.services import PlaylistService
    
    playlist_service = PlaylistService(db)
    pl = playlist_service.get_playlist(playlist_id)
    
    if not pl:
        return HTMLResponse("<h1>Playlist 不存在</h1>", status_code=404)
    
    # 获取全量视频列表（轻量：仅模型对象，不含进度）
    all_videos = playlist_service.get_playlist_videos(playlist_id)
    progress = playlist_service.get_playlist_progress(playlist_id)
    
    # 全量 ID 列表（供 JS 批量操作使用）
    all_video_ids = [v.id for v in all_videos]
    
    # 全量进度查询（单次SQL，供分页渲染 + JS 范围选择共用）
    all_progress_map = db.batch_get_video_progress(all_video_ids) if all_video_ids else {}
    
    # 构建全量视频基础数据（供 JS 范围选择使用，仅含 id/pending/unavailable）
    all_video_data = []
    for v in all_videos:
        vp = all_progress_map.get(v.id, {"completed": 0, "total": 7})
        metadata = v.metadata or {}
        unavailable = metadata.get("unavailable", False)
        pending = (vp["total"] - vp["completed"]) > 0 and not unavailable
        all_video_data.append({"id": v.id, "pending": pending, "unavailable": unavailable})
    
    # 分页
    total = len(all_videos)
    total_pages = (total + per_page - 1) // per_page if total > 0 else 1
    start = (page - 1) * per_page
    end = start + per_page
    page_videos = all_videos[start:end]
    
    video_list = []
    for idx, v in enumerate(page_videos):
        vp = all_progress_map.get(v.id, {"completed": 0, "total": 7, "progress": 0, "has_failed": False, "has_running": False})
        pending_count = vp["total"] - vp["completed"]
        metadata = v.metadata or {}
        duration = metadata.get("duration", 0)
        duration_formatted = format_duration(duration) if duration else ""
        
        # 状态判定：failed > running > completed > pending
        if vp.get("has_failed"):
            status = "failed"
        elif vp.get("has_running"):
            status = "running"
        elif pending_count == 0:
            status = "completed"
        else:
            status = "pending"
        
        video_list.append({
            "id": v.id,
            "title": v.title,
            "playlist_index": v.playlist_index,
            "global_index": start + idx + 1,
            "pending_count": pending_count,
            "status": status,
            "progress": vp["progress"],
            "upload_date": metadata.get("upload_date", ""),
            "upload_date_interpolated": metadata.get("upload_date_interpolated", False),
            "unavailable": metadata.get("unavailable", False),
            "duration": duration,
            "duration_formatted": duration_formatted
        })
    
    return templates.TemplateResponse("playlist_detail.html", {
        "request": request,
        "playlist": pl,
        "videos": video_list,
        "all_video_ids": all_video_ids,
        "all_video_data": all_video_data,
        "progress": progress,
        "current_page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages
    })


# ==================== Tasks 页面路由 ====================

@app.get("/tasks", response_class=HTMLResponse)
async def tasks_page(request: Request):
    """任务列表页"""
    from vat.web.jobs import JobManager
    from vat.config import load_config
    from pathlib import Path
    
    config = load_config()
    log_dir = Path(config.storage.database_path).parent / "job_logs"
    job_manager = JobManager(config.storage.database_path, str(log_dir))
    
    # 先更新所有 running 状态任务的实际状态
    jobs = job_manager.list_jobs(limit=50)
    for j in jobs:
        if j.status == JobStatus.RUNNING:
            job_manager.update_job_status(j.job_id)
    # 重新获取更新后的列表
    jobs = job_manager.list_jobs(limit=50)
    task_list = [j.to_dict() for j in jobs]
    
    return templates.TemplateResponse("tasks.html", {
        "request": request,
        "tasks": task_list
    })


@app.get("/tasks/new", response_class=HTMLResponse)
async def task_new_page(
    request: Request, 
    playlist: Optional[str] = None, 
    video: Optional[str] = None,
    videos: Optional[str] = None  # 逗号分隔的多个视频 ID
):
    """新建任务页"""
    db = get_db()
    
    playlist_id = playlist
    playlist_title = None
    playlist_video_count = 0
    
    # 解析选中的视频 ID 列表
    selected_video_ids = set()
    if video:
        selected_video_ids.add(video)
    if videos:
        selected_video_ids.update(videos.split(','))
    
    if playlist_id:
        from vat.services import PlaylistService
        playlist_service = PlaylistService(db)
        pl = playlist_service.get_playlist(playlist_id)
        if pl:
            playlist_title = pl.title
            videos_in_playlist = playlist_service.get_playlist_videos(playlist_id)
            playlist_video_count = len(videos_in_playlist)
    
    # 获取视频列表（优化：避免加载全部视频到 DOM）
    MAX_VIDEOS_NO_PLAYLIST = 200
    truncated = False
    
    if playlist_id:
        # 有 playlist：加载该 playlist 的全部视频
        all_videos = db.list_videos(playlist_id=playlist_id)
    elif selected_video_ids:
        # 有明确选中的视频：只加载选中的视频
        all_videos = [v for vid in selected_video_ids if (v := db.get_video(vid))]
    else:
        # 无过滤条件（直接访问 /tasks/new）：限制加载数量
        all_videos = db.list_videos()
        if len(all_videos) > MAX_VIDEOS_NO_PLAYLIST:
            truncated = True
            all_videos = all_videos[:MAX_VIDEOS_NO_PLAYLIST]
    
    # 批量获取进度（消除 N+1）
    all_video_ids_list = [v.id for v in all_videos]
    progress_map = db.batch_get_video_progress(all_video_ids_list) if all_video_ids_list else {}
    
    video_list = []
    for v in all_videos:
        vp = progress_map.get(v.id, {"completed": 0, "total": 7})
        completed = vp["completed"]
        # 选中逻辑：指定了视频列表则按列表，否则 playlist 下全选
        if selected_video_ids:
            is_selected = v.id in selected_video_ids
        else:
            is_selected = playlist_id is not None
        video_list.append({
            "id": v.id,
            "title": v.title or v.id,
            "selected": is_selected,
            "progress_text": f"{completed}/7 ({int(completed/7*100)}%)"
        })
    
    # 计算返回链接
    back_url = "/tasks"
    if playlist_id:
        back_url = f"/playlists/{playlist_id}"
    elif video:
        back_url = f"/video/{video}"
    
    return templates.TemplateResponse("task_new.html", {
        "request": request,
        "videos": video_list,
        "playlist_id": playlist_id,
        "playlist_title": playlist_title,
        "playlist_video_count": playlist_video_count,
        "back_url": back_url,
        "truncated": truncated
    })


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
async def task_detail_page(request: Request, task_id: str):
    """任务详情页"""
    from vat.web.jobs import JobManager
    from vat.config import load_config
    from pathlib import Path
    
    config = load_config()
    log_dir = Path(config.storage.database_path).parent / "job_logs"
    job_manager = JobManager(config.storage.database_path, str(log_dir))
    
    # 更新任务状态
    job_manager.update_job_status(task_id)
    
    job = job_manager.get_job(task_id)
    if not job:
        return HTMLResponse("<h1>任务不存在</h1>", status_code=404)
    
    # 获取视频标题信息
    db = get_db()
    video_info_list = []
    for vid in job.video_ids:
        video = db.get_video(vid)
        if video:
            video_info_list.append({"id": vid, "title": video.title or vid})
        else:
            video_info_list.append({"id": vid, "title": vid})
    
    task_dict = job.to_dict()
    task_dict["video_info_list"] = video_info_list
    
    return templates.TemplateResponse("task_detail.html", {
        "request": request,
        "task": task_dict
    })


# ==================== API 路由 ====================

@app.get("/api/videos")
async def api_list_videos():
    """API: 获取视频列表"""
    db = get_db()
    videos = db.list_videos()
    return [{"id": v.id, "title": v.title, "source_type": v.source_type.value} for v in videos]


@app.get("/api/video/{video_id}")
async def api_get_video(video_id: str):
    """API: 获取视频详情"""
    db = get_db()
    video = db.get_video(video_id)
    if not video:
        return JSONResponse({"error": "Video not found"}, status_code=404)
    
    tasks = db.get_tasks(video_id)
    return {
        "id": video.id,
        "title": video.title,
        "source_type": video.source_type.value,
        "source_url": video.source_url,
        "metadata": video.metadata,
        "tasks": [
            {
                "step": t.step.value,
                "status": t.status.value,
                "error_message": t.error_message
            } for t in tasks
        ]
    }


@app.get("/api/stats")
async def api_stats():
    """API: 获取统计信息"""
    db = get_db()
    return db.get_statistics()


@app.get("/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request):
    """Custom Prompts 管理页"""
    return templates.TemplateResponse("prompts.html", {"request": request})


# ==================== 启动自动同步 ====================

def _auto_sync_stale_playlists():
    """检查并自动同步超过 7 天未更新的 playlist（后台线程）"""
    import threading
    from datetime import timedelta
    from vat.services import PlaylistService
    from vat.downloaders import YouTubeDownloader
    
    config = load_config()
    db = Database(config.storage.database_path)
    playlists = db.list_playlists()
    
    if not playlists:
        return
    
    now = datetime.now()
    stale_threshold = timedelta(days=7)
    stale_playlists = []
    
    for pl in playlists:
        if not pl.last_synced_at or (now - pl.last_synced_at) > stale_threshold:
            stale_playlists.append(pl)
    
    if not stale_playlists:
        return
    
    import logging
    logger = logging.getLogger("vat.web.auto_sync")
    logger.info(f"发现 {len(stale_playlists)} 个超过 7 天未同步的 Playlist，启动后台同步...")
    
    def sync_one(pl):
        try:
            sync_db = Database(config.storage.database_path)
            downloader = YouTubeDownloader(
                proxy=config.proxy.get_proxy(),
                video_format=config.downloader.youtube.format
            )
            service = PlaylistService(sync_db, downloader)
            result = service.sync_playlist(
                pl.source_url,
                auto_add_videos=True,
                fetch_upload_dates=True,
                progress_callback=lambda msg: logger.info(f"[{pl.title}] {msg}")
            )
            logger.info(f"[{pl.title}] 同步完成: 新增 {result.new_count}, 已存在 {result.existing_count}")
        except Exception as e:
            logger.error(f"[{pl.title}] 自动同步失败: {e}")
    
    for pl in stale_playlists:
        t = threading.Thread(target=sync_one, args=(pl,), daemon=True, name=f"auto-sync-{pl.id}")
        t.start()


@app.on_event("startup")
async def on_startup():
    """应用启动时执行的任务"""
    import threading
    # 在后台线程中执行，不阻塞启动
    threading.Thread(target=_auto_sync_stale_playlists, daemon=True, name="auto-sync-check").start()


# ==================== 启动入口 ====================

def run_server(host: str | None = None, port: int | None = None):
    """启动服务器
    
    Args:
        host: 监听地址，None 时从配置文件读取
        port: 监听端口，None 时从配置文件读取
    """
    import uvicorn
    config = load_config()
    host = host or config.web.host
    port = port or config.web.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
