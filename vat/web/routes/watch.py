"""
Watch 模式 API 路由

提供 Watch Session 的管理接口：
- 列表查看所有 session
- 启动新 watch session（通过 JobManager 提交）
- 停止 running session
- 查看 round 详情
"""
import json
import os
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vat.database import Database
from vat.services.watch_service import WatchService
from vat.web.deps import get_db, get_web_config

router = APIRouter(prefix="/api/watch", tags=["watch"])


def _get_job_manager():
    """获取 JobManager 实例。"""
    from vat.web.routes.tasks import get_job_manager
    return get_job_manager()


class WatchStartRequest(BaseModel):
    """启动 Watch 请求"""
    playlist_ids: List[str]
    interval: Optional[int] = None
    stages: Optional[str] = None
    gpu_device: str = "auto"
    concurrency: Optional[int] = None
    force: bool = False
    fail_fast: bool = False
    once: bool = False


@router.get("/sessions")
async def list_sessions(
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
):
    """获取 Watch Session 列表"""
    db = get_db()
    
    with db.get_connection() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM watch_sessions WHERE status = ? ORDER BY started_at DESC LIMIT ?",
                (status, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM watch_sessions ORDER BY started_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
    
    sessions = []
    for row in rows:
        sess = dict(row)
        sess['playlist_ids'] = json.loads(sess['playlist_ids']) if sess['playlist_ids'] else []
        sess['config'] = json.loads(sess['config']) if sess.get('config') else {}
        
        # 检查 running session 的进程是否存活
        if sess['status'] == 'running' and sess.get('pid'):
            sess['pid_alive'] = WatchService._is_pid_alive(sess['pid'])
        else:
            sess['pid_alive'] = None
        
        sessions.append(sess)
    
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取单个 Session 详情"""
    db = get_db()
    
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM watch_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    
    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    sess = dict(row)
    sess['playlist_ids'] = json.loads(sess['playlist_ids']) if sess['playlist_ids'] else []
    sess['config'] = json.loads(sess['config']) if sess.get('config') else {}
    
    if sess['status'] == 'running' and sess.get('pid'):
        sess['pid_alive'] = WatchService._is_pid_alive(sess['pid'])
    
    return sess


@router.get("/sessions/{session_id}/rounds")
async def get_session_rounds(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
):
    """获取 Session 的 Round 记录"""
    db = get_db()
    
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM watch_rounds WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
    
    rounds = []
    for row in rows:
        r = dict(row)
        r['submitted_video_ids'] = json.loads(r['submitted_video_ids']) if r.get('submitted_video_ids') else []
        r['submitted_job_ids'] = json.loads(r['submitted_job_ids']) if r.get('submitted_job_ids') else []
        r['retry_video_ids'] = json.loads(r['retry_video_ids']) if r.get('retry_video_ids') else []
        rounds.append(r)
    
    return {"rounds": rounds}


@router.post("/start")
async def start_watch(req: WatchStartRequest):
    """启动新的 Watch Session（通过 JobManager 提交子进程）"""
    from vat.services import PlaylistService
    
    config = get_web_config()
    db = get_db()
    
    # 验证 playlist 存在
    playlist_service = PlaylistService(db, config)
    for pl_id in req.playlist_ids:
        pl = playlist_service.get_playlist(pl_id)
        if not pl:
            return JSONResponse(
                {"error": f"Playlist 不存在: {pl_id}"},
                status_code=400
            )
    
    # 检查是否已有 running session 监控同一 playlist
    with db.get_connection() as conn:
        running = conn.execute(
            "SELECT session_id, playlist_ids, pid FROM watch_sessions WHERE status = 'running'"
        ).fetchall()
    
    for row in running:
        existing_pls = json.loads(row['playlist_ids'])
        overlap = set(existing_pls) & set(req.playlist_ids)
        if overlap and WatchService._is_pid_alive(row['pid'] or 0):
            return JSONResponse(
                {"error": f"Playlist {overlap} 已被 session {row['session_id']} 监控中"},
                status_code=409
            )
    
    # 通过统一入口获取 JobManager，避免 watch 路由再维护一套构造逻辑
    job_manager = _get_job_manager()
    
    task_params = {
        'playlist_ids': req.playlist_ids,
        'once': req.once,
        'force': req.force,
        'fail_fast': req.fail_fast,
    }
    if req.interval is not None:
        task_params['interval'] = req.interval
    if req.stages is not None:
        task_params['stages'] = req.stages
    if req.gpu_device != "auto":
        task_params['gpu'] = req.gpu_device
    if req.concurrency is not None:
        task_params['concurrency'] = req.concurrency
    
    job_id = job_manager.submit_job(
        video_ids=[],
        steps=[],
        task_type='watch',
        task_params=task_params,
    )
    
    return {"job_id": job_id, "message": "Watch 任务已提交"}


@router.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    """停止 Watch Session（发送 SIGTERM）"""
    import signal
    
    db = get_db()
    
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM watch_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    
    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    if row['status'] != 'running':
        return JSONResponse({"error": f"Session 状态为 {row['status']}，无法停止"}, status_code=400)
    
    pid = row['pid']
    if not pid or not WatchService._is_pid_alive(pid):
        # 进程已死，直接标记为 stopped
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE watch_sessions SET status = 'stopped', stopped_at = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id)
            )
        return {"message": "Session 进程已不存在，已标记为 stopped"}
    
    try:
        os.kill(pid, signal.SIGTERM)
        return {"message": f"已发送停止信号到 PID {pid}"}
    except Exception as e:
        return JSONResponse({"error": f"停止失败: {e}"}, status_code=500)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除 Session 记录（仅限已停止的）"""
    db = get_db()
    
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT status FROM watch_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    
    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    if row['status'] == 'running':
        return JSONResponse({"error": "无法删除 running 状态的 session"}, status_code=400)
    
    with db.get_connection() as conn:
        conn.execute("DELETE FROM watch_rounds WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM watch_sessions WHERE session_id = ?", (session_id,))
    
    return {"message": "已删除"}
