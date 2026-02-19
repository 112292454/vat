"""
B站设置相关路由
"""
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ...config import load_config
from ...database import Database
from ...uploaders.bilibili import BilibiliUploader
from ...uploaders.upload_config import get_upload_config_manager, UploadConfigManager
from ...embedder.ffmpeg_wrapper import FFmpegWrapper
from ...utils.logger import setup_logger

logger = setup_logger("web.bilibili")

# 审核修复任务状态（内存存储，key=aid）
_fix_tasks: Dict[int, Dict[str, Any]] = {}

router = APIRouter(prefix="/bilibili", tags=["bilibili"])

# 模板目录
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def _get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent.parent


def _get_uploader(with_upload_params: bool = False) -> BilibiliUploader:
    """获取 B站上传器实例
    
    Args:
        with_upload_params: 是否包含上传参数（line/threads），上传/替换视频时需要
    """
    config = load_config()
    project_root = _get_project_root()
    cookies_file = project_root / config.uploader.bilibili.cookies_file
    kwargs = {'cookies_file': str(cookies_file)}
    if with_upload_params:
        kwargs['line'] = config.uploader.bilibili.line
        kwargs['threads'] = config.uploader.bilibili.threads
    return BilibiliUploader(**kwargs)


def _get_cookies_path() -> Path:
    """获取 cookie 文件路径"""
    config = load_config()
    project_root = _get_project_root()
    return project_root / config.uploader.bilibili.cookies_file


@router.get("", response_class=HTMLResponse)
async def bilibili_settings_page(request: Request):
    """B站设置页面"""
    # 使用独立的上传配置（可在线编辑）
    upload_mgr = get_upload_config_manager()
    upload_config = upload_mgr.load()
    bilibili_config = upload_config.bilibili
    
    # 检查登录状态
    login_status = {
        'logged_in': False,
        'username': None,
        'uid': None,
        'level': None,
    }
    
    cookies_path = _get_cookies_path()
    if cookies_path.exists():
        try:
            uploader = _get_uploader()
            if uploader.validate_credentials():
                login_status['logged_in'] = True
                
                # 获取用户信息
                session = uploader._get_authenticated_session()
                resp = session.get('https://api.bilibili.com/x/web-interface/nav', timeout=5)
                data = resp.json()
                
                if data.get('code') == 0:
                    user_data = data.get('data', {})
                    login_status['username'] = user_data.get('uname')
                    login_status['uid'] = user_data.get('mid')
                    login_status['level'] = user_data.get('level_info', {}).get('current_level', 0)
        except Exception as e:
            logger.error(f"检查登录状态失败: {e}")
    
    # 获取合集列表
    seasons = []
    if login_status['logged_in']:
        try:
            uploader = _get_uploader()
            seasons = uploader.list_seasons()
        except Exception as e:
            logger.error(f"获取合集列表失败: {e}")
    
    # 模板配置
    templates_config = {
        'title': bilibili_config.templates.title,
        'description': bilibili_config.templates.description,
        'custom_vars': bilibili_config.templates.custom_vars,
    }
    
    return templates.TemplateResponse("bilibili.html", {
        "request": request,
        "login_status": login_status,
        "seasons": seasons,
        "config": {
            "copyright": bilibili_config.copyright,
            "default_tid": bilibili_config.default_tid,
            "default_tags": bilibili_config.default_tags,
            "auto_cover": bilibili_config.auto_cover,
            "season_id": bilibili_config.season_id,
        },
        "templates": templates_config,
    })


@router.get("/qrcode")
async def get_login_qrcode():
    """获取登录二维码"""
    try:
        import stream_gears
        
        qr_data = stream_gears.get_qrcode(None)
        qr_info = json.loads(qr_data)
        qr_url = qr_info.get('data', {}).get('url', '')
        
        if not qr_url:
            return JSONResponse({"success": False, "error": "获取二维码失败"})
        
        return JSONResponse({
            "success": True,
            "qr_url": qr_url,
            "qr_data": qr_data,  # 用于后续轮询
        })
        
    except ImportError:
        return JSONResponse({"success": False, "error": "缺少 stream_gears 模块"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@router.post("/login")
async def do_login(qr_data: str = Form(...)):
    """完成登录（在后台线程中轮询二维码状态）"""
    import threading
    
    def login_task():
        try:
            import stream_gears
            result = stream_gears.login_by_qrcode(qr_data, None)
            
            # 保存 cookie
            cookies_path = _get_cookies_path()
            cookies_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cookies_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            logger.info("B站登录成功，Cookie已保存")
        except Exception as e:
            logger.error(f"B站登录失败: {e}")
    
    # 在后台线程执行（因为 login_by_qrcode 是阻塞调用）
    thread = threading.Thread(target=login_task, daemon=True)
    thread.start()
    
    return JSONResponse({"success": True, "message": "登录任务已启动，请等待扫码完成"})


@router.get("/status")
async def check_status():
    """检查登录状态"""
    try:
        cookies_path = _get_cookies_path()
        if not cookies_path.exists():
            return JSONResponse({
                "logged_in": False,
                "error": "Cookie 文件不存在"
            })
        
        uploader = _get_uploader()
        if not uploader.validate_credentials():
            return JSONResponse({
                "logged_in": False,
                "error": "Cookie 无效或已过期"
            })
        
        # 获取用户信息
        session = uploader._get_authenticated_session()
        resp = session.get('https://api.bilibili.com/x/web-interface/nav', timeout=5)
        data = resp.json()
        
        if data.get('code') == 0:
            user_data = data.get('data', {})
            return JSONResponse({
                "logged_in": True,
                "username": user_data.get('uname'),
                "uid": user_data.get('mid'),
                "level": user_data.get('level_info', {}).get('current_level', 0),
            })
        else:
            return JSONResponse({
                "logged_in": False,
                "error": data.get('message', '未知错误')
            })
            
    except Exception as e:
        return JSONResponse({
            "logged_in": False,
            "error": str(e)
        })


@router.get("/seasons")
async def list_seasons():
    """获取合集列表"""
    try:
        uploader = _get_uploader()
        seasons = uploader.list_seasons()
        return JSONResponse({"success": True, "seasons": seasons})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@router.post("/seasons")
async def create_season(title: str = Form(...), description: str = Form("")):
    """创建新合集"""
    try:
        uploader = _get_uploader()
        result = uploader.create_season(title, description)
        
        if result.get('success'):
            return JSONResponse({
                "success": True,
                "season_id": result.get('season_id'),
                "title": title,
            })
        else:
            return JSONResponse({
                "success": False, 
                "error": result.get('error', '创建失败——由于b站api限制，无法从此处创建，请于b站网页端手动创建后使用')
            })
            
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@router.get("/config")
async def get_upload_config():
    """获取上传配置"""
    try:
        mgr = get_upload_config_manager()
        config = mgr.load()
        return JSONResponse({
            "success": True,
            "config": config.bilibili.to_dict()
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@router.put("/config")
async def update_upload_config(request: Request):
    """更新上传配置"""
    try:
        data = await request.json()
        mgr = get_upload_config_manager()
        
        if mgr.update_bilibili(data):
            return JSONResponse({"success": True, "message": "配置已保存"})
        else:
            return JSONResponse({"success": False, "error": "保存失败"})
            
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# =============================================================================
# 审核退回管理
# =============================================================================

@router.get("/rejected")
async def list_rejected_videos(keyword: str = ""):
    """获取被退回的稿件列表"""
    try:
        uploader = _get_uploader()
        rejected = uploader.get_rejected_videos(keyword=keyword)
        
        # 为前端整理数据
        result = []
        for v in rejected:
            all_ranges = []
            is_full = False
            problems_text = []
            for p in v['problems']:
                all_ranges.extend(p['time_ranges'])
                if p['is_full_video']:
                    is_full = True
                problems_text.append({
                    'reason': p['reason'],
                    'violation_time': p['violation_time'],
                    'violation_position': p['violation_position'],
                    'modify_advise': p['modify_advise'],
                    'time_ranges': p['time_ranges'],
                    'is_full_video': p['is_full_video'],
                })
            
            fixable = bool(all_ranges) and not is_full
            result.append({
                'aid': v['aid'],
                'bvid': v['bvid'],
                'title': v['title'],
                'reject_reason': v['reject_reason'],
                'problems': problems_text,
                'all_ranges': all_ranges,
                'is_full_video': is_full,
                'fixable': fixable,
                'fix_status': _fix_tasks.get(v['aid'], {}).get('status'),
            })
        
        return JSONResponse({"success": True, "videos": result})
    except Exception as e:
        logger.error(f"获取退回稿件失败: {e}")
        return JSONResponse({"success": False, "error": str(e)})


def _find_local_video(aid: int) -> Optional[Path]:
    """
    根据 aid 查找本地视频文件路径。
    
    查找策略（按优先级）：
    1. 从 B站稿件 source URL 提取 YouTube video ID → DB 查找视频记录 → 本地 final.mp4
    2. DB 中通过 bilibili_aid 匹配 → 本地 final.mp4
    3. 通过 B站稿件标题匹配 DB 翻译标题 → 本地 final.mp4
    4. 直接按 YouTube video ID 查找 output 目录
    
    Returns:
        找到的本地视频文件路径，或 None
    """
    import re
    
    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    yt_video_id = None  # 记录提取到的 YouTube video ID，供后续 fallback 使用
    bili_title = None    # B站稿件标题，用于标题匹配
    
    # 方法1: 从 B站稿件获取 source URL → 提取 YouTube video ID → DB 匹配
    try:
        uploader = _get_uploader()
        detail = uploader.get_archive_detail(aid)
        if detail:
            archive = detail.get('archive', {})
            bili_title = archive.get('title', '')
            source = archive.get('source', '')
            yt_match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)', source)
            if yt_match:
                yt_video_id = yt_match.group(1)
                logger.info(f"稿件 av{aid} 对应 YouTube 视频: {yt_video_id}")
                
                video = db.get_video(yt_video_id)
                if video:
                    path = _resolve_video_file(video, config)
                    if path:
                        logger.info(f"通过 source URL 找到本地视频: {path}")
                        return path
    except Exception as e:
        logger.warning(f"通过 source URL 查找视频失败: {e}")
    
    # 方法2: DB 中通过 bilibili_aid 匹配
    # 方法3: 通过 B站稿件标题匹配 DB 翻译标题
    # 合并遍历 list_videos，避免重复查询
    videos = db.list_videos()
    for v in videos:
        meta = v.metadata or {}
        
        # 方法2: bilibili_aid
        if str(meta.get('bilibili_aid', '')) == str(aid):
            path = _resolve_video_file(v, config)
            if path:
                logger.info(f"通过 bilibili_aid 找到本地视频: {path}")
                return path
    
    # 方法3: 标题匹配（B站稿件标题通常是我们上传时的翻译标题）
    if bili_title:
        # 从 B站标题中提取核心部分（去掉 " | #N" 等上传模板后缀）
        clean_title = re.sub(r'\s*\|\s*#\d+\s*$', '', bili_title).strip()
        for v in videos:
            meta = v.metadata or {}
            translated = meta.get('translated', {})
            t_title = translated.get('title_translated', '') if translated else ''
            if t_title and clean_title and (clean_title in t_title or t_title in clean_title):
                path = _resolve_video_file(v, config)
                if path:
                    logger.info(f"通过标题匹配找到本地视频: '{clean_title}' → {v.id}, {path}")
                    return path
    
    # 方法4: 如果有 YouTube video ID，直接查找 output 目录（可能 DB 记录已删但文件还在）
    if yt_video_id:
        vid_dir = Path(config.storage.output_dir) / yt_video_id
        for name in ['final.mp4', f'{yt_video_id}.mp4']:
            candidate = vid_dir / name
            if candidate.exists():
                logger.info(f"通过 output 目录直接找到视频: {candidate}")
                return candidate
    
    return None


def _resolve_video_file(video, config) -> Optional[Path]:
    """从视频记录解析本地视频文件路径（final.mp4 优先）"""
    candidates = []
    if video.output_dir:
        candidates.append(Path(video.output_dir) / "final.mp4")
    vid_dir = Path(config.storage.output_dir) / video.id
    candidates.append(vid_dir / "final.mp4")
    candidates.append(vid_dir / f"{video.id}.mp4")
    
    for c in candidates:
        if c.exists():
            return c
    return None


def _download_from_bilibili(aid: int, bvid: str) -> Optional[Path]:
    """
    从 B站下载视频作为 fallback（仅在本地文件完全找不到时使用）。
    
    使用 yt-dlp 下载 B站视频到临时目录。
    
    Args:
        aid: 稿件 AV号
        bvid: 稿件 BV号
        
    Returns:
        下载后的视频文件路径，或 None
    """
    import subprocess
    import tempfile
    
    url = f"https://www.bilibili.com/video/{bvid}" if bvid else f"https://www.bilibili.com/video/av{aid}"
    
    logger.warning(
        f"⚠️ 本地视频文件未找到，将从 B站下载原视频用于遮罩处理: {url}\n"
        f"  这通常意味着本地视频文件已被清理。建议在清理前完成审核修复。"
    )
    
    # 下载到临时目录
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bilibili_fix_{aid}_"))
    output_template = str(tmp_dir / f"av{aid}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        url,
    ]
    
    # 添加代理（B站不需要代理，但保留以防特殊网络环境）
    config = load_config()
    proxy = config.get_stage_proxy('downloader') if hasattr(config, 'get_stage_proxy') else None
    if proxy:
        cmd.extend(["--proxy", proxy])
    
    try:
        logger.info(f"开始从 B站下载视频: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"B站视频下载失败: {result.stderr[-500:]}")
            return None
        
        # 查找下载的文件
        mp4_files = list(tmp_dir.glob("*.mp4"))
        if mp4_files:
            logger.info(f"B站视频下载成功: {mp4_files[0]} ({mp4_files[0].stat().st_size / 1024 / 1024:.1f}MB)")
            return mp4_files[0]
        
        logger.error(f"下载完成但未找到 mp4 文件: {list(tmp_dir.iterdir())}")
        return None
        
    except subprocess.TimeoutExpired:
        logger.error("B站视频下载超时 (>10分钟)")
        return None
    except Exception as e:
        logger.error(f"B站视频下载异常: {e}")
        return None


def _run_fix_task(aid: int, violation_ranges: list, video_path: Path, mask_text: str, margin: float):
    """后台线程执行遮罩+上传替换"""
    task = _fix_tasks[aid]
    try:
        task['status'] = 'masking'
        task['message'] = '正在遮罩违规片段...'
        
        # ffmpeg 遮罩
        ffmpeg = FFmpegWrapper()
        masked_path = video_path.parent / f"{video_path.stem}_masked{video_path.suffix}"
        
        ok = ffmpeg.mask_violation_segments(
            video_path=video_path,
            output_path=masked_path,
            violation_ranges=violation_ranges,
            mask_text=mask_text,
            margin_sec=margin,
        )
        
        if not ok:
            task['status'] = 'failed'
            task['message'] = '遮罩处理失败'
            return
        
        task['status'] = 'uploading'
        task['message'] = f'遮罩完成，正在上传替换 ({masked_path.stat().st_size / 1024 / 1024:.0f}MB)...'
        
        # 上传替换
        uploader = _get_uploader(with_upload_params=True)
        replace_ok = uploader.replace_video(aid, masked_path)
        
        if replace_ok:
            task['status'] = 'completed'
            task['message'] = '修复完成，已重新提交审核'
            # 清理遮罩临时文件
            masked_path.unlink(missing_ok=True)
        else:
            task['status'] = 'failed'
            task['message'] = '上传替换失败，遮罩文件已保留'
            task['masked_path'] = str(masked_path)
            
    except Exception as e:
        logger.error(f"修复任务异常 (aid={aid}): {e}", exc_info=True)
        task['status'] = 'failed'
        task['message'] = f'异常: {e}'


@router.post("/fix/{aid}")
async def fix_rejected_video(aid: int, request: Request):
    """启动审核退回修复任务（遮罩+上传替换）"""
    try:
        # 检查是否已有任务在运行
        existing = _fix_tasks.get(aid, {})
        if existing.get('status') in ('masking', 'uploading'):
            return JSONResponse({
                "success": False,
                "error": f"aid={aid} 已有修复任务在运行: {existing.get('message')}"
            })
        
        body = await request.json()
        video_path_str = body.get('video_path')
        margin = body.get('margin', 1.0)
        mask_text = body.get('mask_text', '此处内容因合规要求已被遮罩')
        violation_ranges = body.get('violation_ranges', [])
        
        if not violation_ranges:
            return JSONResponse({"success": False, "error": "无违规时间段"})
        
        # 转换为 tuple 列表
        ranges = [(r[0], r[1]) for r in violation_ranges]
        
        # 查找本地视频
        bvid = body.get('bvid', '')
        if video_path_str:
            video_path = Path(video_path_str)
            if not video_path.exists():
                return JSONResponse({"success": False, "error": f"文件不存在: {video_path_str}"})
        else:
            video_path = _find_local_video(aid)
            if not video_path:
                # fallback: 从 B站下载原视频
                video_path = _download_from_bilibili(aid, bvid)
                if not video_path:
                    return JSONResponse({
                        "success": False,
                        "error": "未找到本地视频文件，且从B站下载失败。请手动指定路径",
                        "need_path": True
                    })
        
        # 初始化任务状态
        _fix_tasks[aid] = {
            'status': 'pending',
            'message': '任务已创建',
            'video_path': str(video_path),
            'violation_ranges': ranges,
        }
        
        # 启动后台线程
        thread = threading.Thread(
            target=_run_fix_task,
            args=(aid, ranges, video_path, mask_text, margin),
            daemon=True
        )
        thread.start()
        
        return JSONResponse({
            "success": True,
            "message": "修复任务已启动",
            "video_path": str(video_path),
        })
        
    except Exception as e:
        logger.error(f"启动修复任务失败: {e}", exc_info=True)
        return JSONResponse({"success": False, "error": str(e)})


@router.get("/fix/{aid}/status")
async def get_fix_status(aid: int):
    """获取修复任务状态"""
    task = _fix_tasks.get(aid)
    if not task:
        return JSONResponse({"status": "none", "message": "无任务记录"})
    return JSONResponse(task)
