"""
B站设置相关路由
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ...config import load_config
from ...uploaders.bilibili import BilibiliUploader
from ...uploaders.upload_config import get_upload_config_manager, UploadConfigManager
from ...utils.logger import setup_logger

logger = setup_logger("web.bilibili")

router = APIRouter(prefix="/bilibili", tags=["bilibili"])

# 模板目录
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def _get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent.parent


def _get_uploader() -> BilibiliUploader:
    """获取 B站上传器实例"""
    config = load_config()
    project_root = _get_project_root()
    cookies_file = project_root / config.uploader.bilibili.cookies_file
    return BilibiliUploader(cookies_file=str(cookies_file))


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
