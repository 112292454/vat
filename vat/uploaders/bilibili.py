"""
B站上传器实现

直接使用 biliup 库的 Web 端 API 上传（TV 端 API 已停用）
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .base import BaseUploader
from vat.utils.logger import setup_logger

try:
    from biliup.plugins.bili_webup import BiliBili, Data
    BILIUP_AVAILABLE = True
except ImportError:
    BILIUP_AVAILABLE = False

logger = setup_logger("uploader.bilibili")


@dataclass
class UploadResult:
    """上传结果"""
    success: bool
    bvid: str = ""
    error: str = ""
    
    def __bool__(self):
        return self.success


class BilibiliUploader(BaseUploader):
    """
    B站视频上传器
    
    直接使用 biliup 库的 Web 端 API 上传
    Cookie 通过 scripts/bilibili_login.py 获取
    """
    
    def __init__(
        self,
        cookies_file: str,
        line: str = 'AUTO',
        threads: int = 3
    ):
        """
        初始化B站上传器
        
        Args:
            cookies_file: cookies JSON文件路径
            line: 上传线路 (AUTO/bda2/qn/ws)
            threads: 上传线程数
        """
        if not BILIUP_AVAILABLE:
            raise ImportError("biliup 未安装，请运行: pip install biliup")
        
        self.cookies_file = Path(cookies_file).expanduser()
        self.line = line
        self.threads = threads
        self.cookie_data = None
        self._raw_cookie_data = None
        self._cookie_loaded = False
    
    def _load_cookie(self):
        """加载cookie文件"""
        if self._cookie_loaded:
            return
            
        if not self.cookies_file.exists():
            raise FileNotFoundError(
                f"Cookies文件不存在: {self.cookies_file}\n"
                f"请先运行 python scripts/bilibili_login.py 获取cookie"
            )
        
        with open(self.cookies_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 保留原始数据，供 biliup login_by_cookies 使用
        self._raw_cookie_data = data
        
        # 提取关键 cookie（兼容 stream_gears 登录格式）
        keys_to_extract = ["SESSDATA", "bili_jct", "DedeUserID__ckMd5", "DedeUserID", "access_token"]
        self.cookie_data = {}
        
        if 'cookie_info' in data and 'cookies' in data['cookie_info']:
            for cookie in data['cookie_info']['cookies']:
                if cookie.get('name') in keys_to_extract:
                    self.cookie_data[cookie['name']] = cookie['value']
        
        if 'token_info' in data and 'access_token' in data['token_info']:
            self.cookie_data['access_token'] = data['token_info']['access_token']
        
        self._cookie_loaded = True
        logger.info(f"已加载B站cookie: {self.cookies_file}")
    
    def upload(
        self, 
        video_path: Path, 
        title: str,
        description: str,
        tid: int,
        tags: List[str],
        copyright: int = 2,
        source: str = '',
        cover_path: Optional[Path] = None,
        dtime: int = 0,
        dynamic: str = ''
    ) -> UploadResult:
        """
        上传视频到B站
        
        Args:
            video_path: 视频文件路径
            title: 视频标题
            description: 视频描述
            tid: 分区ID
            tags: 标签列表
            copyright: 类型，1=自制，2=转载
            source: 转载来源URL（copyright=2时必填）
            cover_path: 封面图片路径（可选）
            dtime: 定时发布时间戳（0表示立即发布，需>2小时后）
            dynamic: 粉丝动态内容（可选）
            
        Returns:
            UploadResult: 上传结果
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return UploadResult(success=False, error=f"视频文件不存在: {video_path}")
        
        # 加载cookie
        try:
            self._load_cookie()
        except Exception as e:
            return UploadResult(success=False, error=f"加载cookie失败: {e}")
        
        logger.info(f"开始上传视频到B站: {video_path.name}")
        logger.info(f"标题: {title}")
        logger.info(f"分区: {tid}")
        logger.info(f"标签: {tags}")
        logger.info(f"类型: {'自制' if copyright == 1 else '转载'}")
        
        try:
            # 准备上传数据
            data = Data()
            data.copyright = copyright
            data.title = title[:80]  # B站标题限制80字符
            data.desc = description[:2000] if description else ''
            data.tid = tid
            data.set_tag(tags[:12])  # 标签限制12个
            
            # 转载来源
            if copyright == 2 and source:
                data.source = source
            
            # 定时发布
            if dtime and dtime > 0:
                data.delay_time(dtime)
            
            # 粉丝动态
            if dynamic:
                data.dynamic = dynamic
            
            with BiliBili(data) as bili:
                # 登录（biliup 要求原始 JSON 结构，含 cookie_info/token_info）
                bili.login_by_cookies(self._raw_cookie_data)
                if 'access_token' in self.cookie_data:
                    bili.access_token = self.cookie_data['access_token']
                
                # 上传封面
                if cover_path and Path(cover_path).exists():
                    logger.info(f"上传封面: {cover_path}")
                    try:
                        cover_url = bili.cover_up(str(cover_path))
                        data.cover = cover_url.replace('http:', '')
                        logger.info(f"封面上传成功")
                    except Exception as e:
                        logger.warning(f"封面上传失败，继续上传视频: {e}")
                
                # 上传视频文件
                logger.info("上传视频文件...")
                video_part = bili.upload_file(str(video_path), lines=self.line, tasks=self.threads)
                video_part['title'] = 'P1'
                data.append(video_part)
                
                # 使用 Web 端 API 提交（TV 端 API 已停用）
                logger.info("提交视频（Web端API）...")
                ret = bili.submit_web()
                
                if ret.get('code') == 0:
                    bvid = ret.get('data', {}).get('bvid', '')
                    logger.info(f"上传成功: {title}, BV号: {bvid}")
                    return UploadResult(success=True, bvid=bvid)
                else:
                    error_msg = ret.get('message', '未知错误')
                    logger.error(f"上传失败: {error_msg}")
                    return UploadResult(success=False, error=error_msg)
                
        except Exception as e:
            logger.error(f"上传异常: {e}")
            return UploadResult(success=False, error=str(e))
    
    def upload_with_metadata(self, video_path: Path, metadata: Dict[str, Any]) -> UploadResult:
        """
        使用metadata字典上传视频（兼容旧接口）
        
        Args:
            video_path: 视频文件路径
            metadata: 视频元数据字典
                - title: str - 标题
                - desc: str - 描述
                - tags: List[str] - 标签
                - tid: int - 分区ID
                
        Returns:
            UploadResult: 上传结果
        """
        return self.upload(
            video_path=video_path,
            title=metadata.get('title', video_path.stem),
            description=metadata.get('desc', ''),
            tid=metadata.get('tid', 21),
            tags=metadata.get('tags', [])
        )
    
    def validate_credentials(self) -> bool:
        """
        验证cookies是否有效
        
        Returns:
            是否有效
        """
        try:
            self._load_cookie()
            # 检查必要的cookie字段
            required_keys = ["SESSDATA", "bili_jct", "DedeUserID"]
            for key in required_keys:
                if key not in self.cookie_data:
                    logger.warning(f"Cookie缺少必要字段: {key}")
                    return False
            logger.info("Cookie验证通过")
            return True
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False
    
    def get_upload_limit(self) -> Dict[str, Any]:
        """
        获取上传限制信息
        
        Returns:
            限制信息字典
        """
        # B站的上传限制
        return {
            'max_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'max_duration': 4 * 3600,  # 4小时
            'supported_formats': [
                'mp4', 'flv', 'avi', 'wmv', 'mov',
                'webm', 'mkv', 'mpeg', 'mpg', 'rmvb'
            ]
        }
    
    def get_categories(self) -> Dict[int, str]:
        """
        获取分区列表
        
        Returns:
            分区ID到名称的映射
        """
        # B站主要分区
        return {
            1: '动画',
            13: '番剧',
            167: '国创',
            3: '音乐',
            129: '舞蹈',
            4: '游戏',
            36: '知识',
            188: '数码',
            160: '生活',
            211: '美食',
            217: '动物圈',
            119: '鬼畜',
            155: '时尚',
            165: '广告',
            5: '娱乐',
            181: '影视',
            177: '纪录片',
            23: '电影',
            11: '电视剧',
            138: '搬运·转载',
        }
    
    # =========================================================================
    # 合集管理功能
    # =========================================================================
    
    def _get_authenticated_session(self) -> 'requests.Session':
        """获取已认证的 requests session"""
        import requests
        
        self._load_cookie()
        
        session = requests.Session()
        # 设置 cookies 到正确的域名
        for name, value in self.cookie_data.items():
            session.cookies.set(name, value, domain='.bilibili.com')
        
        session.headers.update({
            'user-agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
            'referer': "https://member.bilibili.com/",
            'origin': "https://member.bilibili.com",
        })
        
        return session
    
    def list_seasons(self) -> List[Dict[str, Any]]:
        """
        获取用户的合集列表
        
        Returns:
            合集列表，每个元素包含 {season_id, name, description, cover, total}
        """
        session = self._get_authenticated_session()
        
        try:
            # 获取用户合集列表
            resp = session.get(
                'https://member.bilibili.com/x2/creative/web/seasons',
                params={'pn': 1, 'ps': 50},
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') != 0:
                logger.error(f"获取合集列表失败: {data.get('message')}")
                return []
            
            seasons = []
            # API 返回格式: data.seasons (不是 seasonList)
            for item in data.get('data', {}).get('seasons', []):
                season_info = item.get('season', {})
                # 视频数量在 sections.sections[0].epCount 中
                ep_count = 0
                sections_data = item.get('sections', {}).get('sections', [])
                if sections_data:
                    ep_count = sections_data[0].get('epCount', 0)
                
                seasons.append({
                    'season_id': season_info.get('id'),
                    'name': season_info.get('title'),
                    'description': season_info.get('desc', ''),
                    'cover': season_info.get('cover', ''),
                    'total': ep_count,
                })
            
            return seasons
            
        except Exception as e:
            logger.error(f"获取合集列表异常: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def add_to_season(self, aid: int, season_id: int) -> bool:
        """
        将视频添加到合集
        
        Args:
            aid: 视频AV号（整数）
            season_id: 合集ID
            
        Returns:
            是否成功
        """
        session = self._get_authenticated_session()
        bili_jct = self.cookie_data.get('bili_jct', '')
        
        try:
            # 添加视频到合集
            resp = session.post(
                'https://member.bilibili.com/x2/creative/web/season/section/add',
                data={
                    'season_id': season_id,
                    'aids': str(aid),
                    'csrf': bili_jct,
                },
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') == 0:
                logger.info(f"成功添加视频 av{aid} 到合集 {season_id}")
                return True
            else:
                logger.error(f"添加到合集失败: {data.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"添加到合集异常: {e}")
            return False
    
    def create_season(self, title: str, description: str = '') -> dict:
        """
        创建新合集
        
        Args:
            title: 合集标题
            description: 合集简介
            
        Returns:
            {'success': True, 'season_id': int} 或 {'success': False, 'error': str}
        """
        session = self._get_authenticated_session()
        bili_jct = self.cookie_data.get('bili_jct', '')
        
        try:
            resp = session.post(
                'https://member.bilibili.com/x2/creative/web/season/add',
                data={
                    'title': title,
                    'desc': description,
                    'cover': '',  # 可选封面
                    'csrf': bili_jct,
                },
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') == 0:
                season_id = data.get('data', {}).get('season_id')
                logger.info(f"成功创建合集: {title}, ID: {season_id}")
                return {'success': True, 'season_id': season_id}
            else:
                error_msg = data.get('message', '未知错误')
                error_code = data.get('code', 'N/A')
                logger.error(f"创建合集失败: code={error_code}, message={error_msg}")
                # -400 通常表示请求参数错误或 API 变更
                if error_code == -400:
                    error_msg = f"API 请求错误 (code={error_code})，可能是 B站 API 变更或需要在网页端操作"
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            logger.error(f"创建合集异常: {e}")
            return {'success': False, 'error': str(e)}
    
    def bvid_to_aid(self, bvid: str) -> Optional[int]:
        """
        将BV号转换为AV号
        
        Args:
            bvid: BV号
            
        Returns:
            AV号，失败返回None
        """
        import requests
        
        try:
            resp = requests.get(
                'https://api.bilibili.com/x/web-interface/view',
                params={'bvid': bvid},
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') == 0:
                return data.get('data', {}).get('aid')
            else:
                logger.error(f"BV号转换失败: {data.get('message')}")
                return None
                
        except Exception as e:
            logger.error(f"BV号转换异常: {e}")
            return None


def create_bilibili_uploader(config: Any) -> BilibiliUploader:
    """
    从配置创建B站上传器
    
    Args:
        config: 配置对象
        
    Returns:
        B站上传器实例
    """
    return BilibiliUploader(
        cookies_file=config.uploader.bilibili.cookies_file,
        line=config.uploader.bilibili.line,
        threads=config.uploader.bilibili.threads
    )
