"""
B站上传器实现

直接使用 biliup 库的 Web 端 API 上传（TV 端 API 已停用）
"""
import json
import re
import time
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
    aid: int = 0
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
                    resp_data = ret.get('data', {})
                    bvid = resp_data.get('bvid', '')
                    aid = resp_data.get('aid', 0)
                    logger.info(f"上传成功: {title}, BV号: {bvid}, AV号: {aid}")
                    return UploadResult(success=True, bvid=bvid, aid=aid)
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
        title = metadata.get('title')
        if not title:
            raise ValueError(f"upload_with_metadata: metadata 中缺少 title，不能用文件名 '{video_path.stem}' 替代")
        return self.upload(
            video_path=video_path,
            title=title,
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
        将视频添加到合集（新版合集/SEASON）
        
        通过逆向 B站创作中心前端 JS 发现的正确调用格式：
        - 端点: /x2/creative/web/season/section/episodes/add
        - 参数: sectionId (驼峰), episodes (对象数组含 aid/cid/title), csrf
        - Content-Type: application/json
        - csrf 同时在 query string 和 JSON body 中
        
        Args:
            aid: 视频AV号（整数）
            season_id: 合集ID
            
        Returns:
            是否成功
        """
        session = self._get_authenticated_session()
        bili_jct = self.cookie_data.get('bili_jct', '')
        assert bili_jct, "bili_jct 为空，无法调用需要 CSRF 的 API（cookie 未正确加载？）"
        
        try:
            # 1. 获取 section_id（合集下的分区ID）
            season_info = self.get_season_episodes(season_id)
            if not season_info:
                logger.error(f"无法获取合集 {season_id} 的 section_id")
                return False
            section_id = season_info['section_id']
            
            # 检查视频是否已在合集中
            existing_aids = [ep['aid'] for ep in season_info.get('episodes', [])]
            if aid in existing_aids:
                logger.info(f"视频 av{aid} 已在合集 {season_id} 中，跳过添加")
                return True
            
            # 2. 获取视频的 cid 和 title（API 要求 episodes 对象包含这些字段）
            # 注意：刚上传的视频可能还未被公共 API 索引（需要转码/审核），
            # 此处仅尝试一次，延迟重试由外部 season_sync 负责。
            resp = session.get(
                'https://api.bilibili.com/x/web-interface/view',
                params={'aid': aid},
                timeout=10
            )
            view_data = resp.json()
            if view_data.get('code') != 0:
                logger.warning(f"获取视频信息失败 av{aid}: {view_data.get('message')}（可能尚未索引，稍后通过 upload sync 重试）")
                return False
            
            pages = view_data['data'].get('pages', [])
            if not pages:
                logger.error(f"视频 av{aid} 的 pages 为空，无法获取 cid")
                return False
            cid = pages[0]['cid']
            title = view_data['data'].get('title', '')
            if not title:
                logger.warning(f"视频 av{aid} 的 title 为空")
            
            # 3. 调用 episodes/add（经验证的正确格式）
            payload = {
                'sectionId': section_id,
                'episodes': [{
                    'title': title,
                    'aid': aid,
                    'cid': cid,
                    'charging_pay': 0,
                }],
                'csrf': bili_jct,
            }
            headers = {
                'Content-Type': 'application/json; charset=UTF-8',
                'Referer': 'https://member.bilibili.com/platform/upload-manager/article/season',
                'Origin': 'https://member.bilibili.com',
            }
            resp = session.post(
                f'https://member.bilibili.com/x2/creative/web/season/section/episodes/add?csrf={bili_jct}',
                json=payload,
                headers=headers,
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') == 0:
                logger.info(f"成功添加视频 av{aid} 到合集 {season_id} (section={section_id})")
                return True
            else:
                logger.error(f"添加到合集失败: code={data.get('code')}, message={data.get('message')}")
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
        assert bili_jct, "bili_jct 为空，无法调用需要 CSRF 的 API（cookie 未正确加载？）"
        
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
    
    def get_season_episodes(self, season_id: int) -> Optional[Dict[str, Any]]:
        """
        获取合集内的视频列表（创作中心接口）
        
        需要先获取 section_id，然后才能进行排序/删除操作。
        
        Args:
            season_id: 合集ID
            
        Returns:
            {'section_id': int, 'episodes': [{'aid': int, 'title': str, ...}]}
            失败返回 None
        """
        session = self._get_authenticated_session()
        
        try:
            # 先获取合集信息（包含 section_id）
            # list_seasons 返回的数据中包含 sections.sections[0].id
            resp = session.get(
                'https://member.bilibili.com/x2/creative/web/seasons',
                params={'pn': 1, 'ps': 50},
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') != 0:
                logger.error(f"获取合集列表失败: {data.get('message')}")
                return None
            
            # 找到目标合集的 section_id
            section_id = None
            for item in data.get('data', {}).get('seasons', []):
                season_info = item.get('season', {})
                if season_info.get('id') == season_id:
                    sections_data = item.get('sections', {}).get('sections', [])
                    if sections_data:
                        section_id = sections_data[0].get('id')
                    break
            
            if section_id is None:
                logger.error(f"未找到合集 {season_id} 的 section_id")
                return None
            
            # 用 section_id 获取视频列表
            resp2 = session.get(
                'https://member.bilibili.com/x2/creative/web/season/section',
                params={'id': section_id},
                timeout=10
            )
            data2 = resp2.json()
            
            if data2.get('code') != 0:
                logger.error(f"获取合集视频列表失败: {data2.get('message')}")
                return None
            
            episodes = data2.get('data', {}).get('episodes') or []
            logger.info(f"合集 {season_id} (section={section_id}) 共 {len(episodes)} 个视频")
            
            return {
                'section_id': section_id,
                'episodes': episodes,
            }
            
        except Exception as e:
            logger.error(f"获取合集视频列表异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def remove_from_season(self, aids: List[int], season_id: int) -> bool:
        """
        从合集中移除视频
        
        通过逆向前端 JS 发现的正确调用格式：
        - 端点: /x2/creative/web/season/section/episode/del（注意是单数 episode）
        - 参数: {id: episode_id}，episode_id 是视频在合集中的内部ID，不是 aid
        - 需要先通过 get_season_episodes 查找 aid → episode_id 的映射
        - 每次只能删除一个 episode
        
        Args:
            aids: 要移除的视频 AV 号列表
            season_id: 合集ID
            
        Returns:
            是否全部成功（部分成功也返回 False）
        """
        session = self._get_authenticated_session()
        bili_jct = self.cookie_data.get('bili_jct', '')
        assert bili_jct, "bili_jct 为空，无法调用需要 CSRF 的 API（cookie 未正确加载？）"
        
        try:
            # 获取合集视频列表，建立 aid → episode_id 映射
            season_info = self.get_season_episodes(season_id)
            if not season_info:
                return False
            
            aid_to_episode_id = {ep['aid']: ep['id'] for ep in season_info.get('episodes', [])}
            
            success_count = 0
            for aid in aids:
                episode_id = aid_to_episode_id.get(aid)
                if episode_id is None:
                    logger.warning(f"视频 av{aid} 不在合集 {season_id} 中，跳过")
                    continue
                
                resp = session.post(
                    'https://member.bilibili.com/x2/creative/web/season/section/episode/del',
                    data={'id': episode_id, 'csrf': bili_jct},
                    timeout=10
                )
                data = resp.json()
                
                if data.get('code') == 0:
                    logger.info(f"成功从合集 {season_id} 移除视频 av{aid} (episode={episode_id})")
                    success_count += 1
                else:
                    logger.error(f"从合集移除视频 av{aid} 失败: code={data.get('code')}, message={data.get('message')}")
            
            total = len(aids)
            if success_count == total:
                logger.info(f"成功从合集 {season_id} 移除全部 {total} 个视频")
                return True
            else:
                logger.warning(f"从合集 {season_id} 移除视频: {success_count}/{total} 成功")
                return False
                
        except Exception as e:
            logger.error(f"从合集移除视频异常: {e}")
            return False
    
    def sort_season_episodes(self, season_id: int, aids_in_order: List[int]) -> bool:
        """
        对合集内的视频重新排序
        
        通过浏览器抓包发现的正确调用格式：
        - 端点: /x2/creative/web/season/section/edit
        - Content-Type: application/json
        - csrf 只在 query string 中，body 中不含 csrf
        - 参数: {
            section: {id, type, seasonId, title},  // 必须包含完整的 section 信息
            sorts: [{id: episode_id, sort: 序号(1-indexed)}],  // 必须包含所有视频
            captcha_token: ""
          }
        - episode_id 是视频在合集中的内部ID，不是 aid
        - sorts 必须包含合集中的所有视频，不能只传部分
        
        Args:
            season_id: 合集ID
            aids_in_order: 按期望顺序排列的 aid 列表，必须包含合集中的所有视频
            
        Returns:
            是否成功
        """
        session = self._get_authenticated_session()
        bili_jct = self.cookie_data.get('bili_jct', '')
        assert bili_jct, "bili_jct 为空，无法调用需要 CSRF 的 API（cookie 未正确加载？）"
        
        try:
            # 获取合集当前状态
            season_info = self.get_season_episodes(season_id)
            if not season_info:
                return False
            
            section_id = season_info['section_id']
            all_episodes = season_info.get('episodes', [])
            
            # 建立 aid → episode 映射
            aid_to_episode = {ep['aid']: ep for ep in all_episodes}
            
            # 校验所有 aid 都在合集中
            missing = [aid for aid in aids_in_order if aid not in aid_to_episode]
            if missing:
                logger.error(f"以下 aid 不在合集 {season_id} 中: {missing}")
                return False
            
            # 如果传入的 aids 不是全量，补充未列出的视频到末尾
            listed_aids = set(aids_in_order)
            remaining = [ep['aid'] for ep in all_episodes if ep['aid'] not in listed_aids]
            full_order = list(aids_in_order) + remaining
            
            # 生成 sorts 数组（1-indexed）
            sorts = []
            for idx, aid in enumerate(full_order):
                ep = aid_to_episode[aid]
                sorts.append({'id': ep['id'], 'sort': idx + 1})
            
            # section 对象必须包含 id, type, seasonId, title
            payload = {
                'section': {
                    'id': section_id,
                    'type': 1,
                    'seasonId': season_id,
                    'title': '正片',
                },
                'sorts': sorts,
                'captcha_token': '',
            }
            headers = {
                'Content-Type': 'application/json',
                'Referer': 'https://member.bilibili.com/platform/upload-manager/ep',
                'Origin': 'https://member.bilibili.com',
            }
            resp = session.post(
                f'https://member.bilibili.com/x2/creative/web/season/section/edit?csrf={bili_jct}',
                json=payload,
                headers=headers,
                timeout=15
            )
            data = resp.json()
            
            if data.get('code') == 0:
                logger.info(f"合集 {season_id} 排序成功，共 {len(sorts)} 个视频")
                return True
            else:
                logger.error(f"合集排序失败: code={data.get('code')}, message={data.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"合集排序异常: {e}")
            return False
    
    @staticmethod
    def _extract_title_index(title: str) -> Optional[int]:
        """从标题中提取 #数字，如 '【xxx】翻译标题 | #42' → 42"""
        m = re.search(r'#(\d+)\s*$', title)
        return int(m.group(1)) if m else None

    def auto_sort_season(self, season_id: int, newly_added_aid: Optional[int] = None) -> bool:
        """
        按标题中的 #数字 对合集自动排序
        
        规则：
        - 从每个视频标题末尾解析 #数字 作为排序键
        - 无 #数字 的视频视为最老，排在最前面（按 episode_id 保持相对顺序）
        - 如果新添加的视频 #数字 >= 当前合集中最大的 #数字，
          说明是顺序上传（追加到末尾即为正确位置），跳过排序
        
        Args:
            season_id: 合集ID
            newly_added_aid: 刚添加的视频 aid，用于判断是否需要排序
            
        Returns:
            是否成功（跳过排序也算成功）
        """
        try:
            season_info = self.get_season_episodes(season_id)
            if not season_info:
                return False
            
            episodes = season_info.get('episodes', [])
            if len(episodes) <= 1:
                return True
            
            # 解析每个 episode 的 #数字
            ep_with_idx = []
            for ep in episodes:
                idx = self._extract_title_index(ep.get('title', ''))
                ep_with_idx.append((ep, idx))
            
            # 判断是否需要排序：新视频的 # 是最大的 → 顺序上传，跳过
            if newly_added_aid is not None:
                new_ep_idx = None
                max_existing_idx = -1
                for ep, idx in ep_with_idx:
                    if ep['aid'] == newly_added_aid:
                        new_ep_idx = idx
                    else:
                        if idx is not None and idx > max_existing_idx:
                            max_existing_idx = idx
                
                if new_ep_idx is not None and new_ep_idx >= max_existing_idx:
                    logger.info(f"视频 av{newly_added_aid} (#{new_ep_idx}) 已在合集末尾，无需排序")
                    return True
            
            # 排序：无 #数字 的排最前（用 -1），有 #数字 的按数字升序
            # 同为无 #数字 的保持原始相对顺序（episode_id）
            def sort_key(item):
                ep, idx = item
                if idx is None:
                    return (0, ep['id'])  # 无编号：排最前，按 episode_id 保序
                return (1, idx)           # 有编号：排后面，按 #数字
            
            sorted_eps = sorted(ep_with_idx, key=sort_key)
            sorted_aids = [ep['aid'] for ep, _ in sorted_eps]
            
            # 检查排序前后是否一致，一致则跳过
            current_aids = [ep['aid'] for ep in episodes]
            if sorted_aids == current_aids:
                logger.info(f"合集 {season_id} 已是正确顺序，无需排序")
                return True
            
            logger.info(f"合集 {season_id} 需要排序，当前 {len(episodes)} 个视频")
            return self.sort_season_episodes(season_id, sorted_aids)
            
        except Exception as e:
            logger.error(f"合集自动排序异常: {e}")
            return False

    def delete_video(self, aid: int) -> bool:
        """
        删除自己的视频（稿件）
        
        警告：此操作不可逆！
        
        Args:
            aid: 视频AV号
            
        Returns:
            是否成功
        """
        session = self._get_authenticated_session()
        bili_jct = self.cookie_data.get('bili_jct', '')
        assert bili_jct, "bili_jct 为空，无法调用需要 CSRF 的 API（cookie 未正确加载？）"
        
        try:
            resp = session.post(
                'https://member.bilibili.com/x/web/archive/delete',
                data={
                    'aid': aid,
                    'csrf': bili_jct,
                },
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') == 0:
                logger.info(f"成功删除视频 av{aid}")
                return True
            else:
                logger.error(f"删除视频失败: {data.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"删除视频异常: {e}")
            return False
    
    def get_my_videos(self, page: int = 1, page_size: int = 30) -> Optional[Dict[str, Any]]:
        """
        获取自己的稿件列表
        
        Args:
            page: 页码
            page_size: 每页数量
            
        Returns:
            {'total': int, 'videos': [{'aid': int, 'bvid': str, 'title': str, ...}]}
            失败返回 None
        """
        session = self._get_authenticated_session()
        
        try:
            resp = session.get(
                'https://member.bilibili.com/x/web/archives',
                params={
                    'pn': page,
                    'ps': page_size,
                    'status': '',  # 空=全部
                    'tid': 0,
                    'keyword': '',
                },
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') != 0:
                logger.error(f"获取稿件列表失败: {data.get('message')}")
                return None
            
            arc_data = data.get('data', {})
            videos = []
            for item in arc_data.get('arc_audits', []):
                archive = item.get('Archive', {})
                videos.append({
                    'aid': archive.get('aid'),
                    'bvid': archive.get('bvid'),
                    'title': archive.get('title', ''),
                    'state': archive.get('state', 0),  # 0=正常
                    'state_desc': archive.get('state_desc', ''),
                })
            
            return {
                'total': arc_data.get('page', {}).get('count', 0),
                'videos': videos,
            }
            
        except Exception as e:
            logger.error(f"获取稿件列表异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_video_detail(self, aid: int) -> Optional[Dict[str, Any]]:
        """
        获取视频详情（公共 API），返回完整视频信息。
        
        Args:
            aid: 视频AV号
            
        Returns:
            视频信息字典（含 aid, bvid, title, desc, tid, tag, copyright, source, cover, pages 等），
            失败返回 None
        """
        session = self._get_authenticated_session()
        try:
            resp = session.get(
                'https://api.bilibili.com/x/web-interface/view',
                params={'aid': aid},
                timeout=10
            )
            data = resp.json()
            if data.get('code') != 0:
                logger.error(f"获取视频详情失败 av{aid}: {data.get('message')}")
                return None
            return data['data']
        except Exception as e:
            logger.error(f"获取视频详情异常 av{aid}: {e}")
            return None
    
    def edit_video_info(
        self,
        aid: int,
        title: Optional[str] = None,
        desc: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tid: Optional[int] = None,
    ) -> bool:
        """
        修改已上传视频的标题、简介、标签、分区等信息。
        
        流程：先获取当前视频信息，修改指定字段，然后提交编辑。
        
        API 端点: POST https://member.bilibili.com/x/vu/web/edit?csrf={bili_jct}
        Content-Type: application/json
        
        Args:
            aid: 视频AV号
            title: 新标题（None=不修改）
            desc: 新简介（None=不修改）
            tags: 新标签列表（None=不修改）
            tid: 新分区ID（None=不修改）
            
        Returns:
            是否成功
        """
        session = self._get_authenticated_session()
        bili_jct = self.cookie_data.get('bili_jct', '')
        assert bili_jct, "bili_jct 为空，无法调用需要 CSRF 的 API（cookie 未正确加载？）"
        
        # 1. 从创作中心 API 获取完整稿件信息（含正确的 filename、source 等）
        #    端点: /x/client/archive/view（/x/web/archive/view 已失效返回 404）
        archive_data = None
        try:
            archive_resp = session.get(
                'https://member.bilibili.com/x/client/archive/view',
                params={'aid': aid},
                timeout=10
            )
            resp_data = archive_resp.json()
            if resp_data.get('code') == 0:
                archive_data = resp_data.get('data', {})
        except Exception as e:
            logger.warning(f"创作中心 API 获取 av{aid} 失败: {e}")
        
        if not archive_data:
            logger.error(f"无法从创作中心获取视频 av{aid} 的稿件信息，跳过编辑")
            return False
        
        arc = archive_data.get('archive', {})
        arc_videos = archive_data.get('videos', [])
        
        # 2. 构建 videos 数组（必须包含所有分P，且 filename 必须正确）
        videos = []
        for v in arc_videos:
            videos.append({
                'filename': v.get('filename', ''),
                'title': v.get('title', 'P1'),
                'desc': v.get('desc', ''),
                'cid': v.get('cid'),
            })
        if not videos:
            logger.error(f"视频 av{aid} 的 videos 为空，无法编辑")
            return False
        
        # 3. 获取当前标签和完整简介（需要公共 API）
        #    创作中心 API 会截断 desc 到 250 字符，必须从公共 API 获取完整 desc
        current_tags = ''
        full_desc = arc.get('desc', '')  # 创作中心返回的截断 desc，作为 fallback
        detail = self.get_video_detail(aid)
        if detail:
            if not tags:
                current_tags = detail.get('tag', '')
                if isinstance(current_tags, list):
                    current_tags = ','.join(
                        t.get('tag_name', '') if isinstance(t, dict) else str(t) for t in current_tags
                    )
            # 公共 API 的 desc 通常比创作中心的更完整
            pub_desc = detail.get('desc', '')
            if len(pub_desc) > len(full_desc):
                full_desc = pub_desc
        
        payload = {
            'aid': aid,
            'copyright': arc.get('copyright', 1),
            'title': (title or arc.get('title', ''))[:80],
            'desc': (desc if desc is not None else full_desc)[:2000],
            'tag': ','.join(tags[:12]) if tags else current_tags,
            'tid': tid or arc.get('tid', 21),
            'source': arc.get('source', ''),
            'cover': arc.get('cover', ''),
            'videos': videos,
            'csrf': bili_jct,
        }
        
        # 3. 提交编辑
        try:
            headers = {
                'Content-Type': 'application/json; charset=UTF-8',
                'Referer': 'https://member.bilibili.com/platform/upload/video/frame',
                'Origin': 'https://member.bilibili.com',
            }
            resp = session.post(
                f'https://member.bilibili.com/x/vu/web/edit?csrf={bili_jct}',
                json=payload,
                headers=headers,
                timeout=15
            )
            data = resp.json()
            
            if data.get('code') == 0:
                changes = []
                if title:
                    changes.append(f"标题→{title[:30]}")
                if desc is not None:
                    changes.append(f"简介({len(desc)}字)")
                if tags:
                    changes.append(f"标签({len(tags)}个)")
                if tid:
                    changes.append(f"分区→{tid}")
                logger.info(f"成功编辑视频 av{aid}: {', '.join(changes) or '无变更'}")
                return True
            else:
                logger.error(f"编辑视频失败 av{aid}: code={data.get('code')}, message={data.get('message')}")
                return False
        except Exception as e:
            logger.error(f"编辑视频异常 av{aid}: {e}")
            return False

    def get_rejected_videos(self, keyword: str = '') -> List[Dict[str, Any]]:
        """
        获取被退回的稿件列表及其审核详情（违规时间段等）
        
        通过创作中心 /x/web/archives?status=not_pubed 获取退回稿件，
        解析 problem_detail 中的违规时间段。
        
        Args:
            keyword: 可选的搜索关键词
            
        Returns:
            列表，每项包含:
            {
                'aid': int,
                'bvid': str,
                'title': str,
                'state': int,
                'reject_reason': str,
                'problems': [
                    {
                        'reason': str,          # 退回原因
                        'violation_time': str,   # 原始时间字符串，如 "P1(00:20:18-00:20:24)"
                        'violation_position': str, # "内容" / "口播" / "内容全程"
                        'time_ranges': [(start_sec, end_sec), ...],  # 解析后的秒数
                        'is_full_video': bool,   # 是否全片违规
                        'modify_advise': str,
                    }
                ],
            }
        """
        session = self._get_authenticated_session()
        
        try:
            resp = session.get(
                'https://member.bilibili.com/x/web/archives',
                params={
                    'status': 'not_pubed',
                    'pn': 1,
                    'ps': 50,
                    'keyword': keyword,
                    'interactive': 1,
                },
                timeout=10
            )
            data = resp.json()
            
            if data.get('code') != 0:
                logger.error(f"获取退回稿件失败: {data.get('message')}")
                return []
            
            results = []
            for item in data.get('data', {}).get('arc_audits', []):
                archive = item.get('Archive', {})
                problem_detail = item.get('problem_detail') or []
                
                problems = []
                for pd in problem_detail:
                    vt = pd.get('violation_time', '')
                    vp = pd.get('violation_position', '')
                    reason = pd.get('reject_reason', '')
                    
                    # 优先从 violation_time 解析；若为空，尝试从 reason 中提取
                    time_ranges = self._parse_violation_time(vt) if vt else []
                    if not time_ranges and reason:
                        time_ranges = self._parse_violation_time(reason)
                    
                    # is_full_video: 明确标注「内容全程」或 无任何可解析的时间段
                    is_full = vp == '内容全程' or (not vt and not vp and not time_ranges)
                    
                    problems.append({
                        'reason': reason,
                        'violation_time': vt,
                        'violation_position': vp,
                        'time_ranges': time_ranges,
                        'is_full_video': is_full,
                        'modify_advise': pd.get('modify_advise', ''),
                    })
                
                results.append({
                    'aid': archive.get('aid'),
                    'bvid': archive.get('bvid', ''),
                    'title': archive.get('title', ''),
                    'state': archive.get('state', 0),
                    'reject_reason': archive.get('reject_reason', ''),
                    'problems': problems,
                })
            
            return results
            
        except Exception as e:
            logger.error(f"获取退回稿件异常: {e}")
            return []
    
    @staticmethod
    def _parse_violation_time(text: str) -> List[tuple]:
        """
        解析违规时间字符串为 (start_seconds, end_seconds) 列表
        
        支持的格式（按优先级）：
        - "P1(00:20:18-00:20:24)"  — 标准格式
        - "P1(00:20:18-00:20:24)、P1(00:23:33-00:23:35)"  — 多段
        - "【23:28-23:29】"  — B站新格式（MM:SS 在中括号内）
        - "【00:23:28-00:23:29】"  — B站新格式（HH:MM:SS 在中括号内）
        
        也支持从 reject_reason 文本中提取（如 "您的视频【23:28-23:29】..."）
        
        Returns:
            [(start_sec, end_sec), ...]
        """
        ranges = []
        
        # 格式1: P数字(HH:MM:SS-HH:MM:SS)
        pattern_p = r'P\d+\((\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})\)'
        for m in re.finditer(pattern_p, text):
            start_sec = BilibiliUploader._time_to_seconds(m.group(1))
            end_sec = BilibiliUploader._time_to_seconds(m.group(2))
            if start_sec is not None and end_sec is not None:
                ranges.append((start_sec, end_sec))
        
        # 格式2: 【时间-时间】（支持 MM:SS 或 HH:MM:SS，兼容全角冒号 ： 和冒号后空格）
        if not ranges:
            pattern_bracket = r'【(\d{1,2}[:：]\s*\d{2}(?:[:：]\s*\d{2})?)[\s]*[-–—][\s]*(\d{1,2}[:：]\s*\d{2}(?:[:：]\s*\d{2})?)】'
            for m in re.finditer(pattern_bracket, text):
                start_sec = BilibiliUploader._time_to_seconds(m.group(1))
                end_sec = BilibiliUploader._time_to_seconds(m.group(2))
                if start_sec is not None and end_sec is not None:
                    ranges.append((start_sec, end_sec))
        
        if not ranges and text.strip():
            logger.warning(f"无法解析违规时间: {text}")
        
        return ranges
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> Optional[float]:
        """将 HH:MM:SS 或 MM:SS 转换为秒数（兼容全角冒号 ：）"""
        # 统一全角冒号为半角，再去除多余空格
        normalized = time_str.replace('\uff1a', ':').replace(' ', '')
        parts = normalized.split(':')
        if len(parts) == 3:
            try:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            except ValueError:
                return None
        elif len(parts) == 2:
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except ValueError:
                return None
        return None

    def get_archive_detail(self, aid: int) -> Optional[Dict[str, Any]]:
        """
        获取稿件详情（创作中心接口），含 filename 等上传信息。
        用于获取被退回稿件的完整信息以便重新编辑上传。
        
        Args:
            aid: 稿件AV号
            
        Returns:
            {'archive': {...}, 'videos': [{...}]} 或 None
        """
        session = self._get_authenticated_session()
        try:
            resp = session.get(
                f'https://member.bilibili.com/x/client/archive/view?aid={aid}',
                timeout=10
            )
            data = resp.json()
            if data.get('code') != 0:
                logger.error(f"获取稿件详情失败 av{aid}: {data.get('message')}")
                return None
            return data.get('data')
        except Exception as e:
            logger.error(f"获取稿件详情异常 av{aid}: {e}")
            return None

    def _get_full_desc(self, aid: int, session=None) -> Optional[str]:
        """从公共 API 获取完整的视频简介（创作中心 API 会截断 desc 到 250 字符）
        
        Args:
            aid: 稿件AV号
            session: 已认证的 requests session，未提供则自动获取
            
        Returns:
            完整 desc 字符串，获取失败返回 None
        """
        if session is None:
            session = self._get_authenticated_session()
        try:
            resp = session.get(
                f'https://api.bilibili.com/x/web-interface/view?aid={aid}',
                timeout=10
            )
            data = resp.json()
            if data.get('code') == 0:
                return data.get('data', {}).get('desc', '')
        except Exception as e:
            logger.warning(f"从公共 API 获取 desc 失败 av{aid}: {e}")
        return None

    def replace_video(self, aid: int, new_video_path: Path) -> bool:
        """
        替换被退回稿件的视频文件并重新提交审核。
        
        流程：
        1. 上传新的视频文件到 B站（获取新 filename）
        2. 用新 filename 编辑稿件，替换原视频
        
        Args:
            aid: 稿件 AV号
            new_video_path: 新视频文件路径
            
        Returns:
            是否成功
        """
        session = self._get_authenticated_session()
        bili_jct = self.cookie_data.get('bili_jct', '')
        assert bili_jct, "bili_jct 为空"
        
        # 1. 获取稿件当前信息
        detail = self.get_archive_detail(aid)
        if not detail:
            return False
        
        archive = detail['archive']
        old_videos = detail['videos']
        
        if not old_videos:
            logger.error(f"稿件 av{aid} 无视频信息")
            return False
        
        # 创作中心 API 会截断 desc 到 250 字符，需要从公共 API 获取完整 desc
        creator_desc = archive.get('desc', '')
        full_desc = self._get_full_desc(aid, session)
        if full_desc and len(full_desc) > len(creator_desc):
            archive['desc'] = full_desc
            logger.info(f"  使用公共 API 获取完整 desc ({len(full_desc)} 字符，创作中心仅 {len(creator_desc)})")
        
        logger.info(f"替换视频 av{aid}: {archive.get('title', '')[:50]}")
        logger.info(f"  新视频文件: {new_video_path}")
        
        # 2. 上传新视频文件（只上传文件，不创建稿件）
        try:
            from biliup.plugins.bili_webup import BiliBili, Data
            
            self._load_cookie()
            video_part = Data()
            video_part.title = old_videos[0].get('title', '')
            video_part.desc = old_videos[0].get('desc', '')
            
            with BiliBili(video_part) as bili:
                bili.login_by_cookies(self._raw_cookie_data)
                
                # 上传视频文件
                new_video_path = Path(new_video_path)
                uploaded = bili.upload_file(str(new_video_path), lines=self.line, tasks=self.threads)
                logger.info(f"  upload_file 返回值: {uploaded}")
                
                if not uploaded:
                    logger.error(f"视频文件上传失败: 返回值为空")
                    return False
                
                # biliup upload_file 返回 dict，filename 字段名可能是 'bili_filename' 或 'filename'
                new_filename = uploaded.get('bili_filename') or uploaded.get('filename')
                if not new_filename:
                    logger.error(f"视频文件上传后无 filename 字段: {uploaded}")
                    return False
                
                logger.info(f"  新视频上传成功: filename={new_filename}")
                
        except Exception as e:
            logger.error(f"上传新视频文件异常: {e}")
            return False
        
        # 3. 编辑稿件，替换视频 filename
        try:
            edit_payload = {
                'aid': aid,
                'copyright': archive.get('copyright', 2),
                'title': archive.get('title', ''),
                'tag': archive.get('tag', ''),
                'tid': archive.get('tid', 21),
                'desc': archive.get('desc', ''),
                'source': archive.get('source', ''),
                'cover': archive.get('cover', ''),
                'videos': [{
                    'filename': new_filename,
                    'title': old_videos[0].get('title', ''),
                    'desc': old_videos[0].get('desc', ''),
                }],
            }
            
            resp = session.post(
                f'https://member.bilibili.com/x/vu/web/edit?csrf={bili_jct}',
                json=edit_payload,
                headers={
                    'Referer': 'https://member.bilibili.com/',
                    'Origin': 'https://member.bilibili.com',
                },
                timeout=15
            )
            result = resp.json()
            
            if result.get('code') == 0:
                logger.info(f"  ✅ 稿件 av{aid} 视频已替换，已重新提交审核")
                return True
            else:
                logger.error(f"  编辑稿件失败: code={result.get('code')}, msg={result.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"编辑稿件异常: {e}")
            return False

    def bvid_to_aid(self, bvid: str) -> Optional[int]:
        """
        将BV号转换为AV号
        
        Args:
            bvid: BV号
            
        Returns:
            AV号，失败返回None
        """
        try:
            session = self._get_authenticated_session()
            resp = session.get(
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

    # =========================================================================
    # 违规视频修复功能
    # =========================================================================

    def download_video(self, aid: int, output_path: Path, quality: int = 127) -> bool:
        """
        从 B站下载已上传的视频（DASH 视频+音频流，ffmpeg 合并）。
        
        用于违规视频修复的降级路径：当本地原始文件不存在时，从 B站下载当前版本。
        注意：B站会对上传视频重新编码（分辨率/码率/编码均可能降低），
        下载的版本质量低于原始上传文件。应优先使用本地原始文件。
        
        Args:
            aid: 稿件 AV号
            output_path: 输出文件路径（.mp4）
            quality: 期望画质 ID（127=8K, 120=4K, 116=1080P60, 112=1080P+, 80=1080P）
            
        Returns:
            是否成功
        """
        import subprocess
        import requests as req
        
        session = self._get_authenticated_session()
        
        # 1. 获取 cid
        # 优先从创作中心获取（支持未发布/退回视频），回退到公共 API
        cid = None
        detail = self.get_archive_detail(aid)
        if detail and detail.get('videos'):
            cid = detail['videos'][0].get('cid')
        
        if not cid:
            resp = session.get(
                f'https://api.bilibili.com/x/web-interface/view?aid={aid}',
                timeout=10
            )
            view_data = resp.json()
            if view_data.get('code') == 0:
                pages = view_data.get('data', {}).get('pages', [])
                if pages:
                    cid = pages[0].get('cid')
        
        if not cid:
            logger.error(f"无法获取 av{aid} 的 cid")
            return False
        
        # 2. 获取 DASH 播放流
        resp = session.get(
            'https://api.bilibili.com/x/player/playurl',
            params={
                'avid': aid, 'cid': cid,
                'qn': quality, 'fnval': 16, 'fnver': 0, 'fourk': 1,
            },
            timeout=10
        )
        play_data = resp.json()
        if play_data.get('code') != 0:
            logger.error(f"获取播放流失败 av{aid}: {play_data.get('message')}")
            return False
        
        dash = play_data.get('data', {}).get('dash')
        if not dash:
            logger.error(f"av{aid} 无 DASH 流（可能仅支持 FLV）")
            return False
        
        video_streams = dash.get('video', [])
        audio_streams = dash.get('audio', [])
        if not video_streams or not audio_streams:
            logger.error(f"av{aid} DASH 流为空: video={len(video_streams)} audio={len(audio_streams)}")
            return False
        
        # 选最高码率的视频和音频
        best_video = max(video_streams, key=lambda x: x['bandwidth'])
        best_audio = max(audio_streams, key=lambda x: x['bandwidth'])
        
        logger.info(
            f"下载 av{aid}: 视频 {best_video.get('width','?')}x{best_video.get('height','?')} "
            f"{best_video['codecs']} {best_video['bandwidth']//1000}kbps, "
            f"音频 {best_audio['codecs']} {best_audio['bandwidth']//1000}kbps"
        )
        
        # 3. 用 ffmpeg 下载并合并 DASH 流（-c copy 不重新编码）
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        video_url = best_video['baseUrl']
        audio_url = best_audio['baseUrl']
        
        cmd = [
            'ffmpeg', '-y',
            '-headers', 'Referer: https://www.bilibili.com/\r\nUser-Agent: Mozilla/5.0\r\n',
            '-i', video_url,
            '-headers', 'Referer: https://www.bilibili.com/\r\nUser-Agent: Mozilla/5.0\r\n',
            '-i', audio_url,
            '-c', 'copy',
            '-movflags', '+faststart',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                logger.error(f"ffmpeg 合并失败: {result.stderr[-500:]}")
                return False
            
            if not output_path.exists():
                logger.error("ffmpeg 完成但输出文件不存在")
                return False
            
            size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"✅ 下载完成: {output_path.name} ({size_mb:.1f}MB)")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg 下载合并超时 (>30分钟)")
            return False
        except Exception as e:
            logger.error(f"下载异常: {e}")
            return False

    def fix_violation(
        self,
        aid: int,
        video_path: Optional[Path] = None,
        mask_text: str = "此处内容因平台合规要求已被遮罩",
        margin_sec: float = 2.0,
        previous_ranges: Optional[List[tuple]] = None,
        dry_run: bool = False,
        callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        修复被退回的违规稿件：获取违规信息 → 合并历史mask → 遮罩 → 替换上传。
        
        累积式修复策略：
        - 合并本次新违规 + 之前已 mask 的时间段，全部应用到原始文件
        - 优先使用本地原始文件（避免 B站转码的质量损失）
        - 本地文件不存在时自动从 B站下载（降级路径，有质量损失警告）
        
        Args:
            aid: 稿件 AV号
            video_path: 视频源文件路径。为 None 时自动从 B站下载（降级）
            mask_text: 遮罩区域显示文字
            margin_sec: 违规区间前后安全边距（秒）
            previous_ranges: 之前已 mask 的时间段列表 [(start, end), ...]
            dry_run: True 时只做遮罩不上传
            callback: 进度回调 callback(message: str)
            
        Returns:
            {
                'success': bool,
                'new_ranges': [(start, end), ...],      # 本次新违规
                'all_ranges': [(start, end), ...],      # 所有 mask（旧+新合并后）
                'masked_path': str | None,              # 遮罩文件路径（dry_run 时保留）
                'source': 'local' | 'bilibili',         # 视频来源
                'message': str,
            }
        """
        from ..embedder.ffmpeg_wrapper import FFmpegWrapper
        
        def _cb(msg):
            if callback:
                callback(msg)
            logger.info(msg)
        
        result = {
            'success': False, 'new_ranges': [], 'all_ranges': [],
            'masked_path': None, 'source': 'local', 'message': '',
        }
        
        # Step 1: 获取退回信息
        _cb(f"获取 av{aid} 审核退回信息...")
        rejected = self.get_rejected_videos()
        target = [v for v in rejected if v['aid'] == aid]
        
        if not target:
            result['message'] = f"未找到 aid={aid} 的退回稿件"
            return result
        
        t = target[0]
        _cb(f"  标题: {t['title'][:60]}")
        
        # 收集本次新违规时间段
        new_ranges = []
        for p in t['problems']:
            new_ranges.extend(p['time_ranges'])
            _cb(f"  违规: {p['reason'][:50]}  时间: {p['violation_time']}")
            if p['is_full_video']:
                result['message'] = "全片违规，无法通过遮罩修复"
                return result
        
        if not new_ranges:
            result['message'] = "无具体违规时间段，无法自动修复"
            return result
        
        result['new_ranges'] = new_ranges
        
        # Step 2: 合并历史 mask ranges + 本次新 ranges
        all_ranges = list(previous_ranges or []) + new_ranges
        _cb(f"  本次新违规: {new_ranges}")
        if previous_ranges:
            _cb(f"  历史已 mask: {previous_ranges}")
            _cb(f"  合并后总计: {len(all_ranges)} 段")
        
        # Step 3: 确定视频源
        import tempfile
        source_video = None
        source_type = 'local'
        tmp_download = None
        
        if video_path and Path(video_path).exists():
            source_video = Path(video_path)
            _cb(f"  使用本地文件: {source_video} ({source_video.stat().st_size / 1024 / 1024:.0f}MB)")
        else:
            # 降级：从 B站下载
            source_type = 'bilibili'
            _cb("  ⚠️ 本地文件不可用，从 B站下载（质量会降低）...")
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"bili_fix_{aid}_"))
            tmp_download = tmp_dir / f"av{aid}_source.mp4"
            
            if not self.download_video(aid, tmp_download):
                result['message'] = "从 B站下载视频失败"
                return result
            
            source_video = tmp_download
            _cb(f"  B站下载完成: {source_video.stat().st_size / 1024 / 1024:.0f}MB")
        
        result['source'] = source_type
        
        # Step 4: ffmpeg 遮罩
        _cb("开始遮罩处理...")
        ffmpeg = FFmpegWrapper()
        masked_path = source_video.parent / f"{source_video.stem}_masked{source_video.suffix}"
        
        ok = ffmpeg.mask_violation_segments(
            video_path=source_video,
            output_path=masked_path,
            violation_ranges=all_ranges,
            mask_text=mask_text,
            margin_sec=margin_sec,
        )
        
        if not ok:
            result['message'] = "遮罩处理失败"
            return result
        
        out_size = masked_path.stat().st_size / 1024 / 1024
        _cb(f"  遮罩完成: {masked_path.name} ({out_size:.0f}MB)")
        
        # 保存原始违规范围（不含 margin），供下次累积使用
        # margin 仅在 mask_violation_segments 内部动态应用，不持久化
        video_info = ffmpeg.get_video_info(source_video) or {}
        raw_merged = ffmpeg._merge_ranges(all_ranges, 0, 
                                          video_info.get('duration', 0))
        result['all_ranges'] = raw_merged
        result['masked_path'] = str(masked_path)
        
        # Step 5: 上传替换（除非 dry_run）
        if dry_run:
            _cb(f"  dry-run 模式，跳过上传。遮罩文件: {masked_path}")
            result['success'] = True
            result['message'] = 'dry-run 完成'
            return result
        
        _cb("上传替换...")
        import time as _time
        _upload_start = _time.monotonic()
        replace_ok = self.replace_video(aid, masked_path)
        result['upload_duration'] = _time.monotonic() - _upload_start
        
        if replace_ok:
            result['success'] = True
            result['message'] = '修复完成，已重新提交审核'
            _cb(f"  ✅ av{aid} 修复完成")
            # 清理遮罩临时文件
            masked_path.unlink(missing_ok=True)
            result['masked_path'] = None
        else:
            result['message'] = '上传替换失败，遮罩文件已保留'
            _cb(f"  ❌ 上传替换失败")
        
        # 清理下载的临时文件
        if tmp_download and tmp_download.exists():
            tmp_download.unlink(missing_ok=True)
        
        return result


def season_sync(db, uploader: BilibiliUploader, playlist_id: str) -> Dict[str, Any]:
    """
    批量将已上传但未入集的视频添加到 B站 合集，然后按 #数字 排序。
    
    查找指定 playlist 中所有满足以下条件的视频：
    - metadata 中有 bilibili_aid（已上传到B站）
    - metadata 中有 bilibili_target_season_id（配置了目标合集）
    - bilibili_season_added != True（尚未成功添加到合集）
    
    Args:
        db: 数据库实例
        uploader: B站上传器实例（已认证）
        playlist_id: Playlist ID，限定处理范围
        
    Returns:
        {
            'total': int,       # 待处理总数
            'success': int,     # 成功数
            'failed': int,      # 失败数
            'skipped': int,     # 跳过数（已在合集中）
            'season_ids': set,  # 涉及的合集ID（用于排序）
            'failed_videos': list,  # 失败的 video_id 列表
        }
    """
    from vat.services.playlist_service import PlaylistService
    
    playlist_service = PlaylistService(db)
    videos = playlist_service.get_playlist_videos(playlist_id)
    
    # 筛选待同步的视频
    pending = []
    for v in videos:
        meta = v.metadata or {}
        aid = meta.get('bilibili_aid')
        target_season = meta.get('bilibili_target_season_id')
        already_added = meta.get('bilibili_season_added', False)
        
        if aid and target_season and not already_added:
            pending.append((v, int(aid), int(target_season)))
    
    result = {
        'total': len(pending),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'season_ids': set(),
        'failed_videos': [],
    }
    
    if not pending:
        logger.info(f"Playlist {playlist_id}: 没有待同步的视频")
        return result
    
    logger.info(f"Playlist {playlist_id}: 找到 {len(pending)} 个待同步视频")
    
    for i, (video, aid, season_id) in enumerate(pending):
        result['season_ids'].add(season_id)
        try:
            if uploader.add_to_season(aid, season_id):
                result['success'] += 1
                # 更新 DB 标记
                updated_meta = dict(video.metadata or {})
                updated_meta['bilibili_season_added'] = True
                db.update_video(video.id, metadata=updated_meta)
                logger.info(f"✓ {video.title or video.id} -> 合集 {season_id}")
            else:
                result['failed'] += 1
                result['failed_videos'].append(video.id)
                logger.warning(f"✗ {video.title or video.id} -> 合集 {season_id} 失败")
        except Exception as e:
            result['failed'] += 1
            result['failed_videos'].append(video.id)
            logger.error(f"✗ {video.title or video.id} -> 合集 {season_id} 异常: {e}")
        # B站合集编辑 API 有频率限制（code=20111），请求间需间隔避免触发
        if i < len(pending) - 1:
            time.sleep(3)
    
    # 对涉及的每个合集执行排序
    for season_id in result['season_ids']:
        try:
            if uploader.auto_sort_season(season_id):
                logger.info(f"✓ 合集 {season_id} 排序完成")
            else:
                logger.warning(f"⚠ 合集 {season_id} 排序失败")
        except Exception as e:
            logger.warning(f"⚠ 合集 {season_id} 排序异常: {e}")
    
    logger.info(
        f"Season sync 完成: {result['success']} 成功, "
        f"{result['failed']} 失败, {result['skipped']} 跳过"
    )
    return result


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
