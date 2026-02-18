"""
B站上传器实现

直接使用 biliup 库的 Web 端 API 上传（TV 端 API 已停用）
"""
import json
import re
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
            resp = session.get(
                'https://api.bilibili.com/x/web-interface/view',
                params={'aid': aid},
                timeout=10
            )
            view_data = resp.json()
            if view_data.get('code') != 0:
                logger.error(f"获取视频信息失败 av{aid}: {view_data.get('message')}")
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
            
            episodes = data2.get('data', {}).get('episodes', [])
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
