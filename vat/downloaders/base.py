"""
下载器抽象基类

两层接口层级设计：
- BaseDownloader: 所有源类型的通用接口（LOCAL / DIRECT_URL / YouTube 等）
- PlatformDownloader: 平台下载器接口，在 BaseDownloader 基础上增加 playlist 等平台特有能力
"""
import subprocess
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any, Set

from vat.media import probe_media_info
from vat.utils.logger import setup_logger

_logger = setup_logger("downloader.base")


class BaseDownloader(ABC):
    """基础下载器接口 — 所有源类型都实现
    
    定义最小下载契约：download / validate_source / extract_video_id / guaranteed_fields。
    不包含 playlist 等平台特有方法，避免 LSP 违反。
    """
    
    @abstractmethod
    def download(self, source: str, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """下载/导入视频到 output_dir
        
        Args:
            source: 视频源（URL 或本地文件路径）
            output_dir: 输出目录
            **kwargs: 下载器特定参数（如 YouTube 的 download_subs, sub_langs;
                      LOCAL/DIRECT_URL 的 title, progress_callback 等）
            
        Returns:
            标准化结果字典:
            - video_path: Path — output_dir 内的视频文件路径
            - title: str — 视频标题（平台源必须，LOCAL 可为空）
            - subtitles: Dict[str, Path] — {lang: path}，无字幕则为空 dict
            - metadata: Dict — 元数据（duration, url, description, uploader 等）
        """
        pass
    
    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """验证源是否有效（路径存在 / URL 格式正确）
        
        Args:
            source: 视频源字符串
            
        Returns:
            是否有效
        """
        pass
    
    @abstractmethod
    def extract_video_id(self, source: str) -> str:
        """从源提取/生成稳定的视频 ID
        
        Args:
            source: 视频源字符串
            
        Returns:
            视频 ID（16 字符 hex）
        """
        pass
    
    @property
    @abstractmethod
    def guaranteed_fields(self) -> Set[str]:
        """该下载器保证在 download() 返回值中提供的字段集合
        
        executor 会断言这些字段存在且非空。
        缺失 = 下载器 bug，应 fail-fast。
        
        字段名可以是 result 顶层键（如 'title'）或 metadata 子键（如 'duration'）。
        """
        pass
    
    @staticmethod
    def probe_video_metadata(video_path: Path) -> Optional[Dict[str, Any]]:
        """通过 ffprobe 提取视频元数据（共享工具方法）
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            元数据字典，失败返回 None。
            包含: duration, size, bit_rate, video (codec/width/height/fps), audio (codec/sample_rate/channels)
        """
        return probe_media_info(video_path)


class PlatformDownloader(BaseDownloader):
    """平台下载器接口 — 有平台概念的源（YouTube, Bilibili 等）
    
    在 BaseDownloader 基础上，增加 playlist 等平台特有能力。
    仅在需要 playlist 操作的调用点使用此类型约束。
    """
    
    @abstractmethod
    def get_playlist_urls(self, playlist_url: str) -> List[str]:
        """获取播放列表中的所有视频URL
        
        Args:
            playlist_url: 播放列表URL
            
        Returns:
            视频URL列表
        """
        pass
