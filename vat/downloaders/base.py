"""
下载器抽象基类
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any


class BaseDownloader(ABC):
    """下载器抽象基类，方便后续扩展其他平台"""
    
    @abstractmethod
    def download(self, url: str, output_dir: Path) -> Dict[str, Any]:
        """
        下载视频
        
        Args:
            url: 视频URL
            output_dir: 输出目录
            
        Returns:
            字典包含:
            - video_path: Path - 下载的视频路径
            - title: str - 视频标题
            - metadata: Dict - 其他元数据（描述、时长等）
        """
        pass
    
    @abstractmethod
    def get_playlist_urls(self, playlist_url: str) -> List[str]:
        """
        获取播放列表中的所有视频URL
        
        Args:
            playlist_url: 播放列表URL
            
        Returns:
            视频URL列表
        """
        pass
    
    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """
        验证URL是否为该平台的有效URL
        
        Args:
            url: 要验证的URL
            
        Returns:
            是否有效
        """
        pass
    
    @abstractmethod
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        从URL中提取视频ID
        
        Args:
            url: 视频URL
            
        Returns:
            视频ID，如果无法提取则返回None
        """
        pass
