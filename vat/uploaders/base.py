"""
上传器抽象基类
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List, Any


class BaseUploader(ABC):
    """上传器抽象基类，方便后续扩展其他平台"""
    
    @abstractmethod
    def upload(self, video_path: Path, metadata: Dict[str, Any]) -> str:
        """
        上传视频
        
        Args:
            video_path: 视频文件路径
            metadata: 视频元数据，包含:
                - title: str - 标题
                - desc: str - 描述
                - tags: List[str] - 标签
                - cover: Path - 封面图（可选）
                - category: str - 分类（可选）
                
        Returns:
            视频ID（如B站的BV号）
        """
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """
        验证上传凭证是否有效
        
        Returns:
            凭证是否有效
        """
        pass
    
    @abstractmethod
    def get_upload_limit(self) -> Dict[str, Any]:
        """
        获取上传限制信息
        
        Returns:
            字典包含:
            - max_size: int - 最大文件大小（字节）
            - max_duration: int - 最大时长（秒）
            - supported_formats: List[str] - 支持的格式
        """
        pass
