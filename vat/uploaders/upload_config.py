"""
上传配置管理模块

独立于主配置文件，支持 Web UI 在线编辑
配置存储在 config/upload.yaml
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

from vat.utils.logger import setup_logger

logger = setup_logger("upload_config")

# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "upload.yaml"


@dataclass
class UploadTemplates:
    """上传模板配置"""
    title: str = "${translated_title}"
    description: str = "${translated_desc}"
    custom_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class BilibiliUploadConfig:
    """B站上传配置"""
    copyright: int = 2  # 1=自制, 2=转载
    default_tid: int = 21
    default_tags: List[str] = field(default_factory=lambda: ["VTuber", "日本"])
    auto_cover: bool = True
    cover_source: str = "thumbnail"
    season_id: Optional[int] = None
    templates: UploadTemplates = field(default_factory=UploadTemplates)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'copyright': self.copyright,
            'default_tid': self.default_tid,
            'default_tags': self.default_tags,
            'auto_cover': self.auto_cover,
            'cover_source': self.cover_source,
            'season_id': self.season_id,
            'templates': {
                'title': self.templates.title,
                'description': self.templates.description,
                'custom_vars': self.templates.custom_vars,
            }
        }


@dataclass
class UploadConfig:
    """完整上传配置"""
    bilibili: BilibiliUploadConfig = field(default_factory=BilibiliUploadConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'bilibili': self.bilibili.to_dict()}


class UploadConfigManager:
    """
    上传配置管理器
    
    负责加载、保存、更新上传配置
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: Optional[UploadConfig] = None
    
    def load(self) -> UploadConfig:
        """加载配置"""
        if not self.config_path.exists():
            logger.warning(f"上传配置文件不存在，使用默认配置: {self.config_path}")
            self._config = UploadConfig()
            return self._config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            bilibili_data = data.get('bilibili', {})
            templates_data = bilibili_data.pop('templates', {})
            
            templates = UploadTemplates(
                title=templates_data.get('title', '${translated_title}'),
                description=templates_data.get('description', '${translated_desc}'),
                custom_vars=templates_data.get('custom_vars', {}),
            )
            
            bilibili = BilibiliUploadConfig(
                copyright=bilibili_data.get('copyright', 2),
                default_tid=bilibili_data.get('default_tid', 21),
                default_tags=bilibili_data.get('default_tags', ['VTuber', '日本']),
                auto_cover=bilibili_data.get('auto_cover', True),
                cover_source=bilibili_data.get('cover_source', 'thumbnail'),
                season_id=bilibili_data.get('season_id'),
                templates=templates,
            )
            
            self._config = UploadConfig(bilibili=bilibili)
            logger.debug(f"上传配置已加载: {self.config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"加载上传配置失败: {e}")
            self._config = UploadConfig()
            return self._config
    
    def save(self, config: Optional[UploadConfig] = None) -> bool:
        """保存配置"""
        if config:
            self._config = config
        
        if not self._config:
            logger.error("没有配置可保存")
            return False
        
        try:
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = self._config.to_dict()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            logger.info(f"上传配置已保存: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存上传配置失败: {e}")
            return False
    
    def update_bilibili(self, updates: Dict[str, Any]) -> bool:
        """更新 B站配置"""
        if not self._config:
            self.load()
        
        try:
            bilibili = self._config.bilibili
            
            if 'copyright' in updates:
                bilibili.copyright = int(updates['copyright'])
            if 'default_tid' in updates:
                bilibili.default_tid = int(updates['default_tid'])
            if 'default_tags' in updates:
                bilibili.default_tags = updates['default_tags']
            if 'auto_cover' in updates:
                bilibili.auto_cover = bool(updates['auto_cover'])
            if 'cover_source' in updates:
                bilibili.cover_source = updates['cover_source']
            if 'season_id' in updates:
                bilibili.season_id = int(updates['season_id']) if updates['season_id'] else None
            
            if 'templates' in updates:
                t = updates['templates']
                if 'title' in t:
                    bilibili.templates.title = t['title']
                if 'description' in t:
                    bilibili.templates.description = t['description']
                if 'custom_vars' in t:
                    bilibili.templates.custom_vars = t['custom_vars']
            
            return self.save()
            
        except Exception as e:
            logger.error(f"更新 B站配置失败: {e}")
            return False
    
    def get_config(self) -> UploadConfig:
        """获取当前配置"""
        if not self._config:
            self.load()
        return self._config
    
    def get_bilibili_dict(self) -> Dict[str, Any]:
        """获取 B站配置字典"""
        return self.get_config().bilibili.to_dict()


# 全局实例
_manager: Optional[UploadConfigManager] = None


def get_upload_config_manager() -> UploadConfigManager:
    """获取全局配置管理器"""
    global _manager
    if _manager is None:
        _manager = UploadConfigManager()
    return _manager


def load_upload_config() -> UploadConfig:
    """便捷函数：加载上传配置"""
    return get_upload_config_manager().load()


def save_upload_config(config: UploadConfig) -> bool:
    """便捷函数：保存上传配置"""
    return get_upload_config_manager().save(config)
