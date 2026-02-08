"""
缓存元数据管理模块
用于追踪子步骤的配置快照和缓存失效检测
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime


# 关键配置定义（仅记录影响输出的参数）
WHISPER_KEY_CONFIGS = [
    'model', 'language', 'compute_type', 'vad_filter',
    'enable_chunked', 'chunk_length_sec', 'backend'
]

SPLIT_KEY_CONFIGS = [
    'enable', 'mode', 'max_words_cjk', 'max_words_english', 'model'
]

OPTIMIZE_KEY_CONFIGS = [
    'enable', 'custom_prompt', 'model', 'batch_size', 'thread_num'
]


@dataclass
class SubstepMetadata:
    """子步骤元数据"""
    completed_at: str
    config_snapshot: Dict[str, Any]  # 关键配置的快照
    output_file: str


@dataclass
class CacheMetadata:
    """缓存元数据"""
    version: str  # VAT 版本
    video_id: str
    substeps: Dict[str, SubstepMetadata] = field(default_factory=dict)
    
    @classmethod
    def load(cls, video_output_dir: Path) -> 'CacheMetadata':
        """从输出目录加载元数据"""
        metadata_file = video_output_dir / ".cache_metadata.json"
        if not metadata_file.exists():
            return cls(version="0.2.1", video_id="", substeps={})
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 转换 substeps 字典中的数据为 SubstepMetadata 对象
            substeps = {}
            for key, value in data.get('substeps', {}).items():
                substeps[key] = SubstepMetadata(**value)
            return cls(
                version=data.get('version', '0.2.1'),
                video_id=data.get('video_id', ''),
                substeps=substeps
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # 解析失败时返回空 metadata
            print(f"警告: 无法解析缓存元数据文件，将创建新的元数据: {e}")
            return cls(version="0.2.1", video_id="", substeps={})
    
    def save(self, video_output_dir: Path):
        """保存元数据"""
        metadata_file = video_output_dir / ".cache_metadata.json"
        # 转换 substeps 为可序列化的格式
        data = {
            'version': self.version,
            'video_id': self.video_id,
            'substeps': {
                key: asdict(value) for key, value in self.substeps.items()
            }
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def is_substep_valid(self, substep_name: str, current_config: Dict) -> bool:
        """
        检查子步骤缓存是否有效（配置是否一致）
        
        Args:
            substep_name: 子步骤名称 ('whisper', 'split', 'optimize')
            current_config: 当前配置快照
            
        Returns:
            是否有效
        """
        if substep_name not in self.substeps:
            return False
        
        saved_config = self.substeps[substep_name].config_snapshot
        # 严格比较：任何差异都视为失效
        return saved_config == current_config
    
    def update_substep(self, substep_name: str, config_snapshot: Dict, output_file: str):
        """
        更新子步骤元数据
        
        Args:
            substep_name: 子步骤名称
            config_snapshot: 配置快照
            output_file: 输出文件名（相对于输出目录）
        """
        self.substeps[substep_name] = SubstepMetadata(
            completed_at=datetime.now().isoformat(),
            config_snapshot=config_snapshot,
            output_file=output_file
        )


def extract_key_config(config: Any, key_list: list) -> Dict[str, Any]:
    """
    从配置对象中提取关键配置
    
    Args:
        config: 配置对象
        key_list: 关键配置键列表
        
    Returns:
        关键配置字典
    """
    result = {}
    for key in key_list:
        if hasattr(config, key):
            value = getattr(config, key)
            # 如果值是复杂对象，转换为字典
            if hasattr(value, '__dict__'):
                result[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
            else:
                result[key] = value
    return result
