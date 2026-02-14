"""
上传元数据模板渲染器

支持 ${变量名} 格式的变量替换
"""
import re
from typing import Dict, Any, Optional
from datetime import datetime

from vat.utils.logger import setup_logger

logger = setup_logger("uploader.template")


class TemplateRenderer:
    """
    模板渲染器
    
    将模板字符串中的 ${变量名} 替换为实际值
    """
    
    # 变量匹配模式: ${variable_name}
    VARIABLE_PATTERN = re.compile(r'\$\{(\w+)\}')
    
    def __init__(self, custom_vars: Optional[Dict[str, str]] = None):
        """
        初始化渲染器
        
        Args:
            custom_vars: 自定义变量字典，可在模板中使用
        """
        self.custom_vars = custom_vars or {}
    
    def render(self, template: str, context: Dict[str, Any]) -> str:
        """
        渲染模板
        
        Args:
            template: 模板字符串，包含 ${变量名} 占位符
            context: 变量上下文字典
            
        Returns:
            渲染后的字符串
        """
        if not template:
            return ''
        
        # 合并自定义变量和上下文（上下文优先级更高）
        merged_context = {**self.custom_vars, **context}
        
        def replace_var(match):
            var_name = match.group(1)
            value = merged_context.get(var_name)
            
            if value is None:
                logger.warning(f"模板变量未定义: ${{{var_name}}}")
                return f'${{{var_name}}}'  # 保留原样
            
            return str(value)
        
        return self.VARIABLE_PATTERN.sub(replace_var, template)
    
    def get_available_vars(self, context: Dict[str, Any]) -> list:
        """获取所有可用变量名"""
        merged = {**self.custom_vars, **context}
        return sorted(merged.keys())


def build_upload_context(
    video_record: Any,
    playlist_info: Optional[Dict[str, Any]] = None,
    extra_vars: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    从视频记录构建模板上下文
    
    Args:
        video_record: 数据库中的视频记录
        playlist_info: 播放列表信息 (name, index)
        extra_vars: 额外变量
        
    Returns:
        模板上下文字典
        
    可用变量列表:
        基础信息:
            video_id          - YouTube 视频 ID
            source_url        - 原视频链接
            today             - 今天日期 (YYYY-MM-DD)
        
        频道信息:
            channel_name      - 频道/主播名称
            channel_id        - 频道 ID
            channel_url       - 频道链接
        
        原始内容:
            original_title    - 原标题
            original_desc     - 原简介
            original_date     - 原视频发布日期 (YYYY-MM-DD)
            original_date_raw - 原视频发布日期 (YYYYMMDD)
        
        翻译内容:
            translated_title  - 翻译后标题
            translated_desc   - 翻译后简介
            tldr              - 简介摘要
        
        视频信息:
            duration          - 时长 (HH:MM:SS 格式)
            duration_seconds  - 时长秒数
            duration_minutes  - 时长分钟数 (取整)
            thumbnail         - 缩略图 URL
        
        播放列表:
            playlist_name     - 播放列表名称
            playlist_index    - 在列表中的序号
            playlist_id       - 播放列表 ID
        
        自定义变量:
            可在 config/upload.yaml 的 custom_vars 中定义
    """
    metadata = video_record.metadata or {}
    translated = metadata.get('translated', {})
    
    # 从 metadata 或 _video_info 获取视频信息
    video_info = metadata.get('_video_info', {})
    
    # 解析原始上传日期 (格式: YYYYMMDD)
    original_date_raw = metadata.get('upload_date', '') or video_info.get('upload_date', '')
    original_date = ''
    if original_date_raw and len(original_date_raw) == 8:
        try:
            original_date = f"{original_date_raw[:4]}-{original_date_raw[4:6]}-{original_date_raw[6:8]}"
        except:
            pass
    
    # 频道信息（从 metadata 获取，Video 模型没有这些字段）
    channel_id = metadata.get('channel_id', '') or video_info.get('channel_id', '')
    channel_name = metadata.get('uploader', '') or video_info.get('uploader', '')
    channel_url = f"https://www.youtube.com/channel/{channel_id}" if channel_id else ''
    
    # 时长
    duration_sec = metadata.get('duration', 0) or video_info.get('duration', 0) or 0
    duration_min = round(duration_sec / 60) if duration_sec else 0
    
    # 视频 ID（从 URL 或 metadata 提取）
    video_id = metadata.get('video_id', '') or video_info.get('video_id', '') or video_record.id
    
    context = {
        # 基础信息
        'video_id': video_id,
        'source_url': video_record.source_url or f"https://www.youtube.com/watch?v={video_id}",
        'today': datetime.now().strftime('%Y-%m-%d'),
        
        # 频道信息
        'channel_name': channel_name,
        'channel_id': channel_id,
        'channel_url': channel_url,
        
        # 原始内容
        'original_title': video_record.title or video_info.get('title', ''),
        'original_desc': metadata.get('description', '') or video_info.get('description', ''),
        'original_date': original_date,
        'original_date_raw': original_date_raw,
        
        # 翻译后的标题/简介
        'translated_title': translated.get('title_optimized', '') or translated.get('title_translated', '') or video_record.title or '',
        'translated_desc': translated.get('description_translated', '') or translated.get('description_optimized', ''),
        'tldr': translated.get('description_summary', '') or translated.get('tldr', ''),
        
        # 视频信息
        'duration': _format_duration(duration_sec) if duration_sec else '',
        'duration_seconds': duration_sec,
        'duration_minutes': duration_min,
        'thumbnail': metadata.get('thumbnail', '') or video_info.get('thumbnail', ''),
    }
    
    # 播放列表信息
    if playlist_info:
        context['playlist_name'] = playlist_info.get('name', '')
        context['playlist_index'] = playlist_info.get('index', 0)
        context['playlist_id'] = playlist_info.get('id', '')
        # 自定义主播名称（覆盖原频道名）
        custom_uploader = playlist_info.get('uploader_name', '')
        if custom_uploader:
            context['uploader_name'] = custom_uploader
            context['channel_name'] = custom_uploader  # 同时覆盖 channel_name
        else:
            context['uploader_name'] = channel_name
    else:
        # 默认值
        context['playlist_name'] = ''
        context['playlist_index'] = ''
        context['playlist_id'] = ''
        context['uploader_name'] = channel_name
    
    # 合并额外变量
    if extra_vars:
        context.update(extra_vars)
    
    return context


def _format_duration(seconds: int) -> str:
    """格式化时长为 HH:MM:SS 或 MM:SS"""
    if not seconds:
        return ''
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    else:
        return f'{minutes}:{secs:02d}'


def render_upload_metadata(
    video_record: Any,
    templates: Dict[str, Any],
    playlist_info: Optional[Dict[str, Any]] = None,
    extra_vars: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    渲染上传元数据
    
    Args:
        video_record: 视频记录
        templates: 模板配置 (包含 title, description, custom_vars)
        playlist_info: 播放列表信息
        extra_vars: 额外变量
        
    Returns:
        渲染后的元数据 {'title': ..., 'description': ...}
    """
    # 获取自定义变量
    custom_vars = templates.get('custom_vars', {})
    
    # 构建上下文
    context = build_upload_context(video_record, playlist_info, extra_vars)
    
    # 创建渲染器
    renderer = TemplateRenderer(custom_vars)
    
    # 渲染标题和简介
    title_template = templates.get('title', '${translated_title}')
    desc_template = templates.get('description', '${translated_desc}')
    
    rendered_title = renderer.render(title_template, context)
    rendered_desc = renderer.render(desc_template, context)
    
    return {
        'title': rendered_title,
        'description': rendered_desc,
        'context': context,  # 返回上下文供调试
    }
