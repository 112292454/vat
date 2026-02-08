"""
字幕文件处理工具
"""
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import timedelta


def format_timestamp_srt(seconds: float) -> str:
    """
    格式化时间戳为SRT格式
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间戳 (HH:MM:SS,mmm)
    """
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = td.microseconds // 1000
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_ass(seconds: float) -> str:
    """
    格式化时间戳为ASS格式
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间戳 (H:MM:SS.mm)
    """
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    centisecs = td.microseconds // 10000
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def parse_timestamp_srt(timestamp: str) -> float:
    """
    解析SRT时间戳为秒数
    
    Args:
        timestamp: SRT格式时间戳 (HH:MM:SS,mmm)
        
    Returns:
        秒数
    """
    time_str, millis_str = timestamp.replace(',', '.').split('.')
    hours, minutes, seconds = map(int, time_str.split(':'))
    millis = int(millis_str)
    
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def write_srt(segments: List[Dict[str, Any]], output_path: Path) -> None:
    """
    写入SRT字幕文件
    
    Args:
        segments: 字幕段列表，每项包含 start, end, text
        output_path: 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start_time = format_timestamp_srt(seg['start'])
            end_time = format_timestamp_srt(seg['end'])
            text = seg['text']
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n")
            f.write("\n")


def parse_srt(srt_path: Path) -> List[Dict[str, Any]]:
    """
    解析SRT字幕文件
    
    Args:
        srt_path: SRT文件路径
        
    Returns:
        字幕段列表
    """
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT文件不存在: {srt_path}")
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按空行分割字幕块
    blocks = re.split(r'\n\s*\n', content.strip())
    
    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # 第一行是序号，第二行是时间戳，后续是文本
        try:
            timestamp_line = lines[1]
            start_str, end_str = timestamp_line.split(' --> ')
            
            start = parse_timestamp_srt(start_str.strip())
            end = parse_timestamp_srt(end_str.strip())
            text = '\n'.join(lines[2:])
            
            segments.append({
                'start': start,
                'end': end,
                'text': text
            })
        except:
            continue
    
    return segments


def write_ass(
    segments: List[Dict[str, Any]],
    output_path: Path,
    style_config: Dict[str, Any] = None
) -> None:
    """
    写入ASS字幕文件
    
    Args:
        segments: 字幕段列表
        output_path: 输出文件路径
        style_config: 样式配置
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 默认样式
    if style_config is None:
        style_config = {
            'font': 'Arial',
            'font_size': 54,
            'primary_color': '&H00FFFFFF',
            'outline_color': '&H00000000',
            'back_color': '&H80000000',
            'bold': True,
            'italic': False,
            'outline': 3.5,
            'shadow': 1.5,
            'margin_v': 80
        }
    
    bold = -1 if style_config.get('bold', False) else 0
    italic = -1 if style_config.get('italic', False) else 0
    outline = style_config.get('outline', 3.5)
    shadow = style_config.get('shadow', 1.5)
    margin_v = style_config.get('margin_v', 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # ASS头部
        f.write("[Script Info]\n")
        f.write("Title: VAT Subtitle\n")
        f.write("ScriptType: v4.00+\n")
        f.write("WrapStyle: 0\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("\n")
        
        # 样式定义
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        back_color = style_config.get('back_color', '&H80000000')
        
        f.write(f"Style: Default,{style_config['font']},{style_config['font_size']},"
                f"{style_config['primary_color']},&H000000FF,"
                f"{style_config['outline_color']},{back_color},"
                f"{bold},{italic},0,0,100,100,0,0,1,{outline},{shadow},2,10,10,{margin_v},1\n")
        f.write("\n")
        
        # 事件（字幕内容）
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for seg in segments:
            start_time = format_timestamp_ass(seg['start'])
            end_time = format_timestamp_ass(seg['end'])
            text = seg['text'].replace('\n', '\\N')  # ASS换行符
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")


def merge_srt_files(srt_files: List[Path], output_path: Path) -> None:
    """
    合并多个SRT文件（用于双语字幕）
    
    Args:
        srt_files: SRT文件路径列表
        output_path: 输出文件路径
    """
    all_segments = []
    
    for srt_file in srt_files:
        segments = parse_srt(srt_file)
        all_segments.extend(segments)
    
    # 按开始时间排序
    all_segments.sort(key=lambda x: x['start'])
    
    write_srt(all_segments, output_path)


def create_bilingual_srt(
    original_srt: Path,
    translated_srt: Path,
    output_path: Path
) -> None:
    """
    创建双语字幕SRT文件
    
    Args:
        original_srt: 原文字幕路径
        translated_srt: 译文字幕路径
        output_path: 输出文件路径
    """
    original_segments = parse_srt(original_srt)
    translated_segments = parse_srt(translated_srt)
    
    if len(original_segments) != len(translated_segments):
        print(f"警告: 原文和译文字幕数量不一致 ({len(original_segments)} vs {len(translated_segments)})")
    
    # 合并为双语
    bilingual_segments = []
    for i in range(min(len(original_segments), len(translated_segments))):
        orig = original_segments[i]
        trans = translated_segments[i]
        
        bilingual_segments.append({
            'start': orig['start'],
            'end': orig['end'],
            'text': f"{orig['text']}\n{trans['text']}"
        })
    
    write_srt(bilingual_segments, output_path)


def shift_timestamps(segments: List[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
    """
    偏移所有时间戳
    
    Args:
        segments: 字幕段列表
        offset: 偏移量（秒）
        
    Returns:
        偏移后的字幕段列表
    """
    shifted = []
    for seg in segments:
        shifted.append({
            'start': max(0, seg['start'] + offset),
            'end': max(0, seg['end'] + offset),
            'text': seg['text']
        })
    return shifted
