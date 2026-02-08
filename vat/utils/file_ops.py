"""
文件操作工具

提供统一的处理产物删除逻辑，确保不误删原始下载文件。
"""
from pathlib import Path
from typing import List

from .logger import setup_logger

logger = setup_logger("file_ops")

# 已知的处理产物文件名（精确匹配）
PROCESSED_FILE_NAMES = {
    "original_raw.srt",
    "original_split.srt",
    "original.srt",
    "original.json",
    "optimized.srt",
    "original_optimized.srt",
    "translated.srt",
    "translated.ass",
    "final.mp4",
    "final.mkv",
    "ffmpeg_embed.log",
    ".cache_metadata.json",
}

# 已知的处理产物后缀
# .ass: 字幕样式文件
# .wav: 本地提取的音频（YouTube 下载的音频是 m4a 格式，wav 是 whisper 阶段从视频提取的）
PROCESSED_SUFFIXES = {".ass", ".wav"}


def is_processed_file(filepath: Path) -> bool:
    """
    判断文件是否为处理产物（非原始下载文件）
    
    原始下载文件的特征：
    - {youtube_id}.{ext} (如 9bSJy5Byrfc.mp4, 9bSJy5Byrfc.wav)
    - {youtube_id}.{lang}.vtt (如 9bSJy5Byrfc.ja.vtt)
    
    处理产物的特征：
    - 已知文件名：original_raw.srt, optimized.srt, translated.srt, final.mp4 等
    - .ass 后缀
    - .cache_metadata.json
    
    安全策略：只有明确识别为处理产物的才返回 True，其他一律保留。
    """
    name = filepath.name
    
    # 精确匹配已知处理产物
    if name in PROCESSED_FILE_NAMES:
        return True
    
    # .ass 文件一定是处理产物
    if filepath.suffix in PROCESSED_SUFFIXES:
        return True
    
    # 其他文件一律视为原始文件，不删除
    return False


def delete_processed_files(output_dir: Path) -> List[str]:
    """
    删除指定目录中的处理产物，保留原始下载文件
    
    Args:
        output_dir: 视频输出目录
        
    Returns:
        被删除的文件名列表
    """
    if not output_dir.exists():
        return []
    
    deleted = []
    preserved = []
    
    for f in output_dir.iterdir():
        if not f.is_file():
            continue
        
        if is_processed_file(f):
            try:
                f.unlink()
                deleted.append(f.name)
            except Exception as e:
                logger.warning(f"删除文件失败: {f.name} - {e}")
        else:
            preserved.append(f.name)
    
    if deleted:
        logger.info(f"已删除 {len(deleted)} 个处理产物: {', '.join(deleted[:5])}")
    if preserved:
        logger.info(f"保留 {len(preserved)} 个原始文件: {', '.join(preserved[:5])}")
    
    return deleted
