"""媒体信息探测。"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from vat.utils.logger import setup_logger

logger = setup_logger("media.probe")


def probe_media_info(video_path: Path) -> Optional[Dict[str, Any]]:
    """通过 ffprobe 提取媒体元数据。"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        info = json.loads(result.stdout)
        video_stream = None
        audio_stream = None

        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video' and video_stream is None:
                video_stream = stream
            elif stream.get('codec_type') == 'audio' and audio_stream is None:
                audio_stream = stream

        format_info = info.get('format', {})

        fps = 0
        if video_stream:
            fps_str = video_stream.get('r_frame_rate', '0/1')
            try:
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) != 0 else 0
                else:
                    fps = float(fps_str)
            except (ValueError, ZeroDivisionError):
                fps = 0

        return {
            'duration': float(format_info.get('duration', 0)),
            'size': int(format_info.get('size', 0)),
            'bit_rate': int(format_info.get('bit_rate', 0)),
            'video': {
                'codec': video_stream.get('codec_name', '') if video_stream else '',
                'width': video_stream.get('width', 0) if video_stream else 0,
                'height': video_stream.get('height', 0) if video_stream else 0,
                'fps': fps,
            } if video_stream else None,
            'audio': {
                'codec': audio_stream.get('codec_name', '') if audio_stream else '',
                'sample_rate': audio_stream.get('sample_rate', 0) if audio_stream else 0,
                'channels': audio_stream.get('channels', 0) if audio_stream else 0,
            } if audio_stream else None,
        }
    except FileNotFoundError:
        logger.error("ffprobe 未安装或不在 PATH 中，无法提取视频元数据")
        return None
    except Exception as e:
        logger.error(f"ffprobe 提取视频元数据失败: {e}")
        return None
