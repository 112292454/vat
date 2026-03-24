"""媒体音频基础操作。"""

import subprocess
from pathlib import Path


def extract_audio_ffmpeg(
    video_path: Path,
    audio_path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> None:
    """
    使用 ffmpeg 从视频中提取音频。

    成功时不返回值，失败时抛异常。
    """
    if not video_path.exists():
        raise FileNotFoundError(f"输入视频文件不存在: {video_path}")

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',
        '-af', 'aresample=async=1',
        '-acodec', codec,
        '-ac', str(channels),
        '-ar', str(sample_rate),
        '-y',
        str(audio_path)
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音频提取失败: {e.stderr}") from e

    if not audio_path.exists():
        raise RuntimeError(f"音频提取完成但未生成文件: {audio_path}")
