#!/usr/bin/env python3
"""
下载筛选出的视频的音频和字幕
"""

import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# 配置
SELECTED_FILE = Path("/local/gzy/4090-48/vat/experiments/ground_truth/selected/selected_videos.json")
OUTPUT_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/selected/audio")


def download_video(video_id: str) -> bool:
    """下载单个视频的音频和字幕"""
    try:
        output_template = str(OUTPUT_DIR / f"{video_id}.%(ext)s")
        
        # 下载音频(wav) + 人工字幕
        result = subprocess.run([
            'yt-dlp',
            '-x', '--audio-format', 'wav',
            '--write-sub', '--sub-lang', 'ja',
            '--sub-format', 'vtt',
            '-o', output_template,
            f'https://www.youtube.com/watch?v={video_id}'
        ], capture_output=True, text=True, timeout=300)
        
        # 检查是否成功
        wav_file = OUTPUT_DIR / f"{video_id}.wav"
        vtt_file = OUTPUT_DIR / f"{video_id}.ja.vtt"
        
        if wav_file.exists() and vtt_file.exists():
            return True
        return False
        
    except Exception as e:
        print(f"下载失败 {video_id}: {e}")
        return False


def main():
    print("=" * 60)
    print("下载筛选视频的音频和字幕")
    print("=" * 60)
    
    # 读取筛选结果
    with open(SELECTED_FILE, 'r') as f:
        data = json.load(f)
    
    videos = data['videos']
    print(f"待下载视频数: {len(videos)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 检查已下载的
    existing = set()
    for wav in OUTPUT_DIR.glob("*.wav"):
        vid = wav.stem
        vtt = OUTPUT_DIR / f"{vid}.ja.vtt"
        if vtt.exists():
            existing.add(vid)
    
    to_download = [v for v in videos if v['video_id'] not in existing]
    print(f"已下载: {len(existing)}, 待下载: {len(to_download)}")
    
    if not to_download:
        print("所有视频已下载完成")
        return
    
    # 并行下载（限制并发避免被封）
    success = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_video, v['video_id']): v['video_id'] for v in to_download}
        
        for i, future in enumerate(as_completed(futures)):
            video_id = futures[future]
            if future.result():
                success += 1
                print(f"[{i+1}/{len(to_download)}] ✓ {video_id}")
            else:
                failed.append(video_id)
                print(f"[{i+1}/{len(to_download)}] ✗ {video_id}")
    
    print(f"\n下载完成: 成功 {success}, 失败 {len(failed)}")
    
    if failed:
        print(f"失败列表: {failed[:10]}...")


if __name__ == "__main__":
    main()
