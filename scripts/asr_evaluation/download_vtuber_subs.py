#!/usr/bin/env python3
"""
下载有人工字幕的VTuber视频（音频+字幕）用于ASR评估
"""

import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 输入：find_subtitled_videos.py的输出
INPUT_FILE = Path("/local/gzy/4090-48/vat/experiments/ground_truth/vtuber_channels/subtitled_videos.json")
# 输出目录
OUTPUT_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/vtuber_channels/downloaded")


def download_video(video: dict) -> dict:
    """下载单个视频的音频和字幕"""
    video_id = video['video_id']
    
    # 优先日语，其次英语
    sub_langs = video.get('manual_sub_langs', [])
    if 'ja' in sub_langs:
        lang = 'ja'
    elif 'en' in sub_langs:
        lang = 'en'
    elif sub_langs:
        lang = sub_langs[0]
    else:
        return {'video_id': video_id, 'success': False, 'error': 'no subtitles'}
    
    output_template = str(OUTPUT_DIR / f"{video_id}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--write-sub", "--sub-lang", lang,
        "--sub-format", "vtt",
        "-o", output_template,
        f"https://youtube.com/watch?v={video_id}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # 检查文件是否存在
        audio_file = OUTPUT_DIR / f"{video_id}.wav"
        sub_file = OUTPUT_DIR / f"{video_id}.{lang}.vtt"
        
        if audio_file.exists() and sub_file.exists():
            return {
                'video_id': video_id,
                'channel': video.get('channel', ''),
                'title': video.get('title', ''),
                'sub_lang': lang,
                'audio_path': str(audio_file),
                'sub_path': str(sub_file),
                'success': True,
                'error': None
            }
        else:
            return {
                'video_id': video_id,
                'success': False,
                'error': f'files not found: audio={audio_file.exists()}, sub={sub_file.exists()}'
            }
            
    except Exception as e:
        return {'video_id': video_id, 'success': False, 'error': str(e)}


def main():
    print("=" * 70)
    print("下载有人工字幕的VTuber视频")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 读取找到的视频
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 筛选有人工字幕的视频（优先日文）
    videos_with_sub = [v for v in data['videos'] if v.get('has_manual_sub') or v.get('has_manual_ja_sub')]
    
    # 优先日文字幕
    ja_videos = [v for v in videos_with_sub if v.get('has_manual_ja_sub')]
    other_videos = [v for v in videos_with_sub if not v.get('has_manual_ja_sub')]
    
    print(f"有日文字幕的视频: {len(ja_videos)}")
    print(f"有其他语言字幕的视频: {len(other_videos)}")
    
    # 先下载日文字幕的
    to_download = ja_videos + other_videos[:5]  # 日文全下，其他取5个
    print(f"\n将下载 {len(to_download)} 个视频")
    
    # 并行下载
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_video, v): v for v in to_download}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            
            if result['success']:
                print(f"  ✓ [{result.get('channel', '')}] {result['video_id']} ({result.get('sub_lang', '')})")
            else:
                print(f"  ✗ {result['video_id']}: {result['error']}")
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(to_download)}")
    
    # 统计
    success = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n成功: {len(success)}, 失败: {len(failed)}")
    
    # 保存结果
    result_file = OUTPUT_DIR / "downloaded_videos.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(results),
            'success': len(success),
            'failed': len(failed),
            'videos': success
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {result_file}")
    
    # 列出成功的
    if success:
        print(f"\n成功下载的视频:")
        for v in success:
            print(f"  {v['video_id']} [{v.get('sub_lang', '')}] {v.get('title', '')[:40]}")


if __name__ == "__main__":
    main()
