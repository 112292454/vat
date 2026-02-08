#!/usr/bin/env python3
"""
从JTubeSpeech筛选适合ASR评估的视频

筛选条件:
1. 有人工字幕 (sub=True)
2. 日语内容+日语字幕
3. 时长适中 (1-10分钟)
4. 字幕可下载
"""

import subprocess
import random
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# 配置
JTUBESPEECH_CSV = Path("/local/gzy/4090-48/vat/experiments/jtubespeech/data/ja/202103.csv")
OUTPUT_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/selected")
TARGET_COUNT = 50  # 目标筛选数量
MAX_CHECK = 500    # 最多检查的视频数量


def get_video_info(video_id: str) -> dict | None:
    """获取视频信息，检查是否有日语人工字幕"""
    try:
        # 检查字幕
        result = subprocess.run(
            ['yt-dlp', '--list-subs', f'https://www.youtube.com/watch?v={video_id}'],
            capture_output=True, text=True, timeout=30
        )
        
        output = result.stdout + result.stderr
        
        # 检查是否有日语人工字幕
        has_manual_ja = False
        in_manual_section = False
        for line in output.split('\n'):
            if 'Available subtitles' in line:
                in_manual_section = True
            elif 'Available automatic' in line:
                in_manual_section = False
            elif in_manual_section and line.strip().startswith('ja'):
                has_manual_ja = True
                break
        
        if not has_manual_ja:
            return None
        
        # 获取视频时长
        result = subprocess.run(
            ['yt-dlp', '--print', 'duration', f'https://www.youtube.com/watch?v={video_id}'],
            capture_output=True, text=True, timeout=30
        )
        
        duration = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        
        # 筛选1-10分钟的视频
        if duration < 60 or duration > 600:
            return None
        
        return {
            'video_id': video_id,
            'duration': duration,
            'has_manual_ja': True
        }
        
    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("JTubeSpeech视频筛选")
    print("=" * 60)
    
    # 读取有人工字幕的视频ID
    video_ids = []
    with open(JTUBESPEECH_CSV, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3 and parts[2] == 'True':  # sub=True
                video_ids.append(parts[0])
    
    print(f"JTubeSpeech中有人工字幕的视频: {len(video_ids)}")
    
    # 随机抽样检查
    random.seed(42)
    sample = random.sample(video_ids, min(MAX_CHECK, len(video_ids)))
    
    print(f"随机抽样 {len(sample)} 个视频进行检查...")
    print(f"目标筛选数量: {TARGET_COUNT}")
    
    # 并行检查（使用较小batch避免过多未完成任务）
    selected = []
    checked = 0
    batch_size = 20
    
    for batch_start in range(0, len(sample), batch_size):
        if len(selected) >= TARGET_COUNT:
            break
            
        batch = sample[batch_start:batch_start + batch_size]
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_video_info, vid): vid for vid in batch}
            
            for future in as_completed(futures):
                checked += 1
                result = future.result()
                
                if result:
                    selected.append(result)
                    print(f"[{checked}/{len(sample)}] ✓ {result['video_id']} ({result['duration']}s)")
                
                if len(selected) >= TARGET_COUNT:
                    break
        
        if checked % 50 == 0:
            print(f"[{checked}/{len(sample)}] 已筛选 {len(selected)} 个")
    
    print(f"\n已达到目标数量 {len(selected)}")
    
    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "selected_videos.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'total_checked': checked,
            'total_selected': len(selected),
            'videos': selected
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n筛选完成: {len(selected)} 个视频")
    print(f"结果保存到: {output_file}")
    
    # 保存视频ID列表（便于下载）
    id_file = OUTPUT_DIR / "video_ids.txt"
    with open(id_file, 'w') as f:
        for v in selected:
            f.write(v['video_id'] + '\n')
    print(f"视频ID列表: {id_file}")


if __name__ == "__main__":
    main()
