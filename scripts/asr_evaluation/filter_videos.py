#!/usr/bin/env python3
"""
过滤非日语内容的视频

检查方法：
1. 检查自动生成字幕的语言 - 如果是英语等非日语，说明原声不是日语
2. 检查人工字幕内容是否过短（可能是无语音视频的提示性字幕）
"""

import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


DATA_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/selected/audio")
SELECTED_FILE = Path("/local/gzy/4090-48/vat/experiments/ground_truth/selected/selected_videos.json")


def check_video_language(video_id: str) -> dict:
    """
    检查视频的原声语言
    
    原理：
    - 有日语语音的视频：自动字幕第一行是 "ja" 且 Name 包含 "Japanese, Japanese"
    - 无语音但有人工字幕：自动字幕格式是 "xx-ja ... from Japanese"（基于人工字幕翻译）
    - 其他语言视频：第一行是其他语言代码
    """
    try:
        result = subprocess.run(
            ['yt-dlp', '--list-subs', f'https://www.youtube.com/watch?v={video_id}'],
            capture_output=True, text=True, timeout=30
        )
        
        output = result.stdout + result.stderr
        
        # 解析自动字幕
        first_auto_lang = None
        has_speech_recognition = False  # 是否有真正的语音识别字幕
        in_auto_section = False
        
        for line in output.split('\n'):
            if 'Available automatic captions' in line:
                in_auto_section = True
                continue
            elif 'Available subtitles' in line:
                in_auto_section = False
                continue
            elif in_auto_section and line.strip():
                # 跳过header行
                if line.strip().startswith('Language'):
                    continue
                    
                parts = line.split()
                if not parts:
                    continue
                
                lang_code = parts[0]
                
                # 跳过无效的语言代码（应该是2-3个字母）
                if not lang_code.replace('-', '').isalpha() or len(lang_code) > 6:
                    continue
                
                rest_of_line = ' '.join(parts[1:]) if len(parts) > 1 else ''
                
                # 检查是否是 "from Japanese" 格式（基于人工字幕翻译，非语音识别）
                if 'from Japanese' in rest_of_line or (len(lang_code) > 2 and '-ja' in lang_code):
                    # 这是基于人工字幕的翻译，不是语音识别
                    continue
                
                # 检查是否是真正的语音识别字幕
                if first_auto_lang is None:
                    first_auto_lang = lang_code
                    # 如果Name列包含逗号，说明是真正的语音识别（多来源）
                    name_part = rest_of_line.split('vtt')[0] if 'vtt' in rest_of_line else rest_of_line
                    if ',' in name_part:
                        has_speech_recognition = True
                    else:
                        has_speech_recognition = True  # 单一语言也是语音识别
        
        # 判断是否是日语语音内容
        is_japanese_content = (first_auto_lang == 'ja' and has_speech_recognition)
        
        return {
            'video_id': video_id,
            'original_lang': first_auto_lang,
            'has_speech_recognition': has_speech_recognition,
            'is_japanese_content': is_japanese_content,
            'error': None
        }
        
    except Exception as e:
        return {
            'video_id': video_id,
            'original_lang': None,
            'has_speech_recognition': False,
            'is_japanese_content': None,
            'error': str(e)
        }


def check_subtitle_content(video_id: str) -> dict:
    """检查字幕内容是否正常"""
    vtt_file = DATA_DIR / f"{video_id}.ja.vtt"
    
    if not vtt_file.exists():
        return {'video_id': video_id, 'subtitle_chars': 0, 'is_valid': False}
    
    with open(vtt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 统计实际字幕字符数（去除时间戳和标签）
    import re
    # 移除VTT头和时间戳
    lines = content.split('\n')
    text_lines = []
    for line in lines:
        if '-->' not in line and not line.strip().startswith('WEBVTT') and line.strip():
            # 移除HTML标签
            clean = re.sub(r'<[^>]+>', '', line)
            if clean.strip():
                text_lines.append(clean.strip())
    
    total_chars = sum(len(line) for line in text_lines)
    
    # 少于100字符可能是无语音视频
    is_valid = total_chars >= 100
    
    return {
        'video_id': video_id,
        'subtitle_chars': total_chars,
        'is_valid': is_valid
    }


def main():
    print("=" * 60)
    print("视频语言过滤检查")
    print("=" * 60)
    
    # 获取已下载的视频
    video_ids = []
    for vtt in DATA_DIR.glob("*.ja.vtt"):
        video_ids.append(vtt.stem.replace('.ja', ''))
    
    print(f"已下载视频数: {len(video_ids)}")
    
    # 检查字幕内容
    print("\n检查字幕内容...")
    subtitle_results = {}
    for vid in video_ids:
        result = check_subtitle_content(vid)
        subtitle_results[vid] = result
        if not result['is_valid']:
            print(f"  ⚠ {vid}: 字幕过短 ({result['subtitle_chars']}字符)")
    
    # 检查原声语言
    print("\n检查原声语言...")
    language_results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(check_video_language, vid): vid for vid in video_ids}
        
        for future in as_completed(futures):
            result = future.result()
            vid = result['video_id']
            language_results[vid] = result
            
            if result['is_japanese_content'] == False:
                reason = "无语音识别" if not result.get('has_speech_recognition') else f"原声={result.get('original_lang')}"
                print(f"  ⚠ {vid}: 非日语内容 ({reason})")
            elif result['is_japanese_content'] == True:
                print(f"  ✓ {vid}: 日语内容")
            else:
                print(f"  ? {vid}: 无法判断")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("过滤结果汇总")
    print("=" * 60)
    
    valid_videos = []
    invalid_videos = []
    
    for vid in video_ids:
        sub_ok = subtitle_results.get(vid, {}).get('is_valid', False)
        lang_ok = language_results.get(vid, {}).get('is_japanese_content', None)
        
        if sub_ok and lang_ok != False:
            valid_videos.append(vid)
        else:
            reason = []
            if not sub_ok:
                reason.append("字幕过短")
            if lang_ok == False:
                reason.append("非日语内容")
            invalid_videos.append((vid, ', '.join(reason)))
    
    print(f"\n有效视频: {len(valid_videos)}")
    print(f"无效视频: {len(invalid_videos)}")
    
    if invalid_videos:
        print("\n无效视频列表:")
        for vid, reason in invalid_videos:
            print(f"  - {vid}: {reason}")
    
    # 保存过滤结果
    filter_result = {
        'total': len(video_ids),
        'valid': len(valid_videos),
        'invalid': len(invalid_videos),
        'valid_videos': valid_videos,
        'invalid_videos': [{'video_id': v, 'reason': r} for v, r in invalid_videos],
        'subtitle_results': subtitle_results,
        'language_results': language_results
    }
    
    output_file = DATA_DIR.parent / "filter_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filter_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n过滤结果保存到: {output_file}")
    
    # 询问是否删除无效视频
    if invalid_videos:
        print(f"\n建议删除 {len(invalid_videos)} 个无效视频的文件")
        # 保存删除脚本
        delete_script = DATA_DIR.parent / "delete_invalid.sh"
        with open(delete_script, 'w') as f:
            f.write("#!/bin/bash\n")
            for vid, _ in invalid_videos:
                f.write(f"rm -f {DATA_DIR}/{vid}.wav {DATA_DIR}/{vid}.ja.vtt\n")
        print(f"删除脚本: {delete_script}")


if __name__ == "__main__":
    main()
