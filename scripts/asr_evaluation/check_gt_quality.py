#!/usr/bin/env python
"""
GT（Ground Truth）字幕质量检测脚本

检测项目：
1. 乱码检测 - 检查日语字符比例和有效性
2. 长度异常 - GT长度与音频时长的比例
3. 重复度检测 - 高重复度可能是歌曲/口号
4. 内容有效性 - 检查是否包含有效日语内容
"""

import json
import re
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def is_japanese_char(char):
    """判断是否为日语字符（平假名、片假名、汉字）"""
    code = ord(char)
    # 平假名: 3040-309F, 片假名: 30A0-30FF, 汉字: 4E00-9FFF
    return (0x3040 <= code <= 0x309F or  # 平假名
            0x30A0 <= code <= 0x30FF or  # 片假名
            0x4E00 <= code <= 0x9FFF or  # 汉字
            0x3000 <= code <= 0x303F)    # 日语标点


def is_noise_char(char):
    """判断是否为噪音字符（CSS/HTML等）"""
    return char in '{}:;#.()[]'


def clean_vtt_text(vtt_path):
    """解析VTT文件，提取纯文本"""
    content = vtt_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('WEBVTT') or '-->' in line:
            continue
        if line.startswith('Kind:') or line.startswith('Language:') or line.startswith('Style:'):
            continue
        if line.startswith('::cue') or line.startswith('#'):
            continue
        if line.isdigit():
            continue
        # 移除HTML标签和零宽字符
        line = re.sub(r'<[^>]+>', '', line)
        line = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', line)
        line = line.strip()
        if line and not line.startswith('color') and not line.startswith('{'):
            text_lines.append(line)
    return ' '.join(text_lines)


def check_japanese_ratio(text):
    """检查日语字符比例"""
    if not text:
        return 0.0
    jp_chars = sum(1 for c in text if is_japanese_char(c))
    # 排除空格和标点
    valid_chars = sum(1 for c in text if not c.isspace() and not is_noise_char(c))
    if valid_chars == 0:
        return 0.0
    return jp_chars / valid_chars


def check_repetition_ratio(text):
    """检查重复度（词级别）"""
    if not text:
        return 0.0
    # 简单分词：按空格和标点分割
    words = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+|[a-zA-Z]+', text)
    if len(words) < 2:
        return 0.0
    word_counts = Counter(words)
    # 计算重复词的比例
    repeated = sum(count - 1 for count in word_counts.values() if count > 1)
    return repeated / len(words)


def check_noise_ratio(text):
    """检查噪音字符比例（CSS/HTML残留）"""
    if not text:
        return 0.0
    noise_chars = sum(1 for c in text if is_noise_char(c))
    return noise_chars / len(text)


def check_gt_quality(vtt_path, duration_sec=None):
    """
    检查GT质量，返回质量报告
    
    Returns:
        dict: 质量检测结果
    """
    text = clean_vtt_text(vtt_path)
    
    # 各项检测
    jp_ratio = check_japanese_ratio(text)
    rep_ratio = check_repetition_ratio(text)
    noise_ratio = check_noise_ratio(text)
    
    # 长度相关
    text_len = len(text)
    chars_per_sec = text_len / duration_sec if duration_sec and duration_sec > 0 else None
    
    # 问题标记
    issues = []
    
    # 日语比例过低（可能是英文字幕或乱码）
    if jp_ratio < 0.3:
        issues.append(f"low_japanese_ratio:{jp_ratio:.1%}")
    
    # 高重复度（可能是歌曲/口号）
    if rep_ratio > 0.5:
        issues.append(f"high_repetition:{rep_ratio:.1%}")
    
    # 噪音字符过多（CSS/HTML残留）
    if noise_ratio > 0.1:
        issues.append(f"high_noise:{noise_ratio:.1%}")
    
    # 字符密度异常（正常语音约10-20字/秒）
    if chars_per_sec is not None:
        if chars_per_sec > 30:
            issues.append(f"too_dense:{chars_per_sec:.1f}chars/s")
        elif chars_per_sec < 1:
            issues.append(f"too_sparse:{chars_per_sec:.1f}chars/s")
    
    # 内容过短
    if text_len < 20:
        issues.append(f"too_short:{text_len}chars")
    
    # 质量评级
    if not issues:
        quality = "good"
    elif len(issues) == 1 and issues[0].startswith("high_repetition"):
        quality = "song_or_chant"  # 可能是歌曲，不一定是质量问题
    elif any(i.startswith("low_japanese") or i.startswith("high_noise") for i in issues):
        quality = "bad"
    else:
        quality = "suspect"
    
    return {
        "text_length": text_len,
        "japanese_ratio": jp_ratio,
        "repetition_ratio": rep_ratio,
        "noise_ratio": noise_ratio,
        "chars_per_sec": chars_per_sec,
        "issues": issues,
        "quality": quality,
        "text_preview": text[:100] if text else ""
    }


def main():
    # 读取视频信息
    data_dir = Path('/local/gzy/4090-48/vat/experiments/ground_truth/vtuber_channels')
    subtitled_path = data_dir / 'subtitled_videos.json'
    downloaded_dir = data_dir / 'downloaded'
    
    with open(subtitled_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    videos = data.get('videos', [])
    
    # 检测所有有VTT文件的视频
    results = {
        "good": [],
        "suspect": [],
        "bad": [],
        "song_or_chant": [],
        "missing": []
    }
    
    print("="*70)
    print("GT质量检测")
    print("="*70)
    
    for video in videos:
        vid = video['video_id']
        duration = video.get('duration', 0)
        title = video.get('title', '')
        
        vtt_path = downloaded_dir / f'{vid}.ja.vtt'
        if not vtt_path.exists():
            results["missing"].append(vid)
            continue
        
        report = check_gt_quality(vtt_path, duration)
        report['video_id'] = vid
        report['title'] = title
        report['duration'] = duration
        
        quality = report['quality']
        results[quality].append(report)
    
    # 输出统计
    print(f"\n总视频数: {len(videos)}")
    print(f"VTT文件缺失: {len(results['missing'])}")
    print(f"质量良好: {len(results['good'])}")
    print(f"可疑样本: {len(results['suspect'])}")
    print(f"质量差: {len(results['bad'])}")
    print(f"歌曲/口号: {len(results['song_or_chant'])}")
    
    # 详细输出问题样本
    print("\n" + "="*70)
    print("质量差的样本（可能是乱码/非日语）")
    print("="*70)
    for r in results['bad'][:10]:
        print(f"\n{r['video_id']}: {r['title'][:40]}")
        print(f"  问题: {', '.join(r['issues'])}")
        print(f"  预览: {r['text_preview'][:80]}...")
    
    print("\n" + "="*70)
    print("歌曲/口号类（高重复度）")
    print("="*70)
    for r in results['song_or_chant'][:10]:
        print(f"\n{r['video_id']}: {r['title'][:40]}")
        print(f"  重复度: {r['repetition_ratio']:.1%}")
        print(f"  预览: {r['text_preview'][:80]}...")
    
    # 保存完整结果
    output_path = data_dir / 'param_results' / 'gt_quality_check.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {output_path}")
    
    return results


if __name__ == '__main__':
    main()
