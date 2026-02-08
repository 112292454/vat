#!/usr/bin/env python3
"""
从目标YouTube频道查找有人工字幕的视频

用于ASR参数评估：
- 这些频道的视频原声确定是日语
- 有人工字幕的视频可以作为Ground Truth
- 优先日语字幕，其他语言字幕也保存
"""

import subprocess
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List

# 目标频道及其列表类型
# YouTube频道可用的列表: /videos, /streams (直播录播), /shorts
# 扫描所有视频和直播（不限数量），寻找有人工字幕的内容
TARGET_CHANNELS = [
    # === 优先检查的频道 ===
    ("rurudo_LION", "https://www.youtube.com/@rurudo_LION", ["videos", "streams"]),
    ("ShirakamiFubuki", "https://www.youtube.com/@ShirakamiFubuki", ["videos", "streams"]),
    ("nanoch.3637", "https://www.youtube.com/@nanoch.3637", ["videos", "streams"]),
    
    # === Hololive JP 0期生 ===
    ("TokinoSora", "https://www.youtube.com/@TokinoSora", ["videos", "streams"]),
    ("AZKi", "https://www.youtube.com/@AZKi", ["videos", "streams"]),
    ("RobocoSan", "https://www.youtube.com/@Robocosan", ["videos", "streams"]),
    ("SakuraMiko", "https://www.youtube.com/@SakuraMiko", ["videos", "streams"]),
    ("HoshimachiSuisei", "https://www.youtube.com/@HoshimachiSuisei", ["videos", "streams"]),
    
    # === Hololive JP 1期生 ===
    ("NatsuiroMatsuri", "https://www.youtube.com/@NatsuiroMatsuri", ["videos", "streams"]),
    ("AkiRosenthal", "https://www.youtube.com/@AkiRosenthal", ["videos", "streams"]),
    ("AkaiHaato", "https://www.youtube.com/@AkaiHaato", ["videos", "streams"]),
    ("MelChannel", "https://www.youtube.com/@YozoraMel", ["videos", "streams"]),
    
    # === Hololive JP 2期生 ===
    ("MinatoAqua", "https://www.youtube.com/@MinatoAqua", ["videos", "streams"]),
    ("MurasakiShion", "https://www.youtube.com/@MurasakiShionCh", ["videos", "streams"]),
    ("NakiriAyame", "https://www.youtube.com/@NakiriAyame", ["videos", "streams"]),
    ("YuzukiChoco", "https://www.youtube.com/@YuzukiChoco", ["videos", "streams"]),
    ("OozoraSubaru", "https://www.youtube.com/@OozoraSubaru", ["videos", "streams"]),
    
    # === Hololive JP Gamers ===
    ("OokamiMio", "https://www.youtube.com/@OokamiMio", ["videos", "streams"]),
    ("NekomataOkayu", "https://www.youtube.com/@NekomataOkayu", ["videos", "streams"]),
    ("InugamiKorone", "https://www.youtube.com/@InugamiKorone", ["videos", "streams"]),
    
    # === Hololive JP 3期生 ===
    ("UsadaPekora", "https://www.youtube.com/@usaboruPekora", ["videos", "streams"]),
    ("ShiroganeNoel", "https://www.youtube.com/@ShiroganeNoel", ["videos", "streams"]),
    ("ShiranuiFlare", "https://www.youtube.com/@ShiranuiFlare", ["videos", "streams"]),
    ("HoushouMarine", "https://www.youtube.com/@HoushouMarine", ["videos", "streams"]),
    
    # === Hololive JP 4期生 ===
    ("AmaneKanata", "https://www.youtube.com/@AmaneKanata", ["videos", "streams"]),
    ("TsunomakiWatame", "https://www.youtube.com/@TsunomakiWatame", ["videos", "streams"]),
    ("TokoyamiTowa", "https://www.youtube.com/@TokoyamiTowa", ["videos", "streams"]),
    ("HimemoriLuna", "https://www.youtube.com/@HimemoriLuna", ["videos", "streams"]),
    
    # === Hololive JP 5期生 ===
    ("YukihanaLamy", "https://www.youtube.com/@YukihanaLamy", ["videos", "streams"]),
    ("MomosuzuNene", "https://www.youtube.com/@MomosuzuNene", ["videos", "streams"]),
    ("ShishiroBotan", "https://www.youtube.com/@ShishiroBotan", ["videos", "streams"]),
    ("OmaruPolka", "https://www.youtube.com/@OmaruPolka", ["videos", "streams"]),
    
    # === Hololive JP 6期生 (holoX) ===
    ("LaplaceDarkness", "https://www.youtube.com/@LaplusDarknessch", ["videos", "streams"]),
    ("TakaneLui", "https://www.youtube.com/@TakaneLui", ["videos", "streams"]),
    ("HakuiKoyori", "https://www.youtube.com/@HakuiKoyori", ["videos", "streams"]),
    ("SakamataChloe", "https://www.youtube.com/@SakamataChloe", ["videos", "streams"]),
    ("KazamaIroha", "https://www.youtube.com/@kazamairoha", ["videos", "streams"]),
    
    # === ReGLOSS ===
    ("HiodoshiAo", "https://www.youtube.com/@HiodoshiAo", ["videos", "streams"]),
    ("OtonoseKanade", "https://www.youtube.com/@OtonoseKanade", ["videos", "streams"]),
    ("IchijouRirika", "https://www.youtube.com/@IchijouRirika", ["videos", "streams"]),
    ("JuufuuteiRaden", "https://www.youtube.com/@JuufuuteiRaden", ["videos", "streams"]),
    ("TodorokiHajime", "https://www.youtube.com/@TodorokiHajime", ["videos", "streams"]),
]

# 每个列表最多获取的视频数，0表示不限制
MAX_VIDEOS_PER_LIST = 0  # 扫描所有

# 输出目录
OUTPUT_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/vtuber_channels")


@dataclass
class VideoInfo:
    """视频信息"""
    video_id: str
    channel: str
    title: str
    duration: Optional[int]  # 秒
    list_type: str  # videos, streams, shorts
    has_manual_ja_sub: bool
    has_manual_sub: bool  # 有任何人工字幕
    manual_sub_langs: list
    auto_sub_langs: list


def get_channel_videos(channel_name: str, channel_url: str, list_types: List[str], max_per_list: int = 0) -> list:
    """
    获取频道的视频列表
    
    Args:
        channel_name: 频道名
        channel_url: 频道URL
        list_types: 要获取的列表类型 ["videos", "streams", "shorts"]
        max_per_list: 每个列表最多获取多少个，0表示不限制
    """
    all_videos = []
    
    for list_type in list_types:
        print(f"  获取 {channel_name}/{list_type}...", end=" ", flush=True)
        
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print", "%(id)s|%(title)s|%(duration)s",
        ]
        # 只有在限制数量时才添加--playlist-end
        if max_per_list > 0:
            cmd.extend(["--playlist-end", str(max_per_list)])
        cmd.append(f"{channel_url}/{list_type}")
        
        try:
            # 不限制时可能很慢，增加超时时间
            timeout = 600 if max_per_list == 0 else 120
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            count = 0
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        video_id = parts[0]
                        title = parts[1]
                        try:
                            duration = int(float(parts[2])) if parts[2] and parts[2] not in ['NA', 'None', ''] else None
                        except (ValueError, TypeError):
                            duration = None
                        all_videos.append({
                            'video_id': video_id,
                            'title': title,
                            'duration': duration,
                            'channel': channel_name,
                            'list_type': list_type
                        })
                        count += 1
            print(f"找到 {count} 个")
        except subprocess.TimeoutExpired:
            print(f"超时")
        except Exception as e:
            print(f"错误: {e}")
    
    return all_videos


def check_video_subtitles(video_info: dict) -> VideoInfo:
    """检查视频的字幕情况（只用yt-dlp，不需要ASR）"""
    video_id = video_info['video_id']
    
    cmd = [
        "yt-dlp",
        "--list-subs",
        f"https://youtube.com/watch?v={video_id}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        
        # 解析字幕信息
        manual_langs = []
        auto_langs = []
        
        in_manual_section = False
        in_auto_section = False
        
        for line in output.split('\n'):
            if 'Available subtitles for' in line:
                in_manual_section = True
                in_auto_section = False
            elif 'Available automatic captions for' in line:
                in_auto_section = True
                in_manual_section = False
            elif in_manual_section and line.strip():
                # 解析语言代码 (格式: "ja       vtt, ttml, srv3, srv2, srv1, json3")
                match = re.match(r'^(\w+[-\w]*)\s+', line)
                if match:
                    lang = match.group(1)
                    # 排除live_chat（直播聊天记录，不是字幕）
                    if lang not in ['Language', 'Available', 'live_chat']:
                        manual_langs.append(lang)
            elif in_auto_section and line.strip():
                match = re.match(r'^(\w+[-\w]*)\s+', line)
                if match:
                    lang = match.group(1)
                    if lang not in ['Language', 'Available']:
                        auto_langs.append(lang)
        
        has_manual_ja = 'ja' in manual_langs
        has_manual_sub = len(manual_langs) > 0
        
        return VideoInfo(
            video_id=video_id,
            channel=video_info['channel'],
            title=video_info['title'],
            duration=video_info['duration'],
            list_type=video_info.get('list_type', 'videos'),
            has_manual_ja_sub=has_manual_ja,
            has_manual_sub=has_manual_sub,
            manual_sub_langs=manual_langs,
            auto_sub_langs=auto_langs
        )
        
    except Exception as e:
        return VideoInfo(
            video_id=video_id,
            channel=video_info['channel'],
            title=video_info['title'],
            duration=video_info['duration'],
            list_type=video_info.get('list_type', 'videos'),
            has_manual_ja_sub=False,
            has_manual_sub=False,
            manual_sub_langs=[],
            auto_sub_langs=[]
        )


def main():
    print("=" * 70)
    print("从目标频道查找有人工字幕的视频")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_videos = []
    
    # 获取每个频道的视频列表（包括videos和streams）
    print(f"扫描 {len(TARGET_CHANNELS)} 个频道...")
    print(f"每个列表限制: {'不限制' if MAX_VIDEOS_PER_LIST == 0 else MAX_VIDEOS_PER_LIST}")
    
    for channel_name, channel_url, list_types in TARGET_CHANNELS:
        print(f"\n频道: {channel_name}")
        videos = get_channel_videos(channel_name, channel_url, list_types, max_per_list=MAX_VIDEOS_PER_LIST)
        all_videos.extend(videos)
    
    print(f"\n总计 {len(all_videos)} 个视频待检查")
    
    # 并行检查字幕
    print("\n检查字幕情况...")
    results = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:  # 增加并行数
        futures = {executor.submit(check_video_subtitles, v): v for v in all_videos}
        
        for i, future in enumerate(as_completed(futures)):
            info = future.result()
            results.append(info)
            
            if info.has_manual_ja_sub:
                duration_str = f"{info.duration//60}:{info.duration%60:02d}" if info.duration else "?"
                print(f"  ✓ [{info.channel}] {info.video_id} ({duration_str}) - {info.title[:40]}")
            
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(all_videos)}")
    
    # 统计结果
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    videos_with_ja_sub = [r for r in results if r.has_manual_ja_sub]
    videos_with_any_sub = [r for r in results if r.has_manual_sub]
    videos_other_lang_only = [r for r in results if r.has_manual_sub and not r.has_manual_ja_sub]
    
    print(f"\n总视频数: {len(results)}")
    print(f"有人工日文字幕: {len(videos_with_ja_sub)}")
    print(f"有其他语言人工字幕(无日文): {len(videos_other_lang_only)}")
    print(f"有任何人工字幕: {len(videos_with_any_sub)}")
    
    # 按频道统计
    print("\n按频道统计:")
    for channel_name, _, _ in TARGET_CHANNELS:
        channel_videos = [r for r in results if r.channel == channel_name]
        channel_with_ja = [r for r in channel_videos if r.has_manual_ja_sub]
        channel_with_any = [r for r in channel_videos if r.has_manual_sub]
        print(f"  {channel_name}: 日文{len(channel_with_ja)}/其他{len(channel_with_any)-len(channel_with_ja)}/总{len(channel_videos)}")
    
    # 保存结果
    result_file = OUTPUT_DIR / "subtitled_videos.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(results),
            'with_ja_sub': len(videos_with_ja_sub),
            'with_any_sub': len(videos_with_any_sub),
            'videos': [
                {
                    'video_id': r.video_id,
                    'channel': r.channel,
                    'title': r.title,
                    'duration': r.duration,
                    'list_type': r.list_type,
                    'has_manual_ja_sub': r.has_manual_ja_sub,
                    'has_manual_sub': r.has_manual_sub,
                    'manual_sub_langs': r.manual_sub_langs,
                }
                for r in results
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {result_file}")
    
    # 列出有日文字幕的视频
    if videos_with_ja_sub:
        print(f"\n=== 有人工日文字幕的视频 ({len(videos_with_ja_sub)}个) ===")
        for v in videos_with_ja_sub:
            duration_str = f"{v.duration//60}:{v.duration%60:02d}" if v.duration else "?"
            print(f"  [{v.channel}/{v.list_type}] {v.video_id} ({duration_str}) {v.title[:40]}")
    
    # 列出只有其他语言字幕的视频
    if videos_other_lang_only:
        print(f"\n=== 有其他语言人工字幕的视频 ({len(videos_other_lang_only)}个) ===")
        for v in videos_other_lang_only:
            duration_str = f"{v.duration//60}:{v.duration%60:02d}" if v.duration else "?"
            langs = ', '.join(v.manual_sub_langs)
            print(f"  [{v.channel}/{v.list_type}] {v.video_id} ({duration_str}) [{langs}] {v.title[:30]}")
    
    # 生成下载命令
    if videos_with_ja_sub or videos_other_lang_only:
        print("\n=== 下载命令 ===")
        print("# 日文字幕视频:")
        for v in videos_with_ja_sub[:10]:
            print(f"yt-dlp -x --audio-format wav --write-sub --sub-lang ja 'https://youtube.com/watch?v={v.video_id}'")
        if videos_other_lang_only:
            print("\n# 其他语言字幕视频:")
            for v in videos_other_lang_only[:5]:
                langs = ','.join(v.manual_sub_langs)
                print(f"yt-dlp -x --audio-format wav --write-sub --sub-lang {langs} 'https://youtube.com/watch?v={v.video_id}'")
    else:
        print("\n未找到有人工字幕的视频")


if __name__ == "__main__":
    main()
