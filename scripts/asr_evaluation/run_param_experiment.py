#!/usr/bin/env python3
"""
ASR参数评估实验

实验目标：
1. 评估关键参数对ASR质量的影响
2. 重点验证：hallucination_silence_threshold, initial_prompt, condition_on_previous_text
3. 考虑唱歌视频的特殊性

实验阶段：
- Phase 1: Smoke测试（3个视频，验证流程）
- Phase 2: 小规模测试（15个视频，筛选参数）
- Phase 3: 全面测试（全部视频，最终结论）

使用方法：
    python run_param_experiment.py --phase smoke
    python run_param_experiment.py --phase small
    python run_param_experiment.py --phase full
"""
import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 目录配置
DATA_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/vtuber_channels")
INPUT_FILE = DATA_DIR / "subtitled_videos.json"
DOWNLOAD_DIR = DATA_DIR / "downloaded"
RESULTS_DIR = DATA_DIR / "param_results"

# 歌曲关键词（用于分类）- 扩展列表
SONG_KEYWORDS = [
    # 日语
    '歌', '曲', '唄', 'オリジナル', 'カバー', 'ライブ', '歌ってみた', '踊ってみた',
    # 英语
    'cover', 'Cover', 'COVER', 'MV', 'Music', 'MUSIC', 'song', 'Song', 'SONG',
    'SINGING', 'Sing', 'sing', 'original', 'Original', 'ORIGINAL',
    'official', 'Official', 'OFFICIAL', 'Live', 'LIVE',
    # 特殊标记
    '(official)', '【original】', '/ ', '／',  # 常见MV格式
]


@dataclass
class ParamConfig:
    """ASR参数配置"""
    name: str
    description: str = ""
    # 重点测试参数
    hallucination_silence_threshold: Optional[float] = None
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True
    # 其他参数
    no_speech_threshold: float = 0.6
    temperature: float = 0.0
    log_prob_threshold: float = -1.0
    repetition_penalty: float = 1.0


# ============================================================
# 参数配置定义
# ============================================================

# 基线配置（使用默认值）
BASELINE = ParamConfig(
    name="baseline",
    description="默认配置",
    hallucination_silence_threshold=None,
    initial_prompt=None,
    condition_on_previous_text=True,
)

# 重点测试的参数组合
PRIORITY_CONFIGS = [
    BASELINE,
    
    # === hallucination_silence_threshold 测试 ===
    # 用于检测长静音后的幻觉，值越小越敏感
    ParamConfig(
        name="hst_1",
        description="幻觉静默阈值=1秒（激进过滤）",
        hallucination_silence_threshold=1.0,
    ),
    ParamConfig(
        name="hst_2",
        description="幻觉静默阈值=2秒（适中）",
        hallucination_silence_threshold=2.0,
    ),
    ParamConfig(
        name="hst_3",
        description="幻觉静默阈值=3秒（保守）",
        hallucination_silence_threshold=3.0,
    ),
    
    # === condition_on_previous_text 测试 ===
    # True: 基于前文预测，可能被开头幻觉污染
    # False: 独立预测，更稳定但可能不连贯
    ParamConfig(
        name="copt_false",
        description="不基于前文预测（避免幻觉传播）",
        condition_on_previous_text=False,
    ),
    ParamConfig(
        name="copt_false_hst2",
        description="不基于前文 + 幻觉阈值2秒",
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    ),
    
    # === initial_prompt 测试 ===
    ParamConfig(
        name="prompt_jp",
        description="日语提示词",
        initial_prompt="これは日本語の音声です。",
    ),
    ParamConfig(
        name="prompt_vtuber",
        description="VTuber场景提示词",
        initial_prompt="VTuberの配信動画です。日本語で話しています。",
    ),
    ParamConfig(
        name="prompt_jp_copt_false",
        description="日语提示 + 不基于前文",
        initial_prompt="これは日本語の音声です。",
        condition_on_previous_text=False,
    ),
]

# 扩展测试配置（Phase 2+）
EXTENDED_CONFIGS = PRIORITY_CONFIGS + [
    # no_speech_threshold 测试
    ParamConfig(
        name="nst_0.4",
        description="无语音阈值=0.4（更多识别）",
        no_speech_threshold=0.4,
    ),
    ParamConfig(
        name="nst_0.8",
        description="无语音阈值=0.8（更少误识别）",
        no_speech_threshold=0.8,
    ),
    
    # repetition_penalty 测试
    ParamConfig(
        name="rp_1.1",
        description="重复惩罚=1.1",
        repetition_penalty=1.1,
    ),
    ParamConfig(
        name="rp_1.2",
        description="重复惩罚=1.2（强惩罚）",
        repetition_penalty=1.2,
    ),
    
    # 组合配置
    ParamConfig(
        name="optimized_v1",
        description="优化组合：copt_false + hst2 + rp1.1",
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        repetition_penalty=1.1,
    ),
]

# Smoke测试配置（快速验证）
SMOKE_CONFIGS = [
    BASELINE,
    ParamConfig(name="hst_2", description="hst=2", hallucination_silence_threshold=2.0),
    ParamConfig(name="copt_false", description="copt=false", condition_on_previous_text=False),
]


def detect_japanese_ratio(text: str) -> float:
    """检测文本中日语字符的比例（平假名+片假名+汉字）"""
    if not text:
        return 0.0
    
    jp_count = 0
    for char in text:
        # 平假名: U+3040-U+309F, 片假名: U+30A0-U+30FF, 汉字: U+4E00-U+9FFF
        if ('\u3040' <= char <= '\u309F' or  # 平假名
            '\u30A0' <= char <= '\u30FF' or  # 片假名
            '\u4E00' <= char <= '\u9FFF'):   # 汉字
            jp_count += 1
    
    return jp_count / len(text)


@dataclass
class EvalResult:
    """评估结果"""
    video_id: str
    config_name: str
    is_song: bool = False
    
    # Ground Truth
    gt_text: str = ""
    gt_length: int = 0
    gt_jp_ratio: float = 0.0  # GT中日语比例
    
    # ASR输出
    asr_text: str = ""
    asr_length: int = 0
    asr_jp_ratio: float = 0.0  # ASR中日语比例
    segments_count: int = 0
    
    # 质量指标
    cer: float = 0.0  # 字符错误率
    is_anomaly: bool = False  # 是否为异常样本（非日语内容）
    anomaly_reason: str = ""  # 异常原因
    
    # 问题检测
    hallucination_count: int = 0  # 检测到的幻觉数
    repetition_count: int = 0     # 重复片段数
    single_char_count: int = 0    # 单字符片段数
    
    # 性能
    elapsed_sec: float = 0.0
    
    # 错误
    error: Optional[str] = None


def calculate_cer(reference: str, hypothesis: str) -> float:
    """计算字符错误率 (Character Error Rate)"""
    import Levenshtein
    if not reference:
        return 1.0 if hypothesis else 0.0
    distance = Levenshtein.distance(reference, hypothesis)
    return distance / len(reference)


def parse_vtt(vtt_path: Path) -> str:
    """解析VTT字幕文件，提取纯文本"""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    texts = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            i += 1
            while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                text = lines[i].strip()
                text = re.sub(r'<[^>]+>', '', text)  # 移除HTML标签
                text = re.sub(r'\u200b', '', text)   # 移除零宽空格
                if text and not text.startswith('##'):
                    texts.append(text)
                i += 1
        else:
            i += 1
    
    return ''.join(texts).replace(' ', '')


def is_song_video(title: str) -> bool:
    """判断是否为歌曲视频"""
    return any(kw in title for kw in SONG_KEYWORDS)


def download_video(video: dict) -> dict:
    """下载单个视频的音频和字幕"""
    video_id = video['video_id']
    
    audio_file = DOWNLOAD_DIR / f"{video_id}.wav"
    sub_file = DOWNLOAD_DIR / f"{video_id}.ja.vtt"
    
    # 检查是否已下载
    if audio_file.exists() and sub_file.exists():
        return {
            'video_id': video_id,
            'audio_path': str(audio_file),
            'sub_path': str(sub_file),
            'success': True,
            'cached': True,
        }
    
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--write-sub", "--sub-lang", "ja",
        "--sub-format", "vtt",
        "-o", str(DOWNLOAD_DIR / f"{video_id}.%(ext)s"),
        f"https://youtube.com/watch?v={video_id}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if audio_file.exists() and sub_file.exists():
            return {
                'video_id': video_id,
                'audio_path': str(audio_file),
                'sub_path': str(sub_file),
                'success': True,
                'cached': False,
            }
        else:
            return {
                'video_id': video_id,
                'success': False,
                'error': f'files not found after download'
            }
            
    except Exception as e:
        return {'video_id': video_id, 'success': False, 'error': str(e)}


def run_asr_with_config(
    audio_path: str,
    video_id: str,
    config: ParamConfig,
    gt_text: str,
    is_song: bool,
) -> EvalResult:
    """使用指定配置运行ASR并评估"""
    from vat.config import load_config
    from vat.asr import WhisperASR
    
    result = EvalResult(
        video_id=video_id,
        config_name=config.name,
        is_song=is_song,
        gt_text=gt_text,
        gt_length=len(gt_text),
    )
    
    start_time = time.time()
    
    try:
        # 加载VAT配置
        vat_config = load_config()
        asr_cfg = vat_config.asr
        download_root = str(Path(vat_config.storage.models_dir) / asr_cfg.models_subdir) if asr_cfg.models_subdir else None
        
        # 处理temperature：配置中可能是列表或单值
        temperature = asr_cfg.temperature
        if isinstance(temperature, (int, float)):
            temperature = [temperature]
        
        # 创建ASR实例（所有参数必须提供）
        asr = WhisperASR(
            model_name=asr_cfg.model,
            device="auto",
            compute_type=asr_cfg.compute_type,
            language="ja",
            vad_filter=asr_cfg.vad_filter,
            beam_size=asr_cfg.beam_size,
            download_root=download_root,
            # 高级参数
            word_timestamps=asr_cfg.word_timestamps,
            condition_on_previous_text=config.condition_on_previous_text,
            temperature=temperature,
            compression_ratio_threshold=asr_cfg.compression_ratio_threshold,
            log_prob_threshold=config.log_prob_threshold,
            no_speech_threshold=config.no_speech_threshold,
            initial_prompt=config.initial_prompt or "",
            repetition_penalty=config.repetition_penalty,
            hallucination_silence_threshold=config.hallucination_silence_threshold,
            # VAD参数
            vad_threshold=asr_cfg.vad_threshold,
            vad_min_speech_duration_ms=asr_cfg.vad_min_speech_duration_ms,
            vad_max_speech_duration_s=asr_cfg.vad_max_speech_duration_s,
            vad_min_silence_duration_ms=asr_cfg.vad_min_silence_duration_ms,
            vad_speech_pad_ms=asr_cfg.vad_speech_pad_ms,
            # ChunkedASR参数
            enable_chunked=False,  # 评估时不分块
            chunk_length_sec=asr_cfg.chunk_length_sec,
            chunk_overlap_sec=asr_cfg.chunk_overlap_sec,
            chunk_concurrency=asr_cfg.chunk_concurrency,
            # Pipeline参数
            use_pipeline=asr_cfg.use_pipeline,
            enable_diarization=asr_cfg.enable_diarization,
            enable_punctuation=asr_cfg.enable_punctuation,
            pipeline_batch_size=asr_cfg.pipeline_batch_size,
            pipeline_chunk_length=asr_cfg.pipeline_chunk_length,
            num_speakers=asr_cfg.num_speakers,
            min_speakers=asr_cfg.min_speakers,
            max_speakers=asr_cfg.max_speakers,
        )
        
        # 执行ASR
        asr_data = asr.asr_audio(audio_path, language="ja")
        
        # 提取文本
        asr_text = "".join([s.text for s in asr_data.segments])
        result.asr_text = asr_text
        result.asr_length = len(asr_text)
        result.segments_count = len(asr_data.segments)
        
        # 计算CER
        result.cer = calculate_cer(gt_text, asr_text)
        
        # 检测日语比例
        result.gt_jp_ratio = detect_japanese_ratio(gt_text)
        result.asr_jp_ratio = detect_japanese_ratio(asr_text)
        
        # 异常检测：多种条件
        # 1. CER>100% 说明ASR输出远长于GT，可能是非日语内容或幻觉严重
        if result.cer > 1.0:
            result.is_anomaly = True
            result.anomaly_reason = f"CER过高({result.cer:.1%}),ASR输出远长于GT"
        # 2. GT日语比例很低，可能是非日语内容配日语字幕
        elif result.gt_jp_ratio < 0.3:
            result.is_anomaly = True
            result.anomaly_reason = f"GT日语比例过低({result.gt_jp_ratio:.1%})"
        # 3. ASR日语比例很低但CER很高，可能是非日语音频
        elif result.asr_jp_ratio < 0.3 and result.cer > 0.8:
            result.is_anomaly = True
            result.anomaly_reason = f"ASR日语比例过低({result.asr_jp_ratio:.1%}),可能是非日语音频"
        # 4. ASR输出为空或极短
        elif result.asr_length < 10:
            result.is_anomaly = True
            result.anomaly_reason = f"ASR输出过短({result.asr_length}字符)"
        # 5. GT过短（<100字符）不适合评估
        elif result.gt_length < 100:
            result.is_anomaly = True
            result.anomaly_reason = f"GT过短({result.gt_length}字符),不适合评估"
        
        # 问题检测
        for seg in asr_data.segments:
            text = seg.text.strip()
            
            # 单字符检测
            if len(text) <= 1:
                result.single_char_count += 1
            
            # 重复检测（简单启发式）
            if len(text) > 10:
                half = len(text) // 2
                if text[:half] == text[half:half*2]:
                    result.repetition_count += 1
        
    except Exception as e:
        result.error = str(e)
    
    result.elapsed_sec = time.time() - start_time
    return result


def select_videos(phase: str) -> List[dict]:
    """根据阶段选择视频"""
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    videos = data.get('videos', [])
    ja_videos = [v for v in videos if v.get('has_manual_ja_sub')]
    
    # 过滤：非歌曲，有时长（用于smoke/small阶段）
    talk_videos = [v for v in ja_videos 
                   if not is_song_video(v.get('title', ''))
                   and (v.get('duration') or 0) >= 60]
    
    # 按时长排序（优先中等长度）
    talk_videos.sort(key=lambda v: abs((v.get('duration') or 0) - 300))
    
    # 所有有日语字幕的视频（用于full阶段）
    all_ja_videos = [v for v in ja_videos if (v.get('duration') or 0) >= 10]
    all_ja_videos.sort(key=lambda v: abs((v.get('duration') or 0) - 300))
    
    if phase == "smoke":
        # Smoke: 3个短视频（非歌曲）
        return talk_videos[:3]
    elif phase == "small":
        # Small: 15个视频（非歌曲）
        return talk_videos[:15]
    elif phase == "full":
        # Full: 所有有日语字幕的视频（包括歌曲）
        print(f"  非歌曲视频: {len(talk_videos)}")
        print(f"  全部日语字幕视频: {len(all_ja_videos)}")
        return all_ja_videos
    else:
        return talk_videos[:5]


def run_single_task(args: tuple) -> EvalResult:
    """
    运行单个ASR评估任务（用于多进程）
    
    Args:
        args: (audio_path, video_id, config, gt_text, is_song, gpu_id)
    """
    audio_path, video_id, config, gt_text, is_song, gpu_id = args
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    return run_asr_with_config(
        audio_path=audio_path,
        video_id=video_id,
        config=config,
        gt_text=gt_text,
        is_song=is_song,
    )


def run_experiment(phase: str, num_gpus: int = 5, workers_per_gpu: int = 2):
    """运行实验（多GPU并行）
    
    Args:
        phase: 实验阶段
        num_gpus: GPU数量
        workers_per_gpu: 每张GPU的并发进程数
    """
    total_workers = num_gpus * workers_per_gpu
    print("=" * 70)
    print(f"ASR参数评估实验 - Phase: {phase}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU数: {num_gpus}, 每GPU进程数: {workers_per_gpu}, 总进程数: {total_workers}")
    print("=" * 70)
    
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 选择配置
    if phase == "smoke":
        configs = SMOKE_CONFIGS
    elif phase == "small":
        configs = PRIORITY_CONFIGS
    else:
        configs = EXTENDED_CONFIGS
    
    print(f"\n测试配置数: {len(configs)}")
    for c in configs:
        print(f"  - {c.name}: {c.description}")
    
    # 选择视频
    videos = select_videos(phase)
    print(f"\n测试视频数: {len(videos)}")
    
    # 下载视频（并行下载）
    print("\n下载视频...")
    downloaded = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_video, v): v for v in videos}
        for future in as_completed(futures):
            v = futures[future]
            result = future.result()
            if result['success']:
                result['title'] = v.get('title', '')
                result['duration'] = v.get('duration', 0)
                downloaded.append(result)
                status = "cached" if result.get('cached') else "downloaded"
                print(f"  ✓ {result['video_id']} ({status})")
            else:
                print(f"  ✗ {v['video_id']}: {result.get('error', 'unknown')}")
    
    if not downloaded:
        print("没有可用的视频，退出")
        return
    
    print(f"\n可用视频: {len(downloaded)}")
    
    # 准备所有任务
    tasks = []
    for video in downloaded:
        video_id = video['video_id']
        audio_path = video['audio_path']
        sub_path = video['sub_path']
        is_song = is_song_video(video.get('title', ''))
        gt_text = parse_vtt(Path(sub_path))
        
        for config in configs:
            tasks.append((audio_path, video_id, config, gt_text, is_song))
    
    total_tasks = len(tasks)
    print(f"\n总任务数: {total_tasks}")
    print(f"预计每进程任务数: {total_tasks // total_workers}")
    
    # 分配GPU并运行（多进程）
    all_results = []
    
    # 为每个任务分配GPU（轮询分配到GPU）
    tasks_with_gpu = [(t[0], t[1], t[2], t[3], t[4], i % num_gpus) for i, t in enumerate(tasks)]
    
    print(f"\n开始并行评估（{num_gpus} GPU × {workers_per_gpu} 进程/GPU = {total_workers} 并发）...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=total_workers) as executor:
        futures = {executor.submit(run_single_task, t): t for t in tasks_with_gpu}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            all_results.append(result)
            
            if result.error:
                print(f"  [{completed}/{total_tasks}] {result.video_id}/{result.config_name}: ✗ {result.error[:40]}")
            else:
                print(f"  [{completed}/{total_tasks}] {result.video_id}/{result.config_name}: CER={result.cer:.2%} ({result.elapsed_sec:.1f}s)")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s (平均每任务: {elapsed/total_tasks:.1f}s)")
    
    # 保存结果
    result_file = RESULTS_DIR / f"results_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {result_file}")
    
    # 生成报告
    generate_report(all_results, phase)


def generate_report(results: List[EvalResult], phase: str):
    """生成评估报告"""
    report_file = RESULTS_DIR / f"report_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    lines = []
    lines.append(f"# ASR参数评估报告 - {phase.upper()}")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n## 1. 实验概述")
    lines.append(f"\n- 视频数量: {len(set(r.video_id for r in results))}")
    lines.append(f"- 配置数量: {len(set(r.config_name for r in results))}")
    lines.append(f"- 总测试数: {len(results)}")
    
    # 统计异常样本
    anomaly_results = [r for r in results if r.is_anomaly and not r.error]
    normal_results = [r for r in results if not r.is_anomaly and not r.error]
    
    lines.append(f"\n### 异常样本统计")
    lines.append(f"- 异常样本数: {len(anomaly_results)} (已排除)")
    lines.append(f"- 有效样本数: {len(normal_results)}")
    if anomaly_results:
        lines.append(f"\n异常原因:")
        for r in anomaly_results[:10]:  # 最多显示10个
            lines.append(f"  - {r.video_id}: {r.anomaly_reason}")
    
    # 按配置汇总（只统计正常样本）
    lines.append(f"\n## 2. 配置对比（排除异常样本）")
    lines.append("\n| 配置 | 平均CER | 平均段数 | 平均耗时 | 单字符率 | 样本数 |")
    lines.append("|------|---------|----------|----------|----------|--------|")
    
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in normal_results:  # 只统计正常样本
        by_config[r.config_name].append(r)
    
    config_stats = []
    for config_name, rs in by_config.items():
        avg_cer = sum(r.cer for r in rs) / len(rs)
        avg_segs = sum(r.segments_count for r in rs) / len(rs)
        avg_time = sum(r.elapsed_sec for r in rs) / len(rs)
        single_rate = sum(r.single_char_count for r in rs) / sum(r.segments_count for r in rs) * 100 if sum(r.segments_count for r in rs) > 0 else 0
        rep_rate = sum(r.repetition_count for r in rs) / len(rs)
        
        config_stats.append({
            'name': config_name,
            'avg_cer': avg_cer,
            'avg_segs': avg_segs,
            'avg_time': avg_time,
            'single_rate': single_rate,
            'rep_rate': rep_rate,
        })
        
        lines.append(f"| {config_name} | {avg_cer:.2%} | {avg_segs:.0f} | {avg_time:.1f}s | {single_rate:.1f}% | {len(rs)} |")
    
    # 排序找出最佳配置
    config_stats.sort(key=lambda x: x['avg_cer'])
    
    lines.append(f"\n## 3. 关键发现")
    
    if config_stats:
        best = config_stats[0]
        baseline = next((c for c in config_stats if c['name'] == 'baseline'), config_stats[-1])
        
        lines.append(f"\n### 3.1 最佳配置")
        lines.append(f"- **{best['name']}**: CER = {best['avg_cer']:.2%}")
        
        if best['name'] != 'baseline':
            improvement = (baseline['avg_cer'] - best['avg_cer']) / baseline['avg_cer'] * 100
            lines.append(f"- 相比baseline提升: {improvement:.1f}%")
        
        lines.append(f"\n### 3.2 参数效果分析")
        
        # 分析 hallucination_silence_threshold
        hst_configs = [c for c in config_stats if c['name'].startswith('hst_')]
        if hst_configs:
            lines.append(f"\n**hallucination_silence_threshold:**")
            for c in hst_configs:
                diff = (c['avg_cer'] - baseline['avg_cer']) / baseline['avg_cer'] * 100
                lines.append(f"- {c['name']}: CER变化 {diff:+.1f}%")
        
        # 分析 condition_on_previous_text
        copt_configs = [c for c in config_stats if 'copt_false' in c['name']]
        if copt_configs:
            lines.append(f"\n**condition_on_previous_text=False:**")
            for c in copt_configs:
                diff = (c['avg_cer'] - baseline['avg_cer']) / baseline['avg_cer'] * 100
                lines.append(f"- {c['name']}: CER变化 {diff:+.1f}%")
        
        # 分析 initial_prompt
        prompt_configs = [c for c in config_stats if c['name'].startswith('prompt_')]
        if prompt_configs:
            lines.append(f"\n**initial_prompt:**")
            for c in prompt_configs:
                diff = (c['avg_cer'] - baseline['avg_cer']) / baseline['avg_cer'] * 100
                lines.append(f"- {c['name']}: CER变化 {diff:+.1f}%")
    
    # 详细结果
    lines.append(f"\n## 4. 详细结果")
    lines.append("\n| 视频ID | 配置 | CER | GT长度 | ASR长度 | 段数 | 耗时 |")
    lines.append("|--------|------|-----|--------|---------|------|------|")
    
    for r in sorted(results, key=lambda x: (x.video_id, x.config_name)):
        if not r.error:
            lines.append(f"| {r.video_id[:11]} | {r.config_name} | {r.cer:.2%} | {r.gt_length} | {r.asr_length} | {r.segments_count} | {r.elapsed_sec:.1f}s |")
    
    # 建议
    lines.append(f"\n## 5. 建议")
    lines.append("\n根据实验结果，建议的配置：")
    lines.append("\n```yaml")
    lines.append("asr:")
    if config_stats:
        best = config_stats[0]
        if 'hst' in best['name']:
            hst_val = best['name'].split('_')[1]
            lines.append(f"  hallucination_silence_threshold: {hst_val}")
        if 'copt_false' in best['name']:
            lines.append("  condition_on_previous_text: false")
        if 'prompt' in best['name']:
            lines.append('  initial_prompt: "これは日本語の音声です。"')
    lines.append("```")
    
    # 写入文件
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n报告已生成: {report_file}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("实验摘要")
    print("=" * 60)
    if config_stats:
        print(f"最佳配置: {config_stats[0]['name']} (CER: {config_stats[0]['avg_cer']:.2%})")
        print(f"基线配置: baseline (CER: {baseline['avg_cer']:.2%})")


def main():
    parser = argparse.ArgumentParser(description='ASR参数评估实验')
    parser.add_argument('--phase', choices=['smoke', 'small', 'full'], default='smoke',
                        help='实验阶段: smoke(3视频), small(15视频), full(全部)')
    parser.add_argument('--gpus', type=int, default=5,
                        help='使用的GPU数量（默认5）')
    parser.add_argument('--workers-per-gpu', type=int, default=2,
                        help='每张GPU的并发进程数（默认2）')
    
    args = parser.parse_args()
    
    run_experiment(args.phase, args.gpus, args.workers_per_gpu)


if __name__ == "__main__":
    main()
