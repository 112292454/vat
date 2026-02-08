#!/usr/bin/env python3
"""
针对异常样本的激进参数测试

测试目标：对于ASR输出过短或CER极高的视频，尝试更激进的参数配置
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# 目录配置
DATA_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/vtuber_channels")
DOWNLOAD_DIR = DATA_DIR / "downloaded"
RESULTS_DIR = DATA_DIR / "param_results"

@dataclass
class AggressiveConfig:
    """激进参数配置"""
    name: str
    description: str
    # Whisper参数
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: str = ""
    hallucination_silence_threshold: float = None
    repetition_penalty: float = 1.0
    # VAD参数
    vad_filter: bool = True
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 2000

# 激进配置列表
AGGRESSIVE_CONFIGS = [
    # 基线（对照）
    AggressiveConfig(
        name="baseline",
        description="默认配置",
    ),
    
    # 1. 更低的语音检测阈值 - 捕获更多语音
    AggressiveConfig(
        name="low_vad_threshold",
        description="降低VAD阈值(0.3)捕获更多语音",
        vad_threshold=0.3,
        min_speech_duration_ms=100,
    ),
    
    # 2. 关闭VAD - 强制处理所有音频
    AggressiveConfig(
        name="no_vad",
        description="关闭VAD，处理全部音频",
        vad_filter=False,
    ),
    
    # 3. 更低的no_speech阈值 - 减少静音检测
    AggressiveConfig(
        name="low_no_speech",
        description="降低no_speech阈值(0.3)",
        no_speech_threshold=0.3,
    ),
    
    # 4. 更宽松的压缩比阈值
    AggressiveConfig(
        name="high_compression",
        description="提高压缩比阈值(3.0)",
        compression_ratio_threshold=3.0,
    ),
    
    # 5. 更大的beam_size
    AggressiveConfig(
        name="large_beam",
        description="增大beam_size(10)",
        beam_size=10,
        best_of=10,
    ),
    
    # 6. 组合：关闭VAD + 低no_speech + prompt
    AggressiveConfig(
        name="aggressive_combo1",
        description="关闭VAD+低no_speech+prompt",
        vad_filter=False,
        no_speech_threshold=0.3,
        initial_prompt="これはVTuberの配信です。日本語で話しています。",
    ),
    
    # 7. 组合：低VAD + 大beam + prompt
    AggressiveConfig(
        name="aggressive_combo2",
        description="低VAD+大beam+prompt",
        vad_threshold=0.2,
        min_speech_duration_ms=50,
        beam_size=10,
        initial_prompt="これはVTuberの動画です。",
    ),
    
    # 8. 极端：全部放宽
    AggressiveConfig(
        name="extreme",
        description="极端配置：全部放宽",
        vad_filter=False,
        no_speech_threshold=0.2,
        compression_ratio_threshold=4.0,
        log_prob_threshold=-2.0,
        beam_size=10,
        initial_prompt="これは日本語の音声です。",
    ),
]


def parse_vtt(vtt_path: Path) -> str:
    """解析VTT文件获取纯文本"""
    if not vtt_path.exists():
        return ""
    
    text_lines = []
    with open(vtt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('WEBVTT') or '-->' in line:
                continue
            if line.isdigit():
                continue
            text_lines.append(line)
    
    return ''.join(text_lines)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """计算字符错误率"""
    import Levenshtein
    if not reference:
        return 1.0
    distance = Levenshtein.distance(reference, hypothesis)
    return distance / len(reference)


def run_asr_task(args: tuple) -> dict:
    """运行单个ASR任务"""
    video_id, config, gpu_id = args
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    audio_path = DOWNLOAD_DIR / f"{video_id}.wav"
    vtt_path = DOWNLOAD_DIR / f"{video_id}.ja.vtt"
    
    if not audio_path.exists() or not vtt_path.exists():
        return {
            'video_id': video_id,
            'config_name': config.name,
            'success': False,
            'error': 'files not found'
        }
    
    gt_text = parse_vtt(vtt_path)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from vat.config import load_config
        from vat.asr import WhisperASR
        
        vat_config = load_config()
        asr_cfg = vat_config.asr
        download_root = str(Path(vat_config.storage.models_dir) / asr_cfg.models_subdir) if asr_cfg.models_subdir else None
        
        start_time = time.time()
        
        # 创建ASR实例
        asr = WhisperASR(
            model_name=asr_cfg.model,
            device="cuda",
            compute_type=asr_cfg.compute_type,
            language="ja",
            vad_filter=config.vad_filter,
            beam_size=config.beam_size,
            download_root=download_root,
            word_timestamps=False,
            condition_on_previous_text=config.condition_on_previous_text,
            temperature=list(config.temperature),
            compression_ratio_threshold=config.compression_ratio_threshold,
            log_prob_threshold=config.log_prob_threshold,
            no_speech_threshold=config.no_speech_threshold,
            initial_prompt=config.initial_prompt,
            repetition_penalty=config.repetition_penalty,
            hallucination_silence_threshold=config.hallucination_silence_threshold,
            # VAD参数
            vad_threshold=config.vad_threshold,
            vad_min_speech_duration_ms=config.min_speech_duration_ms,
            vad_max_speech_duration_s=float('inf'),
            vad_min_silence_duration_ms=config.min_silence_duration_ms,
            vad_speech_pad_ms=400,
            # ChunkedASR参数
            enable_chunked=False,
            chunk_length_sec=30,
            chunk_overlap_sec=2,
            chunk_concurrency=1,
            # Pipeline参数
            use_pipeline=False,
            enable_diarization=False,
            enable_punctuation=False,
            pipeline_batch_size=16,
            pipeline_chunk_length=30,
            num_speakers=None,
            min_speakers=None,
            max_speakers=None,
        )
        
        # 执行ASR
        result = asr.asr_audio(str(audio_path), language="ja")
        
        elapsed = time.time() - start_time
        asr_text = "".join([s.text for s in result.segments])
        cer = calculate_cer(gt_text, asr_text)
        
        return {
            'video_id': video_id,
            'config_name': config.name,
            'cer': cer,
            'gt_length': len(gt_text),
            'asr_length': len(asr_text),
            'segments': len(result.segments),
            'elapsed': elapsed,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        import traceback
        return {
            'video_id': video_id,
            'config_name': config.name,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='异常样本激进参数测试')
    parser.add_argument('--gpus', type=int, default=5, help='GPU数量')
    parser.add_argument('--workers-per-gpu', type=int, default=1, help='每GPU进程数')
    args = parser.parse_args()
    
    total_workers = args.gpus * args.workers_per_gpu
    
    print("=" * 70)
    print("异常样本激进参数测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU数: {args.gpus}, 总进程数: {total_workers}")
    print("=" * 70)
    
    # 加载测试样本
    samples_file = RESULTS_DIR / "aggressive_test_samples.json"
    with open(samples_file) as f:
        samples = json.load(f)
    
    print(f"\n测试样本: {len(samples)}个")
    print(f"测试配置: {len(AGGRESSIVE_CONFIGS)}个")
    
    for c in AGGRESSIVE_CONFIGS:
        print(f"  - {c.name}: {c.description}")
    
    # 准备任务
    tasks = []
    for sample in samples:
        video_id = sample['video_id']
        for config in AGGRESSIVE_CONFIGS:
            gpu_id = len(tasks) % args.gpus
            tasks.append((video_id, config, gpu_id))
    
    total_tasks = len(tasks)
    print(f"\n总任务数: {total_tasks}")
    
    # 并行执行
    all_results = []
    print(f"\n开始测试...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=total_workers) as executor:
        futures = {executor.submit(run_asr_task, t): t for t in tasks}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            all_results.append(result)
            
            if result['success']:
                print(f"  [{completed}/{total_tasks}] {result['video_id']}/{result['config_name']}: CER={result['cer']:.0%}")
            else:
                print(f"  [{completed}/{total_tasks}] {result['video_id']}/{result['config_name']}: ✗ {result.get('error', '')[:50]}")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    # 保存结果
    result_file = RESULTS_DIR / f"aggressive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_file}")
    
    # 生成简要报告
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    from collections import defaultdict
    config_stats = defaultdict(list)
    for r in all_results:
        if r['success'] and r.get('cer') is not None:
            config_stats[r['config_name']].append(r['cer'])
    
    print("\n配置效果对比:")
    stats = []
    for name, cers in config_stats.items():
        if cers:
            stats.append({'name': name, 'avg_cer': sum(cers)/len(cers), 'count': len(cers)})
    
    stats.sort(key=lambda x: x['avg_cer'])
    for s in stats:
        print(f"  {s['name']:20s}: CER={s['avg_cer']:.0%} (n={s['count']})")


if __name__ == "__main__":
    main()
