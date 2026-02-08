#!/usr/bin/env python3
"""
ASR参数评估脚本

功能:
1. 支持多参数配置对比
2. 多GPU并行
3. 计算CER并汇总结果
4. 自动检测非日语内容并跳过

评估参数:
- hallucination_silence_threshold
- initial_prompt
- no_speech_threshold
- condition_on_previous_text
- temperature
- log_prob_threshold
- repetition_penalty
"""

import os
import sys
import re
import json
import time
import unicodedata
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# 使用spawn方式启动子进程，确保CUDA_VISIBLE_DEVICES生效
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过

# 配置
DATA_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/selected/audio")
RESULTS_DIR = Path("/local/gzy/4090-48/vat/experiments/ground_truth/results")
NUM_GPUS = 5
WORKERS_PER_GPU = 2  # 每GPU 2个进程
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU  # 10

# 为快速验证，先只测试部分配置
QUICK_TEST = False  # 设为False运行完整测试

# 日语字符检测阈值
MIN_JAPANESE_RATIO = 0.3  # ASR结果中至少30%应该是日语字符


def is_japanese_text(text: str) -> tuple[bool, float]:
    """
    检查文本是否为日语内容
    
    Args:
        text: 待检查的文本
        
    Returns:
        (is_japanese, ratio): 是否日语, 日语字符比例
    """
    if not text:
        return False, 0.0
    
    japanese_count = 0
    total_count = 0
    
    for char in text:
        if char.isspace():
            continue
        total_count += 1
        
        # 检查是否是日语字符（平假名、片假名、汉字）
        try:
            name = unicodedata.name(char, '')
            if any(x in name for x in ['HIRAGANA', 'KATAKANA', 'CJK']):
                japanese_count += 1
        except ValueError:
            pass
    
    if total_count == 0:
        return False, 0.0
    
    ratio = japanese_count / total_count
    return ratio >= MIN_JAPANESE_RATIO, ratio


@dataclass
class ParamConfig:
    """ASR参数配置"""
    name: str
    hallucination_silence_threshold: Optional[float] = None
    initial_prompt: Optional[str] = None
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    temperature: float = 0.0
    log_prob_threshold: float = -1.0
    repetition_penalty: float = 1.0


# 完整参数配置
FULL_PARAM_CONFIGS = [
    # 基线配置
    ParamConfig(name="baseline"),
    
    # hallucination_silence_threshold 测试
    ParamConfig(name="hst_none", hallucination_silence_threshold=None),
    ParamConfig(name="hst_1", hallucination_silence_threshold=1.0),
    ParamConfig(name="hst_2", hallucination_silence_threshold=2.0),
    ParamConfig(name="hst_3", hallucination_silence_threshold=3.0),
    
    # initial_prompt 测试
    ParamConfig(name="prompt_none", initial_prompt=None),
    ParamConfig(name="prompt_jp", initial_prompt="これは日本語の音声です。"),
    ParamConfig(name="prompt_vtuber", initial_prompt="VTuberの配信です。"),
    
    # no_speech_threshold 测试
    ParamConfig(name="nst_0.4", no_speech_threshold=0.4),
    ParamConfig(name="nst_0.6", no_speech_threshold=0.6),
    ParamConfig(name="nst_0.8", no_speech_threshold=0.8),
    
    # condition_on_previous_text 测试
    ParamConfig(name="copt_true", condition_on_previous_text=True),
    ParamConfig(name="copt_false", condition_on_previous_text=False),
    
    # temperature 测试
    ParamConfig(name="temp_0.0", temperature=0.0),
    ParamConfig(name="temp_0.2", temperature=0.2),
    ParamConfig(name="temp_0.4", temperature=0.4),
    
    # log_prob_threshold 测试
    ParamConfig(name="lpt_-0.5", log_prob_threshold=-0.5),
    ParamConfig(name="lpt_-1.0", log_prob_threshold=-1.0),
    ParamConfig(name="lpt_-1.5", log_prob_threshold=-1.5),
    
    # repetition_penalty 测试
    ParamConfig(name="rp_1.0", repetition_penalty=1.0),
    ParamConfig(name="rp_1.1", repetition_penalty=1.1),
    ParamConfig(name="rp_1.2", repetition_penalty=1.2),
]

# 快速测试配置（验证流程）
QUICK_PARAM_CONFIGS = [
    ParamConfig(name="baseline"),
    ParamConfig(name="hst_2", hallucination_silence_threshold=2.0),
    ParamConfig(name="copt_false", condition_on_previous_text=False),
]

# 根据QUICK_TEST选择配置
PARAM_CONFIGS = QUICK_PARAM_CONFIGS if QUICK_TEST else FULL_PARAM_CONFIGS


def parse_vtt(path: Path) -> str:
    """解析VTT字幕，提取纯文本"""
    with open(path, 'r', encoding='utf-8') as f:
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


def calculate_cer(reference: str, hypothesis: str) -> float:
    """计算字符错误率 (CER)"""
    ref = list(reference.replace(' ', '').replace('\n', ''))
    hyp = list(hypothesis.replace(' ', '').replace('\n', ''))
    
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0
    
    # 动态规划计算编辑距离
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    
    return d[len(ref)][len(hyp)] / len(ref)


def run_video_evaluation(args: tuple) -> list:
    """
    评估单个视频的所有参数配置
    
    流程:
    1. 先用baseline跑ASR，检查结果是否为日语
    2. 如果不是日语，标记为无效并跳过
    3. 如果是日语，继续跑其他参数配置
    
    Args:
        args: (audio_path, vtt_path, configs_list, gpu_id)
        
    Returns:
        list of result dicts
    """
    audio_path, vtt_path, configs, gpu_id = args
    video_id = Path(audio_path).stem
    results = []
    
    # 设置指定的GPU（必须在import torch之前）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # 导入
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from vat.config import load_config
        from vat.asr import WhisperASR
        
        # 解析ground truth
        gt_text = parse_vtt(Path(vtt_path))
        
        # 加载配置
        vat_config = load_config()
        asr_cfg = vat_config.asr
        download_root = str(Path(vat_config.storage.models_dir) / asr_cfg.models_subdir) if asr_cfg.models_subdir else None
        
        def create_asr(config):
            """创建ASR实例"""
            # temperature需要是列表格式
            temp = config.temperature
            if isinstance(temp, (int, float)):
                temp = [temp]
            
            # 使用cuda:0因为CUDA_VISIBLE_DEVICES已经设置为指定GPU
            return WhisperASR(
                model_name=asr_cfg.model,
                device="cuda:0",
                compute_type=asr_cfg.compute_type,
                language=asr_cfg.language,
                vad_filter=asr_cfg.vad_filter,
                beam_size=asr_cfg.beam_size,
                download_root=download_root,
                word_timestamps=asr_cfg.word_timestamps,
                condition_on_previous_text=config.condition_on_previous_text,
                temperature=temp,
                compression_ratio_threshold=asr_cfg.compression_ratio_threshold,
                log_prob_threshold=config.log_prob_threshold,
                no_speech_threshold=config.no_speech_threshold,
                initial_prompt=config.initial_prompt or "",
                repetition_penalty=config.repetition_penalty,
                hallucination_silence_threshold=config.hallucination_silence_threshold,
                vad_threshold=asr_cfg.vad_threshold,
                vad_min_speech_duration_ms=asr_cfg.vad_min_speech_duration_ms,
                vad_max_speech_duration_s=asr_cfg.vad_max_speech_duration_s,
                vad_min_silence_duration_ms=asr_cfg.vad_min_silence_duration_ms,
                vad_speech_pad_ms=asr_cfg.vad_speech_pad_ms,
                enable_chunked=False,
                chunk_length_sec=asr_cfg.chunk_length_sec,
                chunk_overlap_sec=asr_cfg.chunk_overlap_sec,
                chunk_concurrency=asr_cfg.chunk_concurrency,
                use_pipeline=asr_cfg.use_pipeline,
                enable_diarization=asr_cfg.enable_diarization,
                enable_punctuation=asr_cfg.enable_punctuation,
                pipeline_batch_size=asr_cfg.pipeline_batch_size,
                pipeline_chunk_length=asr_cfg.pipeline_chunk_length,
                num_speakers=asr_cfg.num_speakers,
                min_speakers=asr_cfg.min_speakers,
                max_speakers=asr_cfg.max_speakers,
            )
        
        def run_single_config(config) -> dict:
            """运行单个配置的ASR"""
            asr = create_asr(config)
            start_time = time.time()
            result = asr.asr_audio(str(audio_path))
            elapsed = time.time() - start_time
            asr_text = ''.join(s.text for s in result.segments)
            cer = calculate_cer(gt_text, asr_text)
            return {
                'video_id': video_id,
                'config_name': config.name,
                'cer': cer,
                'gt_length': len(gt_text),
                'asr_length': len(asr_text),
                'asr_text': asr_text[:200],  # 保存前200字符用于检查
                'segments': len(result.segments),
                'elapsed': elapsed,
                'success': True,
                'error': None
            }
        
        # 1. 先用baseline配置跑ASR检查是否日语
        baseline_config = configs[0]  # 假设第一个是baseline
        baseline_result = run_single_config(baseline_config)
        
        if baseline_result['success']:
            asr_text = baseline_result.get('asr_text', '')
            is_jp, jp_ratio = is_japanese_text(asr_text)
            
            if not is_jp:
                # 不是日语内容，标记并删除文件
                print(f"  ⚠ {video_id}: 非日语内容 (日语比例: {jp_ratio:.1%})，删除文件")
                # 删除文件
                try:
                    Path(audio_path).unlink(missing_ok=True)
                    Path(vtt_path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"    删除失败: {e}")
                
                return [{
                    'video_id': video_id,
                    'config_name': 'SKIPPED',
                    'cer': None,
                    'success': False,
                    'error': f'非日语内容 (日语比例: {jp_ratio:.1%})',
                    'skipped': True
                }]
        
        # 2. baseline通过检查，记录结果并继续其他配置
        results.append(baseline_result)
        
        # 3. 运行其他配置
        for config in configs[1:]:
            try:
                result = run_single_config(config)
                results.append(result)
            except Exception as e:
                results.append({
                    'video_id': video_id,
                    'config_name': config.name,
                    'cer': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
        
    except Exception as e:
        return [{
            'video_id': video_id,
            'config_name': 'ERROR',
            'cer': None,
            'success': False,
            'error': str(e)
        }]


def main():
    print("=" * 70)
    print("ASR参数评估（带日语内容检测）")
    print(f"GPU数量: {NUM_GPUS}, 并行视频数: {TOTAL_WORKERS}")
    print("=" * 70)
    
    # 收集测试文件
    test_files = []
    for vtt in DATA_DIR.glob("*.ja.vtt"):
        wav = vtt.parent / f"{vtt.stem.replace('.ja', '')}.wav"
        if wav.exists():
            test_files.append((wav, vtt))
    
    # 快速测试时只用前5个文件
    if QUICK_TEST:
        test_files = test_files[:5]
    
    print(f"测试视频数: {len(test_files)}")
    print(f"参数配置数: {len(PARAM_CONFIGS)}")
    print(f"每视频任务数: {len(PARAM_CONFIGS)}")
    
    if not test_files:
        print("错误: 没有找到测试文件")
        return
    
    # 构建任务列表（每个视频一个任务，包含所有配置和指定GPU）
    tasks = []
    for i, (wav, vtt) in enumerate(test_files):
        gpu_id = i % NUM_GPUS  # 轮流分配GPU
        tasks.append((wav, vtt, PARAM_CONFIGS, gpu_id))
    
    print(f"\n开始评估（每个视频先检查是否日语内容）...")
    
    # 并行执行
    all_results = []
    skipped_videos = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with ProcessPoolExecutor(max_workers=TOTAL_WORKERS) as executor:
        futures = {executor.submit(run_video_evaluation, task): task[0] for task in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            video_results = future.result()
            video_id = Path(futures[future]).stem
            
            # 检查是否被跳过
            if video_results and video_results[0].get('skipped'):
                skipped_videos.append(video_id)
                print(f"[{i+1}/{len(tasks)}] {video_id}: SKIPPED - {video_results[0].get('error', '')}")
            else:
                # 打印每个配置的结果
                for r in video_results:
                    if r['success']:
                        print(f"[{i+1}/{len(tasks)}] {r['config_name']}/{r['video_id']}: CER={r['cer']:.4f}")
                    else:
                        err = r.get('error', 'Unknown')[:50]
                        print(f"[{i+1}/{len(tasks)}] {r['config_name']}/{r['video_id']}: ERROR - {err}")
                all_results.extend(video_results)
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    
    # 显示跳过的视频
    if skipped_videos:
        print(f"\n跳过的视频 ({len(skipped_videos)}个): {', '.join(skipped_videos)}")
    
    # 按配置汇总
    config_results = {}
    for r in all_results:
        if r['success']:
            name = r['config_name']
            if name not in config_results:
                config_results[name] = []
            config_results[name].append(r['cer'])
    
    print(f"\n{'配置名称':<20} {'平均CER':>10} {'最小CER':>10} {'最大CER':>10} {'样本数':>8}")
    print("-" * 60)
    
    summary = []
    for name in sorted(config_results.keys()):
        cers = config_results[name]
        avg_cer = sum(cers) / len(cers)
        min_cer = min(cers)
        max_cer = max(cers)
        print(f"{name:<20} {avg_cer:>10.4f} {min_cer:>10.4f} {max_cer:>10.4f} {len(cers):>8}")
        summary.append({
            'config': name,
            'avg_cer': avg_cer,
            'min_cer': min_cer,
            'max_cer': max_cer,
            'count': len(cers)
        })
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 详细结果
    detail_file = RESULTS_DIR / f"evaluation_detail_{timestamp}.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 汇总结果
    summary_file = RESULTS_DIR / f"evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'test_files': len(test_files),
            'skipped_videos': skipped_videos,
            'param_configs': len(PARAM_CONFIGS),
            'summary': sorted(summary, key=lambda x: x['avg_cer'])
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果: {detail_file}")
    print(f"汇总结果: {summary_file}")
    
    # 最优配置
    if summary:
        best = min(summary, key=lambda x: x['avg_cer'])
        print(f"\n最优配置: {best['config']} (平均CER: {best['avg_cer']:.4f})")
    else:
        print("\n没有有效的评估结果")


if __name__ == "__main__":
    main()
