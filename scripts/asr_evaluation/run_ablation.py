#!/usr/bin/env python
"""
简化版消融实验脚本
测试人声分离和VAD分块对ASR效果的影响
"""
import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from vat.utils.logger import setup_logger

logger = setup_logger("ablation")

# 实验目录
EXP_DIR = Path("/local/gzy/4090-48/vat/experiments/ablation")

# 测试视频
TEST_VIDEOS = [
    "qmbZIamMXqE",
    "-VUQ534Kvg0",
    "kWAdT8qt8_8",
    "X53mU_mxCDQ",
    "i-EreP4zejg",
]

@dataclass
class Result:
    video_id: str
    config: str
    segments: int
    hallucinations: int
    repetitions: int
    single_char: int
    time_sec: float
    samples: List[str]
    error: str = ""


def get_whisper_asr():
    """从配置创建 WhisperASR 实例"""
    from vat.config import load_config
    from vat.asr import WhisperASR
    
    config = load_config()
    asr_cfg = config.asr
    
    # 获取模型下载目录
    download_root = str(Path(config.storage.models_dir) / asr_cfg.models_subdir) if asr_cfg.models_subdir else None
    
    return WhisperASR(
        model_name=asr_cfg.model,  # 注意：配置中是 model 不是 model_name
        device=asr_cfg.device,
        compute_type=asr_cfg.compute_type,
        language=asr_cfg.language,
        vad_filter=asr_cfg.vad_filter,
        beam_size=asr_cfg.beam_size,
        download_root=download_root,
        word_timestamps=asr_cfg.word_timestamps,
        condition_on_previous_text=asr_cfg.condition_on_previous_text,
        temperature=asr_cfg.temperature,
        compression_ratio_threshold=asr_cfg.compression_ratio_threshold,
        log_prob_threshold=asr_cfg.log_prob_threshold,
        no_speech_threshold=asr_cfg.no_speech_threshold,
        initial_prompt=asr_cfg.initial_prompt,
        repetition_penalty=asr_cfg.repetition_penalty,
        hallucination_silence_threshold=asr_cfg.hallucination_silence_threshold,
        vad_threshold=asr_cfg.vad_threshold,
        vad_min_speech_duration_ms=asr_cfg.vad_min_speech_duration_ms,
        vad_max_speech_duration_s=asr_cfg.vad_max_speech_duration_s,
        vad_min_silence_duration_ms=asr_cfg.vad_min_silence_duration_ms,
        vad_speech_pad_ms=asr_cfg.vad_speech_pad_ms,
        enable_chunked=False,  # 手动控制分块
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


def run_baseline_asr(audio_path: str, video_id: str) -> Result:
    """基准ASR：无人声分离 + 固定分块"""
    from vat.asr import ASRPostProcessor
    
    start = time.time()
    result = Result(video_id=video_id, config="baseline", segments=0, 
                   hallucinations=0, repetitions=0, single_char=0, time_sec=0, samples=[])
    
    try:
        asr = get_whisper_asr()
        asr_data = asr.asr_audio(audio_path)
        
        segments = [{'text': seg.text, 'start': seg.start_time, 'end': seg.end_time} 
                   for seg in asr_data.segments]
        
        result.segments = len(segments)
        result = analyze_segments(result, segments)
        
    except Exception as e:
        result.error = str(e)
    
    result.time_sec = time.time() - start
    return result


def run_vad_asr(audio_path: str, video_id: str) -> Result:
    """VAD分块ASR：无人声分离 + VAD分块"""
    from vat.asr import ASRPostProcessor
    from vat.asr.dynamic_chunker import DynamicChunker
    
    start = time.time()
    result = Result(video_id=video_id, config="vad_only", segments=0,
                   hallucinations=0, repetitions=0, single_char=0, time_sec=0, samples=[])
    
    try:
        chunker = DynamicChunker(method="vad", chunk_length_sec=300)
        chunks = chunker.split_audio(audio_path)
        logger.info(f"VAD分块: {len(chunks)} 块")
        
        asr = get_whisper_asr()
        
        all_segments = []
        for i, chunk in enumerate(chunks):
            # 保存chunk
            chunk_path = EXP_DIR / f"temp_chunk_{i}.wav"
            
            # 使用pydub保存为wav
            from pydub import AudioSegment
            import io
            audio = AudioSegment.from_file(io.BytesIO(chunk.audio_bytes), format="mp3")
            audio.export(str(chunk_path), format="wav")
            
            # 转录
            asr_data = asr.asr_audio(str(chunk_path))
            
            for seg in asr_data.segments:
                all_segments.append({
                    'text': seg.text,
                    'start': seg.start_time + chunk.start_ms / 1000,
                    'end': seg.end_time + chunk.start_ms / 1000,
                })
            
            chunk_path.unlink()
        
        result.segments = len(all_segments)
        result = analyze_segments(result, all_segments)
        
    except Exception as e:
        logger.error(f"VAD ASR失败: {e}")
        result.error = str(e)
    
    result.time_sec = time.time() - start
    return result


def run_vocal_asr(audio_path: str, video_id: str) -> Result:
    """人声分离ASR：人声分离 + 固定分块"""
    from vat.asr import VocalSeparator
    
    start = time.time()
    result = Result(video_id=video_id, config="vocal_only", segments=0,
                   hallucinations=0, repetitions=0, single_char=0, time_sec=0, samples=[])
    
    try:
        # 人声分离
        separator = VocalSeparator()
        sep_result = separator.separate(audio_path, EXP_DIR)
        
        if not sep_result.success:
            result.error = f"人声分离失败: {sep_result.error_message}"
            return result
        
        vocals_path = str(sep_result.vocals_path)
        logger.info(f"人声分离完成: {vocals_path}")
        
        # ASR
        asr = get_whisper_asr()
        asr_data = asr.asr_audio(vocals_path)
        
        segments = [{'text': seg.text, 'start': seg.start_time, 'end': seg.end_time}
                   for seg in asr_data.segments]
        
        result.segments = len(segments)
        result = analyze_segments(result, segments)
        
    except Exception as e:
        logger.error(f"Vocal ASR失败: {e}")
        result.error = str(e)
    
    result.time_sec = time.time() - start
    return result


def run_full_asr(audio_path: str, video_id: str) -> Result:
    """完整ASR：人声分离 + VAD分块"""
    from vat.asr import VocalSeparator
    from vat.asr.dynamic_chunker import DynamicChunker
    
    start = time.time()
    result = Result(video_id=video_id, config="full", segments=0,
                   hallucinations=0, repetitions=0, single_char=0, time_sec=0, samples=[])
    
    try:
        # 人声分离
        separator = VocalSeparator()
        sep_result = separator.separate(audio_path, EXP_DIR)
        
        if not sep_result.success:
            result.error = f"人声分离失败: {sep_result.error_message}"
            return result
        
        vocals_path = str(sep_result.vocals_path)
        
        # VAD分块
        chunker = DynamicChunker(method="vad", chunk_length_sec=300)
        chunks = chunker.split_audio(vocals_path)
        logger.info(f"VAD分块: {len(chunks)} 块")
        
        # ASR
        asr = get_whisper_asr()
        
        all_segments = []
        for i, chunk in enumerate(chunks):
            chunk_path = EXP_DIR / f"temp_chunk_{i}.wav"
            
            from pydub import AudioSegment
            import io
            audio = AudioSegment.from_file(io.BytesIO(chunk.audio_bytes), format="mp3")
            audio.export(str(chunk_path), format="wav")
            
            asr_data = asr.asr_audio(str(chunk_path))
            
            for seg in asr_data.segments:
                all_segments.append({
                    'text': seg.text,
                    'start': seg.start_time + chunk.start_ms / 1000,
                    'end': seg.end_time + chunk.start_ms / 1000,
                })
            
            chunk_path.unlink()
        
        result.segments = len(all_segments)
        result = analyze_segments(result, all_segments)
        
    except Exception as e:
        logger.error(f"Full ASR失败: {e}")
        result.error = str(e)
    
    result.time_sec = time.time() - start
    return result


def analyze_segments(result: Result, segments: List[dict]) -> Result:
    """分析字幕质量"""
    from vat.asr import ASRPostProcessor
    
    processor = ASRPostProcessor()
    
    for seg in segments:
        text = seg['text']
        proc = processor.process_text(text)
        
        if proc.is_hallucination:
            result.hallucinations += 1
        
        for mod in proc.modifications:
            if mod.get('category') == 'repetition_cleaning':
                result.repetitions += 1
        
        if len(text.strip()) <= 1:
            result.single_char += 1
    
    # 抽样
    random.seed(42)
    sample_idx = random.sample(range(len(segments)), min(10, len(segments)))
    result.samples = [segments[i]['text'] for i in sorted(sample_idx)]
    
    return result


def save_results(results: List[Result]):
    """保存结果"""
    output_path = EXP_DIR / "ablation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {output_path}")


def print_report(results: List[Result]):
    """打印报告"""
    print("\n" + "=" * 80)
    print("消融实验报告：人声分离 × VAD分块")
    print("=" * 80)
    
    # 按配置分组统计
    configs = ["baseline", "vad_only", "vocal_only", "full"]
    
    print("\n## 汇总统计\n")
    print("| 配置 | 平均段数 | 平均幻觉 | 平均重复 | 单字符比例 | 平均耗时(s) |")
    print("|------|----------|----------|----------|------------|-------------|")
    
    for config in configs:
        cfg_results = [r for r in results if r.config == config and not r.error]
        if not cfg_results:
            print(f"| {config} | - | - | - | - | - |")
            continue
        
        avg_seg = sum(r.segments for r in cfg_results) / len(cfg_results)
        avg_hall = sum(r.hallucinations for r in cfg_results) / len(cfg_results)
        avg_rep = sum(r.repetitions for r in cfg_results) / len(cfg_results)
        avg_single = sum(r.single_char / r.segments * 100 for r in cfg_results if r.segments > 0) / len(cfg_results)
        avg_time = sum(r.time_sec for r in cfg_results) / len(cfg_results)
        
        print(f"| {config} | {avg_seg:.0f} | {avg_hall:.1f} | {avg_rep:.1f} | {avg_single:.2f}% | {avg_time:.1f} |")
    
    # 详细结果
    print("\n## 详细结果\n")
    
    for video_id in TEST_VIDEOS:
        video_results = [r for r in results if r.video_id == video_id]
        if not video_results:
            continue
        
        print(f"### 视频: {video_id}\n")
        print("| 配置 | 段数 | 幻觉 | 重复 | 单字符 | 耗时(s) |")
        print("|------|------|------|------|--------|---------|")
        
        for r in video_results:
            single_pct = f"{r.single_char/r.segments*100:.1f}%" if r.segments > 0 else "N/A"
            if r.error:
                print(f"| {r.config} | 错误 | - | - | - | - |")
            else:
                print(f"| {r.config} | {r.segments} | {r.hallucinations} | {r.repetitions} | {single_pct} | {r.time_sec:.1f} |")
        
        print()
    
    # 样本对比
    print("\n## 样本对比（baseline vs full）\n")
    
    for video_id in TEST_VIDEOS[:2]:  # 只显示前2个视频
        baseline = next((r for r in results if r.video_id == video_id and r.config == "baseline"), None)
        full = next((r for r in results if r.video_id == video_id and r.config == "full"), None)
        
        if baseline and full and not baseline.error and not full.error:
            print(f"### {video_id}\n")
            print("**baseline:**")
            for i, s in enumerate(baseline.samples[:5], 1):
                print(f"  {i}. {s[:60]}{'...' if len(s) > 60 else ''}")
            print("\n**full:**")
            for i, s in enumerate(full.samples[:5], 1):
                print(f"  {i}. {s[:60]}{'...' if len(s) > 60 else ''}")
            print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=int, default=2, help="测试视频数量")
    parser.add_argument("--configs", nargs="+", default=["baseline", "vad_only"],
                       help="测试配置")
    args = parser.parse_args()
    
    videos = TEST_VIDEOS[:args.videos]
    configs = args.configs
    
    logger.info(f"开始消融实验: {len(videos)} 视频, 配置: {configs}")
    
    results = []
    
    for video_id in videos:
        audio_path = str(EXP_DIR / f"{video_id}.wav")
        
        if not Path(audio_path).exists():
            logger.error(f"音频不存在: {audio_path}")
            continue
        
        for config in configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"视频: {video_id}, 配置: {config}")
            logger.info(f"{'='*60}")
            
            if config == "baseline":
                result = run_baseline_asr(audio_path, video_id)
            elif config == "vad_only":
                result = run_vad_asr(audio_path, video_id)
            elif config == "vocal_only":
                result = run_vocal_asr(audio_path, video_id)
            elif config == "full":
                result = run_full_asr(audio_path, video_id)
            else:
                continue
            
            results.append(result)
            
            if result.error:
                logger.error(f"实验失败: {result.error}")
            else:
                logger.info(f"完成: 段数={result.segments}, 耗时={result.time_sec:.1f}s")
    
    save_results(results)
    print_report(results)


if __name__ == "__main__":
    main()
