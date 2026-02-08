"""
消融实验：测试人声分离和VAD分块对ASR效果的影响

实验设计：
- 变量1：人声分离 (vocal_separation) - 开/关
- 变量2：VAD分块 (vad_chunking) - 开/关

组合方案：
1. baseline: 无人声分离 + 固定分块
2. vocal_only: 人声分离 + 固定分块
3. vad_only: 无人声分离 + VAD分块
4. full: 人声分离 + VAD分块

评估指标：
- 处理时间
- 字幕段数
- 幻觉检测数
- 重复清理数
- 单字符段落比例
- 超长段落比例
- 人工抽样评估（可选）
"""
import os
import sys
import time
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vat.utils.logger import setup_logger

logger = setup_logger("ablation_experiment")

# 测试视频列表（吹雪频道，长度相近）
TEST_VIDEOS = [
    {
        "url": "https://www.youtube.com/watch?v=qmbZIamMXqE",
        "description": "吹雪视频1",
    },
    {
        "url": "https://www.youtube.com/watch?v=-VUQ534Kvg0",
        "description": "吹雪视频2",
    },
    {
        "url": "https://www.youtube.com/watch?v=kWAdT8qt8_8",
        "description": "吹雪视频3",
    },
    {
        "url": "https://www.youtube.com/watch?v=X53mU_mxCDQ",
        "description": "吹雪视频4",
    },
    {
        "url": "https://www.youtube.com/watch?v=i-EreP4zejg",
        "description": "吹雪视频5",
    },
]

# 实验配置组合
EXPERIMENT_CONFIGS = {
    "baseline": {
        "vocal_separation": False,
        "vad_chunking": False,
        "description": "基准：无人声分离 + 固定分块",
    },
    "vocal_only": {
        "vocal_separation": True,
        "vad_chunking": False,
        "description": "仅人声分离 + 固定分块",
    },
    "vad_only": {
        "vocal_separation": False,
        "vad_chunking": True,
        "description": "无人声分离 + VAD分块",
    },
    "full": {
        "vocal_separation": True,
        "vad_chunking": True,
        "description": "完整：人声分离 + VAD分块",
    },
}


@dataclass
class ExperimentResult:
    """单次实验结果"""
    video_id: str
    config_name: str
    config_description: str
    
    # 处理指标
    processing_time_sec: float = 0.0
    audio_duration_sec: float = 0.0
    
    # 字幕统计
    total_segments: int = 0
    hallucinations_removed: int = 0
    repetitions_cleaned: int = 0
    
    # 质量指标
    single_char_segments: int = 0
    long_segments: int = 0  # >100字符
    avg_segment_length: float = 0.0
    
    # 错误信息
    error: Optional[str] = None
    
    # 样本（用于人工评估）
    sample_segments: List[str] = field(default_factory=list)


@dataclass
class VideoInfo:
    """视频信息"""
    video_id: str
    title: str
    duration_sec: float
    audio_path: str


def extract_video_id(url: str) -> str:
    """从URL提取视频ID"""
    import re
    patterns = [
        r'v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url


def download_video_audio(url: str, output_dir: Path) -> Optional[VideoInfo]:
    """下载视频音频"""
    import subprocess
    
    video_id = extract_video_id(url)
    audio_path = output_dir / f"{video_id}.wav"
    
    if audio_path.exists():
        logger.info(f"音频已存在: {audio_path}")
        # 获取时长
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0
        return VideoInfo(
            video_id=video_id,
            title=video_id,
            duration_sec=duration,
            audio_path=str(audio_path),
        )
    
    logger.info(f"下载视频音频: {url}")
    
    try:
        # 使用 yt-dlp 下载音频
        cmd = [
            "yt-dlp",
            "-x",  # 仅提取音频
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(audio_path.with_suffix('.%(ext)s')),
            "--no-playlist",
            url,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"下载失败: {result.stderr}")
            return None
        
        # 查找输出文件
        for ext in ['.wav', '.m4a', '.mp3', '.opus', '.webm']:
            candidate = audio_path.with_suffix(ext)
            if candidate.exists():
                # 如果不是 wav，转换
                if ext != '.wav':
                    subprocess.run([
                        "ffmpeg", "-i", str(candidate),
                        "-ar", "16000", "-ac", "1",
                        str(audio_path), "-y"
                    ], capture_output=True)
                    candidate.unlink()
                break
        
        if not audio_path.exists():
            logger.error(f"音频文件未生成: {audio_path}")
            return None
        
        # 获取时长
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0
        
        return VideoInfo(
            video_id=video_id,
            title=video_id,
            duration_sec=duration,
            audio_path=str(audio_path),
        )
        
    except Exception as e:
        logger.error(f"下载异常: {e}")
        return None


def run_asr_experiment(
    audio_path: str,
    video_id: str,
    config_name: str,
    vocal_separation: bool,
    vad_chunking: bool,
    output_dir: Path,
) -> ExperimentResult:
    """运行单次ASR实验"""
    from vat.asr import WhisperASR, ASRPostProcessor
    from vat.asr.dynamic_chunker import DynamicChunker
    
    config = EXPERIMENT_CONFIGS[config_name]
    result = ExperimentResult(
        video_id=video_id,
        config_name=config_name,
        config_description=config["description"],
    )
    
    start_time = time.time()
    
    try:
        # 准备音频路径
        working_audio = audio_path
        
        # 步骤1: 人声分离（如果启用）
        if vocal_separation:
            logger.info(f"[{config_name}] 执行人声分离...")
            from vat.asr import VocalSeparator, is_vocal_separation_available
            
            if not is_vocal_separation_available():
                logger.warning("人声分离模型不可用，跳过")
            else:
                separator = VocalSeparator()
                sep_result = separator.separate(audio_path, output_dir)
                if sep_result.success and sep_result.vocals_path:
                    working_audio = str(sep_result.vocals_path)
                    logger.info(f"人声分离完成: {working_audio}")
                else:
                    logger.warning(f"人声分离失败: {sep_result.error_message}")
        
        # 步骤2: ASR转录
        logger.info(f"[{config_name}] 执行ASR转录...")
        
        asr = WhisperASR(
            model_name="large-v3",
            device="auto",
            language="ja",
        )
        
        # 根据配置选择分块方式
        if vad_chunking:
            logger.info(f"[{config_name}] 使用VAD分块...")
            chunker = DynamicChunker(method="vad")
            chunks = chunker.split_audio(working_audio)
            
            # 合并所有chunk的转录结果
            all_segments = []
            for chunk in chunks:
                # 保存chunk到临时文件
                chunk_path = output_dir / f"chunk_{chunk.start_ms}.wav"
                with open(chunk_path, 'wb') as f:
                    f.write(chunk.audio_bytes)
                
                # 转录
                asr_data = asr.asr_audio(str(chunk_path))
                
                # 调整时间戳
                for seg in asr_data.segments:
                    seg.start_time += chunk.start_ms / 1000
                    seg.end_time += chunk.start_ms / 1000
                    all_segments.append({
                        'text': seg.text,
                        'start': seg.start_time,
                        'end': seg.end_time,
                    })
                
                # 清理临时文件
                chunk_path.unlink()
        else:
            # 直接转录
            asr_data = asr.asr_audio(working_audio)
            all_segments = [
                {'text': seg.text, 'start': seg.start_time, 'end': seg.end_time}
                for seg in asr_data.segments
            ]
        
        result.total_segments = len(all_segments)
        
        # 步骤3: 后处理统计
        processor = ASRPostProcessor()
        
        for seg in all_segments:
            text = seg['text']
            proc_result = processor.process_text(text)
            
            if proc_result.is_hallucination:
                result.hallucinations_removed += 1
            
            for mod in proc_result.modifications:
                if mod.get('category') == 'repetition_cleaning':
                    result.repetitions_cleaned += 1
            
            # 质量指标
            if len(text.strip()) <= 1:
                result.single_char_segments += 1
            if len(text) > 100:
                result.long_segments += 1
        
        # 计算平均段落长度
        if all_segments:
            total_chars = sum(len(seg['text']) for seg in all_segments)
            result.avg_segment_length = total_chars / len(all_segments)
        
        # 保存样本（前10个和随机10个）
        import random
        samples = all_segments[:10]
        if len(all_segments) > 20:
            random.seed(42)
            samples.extend(random.sample(all_segments[10:], min(10, len(all_segments) - 10)))
        result.sample_segments = [s['text'] for s in samples]
        
        # 保存完整结果
        srt_path = output_dir / f"{video_id}_{config_name}.srt"
        save_srt(all_segments, srt_path)
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        result.error = str(e)
    
    result.processing_time_sec = time.time() - start_time
    return result


def save_srt(segments: List[dict], output_path: Path):
    """保存SRT文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = format_srt_time(seg['start'])
            end = format_srt_time(seg['end'])
            f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")


def format_srt_time(seconds: float) -> str:
    """格式化SRT时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def run_ablation_experiment(
    videos: List[dict],
    output_dir: Path,
    configs: List[str] = None,
) -> List[ExperimentResult]:
    """运行消融实验"""
    if configs is None:
        configs = list(EXPERIMENT_CONFIGS.keys())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    # 下载视频
    video_infos = []
    for video in videos:
        info = download_video_audio(video['url'], output_dir)
        if info:
            video_infos.append(info)
            logger.info(f"视频就绪: {info.video_id} ({info.duration_sec:.1f}s)")
    
    if not video_infos:
        logger.error("没有可用的视频")
        return results
    
    # 对每个视频运行每种配置
    total_experiments = len(video_infos) * len(configs)
    current = 0
    
    for video_info in video_infos:
        for config_name in configs:
            current += 1
            config = EXPERIMENT_CONFIGS[config_name]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"实验 {current}/{total_experiments}")
            logger.info(f"视频: {video_info.video_id}")
            logger.info(f"配置: {config_name} - {config['description']}")
            logger.info(f"{'='*60}")
            
            result = run_asr_experiment(
                audio_path=video_info.audio_path,
                video_id=video_info.video_id,
                config_name=config_name,
                vocal_separation=config["vocal_separation"],
                vad_chunking=config["vad_chunking"],
                output_dir=output_dir,
            )
            
            result.audio_duration_sec = video_info.duration_sec
            results.append(result)
            
            # 打印中间结果
            logger.info(f"结果: 段数={result.total_segments}, "
                       f"幻觉={result.hallucinations_removed}, "
                       f"重复={result.repetitions_cleaned}, "
                       f"耗时={result.processing_time_sec:.1f}s")
    
    return results


def generate_report(results: List[ExperimentResult], output_path: Path):
    """生成实验报告"""
    report = []
    report.append("=" * 80)
    report.append("消融实验报告：人声分离 × VAD分块")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # 按视频分组
    by_video = {}
    for r in results:
        if r.video_id not in by_video:
            by_video[r.video_id] = {}
        by_video[r.video_id][r.config_name] = r
    
    # 汇总统计
    report.append("## 汇总统计")
    report.append("")
    report.append("| 配置 | 平均段数 | 平均幻觉 | 平均重复 | 平均耗时(s) | 单字符比例 |")
    report.append("|------|----------|----------|----------|-------------|------------|")
    
    for config_name in EXPERIMENT_CONFIGS.keys():
        config_results = [r for r in results if r.config_name == config_name and not r.error]
        if not config_results:
            continue
        
        avg_segments = sum(r.total_segments for r in config_results) / len(config_results)
        avg_hall = sum(r.hallucinations_removed for r in config_results) / len(config_results)
        avg_rep = sum(r.repetitions_cleaned for r in config_results) / len(config_results)
        avg_time = sum(r.processing_time_sec for r in config_results) / len(config_results)
        avg_single = sum(r.single_char_segments / r.total_segments * 100 
                        for r in config_results if r.total_segments > 0) / len(config_results)
        
        report.append(f"| {config_name} | {avg_segments:.0f} | {avg_hall:.1f} | {avg_rep:.1f} | {avg_time:.1f} | {avg_single:.2f}% |")
    
    report.append("")
    
    # 详细结果
    report.append("## 详细结果")
    report.append("")
    
    for video_id, configs in by_video.items():
        report.append(f"### 视频: {video_id}")
        report.append("")
        report.append("| 配置 | 段数 | 幻觉 | 重复 | 单字符 | 耗时(s) | 状态 |")
        report.append("|------|------|------|------|--------|---------|------|")
        
        for config_name, r in configs.items():
            status = "✓" if not r.error else f"✗ {r.error[:20]}"
            single_pct = f"{r.single_char_segments/r.total_segments*100:.1f}%" if r.total_segments > 0 else "N/A"
            report.append(f"| {config_name} | {r.total_segments} | {r.hallucinations_removed} | "
                         f"{r.repetitions_cleaned} | {single_pct} | {r.processing_time_sec:.1f} | {status} |")
        
        report.append("")
        
        # 样本对比
        report.append("**样本对比（前5条）:**")
        report.append("")
        for config_name, r in configs.items():
            report.append(f"*{config_name}:*")
            for i, text in enumerate(r.sample_segments[:5], 1):
                report.append(f"  {i}. {text[:60]}{'...' if len(text) > 60 else ''}")
            report.append("")
    
    # 结论
    report.append("## 结论与建议")
    report.append("")
    report.append("基于实验结果，建议采用以下配置：")
    report.append("")
    report.append("（此处需要根据实际结果填写）")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # 同时保存JSON结果
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    
    return '\n'.join(report)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR消融实验")
    parser.add_argument("--output-dir", type=str, default="/local/gzy/4090-48/vat/experiments/ablation",
                       help="输出目录")
    parser.add_argument("--configs", type=str, nargs="+", 
                       default=["baseline", "vocal_only", "vad_only", "full"],
                       help="要测试的配置")
    parser.add_argument("--videos", type=int, default=None,
                       help="测试视频数量（None表示全部）")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 选择测试视频
    videos = TEST_VIDEOS
    if args.videos:
        videos = videos[:args.videos]
    
    logger.info(f"开始消融实验: {len(videos)} 个视频, {len(args.configs)} 种配置")
    
    # 运行实验
    results = run_ablation_experiment(
        videos=videos,
        output_dir=output_dir,
        configs=args.configs,
    )
    
    # 生成报告
    report_path = output_dir / "ablation_report.md"
    report = generate_report(results, report_path)
    
    print("\n" + report)
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
