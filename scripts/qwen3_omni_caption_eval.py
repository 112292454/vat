#!/usr/bin/env python3
"""
Qwen3-Omni Captioner 评测脚本

用途：
- 在 VAT 视频产物目录上复现实验
- 截取指定音频窗口并生成 caption
- 保存独立的 JSON / Markdown 实验报告

注意：
- 这是独立评测脚本，不接入正式 pipeline
- 需要输入目录中已存在可用音频文件
- 如需标题/简介等元信息，可选地只读访问数据库
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
DEFAULT_VIDEO_IDS = ["_QOMPli80JA", "czIBPN1eCbU", "q2B_u_wZWAQ"]
DEFAULT_BATCH_SIZE = 100


@dataclass
class SampleMeta:
    video_id: str
    source_dir: str
    title: str
    uploader: str
    description: str
    duration: float
    original_audio_path: str
    clip_audio_path: str
    clip_start_sec: float
    clip_duration_sec: float
    batch_index: int = 0
    batch_start_sec: float = 0.0
    batch_end_sec: float = 0.0
    batch_duration_sec: float = 0.0
    window_strategy: str = "manual"
    content_type: str = ""
    batch_preview_text: str = ""


@dataclass
class SampleResult:
    meta: SampleMeta
    caption: str
    elapsed_sec: float
    model_path: str
    generation_config: Dict[str, Any]


def _sample_key(meta: SampleMeta) -> str:
    return "::".join(
        [
            meta.video_id,
            str(meta.batch_index),
            meta.window_strategy,
            f"{meta.clip_start_sec:.2f}",
            f"{meta.clip_duration_sec:.2f}",
        ]
    )


def _load_model_and_processor(model_path: str, use_flash_attn2: bool):
    from qwen_omni_utils import process_mm_info  # noqa: F401
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "dtype": "auto",
    }
    if use_flash_attn2:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, **kwargs)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    return model, processor


def _run_ffmpeg_extract(
    src_audio: Path,
    dst_audio: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    dst_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(src_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst_audio),
    ]
    subprocess.run(cmd, check=True)


def _find_audio_path(video_dir: Path, video_id: str) -> Path:
    candidates = [
        video_dir / f"{video_id}.wav",
        video_dir / "audio.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"未找到音频文件: {video_dir}")


def _load_video_metadata(video_dir: Path) -> Dict[str, Any]:
    metadata_path = video_dir / "original.json"
    if not metadata_path.exists():
        return {}

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    # original.json 在 VAT 中可能是字幕数组，而不是 video metadata
    if isinstance(data, dict):
        return data
    return {}


def _load_segments(video_dir: Path) -> List[Dict[str, Any]]:
    data_path = video_dir / "original.json"
    if not data_path.exists():
        return []
    data = json.loads(data_path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _read_title_from_db_fallback(db_path: Optional[Path], video_id: str) -> Dict[str, Any]:
    if db_path is None:
        return {}
    if not db_path.exists():
        return {}

    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "select title, metadata from videos where id = ?",
            (video_id,),
        ).fetchone()
        if not row:
            return {}
        title, metadata_text = row
        metadata = json.loads(metadata_text) if metadata_text else {}
        metadata["title"] = title or metadata.get("title", "")
        return metadata
    finally:
        conn.close()


def _prepare_sample(
    output_dir: Path,
    work_dir: Path,
    video_id: str,
    start_sec: float,
    clip_duration_sec: float,
    database_path: Optional[Path] = None,
    batch_index: int = 0,
    batch_start_sec: float = 0.0,
    batch_end_sec: float = 0.0,
    window_strategy: str = "manual",
    content_type: str = "",
    batch_preview_text: str = "",
) -> SampleMeta:
    video_dir = output_dir / video_id
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    audio_path = _find_audio_path(video_dir, video_id)
    metadata = _load_video_metadata(video_dir)
    if not metadata and database_path is not None:
        metadata = _read_title_from_db_fallback(database_path, video_id)

    title = str(metadata.get("title", "")).strip()
    uploader = str(metadata.get("uploader", "")).strip()
    description = str(metadata.get("description", "")).strip()
    duration = float(metadata.get("duration", 0) or 0)

    clip_dir = work_dir / "clips"
    clip_path = clip_dir / f"{video_id}_{int(start_sec)}_{int(clip_duration_sec)}.wav"
    _run_ffmpeg_extract(audio_path, clip_path, start_sec=start_sec, duration_sec=clip_duration_sec)

    return SampleMeta(
        video_id=video_id,
        source_dir=str(video_dir),
        title=title,
        uploader=uploader,
        description=description,
        duration=duration,
        original_audio_path=str(audio_path),
        clip_audio_path=str(clip_path),
        clip_start_sec=start_sec,
        clip_duration_sec=clip_duration_sec,
        batch_index=batch_index,
        batch_start_sec=batch_start_sec,
        batch_end_sec=batch_end_sec,
        batch_duration_sec=max(0.0, batch_end_sec - batch_start_sec),
        window_strategy=window_strategy,
        content_type=content_type,
        batch_preview_text=batch_preview_text,
    )


def _select_batch_indices(total_segments: int, batch_size: int) -> List[int]:
    if total_segments <= 0:
        return []

    total_batches = (total_segments + batch_size - 1) // batch_size
    first = 1
    middle = (total_batches + 1) // 2
    last = total_batches

    selected: List[int] = []
    for idx in [first, middle, last]:
        if idx not in selected:
            selected.append(idx)
    return selected


def _compute_strategy_start(
    strategy: str,
    batch_start_sec: float,
    batch_end_sec: float,
    clip_duration_sec: float,
) -> float:
    if strategy == "start":
        return batch_start_sec

    if strategy == "midpoint":
        batch_mid = (batch_start_sec + batch_end_sec) / 2
        latest_valid_start = max(batch_start_sec, batch_end_sec - clip_duration_sec)
        return max(batch_start_sec, min(batch_mid - clip_duration_sec / 2, latest_valid_start))

    raise ValueError(f"不支持的窗口策略: {strategy}")


def _build_manifest_samples(
    output_dir: Path,
    work_dir: Path,
    manifest_path: Path,
    batch_size: int,
    clip_duration_sec: float,
    strategies: Sequence[str],
    database_path: Optional[Path],
) -> List[SampleMeta]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: List[SampleMeta] = []

    for item in manifest["videos"]:
        video_id = item["video_id"]
        content_type = item.get("content_type", "")
        video_dir = output_dir / video_id
        segments = _load_segments(video_dir)
        batch_indices = _select_batch_indices(len(segments), batch_size)

        for batch_index in batch_indices:
            start_idx = (batch_index - 1) * batch_size
            chunk = segments[start_idx : start_idx + batch_size]
            if not chunk:
                continue

            batch_start_sec = float(chunk[0]["start"])
            batch_end_sec = float(chunk[-1]["end"])
            batch_preview_text = " ".join(seg["message"].replace("\n", " ") for seg in chunk[:12])[:500]

            for strategy in strategies:
                clip_start_sec = _compute_strategy_start(
                    strategy=strategy,
                    batch_start_sec=batch_start_sec,
                    batch_end_sec=batch_end_sec,
                    clip_duration_sec=clip_duration_sec,
                )
                samples.append(
                    _prepare_sample(
                        output_dir=output_dir,
                        work_dir=work_dir,
                        video_id=video_id,
                        start_sec=clip_start_sec,
                        clip_duration_sec=clip_duration_sec,
                        database_path=database_path,
                        batch_index=batch_index,
                        batch_start_sec=batch_start_sec,
                        batch_end_sec=batch_end_sec,
                        window_strategy=strategy,
                        content_type=content_type,
                        batch_preview_text=batch_preview_text,
                    )
                )

    return samples


def _generate_caption(
    model,
    processor,
    audio_path: str,
) -> str:
    from qwen_omni_utils import process_mm_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids, _ = model.generate(
        **inputs,
        thinker_return_dict_in_generate=True,
    )
    response = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response.strip()


def _save_results(
    results: List[SampleResult],
    work_dir: Path,
    model_path: str,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    json_path = work_dir / "results.json"
    md_path = work_dir / "results.md"

    payload = {
        "model_path": model_path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "meta": asdict(result.meta),
                "caption": result.caption,
                "elapsed_sec": result.elapsed_sec,
                "model_path": result.model_path,
                "generation_config": result.generation_config,
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3-Omni Captioner 实验结果",
        "",
        f"- 模型：`{model_path}`",
        f"- 样本数：`{len(results)}`",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## {item.meta.video_id}",
                "",
                f"- 标题：{item.meta.title or '(缺失)'}",
                f"- 主播：{item.meta.uploader or '(缺失)'}",
                f"- 内容类型：{item.meta.content_type or '(未标注)'}",
                f"- batch：{item.meta.batch_index or '(手动样本)'}",
                f"- batch 时间范围：{item.meta.batch_start_sec:.1f}s - {item.meta.batch_end_sec:.1f}s",
                f"- 窗口策略：{item.meta.window_strategy}",
                f"- 采样区间：{item.meta.clip_start_sec:.1f}s - {item.meta.clip_start_sec + item.meta.clip_duration_sec:.1f}s",
                f"- 推理耗时：{item.elapsed_sec:.2f}s",
                "",
                "### Batch Preview",
                "",
                item.meta.batch_preview_text or "(无)",
                "",
                "### Caption",
                "",
                item.caption or "(空输出)",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _load_existing_results(work_dir: Path) -> List[SampleResult]:
    json_path = work_dir / "results.json"
    if not json_path.exists():
        return []

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results: List[SampleResult] = []
    for item in payload.get("results", []):
        results.append(
            SampleResult(
                meta=SampleMeta(**item["meta"]),
                caption=item["caption"],
                elapsed_sec=item["elapsed_sec"],
                model_path=item["model_path"],
                generation_config=item["generation_config"],
            )
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-Omni Captioner 实验脚本")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_NAME,
        help="模型名或本地目录",
    )
    parser.add_argument(
        "--output-dir",
        help="VAT 视频输出根目录；不提供时尝试从 VAT 配置读取",
    )
    parser.add_argument(
        "--database-path",
        help="VAT 数据库路径；仅用于补充视频元信息，只读访问",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        default=[],
        help="要评测的视频 ID，可重复传入",
    )
    parser.add_argument(
        "--manifest-json",
        help="批量评测清单 JSON 路径；提供后忽略 --video-id 和手动 start-sec",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="按 VAT 实际翻译批次划分窗口时使用的 batch 大小",
    )
    parser.add_argument(
        "--window-strategy",
        action="append",
        dest="window_strategies",
        default=[],
        help="批量模式的窗口策略，可重复传入，如 start / midpoint",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="从音频的哪个时间点开始截取",
    )
    parser.add_argument(
        "--clip-duration-sec",
        type=float,
        default=30.0,
        help="每个样本截取多长音频",
    )
    parser.add_argument(
        "--use-flash-attn2",
        action="store_true",
        help="加载模型时启用 flash_attention_2",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/qwen3_omni_caption_eval",
        help="实验工作目录",
    )
    parser.add_argument(
        "--clean-work-dir",
        action="store_true",
        help="运行前清空实验工作目录",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有 results.json 断点续跑",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir: Optional[Path] = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    database_path: Optional[Path] = Path(args.database_path).expanduser().resolve() if args.database_path else None
    if output_dir is None:
        from vat.config import load_config

        config = load_config()
        output_dir = Path(config.storage.output_dir)
        if database_path is None:
            database_path = Path(config.storage.database_path)

    work_dir = Path(args.work_dir).expanduser().resolve()
    video_ids = args.video_ids or list(DEFAULT_VIDEO_IDS)

    if args.clean_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] output_dir={output_dir}")
    print(f"[info] work_dir={work_dir}")
    print(f"[info] model_path={args.model_path}")
    print(f"[info] video_ids={video_ids}")

    if args.manifest_json:
        samples = _build_manifest_samples(
            output_dir=output_dir,
            work_dir=work_dir,
            manifest_path=Path(args.manifest_json),
            batch_size=args.batch_size,
            clip_duration_sec=args.clip_duration_sec,
            strategies=args.window_strategies or ["start", "midpoint"],
            database_path=database_path,
        )
    else:
        samples = [
            _prepare_sample(
                output_dir=output_dir,
                work_dir=work_dir,
                video_id=video_id,
                start_sec=args.start_sec,
                clip_duration_sec=args.clip_duration_sec,
                database_path=database_path,
            )
            for video_id in video_ids
        ]

    existing_results = _load_existing_results(work_dir) if args.resume else []
    completed_keys = {_sample_key(item.meta) for item in existing_results}
    if completed_keys:
        print(f"[info] resume enabled, already completed {len(completed_keys)} samples")

    load_start = time.perf_counter()
    model, processor = _load_model_and_processor(
        model_path=args.model_path,
        use_flash_attn2=args.use_flash_attn2,
    )
    load_elapsed = time.perf_counter() - load_start
    print(f"[info] model loaded in {load_elapsed:.2f}s")

    results: List[SampleResult] = list(existing_results)
    for sample in samples:
        key = _sample_key(sample)
        if key in completed_keys:
            print(f"[skip] already completed {key}")
            continue
        print(f"[info] running caption for {sample.video_id}")
        start = time.perf_counter()
        caption = _generate_caption(
            model=model,
            processor=processor,
            audio_path=sample.clip_audio_path,
        )
        elapsed = time.perf_counter() - start
        result = SampleResult(
            meta=sample,
            caption=caption,
            elapsed_sec=elapsed,
            model_path=args.model_path,
            generation_config={
                "official_captioner_mode": True,
                "use_flash_attn2": args.use_flash_attn2,
            },
        )
        results.append(result)
        _save_results(results, work_dir=work_dir, model_path=args.model_path)
        print(f"[result] {sample.video_id}: {caption}")
        print(f"[info] elapsed={elapsed:.2f}s")

    _save_results(results, work_dir=work_dir, model_path=args.model_path)
    print(f"[info] saved results to {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
