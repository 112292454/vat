"""字幕编解码与文件 I/O。"""

import json
import platform
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

from langdetect import LangDetectException, detect

if TYPE_CHECKING:
    from vat.asr.asr_data import ASRData


def _handle_long_path(path: str) -> str:
    if (
        platform.system() == "Windows"
        and len(path) > 260
        and not path.startswith(r"\\?\ ")
    ):
        return rf"\\?\{Path(path).absolute()}"
    return path


def save_asr_data(
    asr_data: "ASRData",
    save_path: str,
    ass_style: str | None = None,
    style_name: str = "default",
) -> None:
    save_path = _handle_long_path(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if save_path.endswith(".srt"):
        text = asr_data_to_srt(asr_data)
        Path(save_path).write_text(text, encoding="utf-8")
    elif save_path.endswith(".txt"):
        text = asr_data_to_txt(asr_data)
        Path(save_path).write_text(text, encoding="utf-8")
    elif save_path.endswith(".json"):
        Path(save_path).write_text(
            json.dumps(asr_data_to_json(asr_data), ensure_ascii=False),
            encoding="utf-8",
        )
    elif save_path.endswith(".ass"):
        asr_data.to_ass(
            save_path=save_path,
            style_str=ass_style,
            style_name=style_name,
        )
    else:
        raise ValueError(f"Unsupported file extension: {save_path}")


def asr_data_to_txt(asr_data: "ASRData") -> str:
    result = []
    for seg in asr_data.segments:
        original = seg.text
        translated = seg.translated_text
        text = f"{original}\n{translated}" if translated else original
        result.append(text)
    return "\n".join(result)


def asr_data_to_srt(asr_data: "ASRData") -> str:
    srt_lines = []
    for n, seg in enumerate(asr_data.segments, 1):
        original = seg.text
        translated = seg.translated_text
        text = f"{original}\n{translated}" if translated else original
        srt_lines.append(f"{n}\n{seg.to_srt_ts()}\n{text}\n")
    return "\n".join(srt_lines)


def asr_data_to_json(asr_data: "ASRData") -> dict:
    result_json = {}
    for i, segment in enumerate(asr_data.segments, 1):
        result_json[str(i)] = {
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "original_subtitle": segment.text,
            "translated_subtitle": segment.translated_text,
        }
    return result_json


def load_asr_data_from_file(file_path: str) -> "ASRData":
    from vat.asr.asr_data import ASRData

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path_obj}")

    try:
        content = file_path_obj.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path_obj.read_text(encoding="gbk")

    suffix = file_path_obj.suffix.lower()

    if suffix == ".srt":
        return asr_data_from_srt(content)
    if suffix == ".vtt":
        if "<c>" in content:
            return asr_data_from_youtube_vtt(content)
        return asr_data_from_vtt(content)
    if suffix == ".ass":
        return asr_data_from_ass(content)
    if suffix == ".json":
        return asr_data_from_json(json.loads(content))
    raise ValueError(f"Unsupported file format: {suffix}")


def asr_data_from_json(json_data: dict) -> "ASRData":
    from vat.asr.asr_data import ASRData, ASRDataSeg

    segments = []
    for i in sorted(json_data.keys(), key=int):
        segment_data = json_data[i]
        segment = ASRDataSeg(
            text=segment_data["original_subtitle"],
            translated_text=segment_data["translated_subtitle"],
            start_time=segment_data["start_time"],
            end_time=segment_data["end_time"],
        )
        segments.append(segment)
    return ASRData(segments)


def asr_data_from_srt(srt_str: str) -> "ASRData":
    from vat.asr.asr_data import ASRData, ASRDataSeg

    segments = []
    srt_time_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})"
    )
    blocks = re.split(r"\n\s*\n", srt_str.strip())

    def is_different_lang(block: str) -> bool:
        lines = block.splitlines()
        if len(lines) != 4:
            return False
        try:
            return detect(lines[2]) != detect(lines[3])
        except LangDetectException:
            return False

    all_four_lines = all(len(b.splitlines()) == 4 for b in blocks)
    sample = blocks[:50]
    sample_size = len(sample)
    is_bilingual = (
        sample_size > 0
        and all_four_lines
        and sum(map(is_different_lang, sample)) / sample_size >= 0.7
    )

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue

        match = srt_time_pattern.match(lines[1])
        if not match:
            continue

        time_parts = list(map(int, match.groups()))
        start_time = sum(
            [time_parts[0] * 3600000, time_parts[1] * 60000, time_parts[2] * 1000, time_parts[3]]
        )
        end_time = sum(
            [time_parts[4] * 3600000, time_parts[5] * 60000, time_parts[6] * 1000, time_parts[7]]
        )

        if is_bilingual and len(lines) == 4:
            segments.append(ASRDataSeg(lines[2], start_time, end_time, lines[3]))
        else:
            segments.append(ASRDataSeg(" ".join(lines[2:]), start_time, end_time))

    return ASRData(segments)


def asr_data_from_vtt(vtt_str: str) -> "ASRData":
    from vat.asr.asr_data import ASRData, ASRDataSeg

    segments = []
    content = vtt_str.split("\n\n")[2:]

    timestamp_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
    )

    for block in content:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        timestamp_line = lines[1]
        match = timestamp_pattern.match(timestamp_line)
        if not match:
            continue

        time_parts = list(map(int, match.groups()))
        start_time = sum(
            [time_parts[0] * 3600000, time_parts[1] * 60000, time_parts[2] * 1000, time_parts[3]]
        )
        end_time = sum(
            [time_parts[4] * 3600000, time_parts[5] * 60000, time_parts[6] * 1000, time_parts[7]]
        )

        text_line = " ".join(lines[2:])
        cleaned_text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text_line)
        cleaned_text = re.sub(r"</?c>", "", cleaned_text).strip()

        if cleaned_text and cleaned_text != " ":
            segments.append(ASRDataSeg(cleaned_text, start_time, end_time))

    return ASRData(segments)


def asr_data_from_youtube_vtt(vtt_str: str) -> "ASRData":
    from vat.asr.asr_data import ASRData, ASRDataSeg

    def parse_timestamp(ts: str) -> int:
        h, m, s = ts.split(":")
        return int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)

    def split_timestamped_text(text: str) -> List[Any]:
        pattern = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d{3})>([^<]*)")
        matches = list(pattern.finditer(text))
        word_segments = []

        for i in range(len(matches) - 1):
            current_match = matches[i]
            next_match = matches[i + 1]

            start_time = parse_timestamp(current_match.group(1))
            end_time = parse_timestamp(next_match.group(1))
            word = current_match.group(2).strip()

            if word:
                word_segments.append(ASRDataSeg(word, start_time, end_time))

        return word_segments

    segments = []
    blocks = re.split(r"\n\n+", vtt_str.strip())

    timestamp_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}\.\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}\.\d{3})"
    )
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        match = timestamp_pattern.match(lines[0])
        if not match:
            continue

        text = "\n".join(lines)

        timestamp_row = re.search(r"\n(.*?<c>.*?</c>.*)", block)
        if timestamp_row:
            text = re.sub(r"<c>|</c>", "", timestamp_row.group(1))
            block_start_time_string = f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
            block_end_time_string = f"{match.group(4)}:{match.group(5)}:{match.group(6)}"
            text = f"<{block_start_time_string}>{text}<{block_end_time_string}>"

            word_segments = split_timestamped_text(text)
            segments.extend(word_segments)

    return ASRData(segments)


def asr_data_from_ass(ass_str: str) -> "ASRData":
    from vat.asr.asr_data import ASRData, ASRDataSeg

    segments = []
    ass_time_pattern = re.compile(
        r"Dialogue: \d+,(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),(.*?),.*?,\d+,\d+,\d+,.*?,(.*?)$"
    )

    def parse_ass_time(time_str: str) -> int:
        hours, minutes, seconds = time_str.split(":")
        seconds, centiseconds = seconds.split(".")
        return (
            int(hours) * 3600000
            + int(minutes) * 60000
            + int(seconds) * 1000
            + int(centiseconds) * 10
        )

    has_default = "Dialogue:" in ass_str and ",Default," in ass_str
    has_secondary = ",Secondary," in ass_str
    has_translation = has_default and has_secondary
    temp_segments = {}
    lines = ass_str.splitlines()

    for line in lines:
        if not line.startswith("Dialogue:"):
            continue
        match = ass_time_pattern.match(line)
        if not match:
            continue
        start_time = parse_ass_time(match.group(1))
        end_time = parse_ass_time(match.group(2))
        style = match.group(3).strip()
        text = match.group(4)

        text = re.sub(r"\{[^}]*\}", "", text)
        text = text.replace("\\N", "\n").strip()
        if not text:
            continue

        if has_translation:
            if style not in ("Secondary", "Default"):
                continue

            time_key = f"{start_time}-{end_time}"
            if time_key not in temp_segments:
                temp_segments[time_key] = ASRDataSeg(
                    text="", start_time=start_time, end_time=end_time
                )

            if style == "Default":
                temp_segments[time_key].translated_text = text
            else:
                temp_segments[time_key].text = text
        else:
            segments.append(ASRDataSeg(text, start_time, end_time))

    if has_translation:
        segments.extend(temp_segments.values())

    return ASRData(segments)
