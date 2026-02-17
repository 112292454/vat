import json
import math
import os
import platform
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langdetect import LangDetectException, detect

from ..utils.text_utils import is_mainly_cjk
from ..utils.logger import setup_logger

logger = setup_logger("asr_data")

# 注意：不导入 SubtitleLayoutEnum，to_ass() 使用简化的布局逻辑
# 默认行为：双语字幕，原文在上（小字），译文在下（大字）

# 多语言分词模式(支持词级和字符级语言)
_WORD_SPLIT_PATTERN = (
    r"[a-zA-Z\u00c0-\u00ff\u0100-\u017f']+"  # 拉丁字符(含扩展)
    r"|[\u0400-\u04ff]+"  # 西里尔字母(俄文)
    r"|[\u0370-\u03ff]+"  # 希腊字母
    r"|[\u0600-\u06ff]+"  # 阿拉伯文
    r"|[\u0590-\u05ff]+"  # 希伯来文
    r"|\d+"  # 数字
    r"|[\u4e00-\u9fff]"  # 中文
    r"|[\u3040-\u309f]"  # 日文平假名
    r"|[\u30a0-\u30ff]"  # 日文片假名
    r"|[\uac00-\ud7af]"  # 韩文
    r"|[\u0e00-\u0e7f][\u0e30-\u0e3a\u0e47-\u0e4e]*"  # 泰文
    r"|[\u0900-\u097f]"  # 天城文(印地语)
    r"|[\u0980-\u09ff]"  # 孟加拉文
    r"|[\u0e80-\u0eff]"  # 老挝文
    r"|[\u1000-\u109f]"  # 缅甸文
)


def handle_long_path(path: str) -> str:
    r"""Handle Windows long path limitation by adding \\?\ prefix.

    Args:
        path: Original file path

    Returns:
        Path with \\?\ prefix if needed (Windows only)
    """
    if (
        platform.system() == "Windows"
        and len(path) > 260
        and not path.startswith(r"\\?\ ")
    ):
        return rf"\\?\{os.path.abspath(path)}"
    return path


class ASRDataSeg:
    def __init__(
        self, text: str, start_time: int, end_time: int, translated_text: str = "", speaker_id: str = None
    ):
        self.text = text
        self.translated_text = translated_text
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_id = speaker_id

    def to_srt_ts(self) -> str:
        """Convert to SRT timestamp format"""
        return f"{self._ms_to_srt_time(self.start_time)} --> {self._ms_to_srt_time(self.end_time)}"

    def to_lrc_ts(self) -> str:
        """Convert to LRC timestamp format"""
        return f"[{self._ms_to_lrc_time(self.start_time)}]"

    def to_ass_ts(self) -> Tuple[str, str]:
        """Convert to ASS timestamp format"""
        return self._ms_to_ass_ts(self.start_time), self._ms_to_ass_ts(self.end_time)

    @staticmethod
    def _ms_to_lrc_time(ms: int) -> str:
        """Convert milliseconds to LRC time format (MM:SS.cc)"""
        seconds = ms / 1000
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes):02}:{seconds:.2f}"

    @staticmethod
    def _ms_to_srt_time(ms: int) -> str:
        """Convert milliseconds to SRT time format (HH:MM:SS,mmm)"""
        total_seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

    @staticmethod
    def _ms_to_ass_ts(ms: int) -> str:
        """Convert milliseconds to ASS timestamp format (H:MM:SS.cc)"""
        total_seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        centiseconds = int(milliseconds / 10)
        return f"{int(hours):01}:{int(minutes):02}:{int(seconds):02}.{centiseconds:02}"

    @property
    def transcript(self) -> str:
        """Return segment text
            ?意义不明 ai写的
        """
        return self.text

    def __str__(self) -> str:
        return f"ASRDataSeg({self.text}, {self.start_time}, {self.end_time})"


class ASRData:
    def __init__(self, segments: List[ASRDataSeg]):
        assert isinstance(segments, list), "调用契约错误: segments 必须是列表"
        filtered_segments = [seg for seg in segments if seg.text and seg.text.strip()]
        filtered_segments.sort(key=lambda x: x.start_time)
        
        # 过滤退化段：duration <= 0 或 < 50ms（whisper 幽灵段，播放中不可见）
        # 注意：50ms 是显示下限；ASS 格式精度为 10ms(centisecond)，
        # <10ms 的段在 ASS 转换后会变成 start==end 导致下游崩溃
        MIN_DURATION_MS = 50
        valid_segments = []
        degenerate_count = 0
        for seg in filtered_segments:
            if seg.end_time - seg.start_time < MIN_DURATION_MS:
                degenerate_count += 1
                continue
            valid_segments.append(seg)
        
        if degenerate_count > 0:
            logger.warning(
                f"过滤 {degenerate_count} 个退化字幕段（duration < {MIN_DURATION_MS}ms）"
            )
        
        # 排序后校验
        for i in range(1, len(valid_segments)):
            assert valid_segments[i].start_time >= valid_segments[i-1].start_time, \
                f"逻辑错误: 排序后第 {i} 段开始时间早于前一段"
                
        self.segments = valid_segments

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def has_data(self) -> bool:
        """Check if there are any utterances"""
        return len(self.segments) > 0

    def _is_word_level_segment(self, segment: ASRDataSeg) -> bool:
        """判断单个片段是否为词级

        Args:
            segment: 待判断的字幕片段

        Returns:
            True 如果片段符合词级模式
        """
        text = segment.text.strip()

        # CJK语言：1-2个字符
        if is_mainly_cjk(text):
            return len(text) <= 2

        # 非CJK语言（如英文）：单个单词
        words = text.split()
        return len(words) == 1

    def is_word_timestamp(self) -> bool:
        """检查时间戳是否为词级(非句子级)

        词级判定标准:
        - 英文: 单个单词
        - CJK/亚洲语言: 1-2个字符
        - 允许20%误差容忍

        Returns:
            True 如果80%+的片段符合词级模式
        """
        if not self.segments:
            return False

        # 统计符合词级模式的片段数量
        word_level_count = sum(
            1 for seg in self.segments if self._is_word_level_segment(seg)
        )

        WORD_LEVEL_THRESHOLD = 0.8
        word_level_ratio = word_level_count / len(self.segments)

        return word_level_ratio >= WORD_LEVEL_THRESHOLD

    def split_to_word_segments(self) -> "ASRData":
        """将句子级字幕分割为词级字幕,并按音素估算分配时间戳

        时间戳分配基于音素估算(每4个字符约1个音素)

        Returns:
            修改后的ASRData实例
        """
        CHARS_PER_PHONEME = 4
        new_segments = []

        for seg in self.segments:
            text = seg.text
            duration = seg.end_time - seg.start_time

            # 使用统一的多语言分词模式
            words_list = list(re.finditer(_WORD_SPLIT_PATTERN, text))

            if not words_list:
                continue

            # 计算总音素数
            total_phonemes = sum(
                math.ceil(len(w.group()) / CHARS_PER_PHONEME) for w in words_list
            )
            time_per_phoneme = duration / max(total_phonemes, 1)

            # 为每个词分配时间戳
            current_time = seg.start_time
            for word_match in words_list:
                word = word_match.group()
                word_phonemes = math.ceil(len(word) / CHARS_PER_PHONEME)
                word_duration = int(time_per_phoneme * word_phonemes)

                word_end_time = min(current_time + word_duration, seg.end_time)
                new_segments.append(
                    ASRDataSeg(
                        text=word, start_time=current_time, end_time=word_end_time
                    )
                )
                current_time = word_end_time

        self.segments = new_segments
        return self

    def strip_cjk_spaces(self) -> "ASRData":
        """移除 CJK 文本中的无意义空格
        
        ASR (Whisper) 有时在日语文本中插入空格分隔每个词/字，
        例如 "微熱 に なっ て き た ん だ けど"。
        日语不使用空格分词，这些空格会导致下游断句产生碎片化。
        
        策略：
        - CJK 字符之间的空格：直接移除
        - CJK 与 ASCII 之间的空格：移除
        - ASCII 单词之间的空格：保留（如 "CLIP STUDIO PAINT"）
        
        Returns:
            Self for method chaining
        """
        # CJK 范围：CJK统一汉字 + 平假名 + 片假名 + 全角字符
        _CJK = r'[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]'
        
        cleaned_count = 0
        for seg in self.segments:
            text = seg.text
            if not text or ' ' not in text:
                continue
            
            # 判断是否为 CJK 为主的文本（阈值 30%）
            cjk_chars = len(re.findall(_CJK, text))
            non_space = len(re.findall(r'\S', text))
            if non_space == 0 or cjk_chars / non_space < 0.3:
                continue
            
            # 1. 移除两个 CJK 字符之间的空格
            new_text = re.sub(
                rf'(?<={_CJK})[\s]+(?={_CJK})',
                '', text
            )
            # 2. 移除 CJK 与 ASCII 之间的空格
            new_text = re.sub(
                rf'(?<={_CJK})[\s]+(?=[A-Za-z0-9])',
                '', new_text
            )
            new_text = re.sub(
                rf'(?<=[A-Za-z0-9])[\s]+(?={_CJK})',
                '', new_text
            )
            
            if new_text != text:
                seg.text = new_text.strip()
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"strip_cjk_spaces: 清理了 {cleaned_count} 个片段中的无意义空格")
        
        return self

    def dedup_adjacent_segments(self, max_gap_ms: int = 15000) -> "ASRData":
        """移除相邻的重复/重叠字幕段
        
        处理 Whisper 分块处理时产生的两类重复：
        1. 完全相同文本的相邻段（原有逻辑）
        2. 时间重叠的相邻段（分块边界重叠，文本可能不完全相同）
        
        对于时间重叠段，按以下策略处理：
        - 文本包含关系（一方是另一方的子串）→ 移除被包含的，保留更完整的
        - 重叠比 ≥ 50%（占较短段时长）→ 移除较短段（同一音频区域的重复识别）
        - 重叠比 < 50% → 保留两段，调整边界消除重叠（不同内容，仅时序微偏）
        
        Args:
            max_gap_ms: 最大时间间隔（毫秒），超过此间隔的相同文本视为有意重复
            
        Returns:
            Self for method chaining
        """
        if len(self.segments) < 2:
            return self
        
        kept = [self.segments[0]]
        removed_count = 0
        adjusted_count = 0
        
        for i in range(1, len(self.segments)):
            prev = kept[-1]
            curr = self.segments[i]
            
            # Case 0: 文本完全相同 + 时间间隔在阈值内 → 精确重复
            gap = curr.start_time - prev.end_time
            if curr.text.strip() == prev.text.strip() and gap <= max_gap_ms:
                prev.end_time = max(prev.end_time, curr.end_time)
                removed_count += 1
                continue
            
            # 以下仅处理时间重叠的情况
            if curr.start_time >= prev.end_time:
                # 无时间重叠，正常保留
                kept.append(curr)
                continue
            
            # 存在时间重叠：curr.start_time < prev.end_time
            overlap_ms = prev.end_time - curr.start_time
            dur_prev = prev.end_time - prev.start_time
            dur_curr = curr.end_time - curr.start_time
            min_dur = min(dur_prev, dur_curr)
            overlap_ratio = overlap_ms / min_dur if min_dur > 0 else 1.0
            
            prev_text = prev.text.strip()
            curr_text = curr.text.strip()
            
            # Case 1: 文本包含关系（一方是另一方的子串）
            # Whisper 分块重叠的典型模式：后段文本是前段的后缀
            prev_contains_curr = len(curr_text) >= 2 and curr_text in prev_text
            curr_contains_prev = len(prev_text) >= 2 and prev_text in curr_text
            
            if prev_contains_curr:
                # prev 包含 curr 的全部内容 → 移除 curr
                prev.end_time = max(prev.end_time, curr.end_time)
                removed_count += 1
                logger.debug(
                    f"dedup: 移除被包含段 (prev⊃curr), "
                    f"prev=\"{prev_text[:30]}\" curr=\"{curr_text[:30]}\""
                )
                continue
            
            if curr_contains_prev:
                # curr 包含 prev 的全部内容 → 用 curr 替换 prev
                curr.start_time = min(prev.start_time, curr.start_time)
                kept[-1] = curr
                removed_count += 1
                logger.debug(
                    f"dedup: 替换为更完整段 (curr⊃prev), "
                    f"prev=\"{prev_text[:30]}\" curr=\"{curr_text[:30]}\""
                )
                continue
            
            # Case 2: 高重叠比（≥50%）→ 同一音频区域的不同识别结果
            # 保留时长更长的（通常包含更多上下文，ASR 结果更可靠）
            if overlap_ratio >= 0.5:
                if dur_prev >= dur_curr:
                    # 保留 prev，移除 curr
                    prev.end_time = max(prev.end_time, curr.end_time)
                    removed_count += 1
                else:
                    # 保留 curr，替换 prev
                    curr.start_time = min(prev.start_time, curr.start_time)
                    kept[-1] = curr
                    removed_count += 1
                logger.debug(
                    f"dedup: 移除高重叠段 (ratio={overlap_ratio:.2f}), "
                    f"kept_dur={max(dur_prev,dur_curr)}ms, "
                    f"removed_dur={min(dur_prev,dur_curr)}ms"
                )
                continue
            
            # Case 3: 低重叠比（<50%）→ 不同内容，仅时序微偏
            # 调整边界消除重叠，两段都保留
            prev.end_time = curr.start_time
            adjusted_count += 1
            kept.append(curr)
        
        if removed_count > 0 or adjusted_count > 0:
            logger.info(
                f"dedup_adjacent_segments: 移除 {removed_count} 个重复/重叠段, "
                f"调整 {adjusted_count} 个边界"
            )
            self.segments = kept
        
        return self

    def remove_punctuation(self) -> "ASRData":
        """Remove trailing Chinese punctuation (comma, period) from segments."""
        punctuation = r"[，。]"
        for seg in self.segments:
            seg.text = re.sub(f"{punctuation}+$", "", seg.text.strip())
            seg.translated_text = re.sub(
                f"{punctuation}+$", "", seg.translated_text.strip()
            )
        return self

    def save(
        self,
        save_path: str,
        ass_style: Optional[str] = None,
        style_name: str = "default",
    ) -> None:
        """保存字幕数据到文件
        
        布局固定为：双语字幕，原文在上，译文在下

        Args:
            save_path: 输出文件路径（支持 .srt, .txt, .json, .ass）
            ass_style: ASS 样式字符串（可选）
            style_name: 样式模板名称（可选）
        """
        save_path = handle_long_path(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if save_path.endswith(".srt"):
            self.to_srt(save_path=save_path)
        elif save_path.endswith(".txt"):
            self.to_txt(save_path=save_path)
        elif save_path.endswith(".json"):
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, ensure_ascii=False)
        elif save_path.endswith(".ass"):
            self.to_ass(
                save_path=save_path,
                style_str=ass_style,
                style_name=style_name,
            )
        else:
            raise ValueError(f"Unsupported file extension: {save_path}")

    def to_txt(self, save_path=None) -> str:
        """转换为纯文本格式（无时间戳）
        
        布局：原文在上，译文在下
        """
        result = []
        for seg in self.segments:
            original = seg.text
            translated = seg.translated_text
            text = f"{original}\n{translated}" if translated else original
            result.append(text)
        
        text = "\n".join(result)
        if save_path:
            save_path = handle_long_path(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_srt(self, save_path=None) -> str:
        """转换为 SRT 字幕格式
        
        布局：原文在上，译文在下
        """
        srt_lines = []
        for n, seg in enumerate(self.segments, 1):
            original = seg.text
            translated = seg.translated_text
            text = f"{original}\n{translated}" if translated else original
            srt_lines.append(f"{n}\n{seg.to_srt_ts()}\n{text}\n")

        srt_text = "\n".join(srt_lines)
        if save_path:
            save_path = handle_long_path(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(srt_text)
        return srt_text

    def to_lrc(self, save_path=None) -> str:
        """Convert to LRC subtitle format"""
        raise NotImplementedError("LRC format is not supported")

    def to_json(self) -> dict:
        """Convert to JSON format"""
        result_json = {}
        for i, segment in enumerate(self.segments, 1):
            result_json[str(i)] = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "original_subtitle": segment.text,
                "translated_subtitle": segment.translated_text,
            }
        return result_json

    def to_ass(
        self,
        style_str: Optional[str] = None,
        style_name: str = "default",
        save_path: Optional[str] = None,
        video_width: int = 1280,
        video_height: int = 720,
    ) -> str:
        """转换为 ASS 字幕格式
        
        布局固定为：双语字幕，原文在上（Secondary 小字），译文在下（Default 大字）
        如果没有译文，只显示原文。

        Args:
            style_str: ASS 样式字符串（可选，为 None 时从 style_name 加载）
            style_name: 样式模板名称（对应 resources/subtitle_style/*.txt）
            save_path: ASS 文件保存路径（可选）
            video_width: 视频宽度（默认 1280）
            video_height: 视频高度（默认 720）

        Returns:
            ASS 格式字幕内容
        """
        # 检测说话人数量
        speakers = set(seg.speaker_id for seg in self.segments if seg.speaker_id)
        has_multiple_speakers = len(speakers) > 1

        if not style_str:
            if has_multiple_speakers:
                style_str = self._generate_speaker_styles(speakers)
            else:
                # 尝试从 resources/subtitle_style 加载样式文件
                style_dir = Path(__file__).parent.parent / "resources" / "subtitle_style"
                style_file = style_dir / f"{style_name}.txt"
                if style_file.exists():
                    style_str = style_file.read_text(encoding="utf-8")
                else:
                    # Fallback to a minimal style if nothing is provided
                    style_str = (
                        "[V4+ Styles]\n"
                        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
                        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
                        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
                        "Style: Default,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,"
                        "0,0,1,2,0,2,10,10,15,1\n"
                        "Style: Secondary,Arial,30,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,"
                        "0,0,1,2,0,2,10,10,15,1"
                    )
                    print(f"警告：未找到样式文件，使用默认样式{style_str}")

        ass_content = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {video_width}\n"
            f"PlayResY: {video_height}\n"
            "WrapStyle: 1\n\n"
            f"{style_str}\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        # ASS Layer 碰撞检测机制：
        # - 同一 Layer 内的事件互相参与碰撞检测（自动排开避免重叠）
        # - 不同 Layer 之间不参与碰撞检测
        #
        # 双层字幕策略（发光+本体）：
        # - Layer 0 = 发光底层（_Base 样式 + \blur，与主样式相同 Outline/Shadow 以保证 bounding box 一致）
        # - Layer 1 = 主文字层（主样式，清晰文字）
        # 两层使用相同的事件顺序和数量，确保碰撞检测对称 → 发光层和本体始终重合
        #
        # 对话行顺序（同一片段内）：译文在前、原文在后
        # → 碰撞时译文保持原位，原文被推开（因为 ASS 中先出现的事件有碰撞优先权）
        # _Base 样式与主样式的 Outline/Shadow 完全一致（保证碰撞 bounding box 相同），
        # 发光效果通过 \blur 内联标签实现（后处理特效，不影响碰撞检测布局）
        # 注意：两层都必须有相同结构的 override block，否则部分渲染器（VSFilter等）
        # 在计算自动换行时会因 tag 解析差异导致两层换行位置不同。
        # Main 层使用 \blur0（无模糊）保持 override block 结构对称。
        dlg_base = "Dialogue: 0,{},{},{},,0,0,0,,{{\\blur8}}{}\n"  # Layer 0（发光底层）
        dlg_main = "Dialogue: 1,{},{},{},,0,0,0,,{{\\blur0}}{}\n"  # Layer 1（主文字层，\blur0=无模糊）
        
        # 检测样式表中已有的 _Base 样式，用于决定 base 层使用哪个样式名
        # 如果 X_Base 不存在，base 层使用 X 本身（仍能保证碰撞检测对称）
        style_str_for_check = style_str or ""
        def _base_style(name: str) -> str:
            return f"{name}_Base" if f"Style: {name}_Base," in style_str_for_check else name
        
        for seg in self.segments:
            start_time, end_time = seg.to_ass_ts()
            original = seg.text
            translated = seg.translated_text
            has_translation = bool(translated and translated.strip())

            # 确定样式名称
            if seg.speaker_id and has_multiple_speakers:
                style_name_original = f"Speaker_{seg.speaker_id}"
                style_name_translated = f"Speaker_{seg.speaker_id}_Secondary"
            else:
                # 原文用 Secondary（顶部小字），译文用 Default（底部大字）
                style_name_original = "Secondary"
                style_name_translated = "Default"

            # 固定布局：原文在上，译文在下（双语字幕）
            if has_translation:
                # 译文先写（碰撞优先权：译文保持原位，原文被推开）
                ass_content += dlg_base.format(
                    start_time, end_time, _base_style(style_name_translated), translated
                )
                ass_content += dlg_main.format(
                    start_time, end_time, style_name_translated, translated
                )
                # 原文后写
                ass_content += dlg_base.format(
                    start_time, end_time, _base_style(style_name_original), original
                )
                ass_content += dlg_main.format(
                    start_time, end_time, style_name_original, original
                )
            else:
                # 无译文时只显示原文（仍然两层以保持碰撞检测一致性）
                ass_content += dlg_base.format(
                    start_time, end_time, _base_style(style_name_original), original
                )
                ass_content += dlg_main.format(
                    start_time, end_time, style_name_original, original
                )

        if save_path:
            save_path = handle_long_path(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(ass_content)
        return ass_content

    def _generate_speaker_styles(self, speakers: set) -> str:
        """为每个说话人生成不同颜色的样式"""
        # 预定义颜色方案（BGR格式，适合ASS）
        colors = [
            "&H00FFFFFF",  # 白色
            "&H0000FFFF",  # 黄色
            "&H00FF00FF",  # 洋红
            "&H00FFFF00",  # 青色
            "&H0000FF00",  # 绿色
            "&H000080FF",  # 橙色
        ]
        
        styles = [
            "[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
            "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
            "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        ]
        
        for i, speaker_id in enumerate(sorted(list(speakers))):
            color = colors[i % len(colors)]
            # 原文样式（上方，较大字体）— 发光底层 + 主层
            # _Base 与主样式 Outline/Shadow 完全一致，仅 OutlineColour 不同（发光效果由 \blur 实现）
            styles.append(
                f"Style: Speaker_{speaker_id}_Base,Arial,20,{color},&H000000FF,&H00FFFFFF,&H00000000,"
                f"-1,0,0,0,100,100,0,0,1,2,0,2,10,10,15,1\n"
            )
            styles.append(
                f"Style: Speaker_{speaker_id},Arial,20,{color},&H000000FF,&H00000000,&H00000000,"
                f"-1,0,0,0,100,100,0,0,1,2,0,2,10,10,15,1\n"
            )
            # 译文样式（下方，较小字体，同颜色）— 发光底层 + 主层
            styles.append(
                f"Style: Speaker_{speaker_id}_Secondary_Base,Arial,40,{color},&H000000FF,&H00FFFFFF,&H00000000,"
                f"-1,0,0,0,100,100,0,0,1,2,0,2,10,10,15,1\n"
            )
            styles.append(
                f"Style: Speaker_{speaker_id}_Secondary,Arial,40,{color},&H000000FF,&H00000000,&H00000000,"
                f"-1,0,0,0,100,100,0,0,1,2,0,2,10,10,15,1\n"
            )
        
        return "".join(styles)

    def to_vtt(self, save_path=None) -> str:
        """Convert to WebVTT subtitle format

        Args:
            save_path: Optional save path

        Returns:
            WebVTT format subtitle content
        """
        raise NotImplementedError("WebVTT format is not supported")
        # # WebVTT头部
        # vtt_lines = ["WEBVTT\n"]

        # for n, seg in enumerate(self.segments, 1):
        #     # 转换时间戳格式从毫秒到 HH:MM:SS.mmm
        #     start_time = seg._ms_to_srt_time(seg.start_time).replace(",", ".")
        #     end_time = seg._ms_to_srt_time(seg.end_time).replace(",", ".")

        #     # 添加序号（可选）和时间戳
        #     vtt_lines.append(f"{n}\n{start_time} --> {end_time}\n{seg.transcript}\n")

        # vtt_text = "\n".join(vtt_lines)

        # if save_path:
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         f.write(vtt_text)

        # return vtt_text

    def merge_segments(
        self, start_index: int, end_index: int, merged_text: Optional[str] = None
    ):
        """Merge segments from start_index to end_index (inclusive)."""
        if (
            start_index < 0
            or end_index >= len(self.segments)
            or start_index > end_index
        ):
            raise IndexError("Invalid segment index")
        merged_start_time = self.segments[start_index].start_time
        merged_end_time = self.segments[end_index].end_time
        if merged_text is None:
            merged_text = "".join(
                seg.text for seg in self.segments[start_index : end_index + 1]
            )
        merged_seg = ASRDataSeg(merged_text, merged_start_time, merged_end_time)
        self.segments[start_index : end_index + 1] = [merged_seg]

    def merge_with_next_segment(self, index: int) -> None:
        """Merge segment at index with next segment."""
        if index < 0 or index >= len(self.segments) - 1:
            raise IndexError("Index out of range or no next segment to merge")
        current_seg = self.segments[index]
        next_seg = self.segments[index + 1]
        merged_text = f"{current_seg.text} {next_seg.text}"
        merged_seg = ASRDataSeg(merged_text, current_seg.start_time, next_seg.end_time)
        self.segments[index] = merged_seg
        del self.segments[index + 1]

    def optimize_timing(self, threshold_ms: int = 1000) -> "ASRData":
        """Optimize subtitle display timing by adjusting adjacent segment boundaries.

        If gap between adjacent segments is below threshold, adjust the boundary
        to 3/4 point between them (reduces flicker).

        Args:
            threshold_ms: Time gap threshold in milliseconds (default 1000ms)

        Returns:
            Self for method chaining
        """
        if self.is_word_timestamp() or not self.segments:
            return self

        for i in range(len(self.segments) - 1):
            current_seg = self.segments[i]
            next_seg = self.segments[i + 1]
            time_gap = next_seg.start_time - current_seg.end_time

            if time_gap < threshold_ms:
                mid_time = (
                    current_seg.end_time + next_seg.start_time
                ) // 2 + time_gap // 4
                # 防御性检查：仅在公式结果对两端都合法时才应用
                # 正常（非重叠）输入下此条件恒成立；上游重叠应由 _realign_timestamps 消除
                if mid_time > current_seg.start_time and mid_time < next_seg.end_time:
                    current_seg.end_time = mid_time
                    next_seg.start_time = mid_time

        return self

    def __str__(self):
        return self.to_txt()

    @staticmethod
    def from_subtitle_file(file_path: str) -> "ASRData":
        """Load ASRData from subtitle file.

        Args:
            file_path: Subtitle file path (supports .srt, .vtt, .ass, .json)

        Returns:
            Parsed ASRData instance

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Unsupported file format
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path_obj}")

        try:
            content = file_path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path_obj.read_text(encoding="gbk")

        suffix = file_path_obj.suffix.lower()

        if suffix == ".srt":
            return ASRData.from_srt(content)
        elif suffix == ".vtt":
            if "<c>" in content:
                return ASRData.from_youtube_vtt(content)
            return ASRData.from_vtt(content)
        elif suffix == ".ass":
            return ASRData.from_ass(content)
        elif suffix == ".json":
            return ASRData.from_json(json.loads(content))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def from_json(json_data: dict) -> "ASRData":
        """Create ASRData from JSON data"""
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

    @staticmethod
    def from_srt(srt_str: str) -> "ASRData":
        """Create ASRData from SRT format string.

        Uses language detection to distinguish between bilingual subtitles
        (original + translation) and multiline single-language subtitles.

        Args:
            srt_str: SRT format subtitle string

        Returns:
            Parsed ASRData instance
        """
        segments = []
        srt_time_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})"
        )
        blocks = re.split(r"\n\s*\n", srt_str.strip())

        # Detect bilingual mode: all 4-line + 70% different languages
        def is_different_lang(block: str) -> bool:
            lines = block.splitlines()
            if len(lines) != 4:
                return False
            try:
                return detect(lines[2]) != detect(lines[3])
            except LangDetectException:
                return False

        all_four_lines = all(len(b.splitlines()) == 4 for b in blocks)
        is_bilingual = (
            all_four_lines and sum(map(is_different_lang, blocks[:50])) / 50 >= 0.7
        )

        # Process all blocks based on detected mode
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3:
                continue

            match = srt_time_pattern.match(lines[1])
            if not match:
                continue

            time_parts = list(map(int, match.groups()))
            start_time = sum(
                [
                    time_parts[0] * 3600000,
                    time_parts[1] * 60000,
                    time_parts[2] * 1000,
                    time_parts[3],
                ]
            )
            end_time = sum(
                [
                    time_parts[4] * 3600000,
                    time_parts[5] * 60000,
                    time_parts[6] * 1000,
                    time_parts[7],
                ]
            )

            if is_bilingual and len(lines) == 4:
                segments.append(ASRDataSeg(lines[2], start_time, end_time, lines[3]))
            else:
                segments.append(ASRDataSeg(" ".join(lines[2:]), start_time, end_time))

        return ASRData(segments)

    @staticmethod
    def from_vtt(vtt_str: str) -> "ASRData":
        """Create ASRData from VTT format string.

        Args:
            vtt_str: VTT format subtitle string

        Returns:
            ASRData instance
        """
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
                [
                    time_parts[0] * 3600000,
                    time_parts[1] * 60000,
                    time_parts[2] * 1000,
                    time_parts[3],
                ]
            )
            end_time = sum(
                [
                    time_parts[4] * 3600000,
                    time_parts[5] * 60000,
                    time_parts[6] * 1000,
                    time_parts[7],
                ]
            )

            text_line = " ".join(lines[2:])
            cleaned_text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text_line)
            cleaned_text = re.sub(r"</?c>", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

            if cleaned_text and cleaned_text != " ":
                segments.append(ASRDataSeg(cleaned_text, start_time, end_time))

        return ASRData(segments)

    @staticmethod
    def from_youtube_vtt(vtt_str: str) -> "ASRData":
        """Create ASRData from YouTube VTT format with word-level timestamps.

        Args:
            vtt_str: YouTube VTT format subtitle string (contains <c> tags)

        Returns:
            Parsed ASRData with word-level segments
        """

        def parse_timestamp(ts: str) -> int:
            """Convert timestamp string to milliseconds"""
            h, m, s = ts.split(":")
            return int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)

        def split_timestamped_text(text: str) -> List[ASRDataSeg]:
            """Extract word segments from timestamped text"""
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
                block_start_time_string = (
                    f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
                )
                block_end_time_string = (
                    f"{match.group(4)}:{match.group(5)}:{match.group(6)}"
                )
                text = f"<{block_start_time_string}>{text}<{block_end_time_string}>"

                word_segments = split_timestamped_text(text)
                segments.extend(word_segments)

        return ASRData(segments)

    @staticmethod
    def from_ass(ass_str: str) -> "ASRData":
        """Create ASRData from ASS format string.

        Args:
            ass_str: ASS format subtitle string

        Returns:
            ASRData instance
        """
        segments = []
        ass_time_pattern = re.compile(
            r"Dialogue: \d+,(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),(.*?),.*?,\d+,\d+,\d+,.*?,(.*?)$"
        )

        def parse_ass_time(time_str: str) -> int:
            """Convert ASS timestamp to milliseconds"""
            hours, minutes, seconds = time_str.split(":")
            seconds, centiseconds = seconds.split(".")
            return (
                int(hours) * 3600000
                + int(minutes) * 60000
                + int(seconds) * 1000
                + int(centiseconds) * 10
            )

        # 检查是否有翻译：同时存在Default和Secondary样式
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
          text = text.replace("\\N", "\n")
          text = text.strip()

          if not text:
              continue

          if has_translation:
              # 双语模式：只认 Secondary（原文）和 Default（译文），忽略 Default_Base
              if style not in ("Secondary", "Default"):
                  continue
              
              time_key = f"{start_time}-{end_time}"
              if time_key not in temp_segments:
                  temp_segments[time_key] = ASRDataSeg(
                      text="", start_time=start_time, end_time=end_time
                  )
              
              if style == "Default":
                  temp_segments[time_key].translated_text = text
              else:  # Secondary
                  temp_segments[time_key].text = text
          else:
              segments.append(ASRDataSeg(text, start_time, end_time))

        # 双语模式：将所有 temp_segments 转为 segments
        if has_translation:
            segments.extend(temp_segments.values())

        return ASRData(segments)
