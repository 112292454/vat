"""分块断句处理器"""
import difflib
import re
from typing import List, Tuple

from vat.asr.asr_data import ASRData, ASRDataSeg
from vat.asr.split import split_by_llm
from vat.utils.logger import setup_logger

logger = setup_logger("chunked_split")


def _remove_leading_nchars(text: str, n: int) -> str:
    """
    从文本中移除前 n 个非空白字符，保留剩余部分（包括中间的空白）
    
    例: _remove_leading_nchars("abc def ghi", 4) -> "ef ghi"
    """
    removed = 0
    for i, ch in enumerate(text):
        if ch.strip():  # 非空白字符
            removed += 1
            if removed >= n:
                return text[i + 1:]
    return ""  # 所有非空白字符都被移除


class ChunkedSplitter:
    """分块断句处理器，解决长视频 LLM context 限制"""
    
    def __init__(
        self,
        chunk_size_sentences: int,
        chunk_overlap_sentences: int,
        model: str,
        max_word_count_cjk: int,
        max_word_count_english: int,
        min_word_count_cjk: int = 4,
        min_word_count_english: int = 2,
        recommend_word_count_cjk: int = 12,
        recommend_word_count_english: int = 8,
        scene_prompt: str = "",
        mode: str = "sentence",
        allow_model_upgrade: bool = False,
        model_upgrade_chain: List[str] | None = None,
        api_key: str = "",
        base_url: str = "",
    ):
        """
        初始化分块断句器
        
        Args:
            chunk_size_sentences: 每块句子数
            chunk_overlap_sentences: 重叠句子数
            model: LLM 模型
            max_word_count_cjk: 中文最大字符数（硬性限制）
            max_word_count_english: 英文最大单词数（硬性限制）
            min_word_count_cjk: 中文最小字符数（软性建议）
            min_word_count_english: 英文最小单词数（软性建议）
            recommend_word_count_cjk: 中文推荐字符数（软性建议，理想长度）
            recommend_word_count_english: 英文推荐单词数（软性建议，理想长度）
            scene_prompt: 场景特定提示词（可选）
            mode: 断句模式，"sentence"（句子级）或 "semantic"（语义级）
            allow_model_upgrade: 是否允许模型升级
            model_upgrade_chain: 模型升级顺序列表
        """
        self.chunk_size = chunk_size_sentences
        self.overlap = chunk_overlap_sentences
        self.model = model
        self.max_word_count_cjk = max_word_count_cjk
        self.max_word_count_english = max_word_count_english
        self.min_word_count_cjk = min_word_count_cjk
        self.min_word_count_english = min_word_count_english
        self.recommend_word_count_cjk = recommend_word_count_cjk
        self.recommend_word_count_english = recommend_word_count_english
        self.scene_prompt = scene_prompt
        self.mode = mode
        self.allow_model_upgrade = allow_model_upgrade
        self.model_upgrade_chain = model_upgrade_chain
        self.api_key = api_key
        self.base_url = base_url
        
        # 验证参数合理性
        assert self.overlap < self.chunk_size, \
            f"overlap ({self.overlap}) 必须小于 chunk_size ({self.chunk_size})"
        assert self.overlap >= 1, "overlap 至少为 1"
    
    def split(self, asr_data: ASRData, progress_callback=None) -> ASRData:
        """
        分块断句（并行处理各 chunk，按序合并）
        
        Args:
            asr_data: 原始 ASR 数据（Whisper输出的碎片）
            progress_callback: 进度回调
            
        Returns:
            断句后的 ASRData
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        segments = asr_data.segments
        total = len(segments)
        
        # 分块
        chunks = self._create_chunks(segments)
        num_chunks = len(chunks)
        logger.info(f"将 {total} 个片段分为 {num_chunks} 块并行处理")
        
        def _normalize(s: str) -> str:
            return re.sub(r"\s+", "", s)
        
        # 进度计数器（线程安全）
        completed_count = 0
        count_lock = threading.Lock()
        
        def _process_chunk(chunk_idx: int, chunk_segs: List[ASRDataSeg], start_idx: int, end_idx: int) -> List[ASRDataSeg]:
            """处理单个 chunk：LLM 断句 → 时间戳对齐
            
            仅做计算，不处理 overlap 裁剪（留给合并阶段统一处理）。
            """
            nonlocal completed_count
            
            logger.info(f"处理第 {chunk_idx + 1}/{num_chunks} 块 (片段 {start_idx}-{end_idx})")
            
            # 合并该块的文本
            chunk_text = "".join(seg.text for seg in chunk_segs)
            
            # 调用 LLM 断句
            split_texts = split_by_llm(
                chunk_text,
                model=self.model,
                max_word_count_cjk=self.max_word_count_cjk,
                max_word_count_english=self.max_word_count_english,
                min_word_count_cjk=self.min_word_count_cjk,
                min_word_count_english=self.min_word_count_english,
                recommend_word_count_cjk=self.recommend_word_count_cjk,
                recommend_word_count_english=self.recommend_word_count_english,
                scene_prompt=self.scene_prompt,
                mode=self.mode,
                allow_model_upgrade=self.allow_model_upgrade,
                model_upgrade_chain=self.model_upgrade_chain,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            
            # 重新分配时间戳
            chunk_asr = ASRData(chunk_segs)
            split_asr = self._realign_timestamps(chunk_asr, split_texts)
            
            with count_lock:
                completed_count += 1
                if progress_callback:
                    progress_callback(f"断句进度: {completed_count}/{num_chunks} 块")
            
            return split_asr.segments
        
        # 并行处理所有 chunk（仅 LLM 断句 + 时间戳对齐）
        chunk_results: List[List[ASRDataSeg] | None] = [None] * num_chunks
        
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            future_to_idx = {}
            for idx, (chunk_segs, start_idx, end_idx) in enumerate(chunks):
                future = executor.submit(_process_chunk, idx, chunk_segs, start_idx, end_idx)
                future_to_idx[future] = idx
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunk_results[idx] = future.result()  # 异常会在此抛出
        
        # 按序合并：在主线程中统一处理 overlap 裁剪
        all_split_segments = []
        for idx in range(num_chunks):
            result = chunk_results[idx]
            assert result is not None, "逻辑错误: chunk 结果为 None"
            
            if idx == 0:
                # 第一块，全部保留
                all_split_segments.extend(result)
            else:
                # 后续块：基于字符计数跳过 overlap 部分
                chunk_segs = chunks[idx][0]
                overlap_segs = chunk_segs[:self.overlap]
                overlap_nchars = sum(len(_normalize(seg.text)) for seg in overlap_segs)
                
                non_overlap_segments = self._trim_overlap(
                    result, overlap_nchars, _normalize
                )
                all_split_segments.extend(non_overlap_segments)
        
        logger.info(f"分块断句完成，共 {len(all_split_segments)} 句")
        return ASRData(all_split_segments)
    
    @staticmethod
    def _trim_overlap(
        segments: List[ASRDataSeg],
        overlap_nchars: int,
        normalize_fn,
    ) -> List[ASRDataSeg]:
        """
        从 split segments 中移除 overlap 部分的文本
        
        根据字符计数精确跳过 overlap_nchars 个非空白字符：
        - 完全在 overlap 内的 segment → 跳过
        - 跨越 overlap 边界的 segment → 拆分，保留 overlap 外的部分
        - 完全在 overlap 外的 segment → 保留
        
        Args:
            segments: 当前 chunk 经 _realign_timestamps 后的 split segments
            overlap_nchars: overlap ASR segments 的非空白字符总数
            normalize_fn: 去空白的函数
        """
        if overlap_nchars <= 0:
            return segments
        
        consumed = 0
        result = []
        
        for seg in segments:
            seg_norm = normalize_fn(seg.text)
            seg_nchars = len(seg_norm)
            
            if consumed >= overlap_nchars:
                # 已跳过足够字符，后续全部保留
                result.append(seg)
                continue
            
            if consumed + seg_nchars <= overlap_nchars:
                # 完全在 overlap 内，跳过
                consumed += seg_nchars
                continue
            
            # 跨边界：需要拆分
            chars_to_skip = overlap_nchars - consumed
            consumed = overlap_nchars
            
            # 从原始文本中移除前 chars_to_skip 个非空白字符
            remaining_text = _remove_leading_nchars(seg.text, chars_to_skip)
            remaining_text = remaining_text.strip()
            
            if remaining_text:
                # 用线性插值调整 start_time
                ratio = chars_to_skip / seg_nchars if seg_nchars > 0 else 0
                new_start = int(seg.start_time + (seg.end_time - seg.start_time) * ratio)
                result.append(ASRDataSeg(
                    text=remaining_text,
                    start_time=new_start,
                    end_time=seg.end_time,
                ))
        
        return result
    
    def _create_chunks(
        self, segments: List[ASRDataSeg]
    ) -> List[Tuple[List[ASRDataSeg], int, int]]:
        """
        创建带 overlap 的分块
        
        Returns:
            List[(chunk_segments, start_idx, end_idx)]
        """
        chunks = []
        total = len(segments)
        i = 0
        
        while i < total:
            end = min(i + self.chunk_size, total)
            
            # 如果剩余片段很少，直接合并到当前块，避免产生极小的尾部块
            # 注意：这可能会使当前块略大于 chunk_size
            if total - end <= self.overlap:
                end = total
            
            chunk_segs = segments[i:end]
            chunks.append((chunk_segs, i, end - 1))
            
            # 如果已经处理到最后，直接结束
            if end >= total:
                break
                
            # 下一块的起点：当前块末尾 - overlap
            i = end - self.overlap
        
        return chunks
    
    def _realign_timestamps(self, original_asr: ASRData, split_texts: List[str]) -> ASRData:
        """
        基于 diff 的容错时间戳对齐
        
        核心原则：
        - 原始 ASR 的字符和时间戳是"真理源"
        - split 结果只提供"断行位置"
        - 小范围增删改（≤3字符）自动容错修复
        - 匹配字符直接取 raw 时间戳（不做线性插值）
        - 安全检查：每个最终段落的时间不超出其字符所属原始segment的范围
        """
        MAX_SINGLE_DIFF = 3  # 单个 diff 区域最大字符数
        
        if not split_texts:
            return original_asr
        
        def normalize(s: str) -> str:
            return re.sub(r"\s+", "", s)
        
        # === 1. 构建 raw 字符序列 ===
        # 每个非空白字符: (char_start, char_end, seg_start, seg_end)
        raw_chars = ""
        char_info = []
        
        for seg in original_asr.segments:
            text = seg.text.strip()
            if not text:
                continue
            non_space = normalize(text)
            if not non_space:
                continue
            duration = seg.end_time - seg.start_time
            n = len(non_space)
            for i in range(n):
                ch_start = int(seg.start_time + duration * (i / n))
                ch_end = int(seg.start_time + duration * ((i + 1) / n))
                raw_chars += non_space[i]
                char_info.append((ch_start, ch_end, seg.start_time, seg.end_time))
        
        if not char_info:
            return original_asr
        
        # === 2. 构建 split 字符序列 ===
        split_chars = ""
        char_to_line = []  # split_chars 中每个字符所属的行号
        effective_lines = []  # (原始索引, 原始文本)
        
        for orig_idx, text in enumerate(split_texts):
            stripped = text.strip()
            if not stripped:
                continue
            norm = normalize(stripped)
            if not norm:
                continue
            line_idx = len(effective_lines)
            effective_lines.append(stripped)
            for ch in norm:
                split_chars += ch
                char_to_line.append(line_idx)
        
        if not split_chars or not effective_lines:
            return original_asr
        
        # === 3. diff 对齐 ===
        matcher = difflib.SequenceMatcher(None, raw_chars, split_chars, autojunk=False)
        opcodes = matcher.get_opcodes()
        
        # 为 split 每个字符分配: (start_ms, end_ms) 或 None
        split_times = [None] * len(split_chars)
        # 为 split 每个字符记录其对应的原始 segment 范围: (seg_start, seg_end) 或 None
        split_seg_ranges = [None] * len(split_chars)
        
        total_diff_chars = 0
        diff_details = []
        
        for tag, raw_a, raw_b, split_a, split_b in opcodes:
            if tag == 'equal':
                # 匹配区域：直接取 raw 时间戳
                for k in range(raw_b - raw_a):
                    split_times[split_a + k] = char_info[raw_a + k][:2]
                    split_seg_ranges[split_a + k] = char_info[raw_a + k][2:4]
            
            elif tag == 'delete':
                # raw 有但 split 没有（LLM删除了字符）：跳过 raw 时间
                diff_len = raw_b - raw_a
                total_diff_chars += diff_len
                if diff_len > MAX_SINGLE_DIFF:
                    ctx = raw_chars[max(0, raw_a - 3):min(len(raw_chars), raw_b + 3)]
                    diff_details.append(f"删除{diff_len}字符: ...{ctx}...")
            
            elif tag == 'insert':
                # split 有但 raw 没有（LLM插入了字符）：暂时无时间戳
                diff_len = split_b - split_a
                total_diff_chars += diff_len
                if diff_len > MAX_SINGLE_DIFF:
                    ctx = split_chars[max(0, split_a - 3):min(len(split_chars), split_b + 3)]
                    diff_details.append(f"插入{diff_len}字符: ...{ctx}...")
            
            elif tag == 'replace':
                # raw 和 split 都有但不同：时间戳级别对应
                raw_len = raw_b - raw_a
                split_len = split_b - split_a
                total_diff_chars += max(raw_len, split_len)
                
                if max(raw_len, split_len) > MAX_SINGLE_DIFF:
                    raw_ctx = raw_chars[raw_a:raw_b]
                    split_ctx = split_chars[split_a:split_b]
                    diff_details.append(f"替换: [{raw_ctx}] → [{split_ctx}]")
                
                # 按位置对应，不做线性插值
                for k in range(split_len):
                    raw_idx = raw_a + min(k, raw_len - 1)
                    split_times[split_a + k] = char_info[raw_idx][:2]
                    split_seg_ranges[split_a + k] = char_info[raw_idx][2:4]
        
        # === 4. 填充插入字符的时间（用最近的已分配邻居）===
        for i in range(len(split_times)):
            if split_times[i] is not None:
                continue
            
            # 向前找
            prev_time = None
            prev_seg_range = None
            for p in range(i - 1, -1, -1):
                if split_times[p] is not None:
                    prev_time = split_times[p]
                    prev_seg_range = split_seg_ranges[p]
                    break
            
            # 向后找
            next_time = None
            next_seg_range = None
            for n_idx in range(i + 1, len(split_times)):
                if split_times[n_idx] is not None:
                    next_time = split_times[n_idx]
                    next_seg_range = split_seg_ranges[n_idx]
                    break
            
            # 分配：锚定到邻近时间戳，不插值
            if prev_time is not None and next_time is not None:
                # 用前一个字符的 end 到后一个字符的 start
                t_start = prev_time[1]
                t_end = next_time[0]
                if t_start > t_end:
                    t_start = t_end = prev_time[1]
                split_times[i] = (t_start, t_end)
                # seg_range 取前后邻居的并集（插入字符可能跨越 segment 边界）
                if prev_seg_range and next_seg_range:
                    split_seg_ranges[i] = (
                        min(prev_seg_range[0], next_seg_range[0]),
                        max(prev_seg_range[1], next_seg_range[1]),
                    )
                else:
                    split_seg_ranges[i] = prev_seg_range or next_seg_range
            elif prev_time is not None:
                split_times[i] = prev_time
                split_seg_ranges[i] = prev_seg_range
            elif next_time is not None:
                split_times[i] = next_time
                split_seg_ranges[i] = next_seg_range
            else:
                # 整个序列都没有匹配（不应发生）
                raise ValueError("对齐失败: split 文本与原始文本完全不匹配")
        
        # === 5. 差异统计与警告 ===
        diff_ratio = total_diff_chars / max(len(raw_chars), 1)
        
        if total_diff_chars > 0:
            logger.info(
                f"容错对齐: raw={len(raw_chars)}字符, split={len(split_chars)}字符, "
                f"差异={total_diff_chars}字符 ({diff_ratio:.1%})"
            )
        
        for detail in diff_details:
            logger.warning(f"对齐差异超出阈值({MAX_SINGLE_DIFF}字符): {detail}")
        
        # === 6. 按行号分组，生成最终段落 ===
        # 预先分组：每行包含哪些 split_chars 索引
        line_char_indices = [[] for _ in range(len(effective_lines))]
        for i, line_idx in enumerate(char_to_line):
            line_char_indices[line_idx].append(i)
        
        new_segments = []
        for line_idx, line_text in enumerate(effective_lines):
            indices = line_char_indices[line_idx]
            if not indices:
                continue
            
            # 收集该行所有字符的时间戳
            times = [split_times[i] for i in indices if split_times[i] is not None]
            if not times:
                continue
            
            start_time = times[0][0]
            end_time = times[-1][1]
            
            if start_time >= end_time:
                end_time = start_time + 1  # 最少 1ms
            
            new_segments.append(ASRDataSeg(
                text=line_text,
                start_time=start_time,
                end_time=end_time,
            ))
        
        # === 6.5 解消相邻段时间重叠 ===
        # _realign_timestamps 按行分组时，相邻行可能映射到同一原始 ASR segment，
        # 导致 seg[i].end_time > seg[i+1].start_time（重叠）。
        # 在此处裁剪前段的 end_time，消除重叠，保持后段的 start_time 不变。
        for i in range(len(new_segments) - 1):
            if new_segments[i].end_time > new_segments[i + 1].start_time:
                # 裁剪前段 end_time 到后段 start_time
                clipped_end = new_segments[i + 1].start_time
                # 确保裁剪后前段仍有效（start < end）
                if clipped_end > new_segments[i].start_time:
                    new_segments[i].end_time = clipped_end
                else:
                    # 极端情况：前后段 start_time 相同，给前段最小 1ms
                    new_segments[i].end_time = new_segments[i].start_time + 1
        
        # === 7. 安全检查 ===
        # 检查：每个段落的时间范围不超出其字符所属原始 segment 的时间范围
        safety_violations = 0
        for seg_idx, seg in enumerate(new_segments):
            indices = line_char_indices[seg_idx]
            seg_ranges = [split_seg_ranges[i] for i in indices if split_seg_ranges[i] is not None]
            
            if not seg_ranges:
                continue
            
            # 该段字符来源的原始 segment 时间范围的并集
            min_seg_start = min(r[0] for r in seg_ranges)
            max_seg_end = max(r[1] for r in seg_ranges)
            
            # 容忍 100ms 误差
            if seg.start_time < min_seg_start - 100 or seg.end_time > max_seg_end + 100:
                safety_violations += 1
                logger.error(
                    f"安全检查失败: 第{seg_idx}段时间 [{seg.start_time}-{seg.end_time}] "
                    f"超出原始segment范围 [{min_seg_start}-{max_seg_end}], "
                    f"text='{seg.text[:30]}...'"
                )
        
        if safety_violations > 0:
            raise ValueError(
                f"时间戳对齐安全检查失败: {safety_violations} 个段落时间超出原始范围，"
                f"可能存在严重对齐错误"
            )
        
        return ASRData(new_segments)
