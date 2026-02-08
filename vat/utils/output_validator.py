"""
输出验证模块：检测 ASR/LLM 输出中的崩溃模式

用于检测：
1. 极端重复字符/模式（模型崩溃的典型表现）
2. ASR 输出中的异常长静默（排除开头，根据上下文判断）
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from vat.utils.logger import setup_logger

logger = setup_logger("output_validator")


# ============================================================================
# 崩溃检测配置
# ============================================================================

# 单字符重复阈值：超过此数量视为警告
SINGLE_CHAR_REPEAT_THRESHOLD = 10

# 短模式重复阈值：2-4字符的模式重复超过此次数视为警告
SHORT_PATTERN_REPEAT_THRESHOLD = 6

# 灾难性阈值倍数：超过警告阈值的N倍视为灾难性崩溃
CATASTROPHIC_MULTIPLIER = 2

# 文本长度与独特字符比例阈值：如果 unique_chars / len < 此值，可能是崩溃
UNIQUENESS_RATIO_THRESHOLD = 0.05

# 单个片段的最大合理时长（秒）：超过此时长且文本异常则警告
MAX_REASONABLE_SEGMENT_DURATION = 30


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool = True
    is_catastrophic: bool = False  # 灾难性崩溃，应该丢弃
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        
    def mark_catastrophic(self, reason: str):
        self.is_valid = False
        self.is_catastrophic = True
        self.add_warning(f"[CATASTROPHIC] {reason}")


# ============================================================================
# 崩溃模式检测
# ============================================================================

def detect_char_flood(text: str) -> Optional[Tuple[str, int]]:
    """
    检测单字符洪水（如 うううううう...）
    
    Returns:
        (重复字符, 重复次数) 或 None
    """
    if not text:
        return None
    
    # 匹配任何字符连续重复 N 次以上
    pattern = r'(.)\1{' + str(SINGLE_CHAR_REPEAT_THRESHOLD - 1) + r',}'
    match = re.search(pattern, text)
    if match:
        char = match.group(1)
        count = len(match.group(0))
        return (char, count)
    return None


def detect_pattern_flood(text: str) -> Optional[Tuple[str, int]]:
    """
    检测短模式洪水（如 ほいほいほいほい...）
    
    Returns:
        (重复模式, 重复次数) 或 None
    """
    if not text or len(text) < 10:
        return None
    
    # 检测 2-6 字符的重复模式
    for pattern_len in range(2, 7):
        # 构建正则：捕获 N 个字符，然后匹配重复
        pattern = r'(.{' + str(pattern_len) + r'})\1{' + str(SHORT_PATTERN_REPEAT_THRESHOLD - 1) + r',}'
        match = re.search(pattern, text)
        if match:
            repeated = match.group(1)
            count = len(match.group(0)) // pattern_len
            return (repeated, count)
    return None


def check_uniqueness_ratio(text: str) -> Tuple[float, bool]:
    """
    检查文本的独特字符比例
    
    Returns:
        (比例, 是否异常低)
    """
    if not text or len(text) < 20:
        return (1.0, False)
    
    unique_chars = len(set(text))
    ratio = unique_chars / len(text)
    is_abnormal = ratio < UNIQUENESS_RATIO_THRESHOLD
    return (ratio, is_abnormal)


def validate_text_output(
    text: str,
    context: str = "unknown",
    duration_sec: Optional[float] = None,
) -> ValidationResult:
    """
    验证文本输出是否存在崩溃模式
    
    Args:
        text: 要验证的文本
        context: 上下文描述（用于日志）
        duration_sec: 对应的时长（秒），用于判断是否异常
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    result.details['text_length'] = len(text) if text else 0
    result.details['context'] = context
    
    if not text:
        return result
    
    # 1. 检测单字符洪水
    char_flood = detect_char_flood(text)
    if char_flood:
        char, count = char_flood
        result.details['char_flood'] = {'char': char, 'count': count}
        catastrophic_threshold = SINGLE_CHAR_REPEAT_THRESHOLD * CATASTROPHIC_MULTIPLIER
        if count >= catastrophic_threshold:
            result.mark_catastrophic(
                f"单字符 '{char}' 重复 {count} 次，疑似模型崩溃"
            )
        else:
            result.add_warning(f"单字符 '{char}' 重复 {count} 次")
    
    # 2. 检测模式洪水
    pattern_flood = detect_pattern_flood(text)
    if pattern_flood:
        pattern, count = pattern_flood
        result.details['pattern_flood'] = {'pattern': pattern, 'count': count}
        catastrophic_threshold = SHORT_PATTERN_REPEAT_THRESHOLD * CATASTROPHIC_MULTIPLIER
        if count >= catastrophic_threshold:
            result.mark_catastrophic(
                f"模式 '{pattern}' 重复 {count} 次，疑似模型崩溃"
            )
        else:
            result.add_warning(f"模式 '{pattern}' 重复 {count} 次")
    
    # 3. 检测独特字符比例
    ratio, is_abnormal = check_uniqueness_ratio(text)
    result.details['uniqueness_ratio'] = ratio
    if is_abnormal:
        result.add_warning(f"独特字符比例过低: {ratio:.2%}")
        if ratio < UNIQUENESS_RATIO_THRESHOLD / 2:
            result.mark_catastrophic(
                f"独特字符比例极低 ({ratio:.2%})，疑似模型崩溃"
            )
    
    # 4. 检测时长与文本长度的异常比例
    if duration_sec and duration_sec > MAX_REASONABLE_SEGMENT_DURATION:
        # 长时间片段应该有足够的内容
        chars_per_sec = len(text) / duration_sec
        result.details['chars_per_sec'] = chars_per_sec
        # 日语正常语速约 5-10 字符/秒
        if chars_per_sec > 20:  # 异常密集
            result.add_warning(
                f"文本密度异常高: {chars_per_sec:.1f} 字符/秒 (时长 {duration_sec:.1f}s)"
            )
    
    # 输出警告日志
    if result.warnings:
        level = "ERROR" if result.is_catastrophic else "WARNING"
        logger.warning(f"[{context}] 输出验证{level}: {'; '.join(result.warnings)}")
    
    return result


# ============================================================================
# ASR 静默检测
# ============================================================================

@dataclass
class SilenceGap:
    """静默间隙信息"""
    start: float
    end: float
    duration: float
    prev_segment_end: float
    next_segment_start: float
    is_at_beginning: bool = False


def detect_silence_gaps(
    segments: List[Dict[str, Any]],
    min_gap_ratio: float = 3.0,
    beginning_threshold_sec: float = 10.0,
) -> List[SilenceGap]:
    """
    检测 ASR 输出中的异常静默间隙
    
    使用上下文相对判断：如果某个间隙显著大于周围平均间隙，则标记为异常
    
    Args:
        segments: ASR 片段列表 [{'start': float, 'end': float, 'text': str}, ...]
        min_gap_ratio: 间隙需要超过平均间隙的倍数才视为异常
        beginning_threshold_sec: 开头多少秒内的静默不计入
        
    Returns:
        异常静默间隙列表
    """
    if not segments or len(segments) < 2:
        return []
    
    # 计算所有间隙
    gaps = []
    for i in range(1, len(segments)):
        prev_end = segments[i-1].get('end', 0)
        curr_start = segments[i].get('start', 0)
        gap = curr_start - prev_end
        if gap > 0:
            gaps.append({
                'index': i,
                'gap': gap,
                'prev_end': prev_end,
                'curr_start': curr_start,
            })
    
    if not gaps:
        return []
    
    # 计算平均间隙（排除极端值）
    gap_values = [g['gap'] for g in gaps]
    gap_values_sorted = sorted(gap_values)
    # 使用中位数作为基准，更稳健
    median_gap = gap_values_sorted[len(gap_values_sorted) // 2]
    
    # 检测异常间隙
    silence_gaps = []
    for g in gaps:
        # 判断是否在开头
        is_beginning = g['prev_end'] < beginning_threshold_sec
        
        # 判断是否异常长
        if median_gap > 0 and g['gap'] > median_gap * min_gap_ratio:
            silence_gaps.append(SilenceGap(
                start=g['prev_end'],
                end=g['curr_start'],
                duration=g['gap'],
                prev_segment_end=g['prev_end'],
                next_segment_start=g['curr_start'],
                is_at_beginning=is_beginning,
            ))
    
    return silence_gaps


def warn_silence_gaps(
    segments: List[Dict[str, Any]],
    min_gap_ratio: float = 3.0,
    beginning_threshold_sec: float = 10.0,
    warn_threshold_sec: float = 5.0,
) -> List[str]:
    """
    检测并警告 ASR 输出中的异常静默
    
    Args:
        segments: ASR 片段列表
        min_gap_ratio: 间隙需要超过平均间隙的倍数
        beginning_threshold_sec: 开头多少秒内的静默不警告
        warn_threshold_sec: 只警告超过此秒数的静默
        
    Returns:
        警告消息列表
    """
    gaps = detect_silence_gaps(segments, min_gap_ratio, beginning_threshold_sec)
    
    warnings = []
    for gap in gaps:
        if gap.duration < warn_threshold_sec:
            continue
        if gap.is_at_beginning:
            # 开头的静默只记录，不警告
            logger.debug(f"开头静默: {gap.start:.1f}s - {gap.end:.1f}s ({gap.duration:.1f}s)")
            continue
        
        msg = f"异常静默: {gap.start:.1f}s - {gap.end:.1f}s ({gap.duration:.1f}s)"
        warnings.append(msg)
        logger.warning(f"[ASR] {msg}")
    
    return warnings


# ============================================================================
# 批量验证
# ============================================================================

def validate_asr_segments(
    segments: List[Dict[str, Any]],
    remove_catastrophic: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    验证 ASR 片段列表，检测崩溃模式和静默异常
    
    Args:
        segments: ASR 片段列表 [{'start': float, 'end': float, 'text': str}, ...]
        remove_catastrophic: 是否移除灾难性崩溃的片段
        
    Returns:
        (过滤后的片段列表, 警告消息列表)
    """
    if not segments:
        return [], []
    
    warnings = []
    filtered = []
    catastrophic_count = 0
    
    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        duration = end - start
        
        result = validate_text_output(
            text,
            context=f"segment[{i}] {start:.1f}s-{end:.1f}s",
            duration_sec=duration,
        )
        
        if result.is_catastrophic:
            catastrophic_count += 1
            warnings.extend(result.warnings)
            if not remove_catastrophic:
                filtered.append(seg)
        else:
            if result.warnings:
                warnings.extend(result.warnings)
            filtered.append(seg)
    
    # 检测静默异常
    silence_warnings = warn_silence_gaps(segments)
    warnings.extend(silence_warnings)
    
    if catastrophic_count > 0:
        logger.warning(f"[ASR] 移除 {catastrophic_count} 个灾难性崩溃片段")
    
    return filtered, warnings


def validate_llm_output(
    text: str,
    context: str = "LLM",
) -> ValidationResult:
    """
    验证 LLM 输出是否存在崩溃模式
    
    Args:
        text: LLM 输出文本
        context: 上下文描述
        
    Returns:
        ValidationResult
    """
    return validate_text_output(text, context=context)
