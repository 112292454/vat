"""Text processing utilities"""

import re
from typing import List, Tuple

from .font_utils import FontType

# CJK and Asian languages without spaces
_NO_SPACE_LANGUAGES = r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u0900-\u0dff]"


def has_cjk(text: str) -> bool:
    """Check if text contains ANY CJK or Asian characters (for wrapping strategy)"""
    return bool(re.search(_NO_SPACE_LANGUAGES, text)) if text else False


def is_mainly_cjk(text: str, threshold: float = 0.5) -> bool:
    """Check if text is mainly CJK or Asian languages without spaces"""
    if not text:
        return False

    no_space_count = len(re.findall(_NO_SPACE_LANGUAGES, text))
    total_chars = len("".join(text.split()))

    return no_space_count / total_chars > threshold if total_chars > 0 else False


def hex_to_rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA tuple (#RRGGBB or #RRGGBBAA)"""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
        return (r, g, b, 255)
    elif len(hex_color) == 8:
        r, g, b, a = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
            int(hex_color[6:8], 16),
        )
        return (r, g, b, a)
    return (0, 0, 0, 255)


def _calculate_text_width(text: str, font: FontType, spacing: float) -> int:
    """
    Calculate text width including character spacing

    Args:
        text: Text to measure
        font: Font for measuring
        spacing: Character spacing (for N chars, adds spacing × (N-1) to width)

    Returns:
        Total width in pixels
    """
    if not text:
        return 0
    bbox = font.getbbox(text)
    base_width = bbox[2] - bbox[0]
    # For N characters, there are N-1 spacing gaps
    spacing_width = spacing * (len(text) - 1) if len(text) > 1 else 0
    return int(base_width + spacing_width)


def wrap_text(
    text: str,
    font: FontType,
    max_width: int,
    horizontal_padding: int = 0,
    extra_margin: int = 0,
    spacing: float = 0.0,
) -> List[str]:
    """
    Wrap text to fit within max width with balanced line lengths

    Strategy:
    1. Calculate minimum required lines using greedy algorithm
    2. Calculate target width per line (total_width / num_lines)
    3. Redistribute text to achieve balanced line lengths

    Args:
        text: Text to wrap
        font: Font for measuring text width
        max_width: Maximum width in pixels
        horizontal_padding: Left/right padding (reduces available width by 2x)
        extra_margin: Additional safety margin
        spacing: Character spacing (for N chars, adds spacing × (N-1) to width)
    """
    available_width = max_width - horizontal_padding * 2 - extra_margin

    # 含任何 CJK 字符的文本用字符级换行（正确处理中英混合）
    # 纯英文才用词级换行
    if has_cjk(text):
        return _wrap_cjk_balanced(text, font, available_width, spacing)
    else:
        return _wrap_english_balanced(text, font, available_width, spacing)


def _tokenize_mixed(text: str) -> List[str]:
    """将混合文本分为 token 列表：CJK 字符单独为 token，连续非 CJK 字符（英文单词等）为一个 token"""
    tokens = []
    current_non_cjk = ""
    cjk_re = re.compile(_NO_SPACE_LANGUAGES)
    for char in text:
        if cjk_re.match(char):
            if current_non_cjk:
                tokens.append(current_non_cjk)
                current_non_cjk = ""
            tokens.append(char)
        else:
            current_non_cjk += char
    if current_non_cjk:
        tokens.append(current_non_cjk)
    return tokens


def _wrap_cjk_balanced(
    text: str, font: FontType, available_width: int, spacing: float = 0.0
) -> List[str]:
    """Wrap CJK/mixed text with balanced line lengths
    
    对混合文本（中英混合），英文单词作为不可分割单元，CJK 字符可逐字断开。
    """
    tokens = _tokenize_mixed(text)
    if not tokens:
        return [text]

    # Step 1: Calculate minimum required lines using greedy algorithm
    temp_lines = []
    current_line = ""
    for token in tokens:
        test_line = current_line + token
        if _calculate_text_width(test_line, font, spacing) <= available_width:
            current_line = test_line
        else:
            if current_line:
                temp_lines.append(current_line)
            current_line = token
    if current_line:
        temp_lines.append(current_line)

    if not temp_lines:
        return [text]

    # If only one line, no need to balance
    if len(temp_lines) == 1:
        return temp_lines

    # Step 2: Calculate total width and target width per line
    total_text_width = _calculate_text_width(text, font, spacing)
    num_lines = len(temp_lines)
    target_width = total_text_width / num_lines

    # Step 3: Redistribute tokens to achieve balanced lines
    lines = []
    current_line = ""
    for i, token in enumerate(tokens):
        test_line = current_line + token
        current_width = _calculate_text_width(test_line, font, spacing)

        should_break = False

        if current_width > available_width:
            # Hard limit: must break
            should_break = True
        elif (
            len(lines) + 1 < num_lines
            and current_line
            and current_width >= target_width * 0.9
        ):
            if i + 1 < len(tokens):
                next_test = test_line + tokens[i + 1]
                next_width = _calculate_text_width(next_test, font, spacing)
                if next_width > target_width * 1.1:
                    should_break = True

        if should_break:
            if current_line:
                lines.append(current_line)
                current_line = token
            else:
                current_line = test_line
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    return lines if lines else [text]


def _wrap_english_balanced(
    text: str, font: FontType, available_width: int, spacing: float = 0.0
) -> List[str]:
    """Wrap English text with balanced line lengths"""

    words = text.split()
    if not words:
        return [text]

    # Step 1: Calculate minimum required lines
    temp_lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if _calculate_text_width(test_line, font, spacing) <= available_width:
            current_line = test_line
        else:
            if current_line:
                temp_lines.append(current_line)
            current_line = word
    if current_line:
        temp_lines.append(current_line)

    if not temp_lines:
        return [text]

    # If only one line, no need to balance
    if len(temp_lines) == 1:
        return temp_lines

    # Step 2: Calculate target width
    total_text_width = _calculate_text_width(text, font, spacing)
    num_lines = len(temp_lines)
    target_width = total_text_width / num_lines

    # Step 3: Redistribute words to achieve balanced lines
    # Important: Do not exceed the minimum line count from greedy algorithm
    lines = []
    current_line = ""
    for i, word in enumerate(words):
        test_line = f"{current_line} {word}".strip()
        current_width = _calculate_text_width(test_line, font, spacing)

        should_break = False

        if current_width > available_width:
            # Hard limit: must break
            should_break = True
        elif (
            len(lines) + 1 < num_lines
            and current_line
            and current_width >= target_width * 0.9
        ):
            # Only balance if we haven't reached the minimum line count yet
            # Close to target width (90% threshold)
            # Check if next word would significantly exceed target
            if i + 1 < len(words):
                next_test = f"{test_line} {words[i + 1]}".strip()
                next_width = _calculate_text_width(next_test, font, spacing)
                if next_width > target_width * 1.1:
                    should_break = True

        if should_break:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    return lines if lines else [text]
