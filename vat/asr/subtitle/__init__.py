"""Subtitle rendering module (ASS and rounded background styles)"""

from typing import Optional

from .ass_renderer import render_ass_preview, render_ass_video
from .ass_utils import (
    AssInfo,
    AssStyle,
    auto_wrap_ass_file,
    parse_ass_info,
    wrap_ass_text,
)
from .font_utils import (
    FontType,
    clear_font_cache,
    get_ass_to_pil_ratio,
    get_builtin_fonts,
    get_font,
)
from .rounded_renderer import render_preview, render_rounded_video
from .styles import RoundedBgStyle
from .text_utils import hex_to_rgba, is_mainly_cjk, wrap_text


def get_subtitle_style(style_name: str, style_dir: Optional[str] = None) -> Optional[str]:
    """Get subtitle style content"""
    if not style_dir:
        return None
        
    from pathlib import Path
    style_path = Path(style_dir) / f"{style_name}.txt"
    assert style_path.exists(), f"样式文件不存在: {style_path}"
    return style_path.read_text(encoding="utf-8")


__all__ = [
    "render_ass_video",
    "render_ass_preview",
    "auto_wrap_ass_file",
    "parse_ass_info",
    "wrap_ass_text",
    "AssInfo",
    "AssStyle",
    "render_preview",
    "render_rounded_video",
    "RoundedBgStyle",
    "get_subtitle_style",
    "FontType",
    "get_font",
    "get_ass_to_pil_ratio",
    "get_builtin_fonts",
    "clear_font_cache",
    "hex_to_rgba",
    "is_mainly_cjk",
    "wrap_text",
]
