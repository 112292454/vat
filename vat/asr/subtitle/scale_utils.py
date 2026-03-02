"""字幕缩放因子计算工具"""


def compute_subtitle_scale_factor(width: int, height: int, reference_height: int = 720) -> float:
    """
    计算字幕样式缩放因子。
    
    横屏视频（width >= height）：基于高度缩放（与 reference_height 对比）。
    竖屏视频（height > width）：基于宽度缩放，乘以 1.5 系数以适当放大
    （纯按宽度缩放字体偏小，竖屏纵向空间充裕可多行显示）。
    
    设计依据：字幕换行由视频宽度决定，横屏时宽高成比例所以用高度即可；
    竖屏时高度远大于宽度，用高度缩放会导致字体相对宽度过大、严重换行。
    
    Args:
        width: 视频宽度
        height: 视频高度
        reference_height: 参考高度（样式设计基准，默认720）
    
    Returns:
        缩放因子（1.0 = 不缩放）
    """
    if width >= height:
        # 横屏：与之前一致
        return height / reference_height
    else:
        # 竖屏：直接基于宽度缩放（不额外乘系数）。
        # 例: 1080x1920 竖屏 → scale = 1080/720 = 1.5，与横屏 1080p 绝对字体大小一致，
        # 每行可容纳约 11 个 CJK 字符，在手机竖屏上可读性良好。
        return width / reference_height
