"""测试字幕缩放因子计算（横屏/竖屏）"""

import importlib.util
import os

import pytest

# 直接加载 scale_utils 模块，绕过 vat.asr.subtitle.__init__.py 的循环导入
_module_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "vat", "asr", "subtitle", "scale_utils.py"
)
_spec = importlib.util.spec_from_file_location("scale_utils", _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_subtitle_scale_factor = _mod.compute_subtitle_scale_factor


class TestComputeSubtitleScaleFactor:
    """compute_subtitle_scale_factor 单元测试"""

    def test_landscape_1080p(self):
        """横屏 1080p：scale = 1080 / 720 = 1.5"""
        factor = compute_subtitle_scale_factor(1920, 1080, 720)
        assert factor == pytest.approx(1.5)

    def test_landscape_720p_no_scale(self):
        """横屏 720p：scale = 1.0（不缩放）"""
        factor = compute_subtitle_scale_factor(1280, 720, 720)
        assert factor == pytest.approx(1.0)

    def test_landscape_4k(self):
        """横屏 4K：scale = 2160 / 720 = 3.0"""
        factor = compute_subtitle_scale_factor(3840, 2160, 720)
        assert factor == pytest.approx(3.0)

    def test_portrait_1080x1920(self):
        """竖屏 1080x1920：scale = 1080 / 720 = 1.5（而非 1920/720=2.67）"""
        factor = compute_subtitle_scale_factor(1080, 1920, 720)
        assert factor == pytest.approx(1.5)

    def test_portrait_720x1280(self):
        """竖屏 720x1280：scale = 720 / 720 = 1.0（而非 1280/720=1.78）"""
        factor = compute_subtitle_scale_factor(720, 1280, 720)
        assert factor == pytest.approx(1.0)

    def test_portrait_old_vs_new(self):
        """竖屏视频：新算法应产生更小的缩放因子（相比旧算法 height/ref）"""
        w, h, ref = 1080, 1920, 720
        old_factor = h / ref  # 2.667
        new_factor = compute_subtitle_scale_factor(w, h, ref)  # 2.25
        assert new_factor < old_factor, (
            f"竖屏缩放因子应小于旧算法: new={new_factor}, old={old_factor}"
        )

    def test_square_uses_height(self):
        """正方形视频（width == height）：走横屏分支"""
        factor = compute_subtitle_scale_factor(1080, 1080, 720)
        assert factor == pytest.approx(1080 / 720)

    def test_portrait_shorts_typical(self):
        """典型 Shorts 竖屏 1080x1920：对比旧算法缩放差异"""
        w, h, ref = 1080, 1920, 720
        new = compute_subtitle_scale_factor(w, h, ref)
        old = h / ref
        # 旧: 2.667, 新: 1.5 → 减少约 43.8%
        reduction = 1 - new / old
        assert 0.3 < reduction < 0.5, f"缩放减少比例应在 30%~50% 之间, 实际={reduction:.1%}"

    def test_custom_reference_height(self):
        """自定义参考高度"""
        factor = compute_subtitle_scale_factor(1920, 1080, 1080)
        assert factor == pytest.approx(1.0)
