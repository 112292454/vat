"""ASRData 单元测试：dedup_adjacent_segments 增强 + 退化段过滤"""

import pytest
from vat.asr.asr_data import ASRData, ASRDataSeg


def _seg(text, start, end, translated=""):
    return ASRDataSeg(text=text, start_time=start, end_time=end, translated_text=translated)


class TestASRDataInit:
    """ASRData.__init__ 退化段过滤"""

    def test_filter_zero_duration(self):
        """duration == 0 的段被过滤"""
        segs = [_seg("hello", 1000, 1000), _seg("world", 2000, 3000)]
        asr = ASRData(segs)
        assert len(asr.segments) == 1
        assert asr.segments[0].text == "world"

    def test_filter_negative_duration(self):
        """start > end 的段被过滤"""
        segs = [_seg("bad", 5000, 4000), _seg("good", 6000, 7000)]
        asr = ASRData(segs)
        assert len(asr.segments) == 1
        assert asr.segments[0].text == "good"

    def test_filter_short_duration(self):
        """duration < 50ms 的段被过滤"""
        segs = [
            _seg("ghost", 1000, 1040),    # 40ms → 过滤
            _seg("ok", 2000, 2100),        # 100ms → 保留
            _seg("tiny", 3000, 3001),      # 1ms → 过滤
        ]
        asr = ASRData(segs)
        assert len(asr.segments) == 1
        assert asr.segments[0].text == "ok"

    def test_keep_50ms_segment(self):
        """duration == 50ms 的段保留（边界）"""
        segs = [_seg("边界", 1000, 1050)]
        asr = ASRData(segs)
        assert len(asr.segments) == 1

    def test_filter_empty_text(self):
        """空文本段被过滤"""
        segs = [_seg("", 1000, 2000), _seg("  ", 3000, 4000), _seg("ok", 5000, 6000)]
        asr = ASRData(segs)
        assert len(asr.segments) == 1

    def test_sort_by_start_time(self):
        """段按 start_time 排序"""
        segs = [_seg("b", 5000, 6000), _seg("a", 1000, 2000)]
        asr = ASRData(segs)
        assert asr.segments[0].text == "a"
        assert asr.segments[1].text == "b"


class TestDedupCase0ExactMatch:
    """Case 0: 文本完全相同的相邻段去重（原有逻辑）"""

    def test_exact_duplicate(self):
        """文本完全相同 + 时间相近 → 移除后者"""
        segs = [
            _seg("同じ", 1000, 2000),
            _seg("同じ", 2500, 3500),
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == 1
        assert asr.segments[0].start_time == 1000
        assert asr.segments[0].end_time == 3500  # 扩展覆盖

    def test_exact_duplicate_far_apart(self):
        """文本相同但间隔 > max_gap_ms → 不去重（可能是有意重复）"""
        segs = [
            _seg("同じ", 1000, 2000),
            _seg("同じ", 20000, 21000),
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments(max_gap_ms=15000)
        assert len(asr.segments) == 2


class TestDedupCase1TextContainment:
    """Case 1: 文本包含关系 + 时间重叠 → 移除被包含的"""

    def test_prev_contains_curr(self):
        """前段包含后段文本 → 移除后段"""
        # whisper 典型模式：前段 "ABCDE"，后段 "CDE"（后缀重复）
        segs = [
            _seg("あんまなくてちょっとスタンプ", 1000, 5000),
            _seg("ちょっとスタンプ", 3000, 5000),
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == 1
        assert asr.segments[0].text == "あんまなくてちょっとスタンプ"
        assert asr.segments[0].end_time == 5000

    def test_curr_contains_prev(self):
        """后段包含前段文本 → 用后段替换前段"""
        # whisper 模式：前段 "AB"，后段 "ABCDE"（前段是后段前缀）
        segs = [
            _seg("3年", 1000, 2000),
            _seg("3年いたかなくらい", 1500, 5000),
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == 1
        assert asr.segments[0].text == "3年いたかなくらい"
        assert asr.segments[0].start_time == 1000  # 扩展到前段起始
        assert asr.segments[0].end_time == 5000

    def test_containment_no_overlap_kept(self):
        """文本包含但无时间重叠 → 不触发（Case 1 仅在时间重叠时生效）"""
        segs = [
            _seg("写真集っていうテイのイラスト集ってこと", 1000, 3000),
            _seg("イラスト集", 4000, 5000),  # 文本是子串，但时间不重叠
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == 2

    def test_single_char_not_containment(self):
        """单字符不触发包含检测（len < 2 保护）"""
        segs = [
            _seg("あいうえお", 1000, 3000),
            _seg("あ", 2000, 2500),  # 单字符是子串，但 len < 2 不触发
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # overlap_ratio = 1000/500 = 2.0 >= 0.5 → Case 2 应该捕获
        assert len(asr.segments) == 1

    def test_real_case_pKgDJxAJV_A(self):
        """真实案例：pKgDJxAJV_A #502/#503"""
        segs = [
            _seg("ちょっと今はあの絵のモチベがあんまなくてちょっとスタンプ", 1768119, 1772680),
            _seg("あんまなくてちょっとスタンプ", 1770000, 1772680),
            _seg("スタンプがねないんで", 1772680, 1775120),
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # #503 被移除（子串），#502 和 #504 保留
        assert len(asr.segments) == 2
        assert "あんまなくてちょっとスタンプ" in asr.segments[0].text
        assert asr.segments[1].text == "スタンプがねないんで"


class TestDedupCase2HighOverlap:
    """Case 2: 高重叠比 (≥50%) + 非文本包含 → 移除较短段"""

    def test_high_overlap_keep_longer(self):
        """overlap_ratio > 0.5，保留时长更长的"""
        segs = [
            _seg("趣味なんか欲しいなみたいな", 1000, 3500),   # 2500ms
            _seg("スミなんか欲しいなみたいな", 1500, 3400),    # 1900ms, overlap=2000
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # overlap_ratio = 2000/1900 ≈ 1.05 → Case 2
        assert len(asr.segments) == 1
        assert asr.segments[0].text == "趣味なんか欲しいなみたいな"
        assert asr.segments[0].end_time == 3500  # 保留较长段，end 取 max

    def test_high_overlap_keep_longer_reversed(self):
        """后段更长时保留后段"""
        segs = [
            _seg("短い", 1000, 2000),    # 1000ms
            _seg("もっと長い文章です", 1200, 4000),   # 2800ms, overlap=800
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # overlap_ratio = 800/1000 = 0.8 → Case 2, keep curr (longer)
        assert len(asr.segments) == 1
        assert asr.segments[0].text == "もっと長い文章です"
        assert asr.segments[0].start_time == 1000  # 扩展到 prev 的起始

    def test_borderline_50_percent(self):
        """恰好 50% 重叠 → 仍触发 Case 2"""
        segs = [
            _seg("AAA", 0, 2000),    # 2000ms
            _seg("BBB", 1000, 3000),  # 2000ms, overlap=1000, ratio=0.5
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == 1

    def test_real_case_near_identical_different_kana(self):
        """真实案例：同一音频的不同识别结果（平/片假名差异）"""
        segs = [
            _seg("ゆるキャラみたいなキャラクターデザインうまかった", 8259900, 8262920),
            _seg("ゆるキャラみたいなキャラクターデザインうまかった", 8260339, 8262900),
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # 文本完全相同 → Case 0 精确去重
        assert len(asr.segments) == 1


class TestDedupCase3LowOverlap:
    """Case 3: 低重叠比 (<50%) → 保留两段，调整边界"""

    def test_low_overlap_keep_both(self):
        """不同内容+低重叠 → 两段都保留"""
        segs = [
            _seg("皆さん知っていらっしゃいますか", 589100, 590360),
            _seg("フォローライブの漫画を", 590000, 598160),
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # overlap = 360, min_dur = 1260, ratio = 0.29 < 0.5 → Case 3
        assert len(asr.segments) == 2
        # 边界已调整：prev.end = curr.start
        assert asr.segments[0].end_time == asr.segments[1].start_time

    def test_low_overlap_boundary_adjusted(self):
        """边界调整后无重叠"""
        segs = [
            _seg("A句子", 0, 5000),
            _seg("B句子", 4000, 10000),  # overlap=1000, ratio=1000/5000=0.2
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == 2
        assert asr.segments[0].end_time == 4000
        assert asr.segments[1].start_time == 4000

    def test_real_case_different_content_small_overlap(self):
        """真实案例：不同内容、小重叠"""
        segs = [
            _seg("いい匂いもする", 4129180, 4130400),
            _seg("良い匂いもしそうですし", 4130000, 4133940),
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # overlap = 400, min_dur = 1220, ratio = 0.33 < 0.5 → Case 3
        assert len(asr.segments) == 2


class TestDedupChainEffects:
    """链式效应：多个连续段的处理"""

    def test_three_overlapping_removed_to_one(self):
        """三个重叠段，前两个被移除，最终只剩一个"""
        segs = [
            _seg("AB", 0, 3000),
            _seg("B", 2000, 3000),       # 子串，被 Case 1 移除
            _seg("BC", 2500, 5000),       # 与扩展后的 prev 重叠
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # 第 2 段被移除（子串），prev 扩展到 [0-3000]
        # 第 3 段与 prev 重叠 500ms，ratio=500/2500=0.2 < 0.5 → Case 3 保留
        assert len(asr.segments) == 2

    def test_no_overlap_passthrough(self):
        """无重叠的段不受影响"""
        segs = [
            _seg("第一句", 0, 1000),
            _seg("第二句", 2000, 3000),
            _seg("第三句", 4000, 5000),
        ]
        asr = ASRData(segs)
        original_count = len(asr.segments)
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == original_count

    def test_mixed_cases(self):
        """混合场景：精确重复 + 子串 + 高重叠 + 低重叠"""
        segs = [
            _seg("精确重复", 0, 1000),
            _seg("精确重复", 1500, 2500),      # Case 0: 精确重复
            _seg("文本AB内容CD", 5000, 8000),
            _seg("内容CD", 7000, 8000),         # Case 1: 子串
            _seg("高重叠テスト", 10000, 12000),
            _seg("違うテキスト", 10500, 11800),  # Case 2: ratio=1500/1300≈1.15
            _seg("低重叠A", 15000, 16000),
            _seg("低重叠B", 15800, 17000),       # Case 3: ratio=200/1000=0.2
        ]
        asr = ASRData(segs)
        asr.dedup_adjacent_segments()
        # Case 0: 移除 → 1 段
        # Case 1: 移除 → 1 段
        # Case 2: 移除 → 1 段
        # Case 3: 保留两段
        assert len(asr.segments) == 5  # 8 - 3 = 5

    def test_empty_segments(self):
        """空列表不崩溃"""
        asr = ASRData([])
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == 0

    def test_single_segment(self):
        """单段不崩溃"""
        asr = ASRData([_seg("唯一", 0, 1000)])
        asr.dedup_adjacent_segments()
        assert len(asr.segments) == 1
