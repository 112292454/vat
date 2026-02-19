"""
B站审核违规处理功能测试

测试内容：
1. _parse_violation_time: 违规时间字符串解析
2. _time_to_seconds: 时间格式转换
3. _merge_ranges: 区间合并
4. get_rejected_videos: 退回稿件数据结构解析
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from vat.uploaders.bilibili import BilibiliUploader
from vat.embedder.ffmpeg_wrapper import FFmpegWrapper


class TestParseViolationTime:
    """测试违规时间字符串解析"""
    
    def test_single_segment(self):
        """单个违规时间段"""
        result = BilibiliUploader._parse_violation_time("P1(00:20:18-00:20:24)")
        assert result == [(1218, 1224)]
    
    def test_multiple_segments_chinese_comma(self):
        """多段违规（中文顿号分隔）"""
        result = BilibiliUploader._parse_violation_time(
            "P1(00:20:18-00:20:24)、P1(00:23:33-00:23:35)"
        )
        assert result == [(1218, 1224), (1413, 1415)]
    
    def test_multiple_segments_english_comma(self):
        """多段违规（英文逗号分隔）"""
        result = BilibiliUploader._parse_violation_time(
            "P1(00:20:18-00:20:24), P1(00:23:33-00:23:35)"
        )
        assert result == [(1218, 1224), (1413, 1415)]
    
    def test_different_parts(self):
        """不同 P 编号"""
        result = BilibiliUploader._parse_violation_time(
            "P1(00:10:00-00:10:30)、P2(00:05:00-00:05:10)"
        )
        assert result == [(600, 630), (300, 310)]
    
    def test_empty_string(self):
        """空字符串"""
        result = BilibiliUploader._parse_violation_time("")
        assert result == []
    
    def test_no_match(self):
        """无法匹配的字符串"""
        result = BilibiliUploader._parse_violation_time("内容全程")
        assert result == []
    
    def test_zero_time(self):
        """零时刻"""
        result = BilibiliUploader._parse_violation_time("P1(00:00:00-00:00:05)")
        assert result == [(0, 5)]
    
    def test_long_video(self):
        """超过1小时的时间"""
        result = BilibiliUploader._parse_violation_time("P1(01:30:00-01:30:10)")
        assert result == [(5400, 5410)]


class TestTimeToSeconds:
    """测试时间格式转换"""
    
    def test_basic(self):
        assert BilibiliUploader._time_to_seconds("00:20:18") == 1218
    
    def test_zero(self):
        assert BilibiliUploader._time_to_seconds("00:00:00") == 0
    
    def test_hour(self):
        assert BilibiliUploader._time_to_seconds("01:00:00") == 3600
    
    def test_invalid(self):
        assert BilibiliUploader._time_to_seconds("invalid") is None
    
    def test_two_parts(self):
        assert BilibiliUploader._time_to_seconds("20:18") is None


class TestMergeRanges:
    """测试区间合并"""
    
    def test_no_overlap(self):
        """不重叠的区间"""
        result = FFmpegWrapper._merge_ranges(
            [(10, 20), (30, 40)], margin=0, max_duration=100
        )
        assert result == [(10, 20), (30, 40)]
    
    def test_with_margin_no_overlap(self):
        """有边距但不重叠"""
        result = FFmpegWrapper._merge_ranges(
            [(10, 20), (30, 40)], margin=1.0, max_duration=100
        )
        assert result == [(9, 21), (29, 41)]
    
    def test_margin_causes_overlap(self):
        """边距导致重叠，应合并"""
        result = FFmpegWrapper._merge_ranges(
            [(10, 20), (21, 30)], margin=1.0, max_duration=100
        )
        assert result == [(9, 31)]
    
    def test_already_overlapping(self):
        """本身就重叠"""
        result = FFmpegWrapper._merge_ranges(
            [(10, 25), (20, 30)], margin=0, max_duration=100
        )
        assert result == [(10, 30)]
    
    def test_clamp_to_duration(self):
        """超出视频时长时被裁剪"""
        result = FFmpegWrapper._merge_ranges(
            [(95, 100)], margin=2.0, max_duration=100
        )
        assert result == [(93, 100)]
    
    def test_clamp_to_zero(self):
        """起始点不能为负"""
        result = FFmpegWrapper._merge_ranges(
            [(0, 5)], margin=2.0, max_duration=100
        )
        assert result == [(0, 7)]
    
    def test_empty(self):
        """空列表"""
        result = FFmpegWrapper._merge_ranges([], margin=1.0, max_duration=100)
        assert result == []
    
    def test_unsorted_input(self):
        """乱序输入应正确排序合并"""
        result = FFmpegWrapper._merge_ranges(
            [(30, 40), (10, 20)], margin=0, max_duration=100
        )
        assert result == [(10, 20), (30, 40)]
    
    def test_multiple_merge(self):
        """多段全部合并"""
        result = FFmpegWrapper._merge_ranges(
            [(10, 15), (14, 20), (19, 25)], margin=0, max_duration=100
        )
        assert result == [(10, 25)]


class TestGetRejectedVideosParsing:
    """测试 get_rejected_videos 对 API 响应的解析（mock HTTP）"""
    
    @patch.object(BilibiliUploader, '_get_authenticated_session')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_parse_response(self, mock_load, mock_session):
        """验证 API 响应解析为正确的数据结构"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'code': 0,
            'data': {
                'arc_audits': [{
                    'Archive': {
                        'aid': 12345,
                        'bvid': 'BV1xx',
                        'title': '测试视频',
                        'state': -30,
                        'reject_reason': '退回原因',
                    },
                    'problem_detail': [{
                        'reject_reason': '视频含违规内容',
                        'violation_time': 'P1(00:10:00-00:10:05)',
                        'violation_position': '内容',
                        'modify_advise': '请修改后重新提交',
                    }],
                }],
            },
        }
        
        session = MagicMock()
        session.get.return_value = mock_resp
        mock_session.return_value = session
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {}
        
        result = uploader.get_rejected_videos()
        
        assert len(result) == 1
        v = result[0]
        assert v['aid'] == 12345
        assert v['bvid'] == 'BV1xx'
        assert len(v['problems']) == 1
        
        p = v['problems'][0]
        assert p['time_ranges'] == [(600, 605)]
        assert p['is_full_video'] is False
        assert p['reason'] == '视频含违规内容'
    
    @patch.object(BilibiliUploader, '_get_authenticated_session')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_full_video_violation(self, mock_load, mock_session):
        """全片违规的解析"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'code': 0,
            'data': {
                'arc_audits': [{
                    'Archive': {
                        'aid': 99999,
                        'bvid': 'BV2yy',
                        'title': '全片违规视频',
                        'state': -30,
                        'reject_reason': '',
                    },
                    'problem_detail': [{
                        'reject_reason': '素材不合规',
                        'violation_time': '',
                        'violation_position': '内容全程',
                        'modify_advise': '',
                    }],
                }],
            },
        }
        
        session = MagicMock()
        session.get.return_value = mock_resp
        mock_session.return_value = session
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {}
        
        result = uploader.get_rejected_videos()
        
        assert len(result) == 1
        p = result[0]['problems'][0]
        assert p['is_full_video'] is True
        assert p['time_ranges'] == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
