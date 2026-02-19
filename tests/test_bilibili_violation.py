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


class TestFixViolation:
    """测试 fix_violation 封装逻辑（mock 所有外部依赖）"""
    
    @patch.object(BilibiliUploader, 'replace_video', return_value=True)
    @patch('vat.embedder.ffmpeg_wrapper.FFmpegWrapper')
    @patch.object(BilibiliUploader, 'get_rejected_videos')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_fix_with_local_file(self, mock_load, mock_rejected, mock_ffmpeg_cls, mock_replace, tmp_path):
        """有本地文件时：直接 mask + 上传，不下载"""
        # 准备
        mock_rejected.return_value = [{
            'aid': 12345, 'bvid': 'BV1xx', 'title': '测试', 'state': -2,
            'problems': [{
                'reason': '违规', 'violation_time': 'P1(00:10:00-00:10:05)',
                'violation_position': '', 'modify_advise': '',
                'is_full_video': False, 'time_ranges': [(600, 605)],
            }],
        }]
        
        mock_ffmpeg = MagicMock()
        mock_ffmpeg_cls.return_value = mock_ffmpeg
        mock_ffmpeg.mask_violation_segments.return_value = True
        mock_ffmpeg._merge_ranges.return_value = [(599, 606)]
        mock_ffmpeg.get_video_info.return_value = {'duration': 3600}
        
        # 创建假的本地视频文件
        local_video = tmp_path / "video.mp4"
        local_video.write_bytes(b'\x00' * 1024)
        # 创建假的 masked 输出
        masked = tmp_path / "video_masked.mp4"
        masked.write_bytes(b'\x00' * 1024)
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {}
        
        result = uploader.fix_violation(
            aid=12345,
            video_path=local_video,
            previous_ranges=[(100, 110)],
            dry_run=False,
        )
        
        assert result['success'] is True
        assert result['source'] == 'local'
        assert result['new_ranges'] == [(600, 605)]
        # mask_violation_segments 应收到合并的 ranges（旧+新）
        call_args = mock_ffmpeg.mask_violation_segments.call_args
        assert (100, 110) in call_args.kwargs['violation_ranges']
        assert (600, 605) in call_args.kwargs['violation_ranges']
    
    @patch.object(BilibiliUploader, 'get_rejected_videos')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_fix_full_video_violation(self, mock_load, mock_rejected):
        """全片违规应返回失败"""
        mock_rejected.return_value = [{
            'aid': 99999, 'bvid': 'BV2yy', 'title': '全片违规', 'state': -2,
            'problems': [{
                'reason': '素材不合规', 'violation_time': '',
                'violation_position': '内容全程', 'modify_advise': '',
                'is_full_video': True, 'time_ranges': [],
            }],
        }]
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {}
        
        result = uploader.fix_violation(aid=99999)
        
        assert result['success'] is False
        assert '全片违规' in result['message']
    
    @patch.object(BilibiliUploader, 'get_rejected_videos')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_fix_no_time_ranges(self, mock_load, mock_rejected):
        """无具体违规时间段应返回失败"""
        mock_rejected.return_value = [{
            'aid': 88888, 'bvid': 'BV3zz', 'title': '无时间段', 'state': -2,
            'problems': [{
                'reason': '不合规', 'violation_time': '',
                'violation_position': '', 'modify_advise': '',
                'is_full_video': False, 'time_ranges': [],
            }],
        }]
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {}
        
        result = uploader.fix_violation(aid=88888)
        
        assert result['success'] is False
        assert '无具体违规时间段' in result['message']
    
    @patch.object(BilibiliUploader, 'get_rejected_videos')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_fix_aid_not_found(self, mock_load, mock_rejected):
        """找不到指定 aid 的退回稿件"""
        mock_rejected.return_value = []
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {}
        
        result = uploader.fix_violation(aid=77777)
        
        assert result['success'] is False
        assert '未找到' in result['message']
    
    @patch('vat.embedder.ffmpeg_wrapper.FFmpegWrapper')
    @patch.object(BilibiliUploader, 'get_rejected_videos')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_dry_run_no_upload(self, mock_load, mock_rejected, mock_ffmpeg_cls, tmp_path):
        """dry-run 模式：做 mask 但不上传"""
        mock_rejected.return_value = [{
            'aid': 12345, 'bvid': 'BV1xx', 'title': '测试', 'state': -2,
            'problems': [{
                'reason': '违规', 'violation_time': 'P1(00:10:00-00:10:05)',
                'violation_position': '', 'modify_advise': '',
                'is_full_video': False, 'time_ranges': [(600, 605)],
            }],
        }]
        
        mock_ffmpeg = MagicMock()
        mock_ffmpeg_cls.return_value = mock_ffmpeg
        mock_ffmpeg.mask_violation_segments.return_value = True
        mock_ffmpeg._merge_ranges.return_value = [(599, 606)]
        mock_ffmpeg.get_video_info.return_value = {'duration': 3600}
        
        local_video = tmp_path / "video.mp4"
        local_video.write_bytes(b'\x00' * 1024)
        masked = tmp_path / "video_masked.mp4"
        masked.write_bytes(b'\x00' * 1024)
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {}
        
        # 不应调用 replace_video
        with patch.object(uploader, 'replace_video') as mock_replace:
            result = uploader.fix_violation(aid=12345, video_path=local_video, dry_run=True)
            mock_replace.assert_not_called()
        
        assert result['success'] is True
        assert result['masked_path'] is not None
        assert 'dry-run' in result['message']
    
    @patch.object(BilibiliUploader, 'download_video', return_value=False)
    @patch.object(BilibiliUploader, 'get_rejected_videos')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_fix_download_fallback_fails(self, mock_load, mock_rejected, mock_download):
        """无本地文件且 B站下载失败"""
        mock_rejected.return_value = [{
            'aid': 12345, 'bvid': 'BV1xx', 'title': '测试', 'state': -2,
            'problems': [{
                'reason': '违规', 'violation_time': 'P1(00:10:00-00:10:05)',
                'violation_position': '', 'modify_advise': '',
                'is_full_video': False, 'time_ranges': [(600, 605)],
            }],
        }]
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {}
        
        result = uploader.fix_violation(aid=12345, video_path=None)
        
        assert result['success'] is False
        assert '下载' in result['message']


class TestDownloadVideo:
    """测试 download_video（mock HTTP + subprocess）"""
    
    @patch('subprocess.run')
    @patch.object(BilibiliUploader, 'get_archive_detail')
    @patch.object(BilibiliUploader, '_get_authenticated_session')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_download_success(self, mock_load, mock_session, mock_detail, mock_run, tmp_path):
        """成功下载 DASH 视频"""
        mock_detail.return_value = {
            'videos': [{'cid': 12345678}]
        }
        
        # mock playurl API 响应
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'code': 0,
            'data': {
                'dash': {
                    'video': [
                        {'id': 112, 'codecs': 'avc1', 'bandwidth': 2000000,
                         'width': 1920, 'height': 1080, 'baseUrl': 'https://example.com/video.m4s'},
                    ],
                    'audio': [
                        {'id': 30280, 'codecs': 'mp4a', 'bandwidth': 90000,
                         'baseUrl': 'https://example.com/audio.m4s'},
                    ],
                },
            },
        }
        session = MagicMock()
        session.get.return_value = mock_resp
        mock_session.return_value = session
        
        # mock ffmpeg 成功执行
        output_file = tmp_path / "output.mp4"
        def fake_run(cmd, **kwargs):
            output_file.write_bytes(b'\x00' * 1024)
            return MagicMock(returncode=0)
        mock_run.side_effect = fake_run
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {'SESSDATA': 'test'}
        
        result = uploader.download_video(aid=99999, output_path=output_file)
        
        assert result is True
        assert output_file.exists()
        mock_run.assert_called_once()
        # 验证 ffmpeg 命令包含 -c copy（不重新编码）
        cmd = mock_run.call_args[0][0]
        assert '-c' in cmd and 'copy' in cmd
    
    @patch.object(BilibiliUploader, 'get_archive_detail', return_value=None)
    @patch.object(BilibiliUploader, '_get_authenticated_session')
    @patch.object(BilibiliUploader, '_load_cookie')
    def test_download_no_cid(self, mock_load, mock_session, mock_detail):
        """无法获取 cid 时失败"""
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {'code': -404}
        session.get.return_value = resp
        mock_session.return_value = session
        
        uploader = BilibiliUploader.__new__(BilibiliUploader)
        uploader._cookie_loaded = True
        uploader.cookie_data = {'SESSDATA': 'test'}
        
        result = uploader.download_video(aid=99999, output_path=Path('/tmp/test.mp4'))
        assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
