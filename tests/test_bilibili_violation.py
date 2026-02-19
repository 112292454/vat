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
    
    def test_bracket_format_mmss(self):
        """B站新格式：【MM:SS-MM:SS】"""
        result = BilibiliUploader._parse_violation_time("您的视频【23:28-23:29】【内容】根据相关法律法规")
        assert result == [(1408, 1409)]
    
    def test_bracket_format_hhmmss(self):
        """B站新格式：【HH:MM:SS-HH:MM:SS】"""
        result = BilibiliUploader._parse_violation_time("您的视频【00:23:28-00:23:29】内容违规")
        assert result == [(1408, 1409)]
    
    def test_bracket_format_multiple(self):
        """B站新格式多段"""
        result = BilibiliUploader._parse_violation_time("视频【10:00-10:05】【20:00-20:10】有问题")
        assert result == [(600, 605), (1200, 1210)]
    
    def test_p_format_takes_priority(self):
        """P格式优先于【】格式"""
        result = BilibiliUploader._parse_violation_time("P1(00:10:00-00:10:05) 视频【20:00-20:10】")
        assert result == [(600, 605)]  # 只返回P格式结果


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
    
    def test_two_parts_mmss(self):
        """MM:SS 格式"""
        assert BilibiliUploader._time_to_seconds("20:18") == 1218
    
    def test_two_parts_zero(self):
        assert BilibiliUploader._time_to_seconds("0:05") == 5
    
    def test_single_part(self):
        assert BilibiliUploader._time_to_seconds("123") is None


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


class TestBracketFormatWithSpaces:
    """回归测试：B站返回的时间格式中冒号后可能有空格（如 "17: 21"）"""
    
    def test_space_after_colon_in_start(self):
        """冒号后有空格：【17: 21-17:23】"""
        result = BilibiliUploader._parse_violation_time(
            "根据相关法律法规、政策及《社区公约》，该视频【17: 21-17:23】【内容】不予审核通过"
        )
        assert result == [(1041, 1043)]
    
    def test_space_after_colon_in_both(self):
        """两端都有空格：【17: 21-17: 23】"""
        result = BilibiliUploader._parse_violation_time("视频【17: 21-17: 23】内容违规")
        assert result == [(1041, 1043)]
    
    def test_space_in_hhmmss(self):
        """HH:MM:SS 格式中有空格：【00:17: 21-00:17:23】"""
        result = BilibiliUploader._parse_violation_time("视频【00:17: 21-00:17:23】内容违规")
        assert result == [(1041, 1043)]
    
    def test_two_problems_with_space(self):
        """两个问题各有一个时间段（其中一个带空格）——不应被判为全片违规"""
        # 模拟图中实际场景：两个 problem_detail
        text1 = "您的视频【20:18-20:24】【内容】根据相关法律法规"
        text2 = "该视频【17: 21-17:23】【内容】不予审核通过"
        
        r1 = BilibiliUploader._parse_violation_time(text1)
        r2 = BilibiliUploader._parse_violation_time(text2)
        
        assert r1 == [(1218, 1224)]
        assert r2 == [(1041, 1043)]
        # 两个都能解析出时间段，不应为空
        assert len(r1) > 0 and len(r2) > 0
    
    def test_no_space_still_works(self):
        """无空格的正常格式仍然正常匹配"""
        result = BilibiliUploader._parse_violation_time("视频【20:18-20:24】内容违规")
        assert result == [(1218, 1224)]
    
    def test_bracket_with_content_tag_not_matched(self):
        """【内容】不应被匹配为时间段"""
        result = BilibiliUploader._parse_violation_time("【内容】不予审核")
        assert result == []


class TestIsFullVideoDetection:
    """回归测试：is_full_video 判定逻辑"""
    
    def test_reason_with_parseable_time_not_full(self):
        """reason 中有可解析时间段 → 不是全片违规"""
        # 模拟 get_rejected_videos 中的逻辑
        vt = ''  # violation_time 为空
        vp = ''  # violation_position 为空
        reason = '该视频【17: 21-17:23】【内容】不予审核通过'
        
        time_ranges = BilibiliUploader._parse_violation_time(vt) if vt else []
        if not time_ranges and reason:
            time_ranges = BilibiliUploader._parse_violation_time(reason)
        
        is_full = vp == '内容全程' or (not vt and not vp and not time_ranges)
        
        assert time_ranges == [(1041, 1043)]
        assert is_full is False
    
    def test_truly_full_video(self):
        """真正的全片违规"""
        vt = ''
        vp = '内容全程'
        reason = '全片内容违规'
        
        time_ranges = BilibiliUploader._parse_violation_time(vt) if vt else []
        if not time_ranges and reason:
            time_ranges = BilibiliUploader._parse_violation_time(reason)
        
        is_full = vp == '内容全程' or (not vt and not vp and not time_ranges)
        assert is_full is True
    
    def test_no_info_at_all(self):
        """所有字段为空 → 判为全片违规（保守策略）"""
        vt = ''
        vp = ''
        reason = '审核不通过'
        
        time_ranges = BilibiliUploader._parse_violation_time(vt) if vt else []
        if not time_ranges and reason:
            time_ranges = BilibiliUploader._parse_violation_time(reason)
        
        is_full = vp == '内容全程' or (not vt and not vp and not time_ranges)
        assert time_ranges == []
        assert is_full is True


class TestMarginAccumulation:
    """回归测试：margin 不应在多次修复间累积"""
    
    def test_merge_ranges_margin_zero_for_storage(self):
        """存储用的 merge 应使用 margin=0（只做去重合并）"""
        ranges = [(1041, 1043), (1218, 1224)]
        raw_merged = FFmpegWrapper._merge_ranges(ranges, margin=0, max_duration=2000)
        assert raw_merged == [(1041, 1043), (1218, 1224)]
    
    def test_merge_ranges_with_margin_for_mask(self):
        """mask 操作使用 margin=2.0（扩展边界）"""
        ranges = [(1041, 1043), (1218, 1224)]
        merged = FFmpegWrapper._merge_ranges(ranges, margin=2.0, max_duration=2000)
        assert merged == [(1039, 1045), (1216, 1226)]
    
    def test_no_double_margin_on_second_fix(self):
        """模拟两次修复：第二次不应在历史 ranges 上再加 margin"""
        # 第一次修复：raw range (1041, 1043)
        first_raw = [(1041, 1043)]
        # 存储 raw（无 margin）
        saved = FFmpegWrapper._merge_ranges(first_raw, margin=0, max_duration=2000)
        assert saved == [(1041, 1043)]
        
        # 第二次修复：新增 (1218, 1224)
        previous_from_db = saved  # [(1041, 1043)] — 无 margin
        new_ranges = [(1218, 1224)]
        all_ranges = list(previous_from_db) + new_ranges
        
        # mask 时应用 margin（正确）
        mask_merged = FFmpegWrapper._merge_ranges(all_ranges, margin=2.0, max_duration=2000)
        assert mask_merged == [(1039, 1045), (1216, 1226)]
        
        # 存储时不含 margin（正确）
        save_merged = FFmpegWrapper._merge_ranges(all_ranges, margin=0, max_duration=2000)
        assert save_merged == [(1041, 1043), (1218, 1224)]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
