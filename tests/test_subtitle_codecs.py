"""subtitle codecs 契约测试。"""

from pathlib import Path

from vat.asr.asr_data import ASRData, ASRDataSeg
from vat.subtitle_utils.codecs import (
    asr_data_to_srt,
    asr_data_to_json,
    save_asr_data,
    load_asr_data_from_file,
)


class TestSubtitleCodecs:
    def test_asr_data_to_srt_renders_bilingual_segments(self):
        data = ASRData([
            ASRDataSeg("原文1", 0, 1000, "译文1"),
            ASRDataSeg("原文2", 1000, 2000, ""),
        ])

        text = asr_data_to_srt(data)

        assert "原文1\n译文1" in text
        assert "原文2" in text

    def test_asr_data_to_json_contains_expected_fields(self):
        data = ASRData([
            ASRDataSeg("原文", 0, 1000, "译文"),
        ])

        payload = asr_data_to_json(data)

        assert payload["1"]["original_subtitle"] == "原文"
        assert payload["1"]["translated_subtitle"] == "译文"

    def test_save_and_load_srt_roundtrip(self, tmp_path):
        data = ASRData([
            ASRDataSeg("こんにちは", 0, 1000, "你好"),
            ASRDataSeg("さようなら", 1000, 2000, "再见"),
        ])
        path = tmp_path / "sample.srt"

        save_asr_data(data, str(path))
        restored = load_asr_data_from_file(str(path))

        assert len(restored.segments) == 2
        assert restored.segments[0].text == "こんにちは"
        assert restored.segments[0].translated_text == "你好"
        assert restored.segments[1].text == "さようなら"
        assert restored.segments[1].translated_text == "再见"

    def test_load_ass_parses_secondary_and_default_tracks(self, tmp_path):
        content = """[Script Info]
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,15,1
Style: Secondary,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,15,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:00.00,0:00:01.00,Default,,0,0,0,,译文
Dialogue: 1,0:00:00.00,0:00:01.00,Secondary,,0,0,0,,原文
"""
        path = tmp_path / "sample.ass"
        path.write_text(content, encoding="utf-8")

        restored = load_asr_data_from_file(str(path))

        assert len(restored.segments) == 1
        assert restored.segments[0].text == "原文"
        assert restored.segments[0].translated_text == "译文"
