"""cache_metadata 契约测试。"""

from types import SimpleNamespace

from vat.utils.cache_metadata import CacheMetadata, SubstepMetadata, extract_key_config


class TestCacheMetadataContracts:
    def test_load_returns_empty_metadata_when_file_missing(self, tmp_path):
        metadata = CacheMetadata.load(tmp_path)

        assert metadata.version == "0.2.1"
        assert metadata.video_id == ""
        assert metadata.substeps == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        metadata = CacheMetadata(
            version="1.0",
            video_id="vid1",
            substeps={
                "whisper": SubstepMetadata(
                    completed_at="2026-03-17T00:00:00",
                    config_snapshot={"model": "large-v3"},
                    output_file="original_raw.srt",
                )
            },
        )

        metadata.save(tmp_path)
        restored = CacheMetadata.load(tmp_path)

        assert restored.version == "1.0"
        assert restored.video_id == "vid1"
        assert restored.substeps["whisper"].config_snapshot == {"model": "large-v3"}

    def test_is_substep_valid_requires_exact_snapshot_match(self):
        metadata = CacheMetadata(version="1.0", video_id="vid1")
        metadata.update_substep("split", {"model": "gpt-4o-mini"}, "original.srt")

        assert metadata.is_substep_valid("split", {"model": "gpt-4o-mini"}) is True
        assert metadata.is_substep_valid("split", {"model": "other"}) is False
        assert metadata.is_substep_valid("translate", {"model": "other"}) is False

    def test_load_returns_empty_metadata_when_json_broken(self, tmp_path):
        (tmp_path / ".cache_metadata.json").write_text("{broken", encoding="utf-8")

        metadata = CacheMetadata.load(tmp_path)

        assert metadata.video_id == ""
        assert metadata.substeps == {}

    def test_update_substep_records_output_file_and_snapshot(self):
        metadata = CacheMetadata(version="1.0", video_id="vid1")

        metadata.update_substep("translate", {"model": "gemini"}, "translated.srt")

        substep = metadata.substeps["translate"]
        assert substep.output_file == "translated.srt"
        assert substep.config_snapshot == {"model": "gemini"}
        assert "T" in substep.completed_at


class TestExtractKeyConfigContracts:
    def test_extract_key_config_serializes_nested_object_dict(self):
        config = SimpleNamespace(
            model="large-v3",
            split=SimpleNamespace(enable=True, mode="sentence", _hidden="x"),
        )

        extracted = extract_key_config(config, ["model", "split"])

        assert extracted == {
            "model": "large-v3",
            "split": {"enable": True, "mode": "sentence"},
        }
