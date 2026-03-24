"""scene_identifier 契约测试。"""

from pathlib import Path
from types import SimpleNamespace

from vat.llm.scene_identifier import SceneIdentifier


def _make_identifier():
    identifier = SceneIdentifier.__new__(SceneIdentifier)
    identifier.model = "test-model"
    identifier.api_key = ""
    identifier.base_url = ""
    identifier.proxy = ""
    identifier.scenes_config = {
        "default_scene": "chatting",
        "scenes": [
            {"id": "chatting", "name": "闲聊", "description": "聊天", "keywords": ["雑談"], "prompts": {"translate": "chat"}},
            {"id": "gaming", "name": "游戏", "description": "游戏", "keywords": ["Minecraft"], "prompts": {"translate": "game"}},
        ],
    }
    return identifier


class TestSceneIdentifierHelpers:
    def test_load_scenes_config_returns_default_when_file_missing(self, monkeypatch):
        identifier = SceneIdentifier.__new__(SceneIdentifier)
        missing = Path("/tmp/scene-missing.yaml")
        monkeypatch.setattr("vat.llm.scene_identifier.SCENES_CONFIG_PATH", missing)

        config = identifier._load_scenes_config()

        assert config == {"scenes": [], "default_scene": "chatting"}

    def test_load_scenes_config_reads_yaml_file(self, tmp_path, monkeypatch):
        identifier = SceneIdentifier.__new__(SceneIdentifier)
        config_file = tmp_path / "scenes.yaml"
        config_file.write_text(
            "default_scene: chatting\nscenes:\n  - id: gaming\n    name: 游戏\n    description: 游戏场景\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("vat.llm.scene_identifier.SCENES_CONFIG_PATH", config_file)

        config = identifier._load_scenes_config()

        assert config["default_scene"] == "chatting"
        assert config["scenes"][0]["id"] == "gaming"

    def test_is_valid_scene(self):
        identifier = _make_identifier()

        assert identifier._is_valid_scene("gaming") is True
        assert identifier._is_valid_scene("unknown") is False

    def test_get_scene_name_and_default_scene(self):
        identifier = _make_identifier()

        assert identifier._get_scene_name("chatting") == "闲聊"
        assert identifier._get_default_scene() == {
            "scene_id": "chatting",
            "scene_name": "闲聊",
            "auto_detected": False,
        }

    def test_get_default_scene_returns_unknown_when_default_scene_missing_in_catalog(self):
        identifier = _make_identifier()
        identifier.scenes_config["default_scene"] = "not-found"

        assert identifier._get_default_scene() == {
            "scene_id": "not-found",
            "scene_name": "Unknown",
            "auto_detected": False,
        }

    def test_get_scene_prompts_returns_configured_prompts(self):
        identifier = _make_identifier()

        assert identifier.get_scene_prompts("gaming") == {"translate": "game"}
        assert identifier.get_scene_prompts("unknown") == {}

    def test_build_system_prompt_lists_scenes_and_default_rule(self):
        identifier = _make_identifier()

        prompt = identifier._build_system_prompt()

        assert "chatting - 闲聊" in prompt
        assert "gaming - 游戏" in prompt
        assert 'If uncertain, output "chatting"' in prompt


class TestSceneIdentifierDetectScene:
    def test_detect_scene_uses_default_when_title_missing(self):
        identifier = _make_identifier()

        result = identifier.detect_scene("", "desc")

        assert result == {
            "scene_id": "chatting",
            "scene_name": "闲聊",
            "auto_detected": False,
        }

    def test_detect_scene_returns_default_on_invalid_llm_output(self, monkeypatch):
        identifier = _make_identifier()
        monkeypatch.setattr(
            "vat.llm.scene_identifier.call_text_llm",
            lambda **kwargs: "invalid-scene",
        )

        result = identifier.detect_scene("标题", "desc")

        assert result["scene_id"] == "chatting"
        assert result["auto_detected"] is False

    def test_detect_scene_returns_detected_scene_on_valid_output(self, monkeypatch):
        identifier = _make_identifier()
        monkeypatch.setattr(
            "vat.llm.scene_identifier.call_text_llm",
            lambda **kwargs: "gaming",
        )

        result = identifier.detect_scene("Minecraft 标题", "desc")

        assert result == {
            "scene_id": "gaming",
            "scene_name": "游戏",
            "auto_detected": True,
        }

    def test_detect_scene_uses_no_description_placeholder(self, monkeypatch):
        identifier = _make_identifier()
        captured = {}

        def fake_call_llm(*, messages, model, temperature, api_key, base_url, proxy):
            captured["messages"] = messages
            return "chatting"

        monkeypatch.setattr("vat.llm.scene_identifier.call_text_llm", fake_call_llm)

        result = identifier.detect_scene("标题", "")

        assert result["scene_id"] == "chatting"
        assert "(no description)" in captured["messages"][1]["content"]

    def test_detect_scene_accepts_uppercase_output_after_lowering(self, monkeypatch):
        identifier = _make_identifier()
        monkeypatch.setattr(
            "vat.llm.scene_identifier.call_text_llm",
            lambda **kwargs: "GAMING",
        )

        result = identifier.detect_scene("Minecraft 标题", "desc")

        assert result["scene_id"] == "gaming"
        assert result["auto_detected"] is True

    def test_detect_scene_returns_default_when_llm_choices_empty(self, monkeypatch):
        identifier = _make_identifier()
        monkeypatch.setattr(
            "vat.llm.scene_identifier.call_text_llm",
            lambda **kwargs: (_ for _ in ()).throw(ValueError("Invalid response")),
        )

        result = identifier.detect_scene("标题", "desc")

        assert result["scene_id"] == "chatting"
        assert result["auto_detected"] is False

    def test_detect_scene_returns_default_on_exception(self, monkeypatch):
        identifier = _make_identifier()
        monkeypatch.setattr("vat.llm.scene_identifier.call_text_llm", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

        result = identifier.detect_scene("标题", "desc")

        assert result["scene_id"] == "chatting"
        assert result["auto_detected"] is False
