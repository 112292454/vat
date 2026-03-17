"""上传配置管理契约测试。"""

from pathlib import Path

from vat.uploaders import upload_config as upload_config_module
from vat.uploaders.upload_config import UploadConfigManager, UploadConfig, save_upload_config, load_upload_config


class TestUploadConfigManager:
    def test_load_returns_defaults_when_file_missing(self, tmp_path):
        manager = UploadConfigManager(config_path=tmp_path / "missing.yaml")

        config = manager.load()

        assert config.bilibili.default_tid == 21
        assert config.bilibili.templates.title == "${translated_title}"

    def test_save_and_reload_roundtrip(self, tmp_path):
        path = tmp_path / "upload.yaml"
        manager = UploadConfigManager(config_path=path)
        config = manager.load()
        config.bilibili.default_tid = 17
        config.bilibili.templates.title = "【${uploader_name}】${translated_title}"
        config.bilibili.templates.custom_vars = {"suffix": "测试"}

        assert manager.save(config) is True

        reloaded = UploadConfigManager(config_path=path).load()
        assert reloaded.bilibili.default_tid == 17
        assert reloaded.bilibili.templates.title == "【${uploader_name}】${translated_title}"
        assert reloaded.bilibili.templates.custom_vars == {"suffix": "测试"}

    def test_update_bilibili_updates_nested_template_fields(self, tmp_path):
        path = tmp_path / "upload.yaml"
        manager = UploadConfigManager(config_path=path)
        manager.load()

        ok = manager.update_bilibili({
            "default_tid": 65,
            "templates": {
                "title": "【${channel_name}】${translated_title}",
                "description": "${translated_desc}\n${suffix}",
                "custom_vars": {"suffix": "END"},
            },
        })

        assert ok is True
        config = manager.get_config()
        assert config.bilibili.default_tid == 65
        assert config.bilibili.templates.title == "【${channel_name}】${translated_title}"
        assert config.bilibili.templates.description == "${translated_desc}\n${suffix}"
        assert config.bilibili.templates.custom_vars == {"suffix": "END"}

    def test_save_upload_config_convenience_uses_global_manager(self, tmp_path, monkeypatch):
        path = tmp_path / "upload.yaml"
        manager = UploadConfigManager(config_path=path)
        config = UploadConfig()
        config.bilibili.default_tid = 99
        monkeypatch.setattr("vat.uploaders.upload_config.get_upload_config_manager", lambda: manager)

        assert save_upload_config(config) is True
        assert manager.load().bilibili.default_tid == 99

    def test_load_upload_config_convenience_uses_global_manager(self, tmp_path, monkeypatch):
        path = tmp_path / "upload.yaml"
        manager = UploadConfigManager(config_path=path)
        manager.load()
        manager.update_bilibili({"default_tid": 77})
        monkeypatch.setattr("vat.uploaders.upload_config.get_upload_config_manager", lambda: manager)

        config = load_upload_config()

        assert config.bilibili.default_tid == 77

    def test_get_bilibili_dict_returns_serializable_dict(self, tmp_path):
        manager = UploadConfigManager(config_path=tmp_path / "upload.yaml")
        manager.load()
        manager.update_bilibili({"default_tid": 31})

        data = manager.get_bilibili_dict()

        assert data["default_tid"] == 31
        assert "templates" in data

    def test_load_falls_back_to_default_when_yaml_invalid(self, tmp_path):
        path = tmp_path / "broken.yaml"
        path.write_text("bilibili: [broken", encoding="utf-8")

        config = UploadConfigManager(config_path=path).load()

        assert config.bilibili.default_tid == 21

    def test_save_returns_false_when_no_config_loaded(self, tmp_path):
        manager = UploadConfigManager(config_path=tmp_path / "upload.yaml")

        assert manager.save() is False

    def test_update_bilibili_returns_false_on_invalid_value(self, tmp_path):
        path = tmp_path / "upload.yaml"
        manager = UploadConfigManager(config_path=path)
        manager.load()

        ok = manager.update_bilibili({"default_tid": "not-an-int"})

        assert ok is False

    def test_save_returns_false_on_write_exception(self, tmp_path, monkeypatch):
        path = tmp_path / "upload.yaml"
        manager = UploadConfigManager(config_path=path)
        config = manager.load()

        monkeypatch.setattr(
            "builtins.open",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )

        assert manager.save(config) is False

    def test_get_upload_config_manager_is_singleton(self, monkeypatch):
        monkeypatch.setattr(upload_config_module, "_manager", None)

        first = upload_config_module.get_upload_config_manager()
        second = upload_config_module.get_upload_config_manager()

        assert first is second
