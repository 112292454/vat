"""web app lifespan 契约测试。"""

from fastapi.testclient import TestClient


def test_app_lifespan_starts_auto_sync_thread(monkeypatch):
    from vat.web import app as web_app_module

    calls = []
    monkeypatch.setattr(web_app_module, "_start_auto_sync_thread", lambda: calls.append("started"))

    with TestClient(web_app_module.app):
        assert calls == ["started"]
