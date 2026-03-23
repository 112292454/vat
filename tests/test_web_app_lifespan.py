"""web app lifespan 契约测试。"""

from fastapi.testclient import TestClient


def test_app_lifespan_does_not_start_auto_sync_thread(monkeypatch):
    from vat.web import app as web_app_module

    assert not hasattr(web_app_module, "_start_auto_sync_thread")

    with TestClient(web_app_module.app):
        pass


def test_app_does_not_register_legacy_json_api_routes():
    from vat.web import app as web_app_module

    routes = []
    for route in web_app_module.app.routes:
        path = getattr(route, "path", None)
        methods = tuple(sorted(m for m in (route.methods or []) if m not in {"HEAD", "OPTIONS"}))
        if path and methods:
            routes.append((path, methods))

    assert routes.count(("/api/videos", ("GET",))) == 1
    assert ("/api/video/{video_id}", ("GET",)) not in routes
    assert ("/api/stats", ("GET",)) not in routes
