"""database API 路由契约测试。"""

import sqlite3

import httpx
import pytest
from fastapi import FastAPI

from vat.web.routes.database import router


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO items (id, name) VALUES (?, ?)",
        [(1, "alpha"), (2, "beta")],
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def client(db_path, monkeypatch):
    monkeypatch.setattr("vat.web.routes.database._get_db_path", lambda: db_path)
    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


class TestDatabaseApi:
    @pytest.mark.anyio
    async def test_query_table_accepts_valid_sort_dir(self, client):
        async with client as ac:
            response = await ac.get(
                "/api/database/tables/items",
                params={"sort_by": "id", "sort_dir": "desc"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["table"] == "items"
        assert [row["id"] for row in payload["rows"]] == [2, 1]

    @pytest.mark.anyio
    async def test_query_table_rejects_invalid_sort_dir(self, client):
        async with client as ac:
            response = await ac.get("/api/database/tables/items", params={"sort_dir": "sideways"})

        assert response.status_code == 422
