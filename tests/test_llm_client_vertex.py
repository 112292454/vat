from datetime import datetime, timedelta, timezone
import pytest
from unittest.mock import MagicMock

from vat.llm.client import _get_vertex_access_token, call_llm


class TestVertexNativeClient:
    def test_call_llm_vertex_native_adapts_response(self, monkeypatch):
        monkeypatch.setenv("VAT_LLM_PROVIDER", "vertex_native")
        monkeypatch.setenv("VAT_VERTEX_AUTH_MODE", "api_key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-vertex-key")
        monkeypatch.setenv("VAT_VERTEX_LOCATION", "global")
        monkeypatch.delenv("VAT_VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("VAT_VERTEX_CREDENTIALS", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        captured = {}

        def fake_post(url, json, headers, timeout, proxy=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["proxy"] = proxy
            response = MagicMock()
            response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "vertex response text"}
                            ]
                        }
                    }
                ]
            }
            response.raise_for_status.return_value = None
            return response

        monkeypatch.setattr("vat.llm.client.httpx.post", fake_post)

        response = call_llm(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello from user"},
            ],
            model="gemini-2.5-flash",
            temperature=0.3,
        )

        assert response.choices[0].message.content == "vertex response text"
        assert captured["url"] == (
            "https://aiplatform.googleapis.com/v1/publishers/google/models/"
            "gemini-2.5-flash:generateContent?key=test-vertex-key"
        )
        assert captured["json"]["systemInstruction"] == {
            "parts": [{"text": "You are a helpful assistant."}]
        }
        assert captured["json"]["contents"] == [
            {"role": "user", "parts": [{"text": "Hello from user"}]}
        ]
        assert captured["json"]["generationConfig"]["temperature"] == 0.3

    def test_explicit_base_url_still_uses_openai_compatible_client(self, monkeypatch):
        monkeypatch.setenv("VAT_LLM_PROVIDER", "vertex_native")

        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "openai compatible text"

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        monkeypatch.setattr("vat.llm.client.get_or_create_client", lambda *args, **kwargs: fake_client)

        response = call_llm(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
            api_key="override-key",
            base_url="https://example.com/v1",
        )

        assert response.choices[0].message.content == "openai compatible text"
        fake_client.chat.completions.create.assert_called_once()

    def test_call_llm_vertex_native_adc_uses_project_scoped_endpoint(self, monkeypatch):
        monkeypatch.setenv("VAT_LLM_PROVIDER", "vertex_native")
        monkeypatch.setenv("VAT_VERTEX_AUTH_MODE", "adc")
        monkeypatch.setenv("VAT_VERTEX_LOCATION", "global")
        monkeypatch.setenv("VAT_VERTEX_PROJECT_ID", "vertex-490203")
        monkeypatch.setenv("VAT_VERTEX_CREDENTIALS", "/home/gzy/.ssh/vat_vertex.json")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        captured = {}

        monkeypatch.setattr(
            "vat.llm.client._get_vertex_access_token",
            lambda credentials_path="", proxy="": "test-access-token",
            raising=False,
        )

        def fake_post(url, json, headers, timeout, proxy=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["proxy"] = proxy
            response = MagicMock()
            response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "vertex adc response"}
                            ]
                        }
                    }
                ]
            }
            response.raise_for_status.return_value = None
            return response

        monkeypatch.setattr("vat.llm.client.httpx.post", fake_post)

        response = call_llm(
            messages=[{"role": "user", "content": "Hello from adc"}],
            model="gemini-2.5-flash",
            temperature=0.1,
        )

        assert response.choices[0].message.content == "vertex adc response"
        assert captured["url"] == (
            "https://aiplatform.googleapis.com/v1/projects/vertex-490203/locations/global/"
            "publishers/google/models/gemini-2.5-flash:generateContent"
        )
        assert captured["headers"]["Authorization"] == "Bearer test-access-token"
        assert "key=" not in captured["url"]

    def test_call_llm_vertex_native_adc_passes_proxy_to_access_token_refresh(self, monkeypatch):
        monkeypatch.setenv("VAT_LLM_PROVIDER", "vertex_native")
        monkeypatch.setenv("VAT_VERTEX_AUTH_MODE", "adc")
        monkeypatch.setenv("VAT_VERTEX_LOCATION", "global")
        monkeypatch.setenv("VAT_VERTEX_PROJECT_ID", "vertex-490203")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        captured = {}

        def fake_get_vertex_access_token(credentials_path="", proxy=""):
            captured["credentials_path"] = credentials_path
            captured["proxy"] = proxy
            return "test-access-token"

        def fake_post(url, json, headers, timeout, proxy=None):
            response = MagicMock()
            response.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "vertex adc response"}]}}]
            }
            response.raise_for_status.return_value = None
            return response

        monkeypatch.setattr("vat.llm.client._get_vertex_access_token", fake_get_vertex_access_token)
        monkeypatch.setattr("vat.llm.client.httpx.post", fake_post)

        response = call_llm(
            messages=[{"role": "user", "content": "Hello from adc"}],
            model="gemini-2.5-flash",
            proxy="http://translate-proxy:7890",
        )

        assert response.choices[0].message.content == "vertex adc response"
        assert captured["proxy"] == "http://translate-proxy:7890"

    def test_call_llm_vertex_native_reports_diagnostics_on_empty_response(self, monkeypatch):
        monkeypatch.setenv("VAT_LLM_PROVIDER", "vertex_native")
        monkeypatch.setenv("VAT_VERTEX_AUTH_MODE", "api_key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-vertex-key")
        monkeypatch.setenv("VAT_VERTEX_LOCATION", "global")
        monkeypatch.delenv("VAT_VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("VAT_VERTEX_CREDENTIALS", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        warnings = []

        def fake_post(url, json, headers, timeout, proxy=None):
            response = MagicMock()
            response.json.return_value = {
                "promptFeedback": {
                    "blockReason": "PROHIBITED_CONTENT",
                    "blockReasonMessage": "blocked by safety",
                },
                "candidates": [
                    {
                        "finishReason": "SAFETY",
                        "content": {"parts": []},
                    }
                ],
            }
            response.raise_for_status.return_value = None
            return response

        monkeypatch.setattr("vat.llm.client.httpx.post", fake_post)
        monkeypatch.setattr("vat.llm.client.logger.warning", lambda msg: warnings.append(msg))

        with pytest.raises(ValueError, match="Invalid Vertex API response"):
            call_llm(
                messages=[{"role": "user", "content": "Hello from vertex"}],
                model="gemini-2.5-flash",
                temperature=0.1,
            )

        assert any("PROHIBITED_CONTENT" in msg for msg in warnings)
        assert any("SAFETY" in msg for msg in warnings)


class TestVertexAccessTokenProxyContracts:
    def test_get_vertex_access_token_uses_proxy_session_for_refresh(self, monkeypatch):
        captured = {}

        class FakeCredentials:
            token = "refreshed-token"

            def refresh(self, request):
                captured["request"] = request

        def fake_from_service_account_file(path, scopes):
            captured["path"] = path
            captured["scopes"] = scopes
            return FakeCredentials()

        def fake_request(session=None):
            captured["session"] = session
            return "request-object"

        monkeypatch.setattr("vat.llm.client._resolve_vertex_credentials_path", lambda _path="": "/tmp/fake.json")
        monkeypatch.setattr(
            "vat.llm.client.service_account.Credentials.from_service_account_file",
            fake_from_service_account_file,
        )
        monkeypatch.setattr("vat.llm.client.GoogleAuthRequest", fake_request)

        token = _get_vertex_access_token("/tmp/fake.json", proxy="http://translate-proxy:7890")

        assert token == "refreshed-token"
        assert captured["path"] == "/tmp/fake.json"
        assert captured["session"].proxies["http"] == "http://translate-proxy:7890"
        assert captured["session"].proxies["https"] == "http://translate-proxy:7890"

    def test_get_vertex_access_token_caches_token_until_expiry_buffer(self, monkeypatch):
        calls = []

        class FakeCredentials:
            def __init__(self):
                self.token = ""
                self.expiry = None

            def refresh(self, request):
                calls.append(request)
                self.token = "cached-token"
                self.expiry = datetime.now(timezone.utc) + timedelta(minutes=10)

        monkeypatch.setattr("vat.llm.client._vertex_token_cache", {}, raising=False)
        monkeypatch.setattr("vat.llm.client._resolve_vertex_credentials_path", lambda _path="": "/tmp/fake-cache.json")
        monkeypatch.setattr(
            "vat.llm.client.service_account.Credentials.from_service_account_file",
            lambda path, scopes: FakeCredentials(),
        )
        monkeypatch.setattr("vat.llm.client.GoogleAuthRequest", lambda session=None: "request")

        token1 = _get_vertex_access_token("/tmp/fake-cache.json", proxy="http://translate-proxy:7890")
        token2 = _get_vertex_access_token("/tmp/fake-cache.json", proxy="http://translate-proxy:7890")

        assert token1 == "cached-token"
        assert token2 == "cached-token"
        assert len(calls) == 1

    def test_get_vertex_access_token_refreshes_again_when_expiry_too_close(self, monkeypatch):
        calls = []

        class FakeCredentials:
            def __init__(self):
                self.token = ""
                self.expiry = None

            def refresh(self, request):
                calls.append(request)
                self.token = f"token-{len(calls)}"
                if len(calls) == 1:
                    self.expiry = datetime.now(timezone.utc) + timedelta(seconds=30)
                else:
                    self.expiry = datetime.now(timezone.utc) + timedelta(minutes=10)

        monkeypatch.setattr("vat.llm.client._vertex_token_cache", {}, raising=False)
        monkeypatch.setattr("vat.llm.client._resolve_vertex_credentials_path", lambda _path="": "/tmp/fake-expiring.json")
        monkeypatch.setattr(
            "vat.llm.client.service_account.Credentials.from_service_account_file",
            lambda path, scopes: FakeCredentials(),
        )
        monkeypatch.setattr("vat.llm.client.GoogleAuthRequest", lambda session=None: "request")

        token1 = _get_vertex_access_token("/tmp/fake-expiring.json")
        token2 = _get_vertex_access_token("/tmp/fake-expiring.json")

        assert token1 == "token-1"
        assert token2 == "token-2"
        assert len(calls) == 2

    def test_get_vertex_access_token_retries_without_proxy_when_proxy_refresh_fails(self, monkeypatch):
        calls = []
        warnings = []
        sleeps = []

        class FakeCredentials:
            def __init__(self):
                self.token = ""
                self.expiry = None

            def refresh(self, request):
                calls.append(request)
                if getattr(request, "_session_proxy", None) == "http://translate-proxy:7890":
                    raise RuntimeError("proxy refresh failed")
                self.token = "fallback-token"
                self.expiry = datetime.now(timezone.utc) + timedelta(minutes=10)

        class FakeRequest:
            def __init__(self, session=None):
                self._session_present = session is not None
                self._session_proxy = session.proxies.get("http") if session is not None else None
                self._trust_env = getattr(session, "trust_env", None) if session is not None else None

        monkeypatch.setattr("vat.llm.client._vertex_token_cache", {}, raising=False)
        monkeypatch.setattr("vat.llm.client._resolve_vertex_credentials_path", lambda _path="": "/tmp/fake-fallback.json")
        monkeypatch.setattr(
            "vat.llm.client.service_account.Credentials.from_service_account_file",
            lambda path, scopes: FakeCredentials(),
        )
        monkeypatch.setattr("vat.llm.client.GoogleAuthRequest", FakeRequest)
        monkeypatch.setattr("vat.llm.client.logger.warning", lambda msg: warnings.append(msg))
        monkeypatch.setattr("vat.llm.client.time.sleep", lambda seconds: sleeps.append(seconds))

        token = _get_vertex_access_token("/tmp/fake-fallback.json", proxy="http://translate-proxy:7890")

        assert token == "fallback-token"
        assert len(calls) == 4
        assert sleeps == [1, 2]
        assert calls[-1]._trust_env is False
        assert any("retry without proxy" in msg for msg in warnings)

    def test_get_vertex_access_token_retries_transient_refresh_failures(self, monkeypatch):
        calls = []
        sleeps = []

        class FakeCredentials:
            def __init__(self):
                self.token = ""
                self.expiry = None

            def refresh(self, request):
                calls.append(request)
                if len(calls) < 3:
                    raise RuntimeError("temporary ssl eof")
                self.token = "recovered-token"
                self.expiry = datetime.now(timezone.utc) + timedelta(minutes=10)

        monkeypatch.setattr("vat.llm.client._vertex_token_cache", {}, raising=False)
        monkeypatch.setattr("vat.llm.client._resolve_vertex_credentials_path", lambda _path="": "/tmp/fake-retry.json")
        monkeypatch.setattr(
            "vat.llm.client.service_account.Credentials.from_service_account_file",
            lambda path, scopes: FakeCredentials(),
        )
        monkeypatch.setattr("vat.llm.client.GoogleAuthRequest", lambda session=None: "request")
        monkeypatch.setattr("vat.llm.client.time.sleep", lambda seconds: sleeps.append(seconds))

        token = _get_vertex_access_token("/tmp/fake-retry.json")

        assert token == "recovered-token"
        assert len(calls) == 3
        assert sleeps == [1, 2]
