"""Tests for the WebSearchPlugin real search logic (mocked HTTP)."""
from __future__ import annotations

import os
import types
import pytest

from semantic_kernel_ui.plugins import WebSearchPlugin


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Clear relevant env vars before each test for isolation."""
    for var in ["SERPAPI_KEY", "GOOGLE_CSE_API_KEY", "GOOGLE_CSE_ENGINE_ID"]:
        monkeypatch.delenv(var, raising=False)
    yield


def test_search_no_configuration():
    plugin = WebSearchPlugin()
    out = plugin.search_web("python testing")
    assert "not configured" in out.lower()
    assert "SERPAPI_KEY" in out or "GOOGLE_CSE_API_KEY" in out


def test_search_serpapi(monkeypatch):
    plugin = WebSearchPlugin()

    # Set SERPAPI key only
    monkeypatch.setenv("SERPAPI_KEY", "dummy-serp-key")

    # Mock internal HTTP JSON retrieval
    fake_data = {
        "organic_results": [
            {"title": "Result A", "link": "https://a.example", "snippet": "Snippet A"},
            {"title": "Result B", "link": "https://b.example", "snippet": "Snippet B"},
        ]
    }
    monkeypatch.setattr(plugin, "_http_get_json", lambda url, params: fake_data)

    out = plugin.search_web("pytest", max_results=2)
    assert "SerpAPI" in out
    assert "Result A" in out and "Result B" in out


def test_search_google_cse(monkeypatch):
    plugin = WebSearchPlugin()

    # Set Google CSE keys only
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "dummy-g-key")
    monkeypatch.setenv("GOOGLE_CSE_ENGINE_ID", "dummy-engine")

    fake_data = {
        "items": [
            {"title": "G Result 1", "link": "https://g1.example", "snippet": "G Snippet 1"},
            {"title": "G Result 2", "link": "https://g2.example", "snippet": "G Snippet 2"},
        ]
    }
    monkeypatch.setattr(plugin, "_http_get_json", lambda url, params: fake_data)

    out = plugin.search_web("machine learning", max_results=2)
    assert "Google CSE" in out
    assert "G Result 1" in out and "G Result 2" in out


def test_search_cache(monkeypatch):
    plugin = WebSearchPlugin()
    monkeypatch.setenv("SERPAPI_KEY", "dummy-serp-key")

    call_count = {"n": 0}
    fake_data = {
        "organic_results": [
            {"title": "Cache Test", "link": "https://cache.example", "snippet": "Cache Snippet"},
        ]
    }

    def fake_http(url, params):
        call_count["n"] += 1
        return fake_data

    monkeypatch.setattr(plugin, "_http_get_json", fake_http)

    first = plugin.search_web("cache query", max_results=1)
    second = plugin.search_web("cache query", max_results=1)

    # HTTP called only once due to cache
    assert call_count["n"] == 1
    assert "Cache Test" in first
    assert "(cached)" in second


def test_empty_query():
    plugin = WebSearchPlugin()
    assert "Error" in plugin.search_web("")
