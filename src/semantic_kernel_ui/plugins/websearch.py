"""Web search plugin supporting real queries via Google CSE or SerpAPI.

Environment variables supported (first found is used):

1. SerpAPI (recommended simpler setup)
   SERPAPI_KEY=<your key>

2. Google Custom Search JSON API
   GOOGLE_CSE_API_KEY=<api key>
   GOOGLE_CSE_ENGINE_ID=<custom search engine id>

If no keys are configured, a helpful instructional message is returned.
"""
from __future__ import annotations
from typing import Annotated, Optional, List, Dict
import os
import json
import time

try:  # Prefer requests if installed
    import requests  # type: ignore
except ImportError:  # Fallback to urllib if requests not present
    import urllib.request as _urlreq
    import urllib.error as _urlerr
    requests = None  # type: ignore

try:
    from semantic_kernel.functions import kernel_function
except ImportError:  # Fallback decorator
    def kernel_function(name: str = None, description: str = None):
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func
        return decorator

__all__ = ["WebSearchPlugin"]

# Simple in‑memory cache (query -> (timestamp, results))
_CACHE: Dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300  # seconds


def _cache_get(key: str) -> Optional[str]:
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: str) -> None:
    _CACHE[key] = (time.time(), value)


class WebSearchPlugin:
    """Plugin for real web search (Google / SerpAPI) with graceful fallback.
    
    Optional override: provider preference order via provider_hint: 'serpapi'|'google'.
    Caching NOTE: cache key includes resolved provider so provider-specific
    results are not mixed if different providers are requested for the same query.
    """

    @kernel_function(name="search_web", description="Search the web for information on a given topic")
    def search_web(
        self,
        query: Annotated[str, "The search query to look for"],
        max_results: Annotated[int, "Maximum number of results (1-10)"] = 5,
        provider_hint: Annotated[str, "Force provider: serpapi|google or auto"] = "auto",
    ) -> Annotated[str, "Search results"]:
        query = (query or "").strip()
        if not query:
            return "Error: Empty query"
        max_results = max(1, min(10, max_results))
        provider_hint = (provider_hint or "auto").lower()

        serp_key = os.getenv("SERPAPI_KEY")
        g_key = os.getenv("GOOGLE_CSE_API_KEY")
        g_cx = os.getenv("GOOGLE_CSE_ENGINE_ID")

        # Resolve provider first so cache key can include it
        if serp_key and provider_hint in ("auto", "serpapi"):
            provider = "serpapi"
        elif g_key and g_cx and provider_hint in ("auto", "google"):
            provider = "google"
        else:
            provider = None

        if not provider:
            return (
                "Web search is not configured or provider unavailable. Set one of:\n"
                " - SERPAPI_KEY (preferred)\n"
                " - GOOGLE_CSE_API_KEY and GOOGLE_CSE_ENGINE_ID\n"
                "(Optional) provider_hint='serpapi' or 'google' to force a provider.\n"
                "Visit https://serpapi.com/ or https://developers.google.com/custom-search for keys."
            )

        # Provider-aware cache key avoids cross-provider pollution
        cache_key = f"{provider}:{query}:{max_results}"
        cached = _cache_get(cache_key)
        if cached:
            return cached + "\n\n(cached)"

        if provider == "serpapi":
            result = self._search_serpapi(query, serp_key, max_results)  # type: ignore[arg-type]
        else:
            result = self._search_google_cse(query, g_key, g_cx, max_results)  # type: ignore[arg-type]

        _cache_set(cache_key, result)
        return result

    # -------- Internal search methods -------- #
    def _search_serpapi(self, query: str, api_key: str, max_results: int) -> str:
        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "engine": "google",
            "api_key": api_key,
            "num": max_results,
        }
        try:
            data = self._http_get_json(url, params)
            organic = data.get("organic_results", [])[:max_results]
            # Normalization
            if not isinstance(organic, list):
                organic = []
            if not organic:
                return f"No results for '{query}'."
            lines = [f"Search results for: '{query}' (SerpAPI)"]
            for i, item in enumerate(organic, 1):
                title = item.get("title", "(no title)")
                link = item.get("link", "")
                snippet = item.get("snippet", "").replace('\n', ' ')
                lines.append(f"{i}. {title}\n{link}\n{snippet}")
            return "\n\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"SerpAPI search failed: {self._sanitize_error(e)}"  # Already sanitized

    def _search_google_cse(self, query: str, api_key: str, engine_id: str, max_results: int) -> str:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": engine_id,
            "q": query,
            "num": max_results,
        }
        try:
            data = self._http_get_json(url, params)
            items = data.get("items", [])[:max_results]
            if not isinstance(items, list):
                items = []
            if not items:
                return f"No results for '{query}'."
            lines = [f"Search results for: '{query}' (Google CSE)"]
            for i, item in enumerate(items, 1):
                title = item.get("title", "(no title)")
                link = item.get("link", "")
                snippet = item.get("snippet", "").replace('\n', ' ')
                lines.append(f"{i}. {title}\n{link}\n{snippet}")
            return "\n\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"Google CSE search failed: {self._sanitize_error(e)}"  # Already sanitized

    # -------- HTTP helper -------- #
    def _http_get_json(self, url: str, params: Dict[str, str]) -> Dict[str, any]:
        if requests:  # type: ignore
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        # Fallback using urllib
        import urllib.parse as _urlparse

        full_url = f"{url}?{_urlparse.urlencode(params)}"
        try:
            with _urlreq.urlopen(full_url, timeout=10) as r:  # type: ignore
                raw = r.read().decode("utf-8")
                # Add basic rate-limit header look for future use (ignored now)
                return json.loads(raw)
        except _urlerr.HTTPError as e:  # type: ignore
            raise RuntimeError(f"HTTP error {e.code}") from e
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(str(e)) from e

    def _sanitize_error(self, exc: Exception) -> str:
        """Return a minimal safe error string without leaking internals."""
        msg = str(exc)
        if len(msg) > 120:
            msg = msg[:117] + "..."
        return msg.replace('\n', ' ').strip()
