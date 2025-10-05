"""HTTP/API plugin for making web requests."""
from __future__ import annotations

import json
from typing import Annotated
from urllib.parse import urlparse

try:
    from semantic_kernel.functions import kernel_function
except ImportError:
    def kernel_function(name: str = None, description: str = None):
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func
        return decorator


class HttpApiPlugin:
    """Plugin for HTTP/API operations (read-only for security)."""

    def __init__(self, timeout: int = 10, max_size: int = 1024 * 1024):
        """Initialize HTTP plugin.

        Args:
            timeout: Request timeout in seconds
            max_size: Maximum response size in bytes (default 1MB)
        """
        self.timeout = timeout
        self.max_size = max_size

    def _is_safe_url(self, url: str) -> tuple[bool, str]:
        """Validate URL is safe to fetch.

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)

            if parsed.scheme not in ['http', 'https']:
                return False, "Only HTTP/HTTPS URLs are allowed"

            if not parsed.netloc:
                return False, "Invalid URL format"

            blocked_domains = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
            if any(blocked in parsed.netloc.lower() for blocked in blocked_domains):
                return False, "Local URLs are not allowed"

            return True, ""
        except Exception as e:
            return False, f"Invalid URL: {str(e)}"

    @kernel_function(name="http_get", description="Make HTTP GET request")
    def http_get(
        self,
        url: Annotated[str, "URL to fetch"]
    ) -> Annotated[str, "Response content"]:
        """Make HTTP GET request to a URL.

        Args:
            url: URL to fetch

        Returns:
            Response content or error message
        """
        is_valid, error = self._is_safe_url(url)
        if not is_valid:
            return f"Error: {error}"

        try:
            import requests

            response = requests.get(
                url,
                timeout=self.timeout,
                headers={'User-Agent': 'SemanticKernelUI/1.0'},
                stream=True
            )

            content_length = int(response.headers.get('content-length', 0))
            if content_length > self.max_size:
                return f"Error: Response too large ({content_length} bytes, max {self.max_size})"

            response.raise_for_status()

            content_type = response.headers.get('content-type', '')

            if 'application/json' in content_type:
                try:
                    data = response.json()
                    return f"Status: {response.status_code}\nContent-Type: {content_type}\n\n{json.dumps(data, indent=2)}"
                except json.JSONDecodeError:
                    return f"Status: {response.status_code}\nError: Invalid JSON response"

            text = response.text[:self.max_size]
            return f"Status: {response.status_code}\nContent-Type: {content_type}\n\n{text[:1000]}{'...' if len(text) > 1000 else ''}"

        except requests.Timeout:
            return f"Error: Request timed out after {self.timeout} seconds"
        except requests.RequestException as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @kernel_function(name="check_url_status", description="Check HTTP status of a URL")
    def check_url_status(
        self,
        url: Annotated[str, "URL to check"]
    ) -> Annotated[str, "URL status information"]:
        """Check the HTTP status of a URL.

        Args:
            url: URL to check

        Returns:
            Status information
        """
        is_valid, error = self._is_safe_url(url)
        if not is_valid:
            return f"Error: {error}"

        try:
            import requests

            response = requests.head(url, timeout=self.timeout, allow_redirects=True)

            info = [
                f"URL: {url}",
                f"Status: {response.status_code} {response.reason}",
                f"Content-Type: {response.headers.get('content-type', 'Unknown')}",
                f"Content-Length: {response.headers.get('content-length', 'Unknown')} bytes",
                f"Server: {response.headers.get('server', 'Unknown')}",
            ]

            if response.history:
                info.append(f"Redirects: {len(response.history)}")
                info.append(f"Final URL: {response.url}")

            return "\n".join(info)

        except requests.Timeout:
            return f"Error: Request timed out after {self.timeout} seconds"
        except requests.RequestException as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @kernel_function(name="parse_json_response", description="Parse JSON from API response")
    def parse_json_response(
        self,
        json_text: Annotated[str, "JSON text to parse"]
    ) -> Annotated[str, "Parsed JSON"]:
        """Parse and format JSON response.

        Args:
            json_text: JSON text to parse

        Returns:
            Formatted JSON or error
        """
        try:
            data = json.loads(json_text)
            formatted = json.dumps(data, indent=2)

            summary = []
            if isinstance(data, dict):
                summary.append(f"Object with {len(data)} keys")
                summary.append(f"Keys: {', '.join(list(data.keys())[:10])}")
            elif isinstance(data, list):
                summary.append(f"Array with {len(data)} items")

            return f"{' - '.join(summary)}\n\n{formatted[:2000]}{'...' if len(formatted) > 2000 else ''}"
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON - {str(e)}"

    @kernel_function(name="fetch_public_api", description="Fetch data from common public APIs")
    def fetch_public_api(
        self,
        api_name: Annotated[str, "API name: 'ipify', 'time', 'jokes'"]
    ) -> Annotated[str, "API response"]:
        """Fetch data from common public APIs.

        Args:
            api_name: Name of public API to call

        Returns:
            API response
        """
        apis = {
            "ipify": "https://api.ipify.org?format=json",
            "time": "https://worldtimeapi.org/api/timezone/Etc/UTC",
            "jokes": "https://official-joke-api.appspot.com/random_joke",
        }

        if api_name.lower() not in apis:
            return f"Error: Unknown API. Available: {', '.join(apis.keys())}"

        url = apis[api_name.lower()]
        return self.http_get(url)
