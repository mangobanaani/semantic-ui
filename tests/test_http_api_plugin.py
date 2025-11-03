"""Tests for HTTP API Plugin."""

import json
from unittest.mock import Mock, patch

from semantic_kernel_ui.plugins.http_api import HttpApiPlugin


class TestHttpApiPlugin:
    """Test HttpApiPlugin functionality."""

    def test_initialization_defaults(self):
        """Test plugin initialization with defaults."""
        plugin = HttpApiPlugin()

        assert plugin.timeout == 10
        assert plugin.max_size == 1024 * 1024

    def test_initialization_custom(self):
        """Test plugin initialization with custom values."""
        plugin = HttpApiPlugin(timeout=30, max_size=2048)

        assert plugin.timeout == 30
        assert plugin.max_size == 2048

    def test_is_safe_url_valid_https(self):
        """Test safe URL validation with HTTPS."""
        plugin = HttpApiPlugin()

        is_valid, error = plugin._is_safe_url("https://example.com")

        assert is_valid is True
        assert error == ""

    def test_is_safe_url_valid_http(self):
        """Test safe URL validation with HTTP."""
        plugin = HttpApiPlugin()

        is_valid, error = plugin._is_safe_url("http://example.com/api")

        assert is_valid is True
        assert error == ""

    def test_is_safe_url_invalid_scheme(self):
        """Test safe URL validation with invalid scheme."""
        plugin = HttpApiPlugin()

        is_valid, error = plugin._is_safe_url("ftp://example.com")

        assert is_valid is False
        assert "HTTP/HTTPS" in error

    def test_is_safe_url_localhost(self):
        """Test safe URL validation blocks localhost."""
        plugin = HttpApiPlugin()

        is_valid, error = plugin._is_safe_url("http://localhost:8080")

        assert is_valid is False
        assert "Local URLs" in error

    def test_is_safe_url_127_0_0_1(self):
        """Test safe URL validation blocks 127.0.0.1."""
        plugin = HttpApiPlugin()

        is_valid, error = plugin._is_safe_url("http://127.0.0.1")

        assert is_valid is False
        assert "Local URLs" in error

    def test_is_safe_url_no_netloc(self):
        """Test safe URL validation with invalid format."""
        plugin = HttpApiPlugin()

        is_valid, error = plugin._is_safe_url("http://")

        assert is_valid is False
        assert "Invalid URL format" in error

    @patch("requests.get")
    def test_http_get_success_json(self, mock_get):
        """Test successful HTTP GET with JSON response."""
        plugin = HttpApiPlugin()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json",
            "content-length": "100"
        }
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        result = plugin.http_get("https://api.example.com/data")

        assert "Status: 200" in result
        assert "application/json" in result
        assert '"key": "value"' in result

    @patch("requests.get")
    def test_http_get_success_text(self, mock_get):
        """Test successful HTTP GET with text response."""
        plugin = HttpApiPlugin()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "text/plain",
            "content-length": "50"
        }
        mock_response.text = "Plain text response"
        mock_get.return_value = mock_response

        result = plugin.http_get("https://example.com")

        assert "Status: 200" in result
        assert "text/plain" in result
        assert "Plain text response" in result

    @patch("requests.get")
    def test_http_get_response_too_large(self, mock_get):
        """Test HTTP GET with response too large."""
        plugin = HttpApiPlugin(max_size=100)

        mock_response = Mock()
        mock_response.headers = {"content-length": "200"}
        mock_get.return_value = mock_response

        result = plugin.http_get("https://example.com")

        assert "Response too large" in result

    @patch("requests.get")
    def test_http_get_timeout(self, mock_get):
        """Test HTTP GET with timeout."""
        import requests
        plugin = HttpApiPlugin(timeout=5)

        mock_get.side_effect = requests.Timeout

        result = plugin.http_get("https://example.com")

        assert "timed out after 5 seconds" in result

    @patch("requests.get")
    def test_http_get_request_exception(self, mock_get):
        """Test HTTP GET with request exception."""
        import requests
        plugin = HttpApiPlugin()

        mock_get.side_effect = requests.RequestException("Network error")

        result = plugin.http_get("https://example.com")

        assert "Error:" in result
        assert "Network error" in result

    def test_http_get_unsafe_url(self):
        """Test HTTP GET with unsafe URL."""
        plugin = HttpApiPlugin()

        result = plugin.http_get("http://localhost:8080")

        assert "Error:" in result
        assert "Local URLs" in result

    @patch("requests.get")
    def test_http_get_invalid_json(self, mock_get):
        """Test HTTP GET with invalid JSON response."""
        plugin = HttpApiPlugin()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json",
            "content-length": "50"
        }
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        mock_get.return_value = mock_response

        result = plugin.http_get("https://api.example.com")

        assert "Invalid JSON response" in result

    @patch("requests.get")
    def test_http_get_long_response_truncated(self, mock_get):
        """Test HTTP GET with long response gets truncated."""
        plugin = HttpApiPlugin()

        long_text = "a" * 2000
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "text/plain",
            "content-length": "2000"
        }
        mock_response.text = long_text
        mock_get.return_value = mock_response

        result = plugin.http_get("https://example.com")

        assert "Status: 200" in result
        assert "..." in result

    @patch("requests.head")
    def test_check_url_status_success(self, mock_head):
        """Test check URL status success."""
        plugin = HttpApiPlugin()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.headers = {
            "content-type": "text/html",
            "content-length": "1000",
            "server": "nginx"
        }
        mock_response.history = []
        mock_head.return_value = mock_response

        result = plugin.check_url_status("https://example.com")

        assert "Status: 200 OK" in result
        assert "text/html" in result
        assert "1000 bytes" in result
        assert "nginx" in result

    @patch("requests.head")
    def test_check_url_status_with_redirects(self, mock_head):
        """Test check URL status with redirects."""
        plugin = HttpApiPlugin()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.headers = {
            "content-type": "text/html",
            "content-length": "1000",
            "server": "nginx"
        }
        mock_response.history = [Mock(), Mock()]
        mock_response.url = "https://example.com/final"
        mock_head.return_value = mock_response

        result = plugin.check_url_status("https://example.com")

        assert "Redirects: 2" in result
        assert "Final URL:" in result

    @patch("requests.head")
    def test_check_url_status_timeout(self, mock_head):
        """Test check URL status with timeout."""
        import requests
        plugin = HttpApiPlugin(timeout=5)

        mock_head.side_effect = requests.Timeout

        result = plugin.check_url_status("https://example.com")

        assert "timed out after 5 seconds" in result

    def test_check_url_status_unsafe_url(self):
        """Test check URL status with unsafe URL."""
        plugin = HttpApiPlugin()

        result = plugin.check_url_status("http://localhost")

        assert "Error:" in result

    def test_parse_json_response_object(self):
        """Test parsing JSON object."""
        plugin = HttpApiPlugin()

        json_text = '{"name": "test", "value": 123}'

        result = plugin.parse_json_response(json_text)

        assert "Object with 2 keys" in result
        assert "name, value" in result
        assert '"name": "test"' in result

    def test_parse_json_response_array(self):
        """Test parsing JSON array."""
        plugin = HttpApiPlugin()

        json_text = '[1, 2, 3, 4, 5]'

        result = plugin.parse_json_response(json_text)

        assert "Array with 5 items" in result

    def test_parse_json_response_invalid(self):
        """Test parsing invalid JSON."""
        plugin = HttpApiPlugin()

        json_text = 'not valid json'

        result = plugin.parse_json_response(json_text)

        assert "Error: Invalid JSON" in result

    def test_parse_json_response_large_truncated(self):
        """Test parsing large JSON gets truncated."""
        plugin = HttpApiPlugin()

        large_obj = {"key": "value" * 1000}
        json_text = json.dumps(large_obj)

        result = plugin.parse_json_response(json_text)

        assert "..." in result

    @patch("requests.get")
    def test_fetch_public_api_ipify(self, mock_get):
        """Test fetching from ipify API."""
        plugin = HttpApiPlugin()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json", "content-length": "20"}
        mock_response.json.return_value = {"ip": "1.2.3.4"}
        mock_get.return_value = mock_response

        result = plugin.fetch_public_api("ipify")

        assert "Status: 200" in result

    def test_fetch_public_api_unknown(self):
        """Test fetching from unknown API."""
        plugin = HttpApiPlugin()

        result = plugin.fetch_public_api("unknown")

        assert "Error: Unknown API" in result
        assert "ipify" in result

    def test_fetch_public_api_case_insensitive(self):
        """Test fetching with case insensitive API name."""
        plugin = HttpApiPlugin()

        # Will call http_get which will fail on actual request,
        # but the API name lookup should work
        result = plugin.fetch_public_api("IPIFY")

        # Should not get "Unknown API" error
        assert "Unknown API" not in result
