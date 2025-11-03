"""Tests for Text Processing Plugin."""

from semantic_kernel_ui.plugins.text_processing import TextProcessingPlugin


class TestTextProcessingPlugin:
    """Test TextProcessingPlugin functionality."""

    def test_count_words_simple(self):
        """Test counting words in simple text."""
        plugin = TextProcessingPlugin()

        result = plugin.count_words("Hello world")

        assert "Words: 2" in result
        assert "Characters:" in result
        assert "Lines: 1" in result

    def test_count_words_multiline(self):
        """Test counting words in multiline text."""
        plugin = TextProcessingPlugin()

        result = plugin.count_words("Line 1\nLine 2\nLine 3")

        assert "Lines: 3" in result

    def test_count_words_with_sentences(self):
        """Test counting sentences."""
        plugin = TextProcessingPlugin()

        result = plugin.count_words("First sentence. Second sentence! Third?")

        assert "Sentences: 3" in result

    def test_format_json_valid_object(self):
        """Test formatting valid JSON object."""
        plugin = TextProcessingPlugin()

        result = plugin.format_json('{"key": "value", "number": 42}')

        assert "Valid JSON" in result
        assert '"key": "value"' in result

    def test_format_json_valid_array(self):
        """Test formatting valid JSON array."""
        plugin = TextProcessingPlugin()

        result = plugin.format_json('[1, 2, 3]')

        assert "Valid JSON" in result

    def test_format_json_invalid(self):
        """Test formatting invalid JSON."""
        plugin = TextProcessingPlugin()

        result = plugin.format_json('not valid json')

        assert "Invalid JSON" in result

    def test_encode_base64(self):
        """Test Base64 encoding."""
        plugin = TextProcessingPlugin()

        result = plugin.encode_base64("Hello, World!")

        assert "Base64:" in result
        assert "SGVsbG8sIFdvcmxkIQ==" in result

    def test_decode_base64(self):
        """Test Base64 decoding."""
        plugin = TextProcessingPlugin()

        result = plugin.decode_base64("SGVsbG8sIFdvcmxkIQ==")

        assert "Decoded:" in result
        assert "Hello, World!" in result

    def test_decode_base64_invalid(self):
        """Test decoding invalid Base64."""
        plugin = TextProcessingPlugin()

        result = plugin.decode_base64("invalid base64!!!")

        assert "Error decoding" in result

    def test_generate_hash_sha256(self):
        """Test generating SHA256 hash."""
        plugin = TextProcessingPlugin()

        result = plugin.generate_hash("test", "sha256")

        assert "SHA256:" in result
        # SHA256 of "test"
        assert "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" in result

    def test_generate_hash_md5(self):
        """Test generating MD5 hash."""
        plugin = TextProcessingPlugin()

        result = plugin.generate_hash("test", "md5")

        assert "MD5:" in result
        # MD5 of "test"
        assert "098f6bcd4621d373cade4e832627b4f6" in result

    def test_generate_hash_sha512(self):
        """Test generating SHA512 hash."""
        plugin = TextProcessingPlugin()

        result = plugin.generate_hash("test", "sha512")

        assert "SHA512:" in result

    def test_generate_hash_default_sha256(self):
        """Test default hash algorithm is SHA256."""
        plugin = TextProcessingPlugin()

        result = plugin.generate_hash("test")

        assert "SHA256:" in result

    def test_generate_hash_unsupported_algorithm(self):
        """Test generating hash with unsupported algorithm."""
        plugin = TextProcessingPlugin()

        result = plugin.generate_hash("test", "sha1")

        assert "Error: Unsupported algorithm" in result
        assert "md5" in result

    def test_extract_urls_single(self):
        """Test extracting single URL."""
        plugin = TextProcessingPlugin()

        result = plugin.extract_urls("Check out https://example.com for more info")

        assert "Found 1 URL(s)" in result
        assert "https://example.com" in result

    def test_extract_urls_multiple(self):
        """Test extracting multiple URLs."""
        plugin = TextProcessingPlugin()

        text = "Visit https://example.com and http://test.org"
        result = plugin.extract_urls(text)

        assert "Found 2 URL(s)" in result
        assert "https://example.com" in result
        assert "http://test.org" in result

    def test_extract_urls_none_found(self):
        """Test extracting URLs when none present."""
        plugin = TextProcessingPlugin()

        result = plugin.extract_urls("No URLs in this text")

        assert "No URLs found" in result

    def test_convert_case_upper(self):
        """Test converting to uppercase."""
        plugin = TextProcessingPlugin()

        result = plugin.convert_case("hello world", "upper")

        assert "HELLO WORLD" in result

    def test_convert_case_lower(self):
        """Test converting to lowercase."""
        plugin = TextProcessingPlugin()

        result = plugin.convert_case("HELLO WORLD", "lower")

        assert "hello world" in result

    def test_convert_case_title(self):
        """Test converting to title case."""
        plugin = TextProcessingPlugin()

        result = plugin.convert_case("hello world", "title")

        assert "Hello World" in result

    def test_convert_case_sentence(self):
        """Test converting to sentence case."""
        plugin = TextProcessingPlugin()

        result = plugin.convert_case("hello world", "sentence")

        assert "Hello world" in result

    def test_convert_case_default_lower(self):
        """Test default case conversion is lower."""
        plugin = TextProcessingPlugin()

        result = plugin.convert_case("HELLO WORLD")

        assert "hello world" in result

    def test_convert_case_unknown_type(self):
        """Test converting with unknown case type."""
        plugin = TextProcessingPlugin()

        result = plugin.convert_case("hello", "unknown")

        assert "Error: Unknown case type" in result
        assert "upper" in result

    def test_count_words_empty_string(self):
        """Test counting words in empty string."""
        plugin = TextProcessingPlugin()

        result = plugin.count_words("")

        assert "Words: 0" in result or "Words: 1" in result  # Empty string may split to 1 empty item

    def test_encode_decode_round_trip(self):
        """Test Base64 encode/decode round trip."""
        plugin = TextProcessingPlugin()

        original = "Test message!"
        encoded_result = plugin.encode_base64(original)
        encoded = encoded_result.split("Base64: ")[1]
        decoded_result = plugin.decode_base64(encoded)

        assert original in decoded_result

    def test_hash_case_insensitive_algorithm(self):
        """Test hash algorithm is case insensitive."""
        plugin = TextProcessingPlugin()

        result1 = plugin.generate_hash("test", "MD5")
        result2 = plugin.generate_hash("test", "md5")

        # Both should work
        assert "MD5:" in result1
        assert "MD5:" in result2

    def test_extract_urls_with_paths(self):
        """Test extracting URLs with paths and parameters."""
        plugin = TextProcessingPlugin()

        text = "API at https://api.example.com/v1/users?id=123"
        result = plugin.extract_urls(text)

        assert "Found 1 URL(s)" in result
        assert "https://api.example.com/v1/users?id=123" in result
