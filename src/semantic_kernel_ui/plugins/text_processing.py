"""Text processing utilities plugin."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Annotated

try:
    from semantic_kernel.functions import kernel_function
except ImportError:
    from typing import Optional

    def kernel_function(name: Optional[str] = None, description: Optional[str] = None):  # type: ignore[misc]
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func

        return decorator


class TextProcessingPlugin:
    """Plugin for text processing operations."""

    @kernel_function(name="count_words", description="Count words and characters in text")  # type: ignore[misc]
    def count_words(
        self, text: Annotated[str, "Text to analyze"]
    ) -> Annotated[str, "Word and character counts"]:
        """Count words and characters in text.

        Args:
            text: Input text

        Returns:
            Statistics about the text
        """
        words = text.split()
        chars = len(text)
        chars_no_space = len(text.replace(" ", ""))
        lines = len(text.split("\n"))
        sentences = len(
            [
                s
                for s in text.replace("!", ".").replace("?", ".").split(".")
                if s.strip()
            ]
        )

        return (
            f"Words: {len(words)}\n"
            f"Characters: {chars}\n"
            f"Characters (no spaces): {chars_no_space}\n"
            f"Lines: {lines}\n"
            f"Sentences: {sentences}"
        )

    @kernel_function(name="format_json", description="Format and validate JSON")  # type: ignore[misc]
    def format_json(
        self, json_text: Annotated[str, "JSON string to format"]
    ) -> Annotated[str, "Formatted JSON"]:
        """Format and validate JSON text.

        Args:
            json_text: JSON string

        Returns:
            Formatted JSON or error message
        """
        try:
            obj = json.loads(json_text)
            formatted = json.dumps(obj, indent=2, sort_keys=False)
            return f"Valid JSON ({len(obj) if isinstance(obj, (dict, list)) else 'N/A'} items):\n\n{formatted}"
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {str(e)}"

    @kernel_function(name="encode_base64", description="Encode text to Base64")  # type: ignore[misc]
    def encode_base64(
        self, text: Annotated[str, "Text to encode"]
    ) -> Annotated[str, "Base64 encoded string"]:
        """Encode text to Base64.

        Args:
            text: Text to encode

        Returns:
            Base64 encoded string
        """
        try:
            encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
            return f"Base64: {encoded}"
        except Exception as e:
            return f"Error encoding: {str(e)}"

    @kernel_function(name="decode_base64", description="Decode Base64 to text")  # type: ignore[misc]
    def decode_base64(
        self, encoded_text: Annotated[str, "Base64 string to decode"]
    ) -> Annotated[str, "Decoded text"]:
        """Decode Base64 to text.

        Args:
            encoded_text: Base64 encoded string

        Returns:
            Decoded text
        """
        try:
            decoded = base64.b64decode(encoded_text).decode("utf-8")
            return f"Decoded: {decoded}"
        except Exception as e:
            return f"Error decoding: {str(e)}"

    @kernel_function(name="generate_hash", description="Generate hash of text")  # type: ignore[misc]
    def generate_hash(
        self,
        text: Annotated[str, "Text to hash"],
        algorithm: Annotated[str, "Hash algorithm: md5, sha256, sha512"] = "sha256",
    ) -> Annotated[str, "Hash value"]:
        """Generate hash of text.

        Args:
            text: Input text
            algorithm: Hash algorithm (md5, sha256, sha512)

        Returns:
            Hash value
        """
        try:
            algorithms = {
                "md5": hashlib.md5,
                "sha256": hashlib.sha256,
                "sha512": hashlib.sha512,
            }

            if algorithm.lower() not in algorithms:
                return (
                    f"Error: Unsupported algorithm. Use: {', '.join(algorithms.keys())}"
                )

            hash_func = algorithms[algorithm.lower()]
            hash_value = hash_func(text.encode("utf-8")).hexdigest()

            return f"{algorithm.upper()}: {hash_value}"
        except Exception as e:
            return f"Error: {str(e)}"

    @kernel_function(name="extract_urls", description="Extract URLs from text")  # type: ignore[misc]
    def extract_urls(
        self, text: Annotated[str, "Text containing URLs"]
    ) -> Annotated[str, "Extracted URLs"]:
        """Extract URLs from text.

        Args:
            text: Input text

        Returns:
            List of URLs found
        """
        import re

        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)

        if urls:
            return f"Found {len(urls)} URL(s):\n" + "\n".join(
                f"  {i+1}. {url}" for i, url in enumerate(urls)
            )
        return "No URLs found"

    @kernel_function(name="convert_case", description="Convert text case")  # type: ignore[misc]
    def convert_case(
        self,
        text: Annotated[str, "Text to convert"],
        case_type: Annotated[str, "Case type: upper, lower, title, sentence"] = "lower",
    ) -> Annotated[str, "Converted text"]:
        """Convert text to different cases.

        Args:
            text: Input text
            case_type: Target case (upper, lower, title, sentence)

        Returns:
            Converted text
        """
        conversions = {
            "upper": text.upper(),
            "lower": text.lower(),
            "title": text.title(),
            "sentence": text.capitalize(),
        }

        result = conversions.get(case_type.lower())
        if result is None:
            return f"Error: Unknown case type. Use: {', '.join(conversions.keys())}"

        return f"{case_type.title()} case: {result}"

    @kernel_function(name="remove_extra_spaces", description="Clean up whitespace in text")  # type: ignore[misc]
    def remove_extra_spaces(
        self, text: Annotated[str, "Text to clean"]
    ) -> Annotated[str, "Cleaned text"]:
        """Remove extra whitespace from text.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        import re

        cleaned = re.sub(r"\s+", " ", text.strip())
        return cleaned

    @kernel_function(name="text_statistics", description="Get detailed text statistics")  # type: ignore[misc]
    def text_statistics(
        self, text: Annotated[str, "Text to analyze"]
    ) -> Annotated[str, "Detailed statistics"]:
        """Get detailed text statistics.

        Args:
            text: Input text

        Returns:
            Comprehensive text statistics
        """
        words = text.split()
        chars = len(text)
        unique_words = len(set(word.lower() for word in words))
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        stats = [
            f"Total words: {len(words)}",
            f"Unique words: {unique_words}",
            f"Total characters: {chars}",
            f"Average word length: {avg_word_length:.1f}",
            f"Longest word: {max(words, key=len) if words else 'N/A'} ({len(max(words, key=len)) if words else 0} chars)",
            f"Lines: {len(text.split(chr(10)))}",
        ]

        return "\n".join(stats)
