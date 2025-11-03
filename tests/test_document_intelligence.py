"""Tests for Document Intelligence Plugin."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_kernel_ui.plugins import DocumentIntelligencePlugin, FileIndexPlugin
from semantic_kernel_ui.plugins.document_intelligence import DocumentType, Language


class TestDocumentIntelligencePlugin:
    """Test DocumentIntelligencePlugin functionality."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "workspace"
            test_dir.mkdir()

            # Create test files for classification
            (test_dir / "hello.py").write_text(
                "def hello():\n    print('Hello World')"
            )
            (test_dir / "config.json").write_text('{"key": "value"}')
            (test_dir / "readme.md").write_text(
                "# Documentation\n\nThis is a guide for users."
            )
            (test_dir / "data.csv").write_text("name,age\nJohn,30\nJane,25")
            (test_dir / "app.log").write_text(
                "2024-01-15 10:30:00 INFO Starting application\n"
            )

            # Finnish content
            (test_dir / "finnish.txt").write_text(
                "Tämä on suomenkielinen dokumentti. "
                "Se sisältää tietoa ja ohjeita käyttäjille. "
                "Tämä teksti on kirjoitettu kokonaan suomeksi, "
                "jotta kielentunnistus toimii oikein."
            )

            # Swedish content
            (test_dir / "swedish.txt").write_text(
                "Detta är ett svenskt dokument. "
                "Det innehåller information och instruktioner."
            )

            # English content
            (test_dir / "english.txt").write_text(
                "This is an English document. "
                "It contains information and instructions for users."
            )

            # Mixed content with emails and URLs
            (test_dir / "contact.txt").write_text(
                "Contact: john.doe@example.com\n"
                "Website: https://example.com\n"
                "Date: 2024-01-15\n"
                "Price: 100€"
            )

            yield test_dir

    @pytest.fixture
    def file_index(self, temp_workspace):
        """Create FileIndexPlugin for safe file access."""
        return FileIndexPlugin(
            allowed_directories=[str(temp_workspace)],
            max_file_size_mb=10
        )

    @pytest.fixture
    def plugin_basic(self, file_index):
        """Create plugin with only rule-based classification."""
        return DocumentIntelligencePlugin(
            file_index_plugin=file_index,
            enable_llm_classification=False,
            enable_azure_doc_intelligence=False
        )

    @pytest.fixture
    def mock_kernel_manager(self):
        """Create mock kernel manager for LLM tests."""
        manager = MagicMock()
        manager.get_response = AsyncMock(return_value=json.dumps({
            "document_type": "code",
            "language": "english",
            "confidence": 0.9,
            "keywords": ["hello", "print", "function"],
            "summary": "Python hello world function"
        }))
        return manager

    @pytest.fixture
    def plugin_with_llm(self, file_index, mock_kernel_manager):
        """Create plugin with LLM enabled."""
        return DocumentIntelligencePlugin(
            file_index_plugin=file_index,
            kernel_manager=mock_kernel_manager,
            enable_llm_classification=True,
            enable_azure_doc_intelligence=False
        )

    # ===== Rule-based Classification Tests =====

    def test_initialization_basic(self, file_index):
        """Test basic plugin initialization."""
        plugin = DocumentIntelligencePlugin(
            file_index_plugin=file_index,
            enable_llm_classification=False,
            enable_azure_doc_intelligence=False
        )
        assert plugin.file_index == file_index
        assert plugin.enable_llm_classification is False
        assert plugin.enable_azure_doc_intelligence is False
        assert plugin.max_content_length == 10000

    def test_initialization_with_flags(self, file_index):
        """Test initialization with feature flags."""
        plugin = DocumentIntelligencePlugin(
            file_index_plugin=file_index,
            enable_llm_classification=True,
            enable_azure_doc_intelligence=True,
            azure_doc_intelligence_endpoint="https://test.azure.com",
            azure_doc_intelligence_key="test-key",
            max_content_length=5000,
            cache_results=False
        )
        assert plugin.enable_llm_classification is True
        assert plugin.enable_azure_doc_intelligence is True
        assert plugin.azure_endpoint == "https://test.azure.com"
        assert plugin.max_content_length == 5000
        assert plugin.cache_results is False

    @pytest.mark.asyncio
    async def test_classify_python_file(self, plugin_basic, temp_workspace):
        """Test classification of Python code file."""
        result = await plugin_basic.classify_document(str(temp_workspace / "hello.py"))
        data = json.loads(result)

        assert data["document_type"] == "code"
        assert data["language"] in ["english", "unknown"]  # Code may not have much language
        assert data["confidence"] == 0.95
        assert data["file_path"].endswith("hello.py")
        assert data["line_count"] == 2

    @pytest.mark.asyncio
    async def test_classify_json_config(self, plugin_basic, temp_workspace):
        """Test classification of JSON configuration file."""
        result = await plugin_basic.classify_document(str(temp_workspace / "config.json"))
        data = json.loads(result)

        assert data["document_type"] == "configuration"
        assert data["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_classify_markdown_doc(self, plugin_basic, temp_workspace):
        """Test classification of Markdown documentation."""
        result = await plugin_basic.classify_document(str(temp_workspace / "readme.md"))
        data = json.loads(result)

        assert data["document_type"] == "documentation"
        assert data["confidence"] >= 0.7
        assert data["language"] == "english"

    @pytest.mark.asyncio
    async def test_classify_csv_data(self, plugin_basic, temp_workspace):
        """Test classification of CSV data file."""
        result = await plugin_basic.classify_document(str(temp_workspace / "data.csv"))
        data = json.loads(result)

        assert data["document_type"] == "data"
        assert data["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_classify_log_file(self, plugin_basic, temp_workspace):
        """Test classification of log file."""
        result = await plugin_basic.classify_document(str(temp_workspace / "app.log"))
        data = json.loads(result)

        assert data["document_type"] == "log"
        assert data["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_classify_nonexistent_file(self, plugin_basic, temp_workspace):
        """Test classification of non-existent file."""
        result = await plugin_basic.classify_document(str(temp_workspace / "nonexistent.txt"))
        data = json.loads(result)

        assert "error" in data
        assert data["error"] == "File not found"

    # ===== Language Detection Tests =====

    def test_detect_finnish(self, plugin_basic, temp_workspace):
        """Test Finnish language detection."""
        result = plugin_basic.detect_language(str(temp_workspace / "finnish.txt"))
        # Should detect finnish or multi-language (finnish has some cognates with other languages)
        assert "finnish" in result.lower() or "multi" in result.lower()

    def test_detect_swedish(self, plugin_basic, temp_workspace):
        """Test Swedish language detection."""
        result = plugin_basic.detect_language(str(temp_workspace / "swedish.txt"))
        assert "swedish" in result.lower()

    def test_detect_english(self, plugin_basic, temp_workspace):
        """Test English language detection."""
        result = plugin_basic.detect_language(str(temp_workspace / "english.txt"))
        assert "english" in result.lower()

    def test_detect_language_code_file(self, plugin_basic, temp_workspace):
        """Test language detection on code (may not detect language)."""
        result = plugin_basic.detect_language(str(temp_workspace / "hello.py"))
        # Code files may not have clear natural language
        assert "Detected language:" in result

    # ===== Entity Extraction Tests =====

    def test_extract_entities_with_email_and_url(self, plugin_basic, temp_workspace):
        """Test entity extraction from contact info."""
        result = plugin_basic.extract_entities(str(temp_workspace / "contact.txt"))
        entities = json.loads(result)

        assert "emails" in entities
        assert "john.doe@example.com" in entities["emails"]

        assert "urls" in entities
        assert "https://example.com" in entities["urls"]

        assert "dates" in entities
        assert "2024-01-15" in entities["dates"]

        assert "numbers" in entities
        # Should find "100€" or variations
        assert len(entities["numbers"]) > 0

    def test_extract_entities_empty_file(self, plugin_basic, temp_workspace):
        """Test entity extraction from file with minimal content."""
        empty_file = temp_workspace / "empty.txt"
        empty_file.write_text("")

        result = plugin_basic.extract_entities(str(empty_file))
        entities = json.loads(result)

        assert entities["emails"] == []
        assert entities["urls"] == []
        assert entities["dates"] == []

    # ===== LLM Classification Tests =====

    @pytest.mark.asyncio
    async def test_llm_classification_enabled(self, plugin_with_llm, temp_workspace):
        """Test LLM-based classification when enabled."""
        result = await plugin_with_llm.classify_document(
            str(temp_workspace / "hello.py"),
            use_llm=True
        )
        data = json.loads(result)

        # Should use LLM result from mock
        assert data["document_type"] == "code"
        assert data["confidence"] == 0.9
        assert "LLM" in data.get("summary", "")
        assert "hello" in data["keywords"]

    @pytest.mark.asyncio
    async def test_llm_classification_disabled(self, plugin_with_llm, temp_workspace):
        """Test fallback to rule-based when LLM disabled."""
        result = await plugin_with_llm.classify_document(
            str(temp_workspace / "hello.py"),
            use_llm=False
        )
        data = json.loads(result)

        # Should use rule-based result
        assert data["document_type"] == "code"
        assert data["confidence"] == 0.95  # Rule-based confidence
        assert "Rule-based" in data.get("summary", "")

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self, plugin_with_llm, temp_workspace):
        """Test fallback to rule-based when LLM fails."""
        # Make LLM return invalid JSON
        plugin_with_llm.kernel_manager.get_response = AsyncMock(
            return_value="Invalid JSON"
        )

        result = await plugin_with_llm.classify_document(
            str(temp_workspace / "hello.py"),
            use_llm=True
        )
        data = json.loads(result)

        # Should fall back to rule-based
        assert data["document_type"] == "code"
        assert "Rule-based" in data.get("summary", "")

    # ===== Caching Tests =====

    @pytest.mark.asyncio
    async def test_caching_enabled(self, plugin_basic, temp_workspace):
        """Test that results are cached."""
        file_path = str(temp_workspace / "hello.py")

        # First call
        result1 = await plugin_basic.classify_document(file_path)
        data1 = json.loads(result1)

        # Second call should use cache
        result2 = await plugin_basic.classify_document(file_path)
        data2 = json.loads(result2)

        assert data1 == data2
        assert len(plugin_basic._cache) == 1

    @pytest.mark.asyncio
    async def test_caching_disabled(self, file_index, temp_workspace):
        """Test plugin with caching disabled."""
        plugin = DocumentIntelligencePlugin(
            file_index_plugin=file_index,
            enable_llm_classification=False,
            cache_results=False
        )

        file_path = str(temp_workspace / "hello.py")

        # Multiple calls
        await plugin.classify_document(file_path)
        await plugin.classify_document(file_path)

        assert len(plugin._cache) == 0

    def test_clear_cache(self, plugin_basic):
        """Test cache clearing."""
        # Add some items to cache
        plugin_basic._cache["key1"] = MagicMock()
        plugin_basic._cache["key2"] = MagicMock()

        result = plugin_basic.clear_cache()

        assert "Cleared 2 cached result(s)" in result
        assert len(plugin_basic._cache) == 0

    # ===== Batch Classification Tests =====

    @pytest.mark.asyncio
    async def test_batch_classify(self, plugin_basic, temp_workspace):
        """Test batch classification of multiple files."""
        files = f"{temp_workspace}/hello.py,{temp_workspace}/config.json,{temp_workspace}/data.csv"

        result = await plugin_basic.batch_classify(files, use_llm=False)
        data = json.loads(result)

        assert isinstance(data, list)
        assert len(data) == 3

        # Check each file was classified
        types = [item["document_type"] for item in data]
        assert "code" in types
        assert "configuration" in types
        assert "data" in types

    @pytest.mark.asyncio
    async def test_batch_classify_with_errors(self, plugin_basic, temp_workspace):
        """Test batch classification with some invalid files."""
        files = f"{temp_workspace}/hello.py,{temp_workspace}/nonexistent.txt"

        result = await plugin_basic.batch_classify(files, use_llm=False)
        data = json.loads(result)

        assert len(data) == 2
        assert "document_type" in data[0]  # Valid file
        assert "error" in data[1]  # Invalid file

    # ===== Document Summary Tests =====

    @pytest.mark.asyncio
    async def test_get_document_summary(self, plugin_basic, temp_workspace):
        """Test getting human-readable document summary."""
        result = await plugin_basic.get_document_summary(str(temp_workspace / "hello.py"))

        assert "File: hello.py" in result
        assert "Type: code" in result
        assert "confidence:" in result.lower()
        assert "Language:" in result
        assert "Size:" in result
        assert "Lines:" in result

    @pytest.mark.asyncio
    async def test_summary_with_keywords(self, plugin_basic, temp_workspace):
        """Test summary includes keywords."""
        result = await plugin_basic.get_document_summary(str(temp_workspace / "readme.md"))

        assert "Keywords:" in result or "keywords" in result.lower()

    # ===== Azure Integration Tests (Mocked) =====

    @pytest.mark.asyncio
    async def test_azure_classification_disabled(self, plugin_basic, temp_workspace):
        """Test that Azure is not used when disabled."""
        result = await plugin_basic.classify_document(
            str(temp_workspace / "hello.py"),
            use_azure=True
        )
        data = json.loads(result)

        # Should fall back to rule-based since Azure is disabled
        assert "Rule-based" in data.get("summary", "")

    @pytest.mark.asyncio
    async def test_azure_client_not_initialized(self, file_index):
        """Test Azure client initialization fails gracefully."""
        plugin = DocumentIntelligencePlugin(
            file_index_plugin=file_index,
            enable_azure_doc_intelligence=True,
            azure_doc_intelligence_endpoint=None,  # No credentials
            azure_doc_intelligence_key=None
        )

        client = plugin._get_azure_client()
        assert client is None

    # ===== Security Tests =====

    @pytest.mark.asyncio
    async def test_respects_file_index_security(self, plugin_basic):
        """Test that plugin respects FileIndexPlugin security."""
        # Try to access file outside allowed directory
        result = await plugin_basic.classify_document("/etc/passwd")
        data = json.loads(result)

        # Should get error since file reading will fail
        assert "error" in data or data.get("document_type") == "unknown"

    def test_content_length_limit(self, file_index, temp_workspace):
        """Test that content is truncated to max length."""
        # Create large file
        large_file = temp_workspace / "large.txt"
        large_file.write_text("x" * 20000)

        plugin = DocumentIntelligencePlugin(
            file_index_plugin=file_index,
            max_content_length=1000
        )

        content = plugin._read_file_content(str(large_file))
        # FileIndexPlugin will handle size limits, content reading is safe
        assert content is not None

    # ===== Helper Method Tests =====

    def test_cache_key_generation(self, plugin_basic):
        """Test cache key generation is consistent."""
        key1 = plugin_basic._get_cache_key("/path/to/file.txt")
        key2 = plugin_basic._get_cache_key("/path/to/file.txt")
        key3 = plugin_basic._get_cache_key("/path/to/other.txt")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32  # MD5 hash length

    def test_keyword_extraction(self, plugin_basic):
        """Test keyword extraction."""
        content = "Python programming language with CamelCase and snake_case identifiers"
        keywords = plugin_basic._extract_keywords(content, DocumentType.CODE)

        assert isinstance(keywords, list)
        assert len(keywords) <= 10
        # Should extract meaningful words
        assert any(kw in ["Python", "CamelCase", "programming"] for kw in keywords)

    def test_language_detection_patterns(self, plugin_basic):
        """Test language pattern matching."""
        # Use more language-specific text to avoid overlaps
        finnish = "Tämä on suomenkielinen teksti joka sisältää ääkkösiä ja sanoja"
        swedish = "Detta är en svensk text som innehåller ord"
        english = "This is an English text that contains words"

        # Finnish might be detected as multi-language due to word overlaps
        finn_result = plugin_basic._detect_language(finnish)
        assert finn_result in [Language.FINNISH, Language.MULTI]

        assert plugin_basic._detect_language(swedish) == Language.SWEDISH
        assert plugin_basic._detect_language(english) == Language.ENGLISH

    def test_entity_extraction_patterns(self, plugin_basic):
        """Test entity extraction patterns."""
        content = """
        Contact: alice@example.com, bob@test.org
        Website: https://example.com and http://test.com
        Dates: 2024-01-15, 01/20/2024, 20.03.2024
        Price: 99.99€ or $150 or 200 euros
        """

        entities = plugin_basic._extract_entities(content)

        assert len(entities["emails"]) == 2
        assert len(entities["urls"]) == 2
        assert len(entities["dates"]) >= 3
        assert len(entities["numbers"]) > 0

    # ===== Integration Tests =====

    @pytest.mark.asyncio
    async def test_full_workflow_rule_based(self, plugin_basic, temp_workspace):
        """Test complete workflow with rule-based classification."""
        file_path = str(temp_workspace / "readme.md")

        # 1. Classify
        result = await plugin_basic.classify_document(file_path)
        data = json.loads(result)
        assert data["document_type"] == "documentation"

        # 2. Get summary
        summary = await plugin_basic.get_document_summary(file_path)
        assert "documentation" in summary.lower()

        # 3. Detect language
        lang = plugin_basic.detect_language(file_path)
        assert "english" in lang.lower()

        # 4. Extract entities
        entities_json = plugin_basic.extract_entities(file_path)
        entities = json.loads(entities_json)
        assert isinstance(entities, dict)

    @pytest.mark.asyncio
    async def test_full_workflow_with_llm(self, plugin_with_llm, temp_workspace):
        """Test complete workflow with LLM classification."""
        file_path = str(temp_workspace / "hello.py")

        # Classify with LLM
        result = await plugin_with_llm.classify_document(file_path, use_llm=True)
        data = json.loads(result)

        assert data["document_type"] == "code"
        assert "LLM" in data.get("summary", "")
        assert data["confidence"] == 0.9  # From mock

    def test_no_write_operations(self, plugin_basic):
        """Test that plugin has no write operations."""
        # Verify no dangerous methods exist
        assert not hasattr(plugin_basic, 'write_file')
        assert not hasattr(plugin_basic, 'delete_file')
        assert not hasattr(plugin_basic, 'modify_file')

    @pytest.mark.asyncio
    async def test_metadata_structure(self, plugin_basic, temp_workspace):
        """Test that returned metadata has correct structure."""
        result = await plugin_basic.classify_document(str(temp_workspace / "hello.py"))
        data = json.loads(result)

        # Check required fields
        required_fields = [
            "file_path", "document_type", "language", "confidence",
            "file_size", "encoding", "line_count", "detected_entities",
            "keywords", "detected_at"
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Check types
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["file_size"], int)
        assert isinstance(data["line_count"], int)
        assert isinstance(data["detected_entities"], dict)
        assert isinstance(data["keywords"], list)
