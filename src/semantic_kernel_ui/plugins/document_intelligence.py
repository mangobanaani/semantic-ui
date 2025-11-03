"""Document Intelligence Plugin with multiple backends.

Supports four classification modes via feature flags:
1. Rule-based classification (fast, no API calls)
2. LLM-based classification (using existing Semantic Kernel)
3. Tesseract OCR (free, local OCR for images/PDFs)
4. Azure AI Document Intelligence (cloud, production-grade OCR)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

try:
    from semantic_kernel.functions import kernel_function
except ImportError:  # Fallback decorator

    def kernel_function(name: Optional[str] = None, description: Optional[str] = None):  # type: ignore[misc]
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func

        return decorator


# Import OCR backends
from .ocr_backends import OCRBackendFactory, OCRBackendType


class DocumentType(str, Enum):
    """Supported document types."""

    UNKNOWN = "unknown"
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    DATA = "data"
    CONTRACT = "contract"
    REPORT = "report"
    EMAIL = "email"
    SPECIFICATION = "specification"
    DRAWING = "drawing"
    LOG = "log"


class Language(str, Enum):
    """Supported languages."""

    UNKNOWN = "unknown"
    ENGLISH = "english"
    FINNISH = "finnish"
    SWEDISH = "swedish"
    MULTI = "multi-language"


@dataclass
class DocumentMetadata:
    """Document metadata structure."""

    file_path: str
    document_type: DocumentType
    language: Language
    confidence: float
    file_size: int
    encoding: str
    line_count: int
    detected_entities: Dict[str, List[str]]
    keywords: List[str]
    summary: Optional[str] = None
    detected_at: Optional[str] = None

    def __post_init__(self):  # type: ignore[misc]
        if self.detected_at is None:
            self.detected_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "document_type": (
                self.document_type.value
                if isinstance(self.document_type, Enum)
                else self.document_type
            ),
            "language": (
                self.language.value
                if isinstance(self.language, Enum)
                else self.language
            ),
            "confidence": self.confidence,
            "file_size": self.file_size,
            "encoding": self.encoding,
            "line_count": self.line_count,
            "detected_entities": self.detected_entities,
            "keywords": self.keywords,
            "summary": self.summary,
            "detected_at": self.detected_at,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class DocumentIntelligencePlugin:
    """Document intelligence with multiple classification backends.

    Features (controlled by flags):
    - Rule-based classification (always available, no API calls)
    - LLM-based classification (requires Semantic Kernel)
    - Azure AI Document Intelligence (requires Azure subscription)

    Security features:
    - Read-only operations
    - Path validation via FileIndexPlugin
    - Size limits for text extraction
    - Caching to avoid redundant processing
    """

    def __init__(
        self,
        file_index_plugin: Optional[Any] = None,
        kernel_manager: Optional[Any] = None,
        enable_llm_classification: bool = True,
        enable_tesseract_ocr: bool = True,
        enable_azure_doc_intelligence: bool = False,
        azure_doc_intelligence_endpoint: Optional[str] = None,
        azure_doc_intelligence_key: Optional[str] = None,
        tesseract_cmd: Optional[str] = None,
        tesseract_languages: str = "eng+fin+swe",
        max_content_length: int = 10000,
        cache_results: bool = True,
    ):
        """Initialize document intelligence plugin.

        Args:
            file_index_plugin: FileIndexPlugin instance for file access
            kernel_manager: KernelManager instance for LLM classification
            enable_llm_classification: Enable LLM-based classification
            enable_tesseract_ocr: Enable Tesseract OCR (free, local)
            enable_azure_doc_intelligence: Enable Azure AI Document Intelligence
            azure_doc_intelligence_endpoint: Azure endpoint URL
            azure_doc_intelligence_key: Azure API key
            tesseract_cmd: Path to tesseract executable (None for auto-detect)
            tesseract_languages: Languages for Tesseract (e.g., "eng+fin+swe")
            max_content_length: Maximum content length for analysis
            cache_results: Enable result caching
        """
        self.file_index = file_index_plugin
        self.kernel_manager = kernel_manager
        self.max_content_length = max_content_length
        self.cache_results = cache_results

        # Feature flags
        self.enable_llm_classification = enable_llm_classification
        self.enable_tesseract_ocr = enable_tesseract_ocr
        self.enable_azure_doc_intelligence = enable_azure_doc_intelligence

        # Initialize OCR factory
        self.ocr_factory = OCRBackendFactory()

        # Register Tesseract if enabled
        if enable_tesseract_ocr:
            self.ocr_factory.register_tesseract(
                tesseract_cmd=tesseract_cmd, language=tesseract_languages
            )

        # Register Azure if enabled
        if enable_azure_doc_intelligence:
            endpoint = azure_doc_intelligence_endpoint or os.getenv(
                "AZURE_DOC_INTELLIGENCE_ENDPOINT"
            )
            key = azure_doc_intelligence_key or os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
            if endpoint and key:
                self.ocr_factory.register_azure(endpoint=endpoint, api_key=key)

        # Legacy Azure client support (deprecated, use OCR factory instead)
        self.azure_endpoint = azure_doc_intelligence_endpoint or os.getenv(
            "AZURE_DOC_INTELLIGENCE_ENDPOINT"
        )
        self.azure_key = azure_doc_intelligence_key or os.getenv(
            "AZURE_DOC_INTELLIGENCE_KEY"
        )
        self._azure_client = None

        # Result cache
        self._cache: Dict[str, DocumentMetadata] = {}

        # Language patterns
        self._language_patterns = {
            Language.FINNISH: [
                r"\b(ja|tai|ei|on|oli|olla|että|jos|kun|vaan)\b",
                r"\b(tämä|tuo|se|nämä|nuo|ne)\b",
                r"[äöåÄÖÅ]",
            ],
            Language.SWEDISH: [
                r"\b(och|eller|inte|är|var|att|om|när|men)\b",
                r"\b(denna|detta|den|dessa|de)\b",
                r"[åäöÅÄÖ]",
            ],
            Language.ENGLISH: [
                r"\b(the|and|or|not|is|was|be|that|if|when|but)\b",
                r"\b(this|that|these|those)\b",
            ],
        }

    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for file."""
        return hashlib.md5(file_path.encode()).hexdigest()

    def _get_azure_client(self):
        """Lazy initialization of Azure Document Intelligence client."""
        if not self.enable_azure_doc_intelligence:
            return None

        if self._azure_client is None and self.azure_endpoint and self.azure_key:
            try:
                from azure.ai.formrecognizer import DocumentAnalysisClient
                from azure.core.credentials import AzureKeyCredential

                self._azure_client = DocumentAnalysisClient(
                    endpoint=self.azure_endpoint,
                    credential=AzureKeyCredential(self.azure_key),
                )
            except ImportError:
                return None
            except Exception as e:
                print(f"Warning: Failed to initialize Azure client: {e}")
                return None

        return self._azure_client

    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content safely using FileIndexPlugin."""
        if self.file_index:
            result = self.file_index.read_file(file_path)
            if (
                "Access denied" in result
                or "Error" in result
                or "File not found" in result
            ):
                return None
            # Extract content from FileIndexPlugin output
            # Format: "File: path (N lines, M bytes)\n\ncontent"
            parts = result.split("\n\n", 1)
            if len(parts) == 2:
                return parts[1]
            return result
        else:
            # Fallback: direct file read
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read(self.max_content_length)
            except Exception:
                return None

    def _rule_based_classify(
        self, file_path: str, content: Optional[str] = None
    ) -> DocumentMetadata:
        """Fast rule-based classification without API calls."""
        path = Path(file_path)
        ext = path.suffix.lower()

        # Read content if not provided
        if content is None:
            content = self._read_file_content(file_path) or ""

        # Truncate content for analysis
        content_sample = content[: self.max_content_length]

        # Determine document type by extension and content
        doc_type = DocumentType.UNKNOWN
        confidence = 0.5

        # Code files
        if ext in [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
        ]:
            doc_type = DocumentType.CODE
            confidence = 0.95
        # Configuration
        elif ext in [
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".conf",
            ".cfg",
            ".env",
        ]:
            doc_type = DocumentType.CONFIGURATION
            confidence = 0.95
        # Documentation
        elif ext in [".md", ".txt", ".rst", ".adoc"]:
            if any(
                keyword in content_sample.lower()
                for keyword in ["readme", "documentation", "guide", "manual"]
            ):
                doc_type = DocumentType.DOCUMENTATION
                confidence = 0.9
            else:
                doc_type = DocumentType.DOCUMENTATION
                confidence = 0.7
        # Data files
        elif ext in [".csv", ".tsv", ".json", ".xml"]:
            doc_type = DocumentType.DATA
            confidence = 0.9
        # Logs
        elif ext in [".log"]:
            doc_type = DocumentType.LOG
            confidence = 0.95
        # Office/PDF documents
        elif ext in [".pdf", ".docx", ".doc", ".odt"]:
            # Need content analysis or Azure to determine if contract/report/spec
            if any(
                keyword in path.name.lower()
                for keyword in ["contract", "agreement", "terms"]
            ):
                doc_type = DocumentType.CONTRACT
                confidence = 0.7
            elif any(
                keyword in path.name.lower()
                for keyword in ["report", "analysis", "summary"]
            ):
                doc_type = DocumentType.REPORT
                confidence = 0.7
            elif any(
                keyword in path.name.lower()
                for keyword in ["spec", "requirement", "design"]
            ):
                doc_type = DocumentType.SPECIFICATION
                confidence = 0.7

        # Detect language
        language = self._detect_language(content_sample)

        # Extract basic entities
        entities = self._extract_entities(content_sample)

        # Extract keywords
        keywords = self._extract_keywords(content_sample, doc_type)

        # Get file stats
        try:
            file_size = os.path.getsize(file_path)
            line_count = content.count("\n") + 1
        except Exception:
            file_size = 0
            line_count = 0

        return DocumentMetadata(
            file_path=file_path,
            document_type=doc_type,
            language=language,
            confidence=confidence,
            file_size=file_size,
            encoding="utf-8",
            line_count=line_count,
            detected_entities=entities,
            keywords=keywords,
        )

    def _detect_language(self, content: str) -> Language:
        """Detect text language using pattern matching."""
        if not content:
            return Language.UNKNOWN

        content_lower = content.lower()[:2000]  # Sample first 2000 chars

        scores = {
            lang: 0 for lang in [Language.FINNISH, Language.SWEDISH, Language.ENGLISH]
        }

        for lang, patterns in self._language_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                scores[lang] += matches

        # Determine primary language
        max_score = max(scores.values())
        if max_score == 0:
            return Language.UNKNOWN

        # Check for multi-language content (threshold: 50% of max score)
        high_scores = [
            lang for lang, score in scores.items() if score > max_score * 0.5
        ]
        if len(high_scores) > 1:
            return Language.MULTI

        return max(scores, key=scores.get)  # type: ignore[arg-type, return-value]

    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns."""
        entities: Dict[str, List[str]] = {
            "emails": [],
            "urls": [],
            "dates": [],
            "numbers": [],
        }

        if not content:
            return entities

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities["emails"] = list(set(re.findall(email_pattern, content)))[:10]

        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        entities["urls"] = list(set(re.findall(url_pattern, content)))[:10]

        # Dates (various formats)
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",  # 2024-01-15
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # 01/15/2024
            r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",  # 15.01.2024
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content))
        entities["dates"] = list(set(dates))[:10]

        # Numbers with context (prices, percentages, etc.)
        number_pattern = r"\b\d+[.,]?\d*\s*(?:%|€|USD|EUR|dollars?|euros?)?\b"
        entities["numbers"] = list(
            set(re.findall(number_pattern, content, re.IGNORECASE))
        )[:10]

        return entities

    def _extract_keywords(self, content: str, doc_type: DocumentType) -> List[str]:
        """Extract relevant keywords based on document type."""
        if not content:
            return []

        # Simple keyword extraction: common important words
        words = re.findall(
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", content
        )  # CamelCase words
        words.extend(re.findall(r"\b[a-z_]+\b", content))  # snake_case words

        # Filter by length and frequency
        from collections import Counter

        word_counts = Counter(words)

        # Get top keywords
        common_words = {
            "the",
            "and",
            "or",
            "is",
            "in",
            "to",
            "of",
            "for",
            "with",
            "on",
            "at",
            "by",
            "from",
        }
        keywords = [
            word
            for word, count in word_counts.most_common(20)
            if len(word) > 3 and word.lower() not in common_words
        ]

        return keywords[:10]

    async def _llm_based_classify(
        self, file_path: str, content: str
    ) -> Optional[DocumentMetadata]:
        """LLM-based classification using Semantic Kernel."""
        if not self.enable_llm_classification or not self.kernel_manager:
            return None

        # Truncate content for LLM
        content_sample = content[: self.max_content_length]

        prompt = f"""Analyze this document and provide classification:

File: {Path(file_path).name}
Content sample (first {len(content_sample)} characters):
---
{content_sample}
---

Provide a JSON response with:
1. document_type: one of [code, documentation, configuration, data, contract, report, email, specification, drawing, log, unknown]
2. language: one of [english, finnish, swedish, multi-language, unknown]
3. confidence: float 0.0-1.0
4. keywords: list of 5-10 important keywords
5. summary: brief 1-2 sentence summary (optional)

Response (JSON only):"""

        try:
            response = await self.kernel_manager.get_response(
                prompt=prompt,
                temperature=0.1,  # Low temperature for factual classification
                max_tokens=500,
            )

            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
            )
            if json_match:
                response = json_match.group(1)

            result = json.loads(response)

            # Get file stats
            file_size = os.path.getsize(file_path)
            line_count = content.count("\n") + 1

            return DocumentMetadata(
                file_path=file_path,
                document_type=DocumentType(result.get("document_type", "unknown")),
                language=Language(result.get("language", "unknown")),
                confidence=float(result.get("confidence", 0.5)),
                file_size=file_size,
                encoding="utf-8",
                line_count=line_count,
                detected_entities={},  # LLM doesn't extract entities in this flow
                keywords=result.get("keywords", []),
                summary=result.get("summary"),
            )
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return None

    async def _azure_classify(self, file_path: str) -> Optional[DocumentMetadata]:
        """Classify using Azure AI Document Intelligence."""
        if not self.enable_azure_doc_intelligence:
            return None

        client = self._get_azure_client()
        if not client:
            return None

        try:
            # Read file as bytes
            with open(file_path, "rb") as f:
                document = f.read()

            # Analyze document with Azure
            poller = client.begin_analyze_document("prebuilt-document", document)
            result = poller.result()

            # Extract text content
            content = (
                " ".join([page.content for page in result.pages])
                if result.pages
                else ""
            )

            # Determine document type based on Azure analysis
            doc_type = DocumentType.UNKNOWN
            confidence = 0.8

            # Check for specific document patterns
            if result.key_value_pairs and len(result.key_value_pairs) > 5:
                # Likely a form or structured document
                doc_type = DocumentType.SPECIFICATION
            elif result.tables:
                doc_type = DocumentType.REPORT

            # Extract entities from Azure results
            entities = {
                "key_value_pairs": [
                    f"{kv.key.content}: {kv.value.content if kv.value else ''}"
                    for kv in (result.key_value_pairs or [])[:10]
                ],
                "tables": [
                    f"Table with {len(table.cells)} cells"
                    for table in (result.tables or [])[:5]
                ],
            }

            # Detect language
            language = self._detect_language(content)

            return DocumentMetadata(
                file_path=file_path,
                document_type=doc_type,
                language=language,
                confidence=confidence,
                file_size=os.path.getsize(file_path),
                encoding="utf-8",
                line_count=content.count("\n") + 1,
                detected_entities=entities,
                keywords=self._extract_keywords(content, doc_type),
                summary=(
                    f"Document with {len(result.pages)} pages" if result.pages else None
                ),
            )
        except Exception as e:
            print(f"Azure classification failed: {e}")
            return None

    @kernel_function(  # type: ignore[misc]
        name="classify_document",
        description="Classify document type, language, and extract metadata",
    )
    async def classify_document(
        self,
        file_path: Annotated[str, "Path to file to classify"],
        use_llm: Annotated[bool, "Use LLM for classification"] = True,
        use_azure: Annotated[bool, "Use Azure AI Document Intelligence"] = False,
    ) -> Annotated[str, "Document classification result (JSON)"]:
        """Classify document and extract metadata.

        Classification strategy (in order):
        1. Check cache if enabled
        2. Try Azure AI Document Intelligence if enabled and requested
        3. Try LLM-based classification if enabled and requested
        4. Fall back to rule-based classification

        Args:
            file_path: Path to document to classify
            use_llm: Whether to use LLM classification
            use_azure: Whether to use Azure classification

        Returns:
            JSON string with classification results
        """
        # Check cache
        cache_key = self._get_cache_key(file_path)
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key].to_json()

        # Check file exists
        if not os.path.exists(file_path):
            return json.dumps({"error": "File not found"})

        result: Optional[DocumentMetadata] = None

        # Try Azure first (most accurate for PDFs/scans)
        if use_azure and self.enable_azure_doc_intelligence:
            result = await self._azure_classify(file_path)
            if result:
                result.summary = f"{result.summary or ''} [Azure AI]".strip()

        # Try LLM if Azure failed or not requested
        if not result and use_llm and self.enable_llm_classification:
            content = self._read_file_content(file_path)
            if content:
                result = await self._llm_based_classify(file_path, content)
                if result:
                    result.summary = f"{result.summary or ''} [LLM]".strip()

        # Fall back to rule-based
        if not result:
            result = self._rule_based_classify(file_path)
            result.summary = f"{result.summary or ''} [Rule-based]".strip()

        # Cache result
        if self.cache_results:
            self._cache[cache_key] = result

        return result.to_json()

    @kernel_function(  # type: ignore[misc]
        name="batch_classify", description="Classify multiple documents at once"
    )
    async def batch_classify(
        self,
        file_paths: Annotated[str, "Comma-separated list of file paths"],
        use_llm: Annotated[bool, "Use LLM for classification"] = False,
    ) -> Annotated[str, "Batch classification results (JSON)"]:
        """Classify multiple documents.

        Args:
            file_paths: Comma-separated file paths
            use_llm: Whether to use LLM (slower but more accurate)

        Returns:
            JSON array of classification results
        """
        paths = [p.strip() for p in file_paths.split(",")]
        results = []

        for path in paths:
            try:
                result = await self.classify_document(path, use_llm=use_llm)
                results.append(json.loads(result))
            except Exception as e:
                results.append({"file_path": path, "error": str(e)})

        return json.dumps(results, indent=2)

    @kernel_function(  # type: ignore[misc]
        name="detect_language", description="Detect the language of a document"
    )
    def detect_language(
        self, file_path: Annotated[str, "Path to file"]
    ) -> Annotated[str, "Detected language"]:
        """Detect document language.

        Args:
            file_path: Path to document

        Returns:
            Detected language name
        """
        content = self._read_file_content(file_path)
        if not content:
            return "Error: Could not read file"

        language = self._detect_language(content)
        return f"Detected language: {language.value}"

    @kernel_function(  # type: ignore[misc]
        name="extract_entities",
        description="Extract entities (emails, URLs, dates) from document",
    )
    def extract_entities(
        self, file_path: Annotated[str, "Path to file"]
    ) -> Annotated[str, "Extracted entities (JSON)"]:
        """Extract entities from document.

        Args:
            file_path: Path to document

        Returns:
            JSON string with extracted entities
        """
        content = self._read_file_content(file_path)
        if content is None:
            return json.dumps({"error": "Could not read file"})

        # Empty file is valid, just return empty entities
        entities = self._extract_entities(content)
        return json.dumps(entities, indent=2)

    @kernel_function(  # type: ignore[misc]
        name="get_document_summary",
        description="Get a brief summary of document metadata",
    )
    async def get_document_summary(
        self, file_path: Annotated[str, "Path to file"]
    ) -> Annotated[str, "Document summary"]:
        """Get a human-readable summary of document.

        Args:
            file_path: Path to document

        Returns:
            Formatted summary string
        """
        result = await self.classify_document(file_path, use_llm=False)
        metadata = json.loads(result)

        if "error" in metadata:
            return metadata["error"]

        summary_lines = [
            f"File: {Path(file_path).name}",
            f"Type: {metadata['document_type']} (confidence: {metadata['confidence']:.0%})",
            f"Language: {metadata['language']}",
            f"Size: {metadata['file_size']:,} bytes",
            f"Lines: {metadata['line_count']:,}",
        ]

        if metadata.get("keywords"):
            summary_lines.append(f"Keywords: {', '.join(metadata['keywords'][:5])}")

        if metadata.get("summary"):
            summary_lines.append(f"Summary: {metadata['summary']}")

        return "\n".join(summary_lines)

    @kernel_function(  # type: ignore[misc]
        name="clear_cache", description="Clear the classification result cache"
    )
    def clear_cache(self) -> Annotated[str, "Cache clear status"]:
        """Clear the classification cache.

        Returns:
            Status message
        """
        count = len(self._cache)
        self._cache.clear()
        return f"Cleared {count} cached result(s)"

    @kernel_function(  # type: ignore[misc]
        name="ocr_document",
        description="Extract text from images and PDFs using OCR (Tesseract or Azure AI)",
    )
    def ocr_document(
        self,
        file_path: Annotated[str, "Path to image or PDF file"],
        preferred_backend: Annotated[
            str, "Preferred OCR backend: 'tesseract' or 'azure'"
        ] = "auto",
    ) -> Annotated[str, "OCR result with extracted text (JSON)"]:
        """Extract text from images and PDFs using OCR.

        Supports multiple backends via factory pattern:
        - Tesseract OCR (free, local)
        - Azure AI Document Intelligence (cloud, production-grade)

        Args:
            file_path: Path to document (image or PDF)
            preferred_backend: 'tesseract', 'azure', or 'auto' for automatic selection

        Returns:
            JSON string with OCR results
        """
        try:
            # Determine backend preference
            backend_type = None
            if preferred_backend.lower() == "tesseract":
                backend_type = OCRBackendType.TESSERACT
            elif preferred_backend.lower() == "azure":
                backend_type = OCRBackendType.AZURE

            # Process document
            result = self.ocr_factory.process_document(
                file_path, preferred_backend=backend_type
            )

            # Format result
            output = {
                "file_path": file_path,
                "backend": result.backend,
                "page_count": result.page_count,
                "average_confidence": f"{result.average_confidence:.2%}",
                "full_text": (
                    result.full_text[:1000] + "..."
                    if len(result.full_text) > 1000
                    else result.full_text
                ),
                "full_text_length": len(result.full_text),
                "metadata": result.metadata,
                "pages": [
                    {
                        "page_number": p.page_number,
                        "confidence": f"{p.confidence:.2%}",
                        "text_preview": (
                            p.text[:200] + "..." if len(p.text) > 200 else p.text
                        ),
                        "text_length": len(p.text),
                    }
                    for p in result.pages
                ],
            }

            return json.dumps(output, indent=2)

        except Exception as e:
            return json.dumps(
                {
                    "error": str(e),
                    "file_path": file_path,
                    "available_backends": [
                        b.get_name() for b in self.ocr_factory.get_available_backends()
                    ],
                },
                indent=2,
            )

    @kernel_function(  # type: ignore[misc]
        name="get_ocr_status",
        description="Check status of OCR backends (Tesseract and Azure AI)",
    )
    def get_ocr_status(self) -> Annotated[str, "OCR backend status information"]:
        """Get status of available OCR backends.

        Returns:
            Formatted status string with backend availability
        """
        available = self.ocr_factory.get_available_backends()

        status_lines = ["OCR Backend Status:", "=" * 40]

        # Tesseract status
        tesseract = self.ocr_factory.get_backend(OCRBackendType.TESSERACT)
        if tesseract:
            status_lines.append("✓ Tesseract OCR: Available")
            status_lines.append(f"  Name: {tesseract.get_name()}")
        else:
            status_lines.append("✗ Tesseract OCR: Not available")
            if self.enable_tesseract_ocr:
                status_lines.append(
                    "  Hint: Install with: pip install pytesseract pillow pdf2image"
                )
                status_lines.append("  System: Install tesseract-ocr binary")

        # Azure status
        azure = self.ocr_factory.get_backend(OCRBackendType.AZURE)
        if azure:
            status_lines.append("✓ Azure AI: Available")
            status_lines.append(f"  Name: {azure.get_name()}")
        else:
            status_lines.append("✗ Azure AI: Not available")
            if self.enable_azure_doc_intelligence:
                status_lines.append(
                    "  Hint: Install with: pip install azure-ai-formrecognizer"
                )
                status_lines.append(
                    "  Config: Set AZURE_DOC_INTELLIGENCE_ENDPOINT and KEY"
                )

        status_lines.append("")
        status_lines.append(f"Total available backends: {len(available)}")

        return "\n".join(status_lines)
