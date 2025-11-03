"""OCR Backend abstraction and implementations.

Provides a factory pattern for different OCR backends:
- Tesseract OCR (free, local)
- Azure AI Document Intelligence (cloud, production-grade)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class OCRBackendType(str, Enum):
    """Available OCR backend types."""

    TESSERACT = "tesseract"
    AZURE = "azure"


@dataclass
class OCRPage:
    """Represents a page of OCR results."""

    page_number: int
    text: str
    confidence: float
    language: Optional[str] = None
    bounding_boxes: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if self.bounding_boxes is None:
            self.bounding_boxes = []


@dataclass
class OCRResult:
    """OCR processing result."""

    pages: List[OCRPage]
    full_text: str
    metadata: Dict[str, Any]
    backend: str

    @property
    def page_count(self) -> int:
        """Get number of pages."""
        return len(self.pages)

    @property
    def average_confidence(self) -> float:
        """Get average confidence across all pages."""
        if not self.pages:
            return 0.0
        return sum(p.confidence for p in self.pages) / len(self.pages)


class OCRBackend(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and configured."""
        pass

    @abstractmethod
    def supports_format(self, file_path: str) -> bool:
        """Check if backend supports the file format."""
        pass

    @abstractmethod
    def process_document(self, file_path: str, **kwargs: Any) -> OCRResult:
        """Process document and return OCR results."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get backend name for identification."""
        pass


class TesseractOCRBackend(OCRBackend):
    """Tesseract OCR backend (free, local).

    Requires: pytesseract and Pillow
    Installation: pip install pytesseract pillow
    System requirement: tesseract-ocr binary
    """

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        language: str = "eng+fin+swe",
        config: str = "",
    ):
        """Initialize Tesseract backend.

        Args:
            tesseract_cmd: Path to tesseract executable (None for auto-detect)
            language: Languages to use (e.g., "eng+fin+swe")
            config: Additional tesseract config options
        """
        self.language = language
        self.config = config
        self._tesseract = None
        self._pil = None
        self._pdf2image = None

        # Try to import dependencies
        try:
            import pytesseract

            self._tesseract = pytesseract

            # Set custom tesseract command if provided
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        except ImportError:
            pass

        try:
            from PIL import Image

            self._pil = Image
        except ImportError:
            pass

        try:
            from pdf2image import convert_from_path

            self._pdf2image = convert_from_path
        except ImportError:
            pass

    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        if not self._tesseract or not self._pil:
            return False

        try:
            # Test if tesseract binary is accessible
            version = self._tesseract.get_tesseract_version()
            return version is not None
        except Exception:
            return False

    def supports_format(self, file_path: str) -> bool:
        """Check if file format is supported."""
        ext = Path(file_path).suffix.lower()
        # Tesseract supports images directly
        image_formats = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
        # PDF requires pdf2image
        if ext == ".pdf":
            return self._pdf2image is not None
        return ext in image_formats

    def process_document(self, file_path: str, **kwargs: Any) -> OCRResult:
        """Process document with Tesseract.

        Args:
            file_path: Path to document (image or PDF)
            **kwargs: Additional options (dpi for PDF conversion)

        Returns:
            OCRResult with extracted text and metadata
        """
        if not self.is_available():
            raise RuntimeError("Tesseract OCR is not available")

        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return self._process_pdf(file_path, **kwargs)
        else:
            return self._process_image(file_path)

    def _process_image(self, file_path: str) -> OCRResult:
        """Process a single image."""
        if not self._pil or not self._tesseract:
            raise RuntimeError("PIL and pytesseract are not available")

        try:
            image = self._pil.open(file_path)

            # Extract text
            text = self._tesseract.image_to_string(
                image, lang=self.language, config=self.config
            )

            # Get confidence data
            data = self._tesseract.image_to_data(
                image, lang=self.language, output_type=self._tesseract.Output.DICT
            )

            # Calculate average confidence (filter out -1 which means no text)
            confidences = [c for c in data["conf"] if c != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            page = OCRPage(
                page_number=1,
                text=text,
                confidence=avg_confidence / 100.0,  # Convert to 0-1 range
                language=self.language,
            )

            return OCRResult(
                pages=[page],
                full_text=text,
                metadata={
                    "backend": "tesseract",
                    "language": self.language,
                    "file_format": Path(file_path).suffix,
                },
                backend="tesseract",
            )

        except Exception as e:
            raise RuntimeError(f"Tesseract OCR failed: {e}")

    def _process_pdf(self, file_path: str, dpi: int = 300) -> OCRResult:
        """Process PDF by converting to images first."""
        if not self._pdf2image or not self._tesseract:
            raise RuntimeError(
                "pdf2image and pytesseract are required for PDF processing"
            )

        try:
            # Convert PDF to images
            images = self._pdf2image(file_path, dpi=dpi)

            pages = []
            all_text = []

            for i, image in enumerate(images, start=1):
                # Extract text from each page
                text = self._tesseract.image_to_string(
                    image, lang=self.language, config=self.config
                )

                # Get confidence
                data = self._tesseract.image_to_data(
                    image, lang=self.language, output_type=self._tesseract.Output.DICT
                )
                confidences = [c for c in data["conf"] if c != -1]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )

                pages.append(
                    OCRPage(
                        page_number=i,
                        text=text,
                        confidence=avg_confidence / 100.0,
                        language=self.language,
                    )
                )

                all_text.append(text)

            return OCRResult(
                pages=pages,
                full_text="\n\n".join(all_text),
                metadata={
                    "backend": "tesseract",
                    "language": self.language,
                    "page_count": len(pages),
                    "dpi": dpi,
                    "file_format": ".pdf",
                },
                backend="tesseract",
            )

        except Exception as e:
            raise RuntimeError(f"PDF OCR failed: {e}")

    def get_name(self) -> str:
        """Get backend name."""
        return "Tesseract OCR"


class AzureOCRBackend(OCRBackend):
    """Azure AI Document Intelligence backend (cloud, production-grade).

    Requires: azure-ai-formrecognizer
    Installation: pip install azure-ai-formrecognizer
    """

    def __init__(self, endpoint: str, api_key: str, model: str = "prebuilt-read"):
        """Initialize Azure backend.

        Args:
            endpoint: Azure endpoint URL
            api_key: Azure API key
            model: Model to use (prebuilt-read, prebuilt-document, etc.)
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self._client = None

        # Try to import Azure SDK
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential

            if endpoint and api_key:
                self._client = DocumentAnalysisClient(
                    endpoint=endpoint, credential=AzureKeyCredential(api_key)
                )
        except ImportError:
            pass
        except Exception:
            pass

    def is_available(self) -> bool:
        """Check if Azure backend is available."""
        return self._client is not None

    def supports_format(self, file_path: str) -> bool:
        """Check if file format is supported."""
        ext = Path(file_path).suffix.lower()
        # Azure supports many formats
        supported = {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".bmp",
            ".heif",
            ".docx",
            ".xlsx",
            ".pptx",
            ".html",
        }
        return ext in supported

    def process_document(self, file_path: str, **kwargs: Any) -> OCRResult:
        """Process document with Azure AI.

        Args:
            file_path: Path to document
            **kwargs: Additional options

        Returns:
            OCRResult with extracted text and metadata
        """
        if not self.is_available() or not self._client:
            raise RuntimeError("Azure AI Document Intelligence is not available")

        try:
            # Read file
            with open(file_path, "rb") as f:
                document = f.read()

            # Analyze document
            poller = self._client.begin_analyze_document(self.model, document)
            result = poller.result()

            # Extract pages
            pages = []
            all_text = []

            for i, page in enumerate(result.pages, start=1):
                page_text = []

                # Extract text from lines
                for line in page.lines:
                    page_text.append(line.content)

                text = "\n".join(page_text)
                all_text.append(text)

                # Get confidence (Azure provides per-word confidence)
                confidences = []
                for word in page.words:
                    if hasattr(word, "confidence") and word.confidence is not None:
                        confidences.append(word.confidence)

                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.9
                )

                pages.append(
                    OCRPage(
                        page_number=i,
                        text=text,
                        confidence=avg_confidence,
                        language=getattr(page, "language", None),
                    )
                )

            # Extract additional metadata
            metadata = {
                "backend": "azure",
                "model": self.model,
                "page_count": len(result.pages),
                "file_format": Path(file_path).suffix,
            }

            # Add key-value pairs if available
            if result.key_value_pairs:
                metadata["key_value_pair_count"] = len(result.key_value_pairs)

            # Add tables if available
            if result.tables:
                metadata["table_count"] = len(result.tables)

            return OCRResult(
                pages=pages,
                full_text="\n\n".join(all_text),
                metadata=metadata,
                backend="azure",
            )

        except Exception as e:
            raise RuntimeError(f"Azure OCR failed: {e}")

    def get_name(self) -> str:
        """Get backend name."""
        return "Azure AI Document Intelligence"


class OCRBackendFactory:
    """Factory for creating and managing OCR backends."""

    def __init__(self) -> None:
        """Initialize factory."""
        self._backends: Dict[OCRBackendType, OCRBackend] = {}

    def register_tesseract(
        self,
        tesseract_cmd: Optional[str] = None,
        language: str = "eng+fin+swe",
        config: str = "",
    ) -> None:
        """Register Tesseract OCR backend.

        Args:
            tesseract_cmd: Path to tesseract executable
            language: Languages to use
            config: Additional config
        """
        backend = TesseractOCRBackend(
            tesseract_cmd=tesseract_cmd, language=language, config=config
        )
        self._backends[OCRBackendType.TESSERACT] = backend

    def register_azure(
        self, endpoint: str, api_key: str, model: str = "prebuilt-read"
    ) -> None:
        """Register Azure AI backend.

        Args:
            endpoint: Azure endpoint URL
            api_key: Azure API key
            model: Model to use
        """
        backend = AzureOCRBackend(endpoint=endpoint, api_key=api_key, model=model)
        self._backends[OCRBackendType.AZURE] = backend

    def get_backend(self, backend_type: OCRBackendType) -> Optional[OCRBackend]:
        """Get a registered backend.

        Args:
            backend_type: Type of backend to retrieve

        Returns:
            OCR backend if registered and available, None otherwise
        """
        backend = self._backends.get(backend_type)
        if backend and backend.is_available():
            return backend
        return None

    def get_available_backends(self) -> List[OCRBackend]:
        """Get list of all available backends.

        Returns:
            List of available OCR backends
        """
        return [
            backend for backend in self._backends.values() if backend.is_available()
        ]

    def get_best_backend_for_file(self, file_path: str) -> Optional[OCRBackend]:
        """Get the best available backend for a file.

        Priority: Azure (most accurate) -> Tesseract (local/free)

        Args:
            file_path: Path to file

        Returns:
            Best available backend for the file format
        """
        # Try Azure first (most accurate)
        azure = self.get_backend(OCRBackendType.AZURE)
        if azure and azure.supports_format(file_path):
            return azure

        # Fall back to Tesseract
        tesseract = self.get_backend(OCRBackendType.TESSERACT)
        if tesseract and tesseract.supports_format(file_path):
            return tesseract

        return None

    def process_document(
        self,
        file_path: str,
        preferred_backend: Optional[OCRBackendType] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Process document with automatic backend selection.

        Args:
            file_path: Path to document
            preferred_backend: Preferred backend (will fall back if not available)
            **kwargs: Additional options passed to backend

        Returns:
            OCRResult from processing

        Raises:
            RuntimeError: If no backend can process the file
        """
        # Try preferred backend first
        if preferred_backend:
            backend = self.get_backend(preferred_backend)
            if backend and backend.supports_format(file_path):
                return backend.process_document(file_path, **kwargs)

        # Auto-select best backend
        backend = self.get_best_backend_for_file(file_path)
        if not backend:
            raise RuntimeError(
                f"No OCR backend available for file: {file_path}. "
                f"Available backends: {[b.get_name() for b in self.get_available_backends()]}"
            )

        return backend.process_document(file_path, **kwargs)
