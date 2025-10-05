"""Configuration management for Semantic Kernel UI."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class ConversationStyle(str, Enum):
    """Multi-agent conversation styles."""
    
    COLLABORATIVE = "collaborative"
    DEBATE = "debate"
    SEQUENTIAL = "sequential"
    BRAINSTORMING = "brainstorming"


class AppSettings(BaseSettings):
    """Application configuration settings."""
    
    model_config = SettingsConfigDict(
        env_file=[".env", "../.env", "../../.env"],  # Look in multiple locations
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields for backward compatibility
    )
    
    # Application settings
    app_title: str = Field(default="Semantic Kernel LLM UI")
    page_icon: str = Field(default="🤖")
    layout: str = Field(default="wide")
    debug_mode: bool = Field(default=False)
    
    # API settings
    openai_api_key: Optional[str] = Field(default=None)
    azure_openai_api_key: Optional[str] = Field(default=None)
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_deployment: Optional[str] = Field(default=None)
    azure_openai_deployment_name: Optional[str] = Field(default=None)  # Legacy field
    azure_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    # Search API settings (optional)
    bing_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    serpapi_key: Optional[str] = Field(default=None)
    google_cse_api_key: Optional[str] = Field(default=None)
    google_cse_engine_id: Optional[str] = Field(default=None)
    
    # Model settings
    default_provider: Provider = Field(default=Provider.OPENAI)
    default_model: str = Field(default="gpt-4")
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # Multi-agent settings
    max_conversation_rounds: int = Field(default=15, ge=1, le=50)
    default_conversation_style: ConversationStyle = Field(default=ConversationStyle.COLLABORATIVE)
    
    # Memory settings
    max_conversation_history: int = Field(default=100, ge=1)
    memory_persist_directory: str = Field(default="./memory")
    vector_db_directory: str = Field(default="./memory/vector_db")
    use_vector_db: bool = Field(default=True)

    # UI settings
    default_search_results: int = Field(default=5, ge=1, le=50)
    max_pagination_limit: int = Field(default=50, ge=1, le=200)
        
    @field_validator("openai_api_key", "azure_openai_api_key")
    @classmethod
    def validate_api_key(cls, value: Optional[str]) -> Optional[str]:
        """Validate API key format."""
        if value is None:
            return value
        # Allow placeholder values and test values
        if value.startswith(("your_", "test-", "sk-")) or len(value) >= 20:
            return value
        if len(value) < 8:  # Very short values are probably invalid
            raise ValueError("API key appears to be too short")
        return value
    
    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_azure_endpoint(cls, value: Optional[str]) -> Optional[str]:
        """Validate Azure endpoint format."""
        if value is None:
            return value
        # Allow placeholder values
        if value.startswith("your_") or value.startswith("https://"):
            return value
        raise ValueError("Azure endpoint must start with https://")
    
    @model_validator(mode="after")
    def validate_provider(self):
        # Basic provider sanity; extend if needed
        if self.default_provider not in {Provider.OPENAI, Provider.AZURE_OPENAI, Provider.LOCAL}:
            raise ValueError("Unsupported provider configured")
        # Additional strict validation used in tests: if provider selected ensure minimal config
        if self.default_provider == Provider.OPENAI and self.openai_api_key is not None:
            # If provided must look valid (handled by field validator) else error already raised
            pass
        if self.default_provider == Provider.AZURE_OPENAI:
            # If azure chosen ensure either all three or none (allows lazy env provisioning)
            provided = [self.azure_openai_api_key, self.azure_openai_endpoint, self.azure_openai_deployment]
            if any(provided) and not all(provided):
                raise ValueError("Incomplete Azure OpenAI configuration")
        return self
    
    def get_api_key(self, provider: Provider) -> Optional[str]:
        """Get API key for the specified provider."""
        if provider == Provider.OPENAI:
            return self.openai_api_key or os.getenv("OPENAI_API_KEY")
        elif provider == Provider.AZURE_OPENAI:
            return self.azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        elif provider == Provider.ANTHROPIC:
            return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        elif provider == Provider.GOOGLE:
            return self.google_api_key or os.getenv("GOOGLE_API_KEY")
        return None
    
    def is_provider_configured(self, provider: Provider) -> bool:
        """Check if a provider is properly configured."""
        if provider == Provider.OPENAI:
            return (self.openai_api_key or os.getenv("OPENAI_API_KEY")) is not None
        elif provider == Provider.AZURE_OPENAI:
            return all([
                self.azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                self.azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                self.azure_openai_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            ])
        elif provider == Provider.ANTHROPIC:
            return (self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")) is not None
        elif provider == Provider.GOOGLE:
            return (self.google_api_key or os.getenv("GOOGLE_API_KEY")) is not None
        return False

    # Re-validation on attribute assignment for tests expecting runtime validation
    def __setattr__(self, name: str, value: Any) -> None:  # type: ignore[override]
        if name in {"openai_api_key", "azure_openai_api_key"} and value is not None:
            # Manually trigger validator logic
            if not str(value).startswith(("sk-", "test-")) and len(str(value)) < 20:
                raise ValueError("API key appears to be too short")
        if name == "azure_openai_endpoint" and value is not None:
            if str(value) and not str(value).startswith("https://"):
                raise ValueError("Azure endpoint must start with https://")
        super().__setattr__(name, value)


# Backward compatibility alias
AppConfig = AppSettings  # Deprecated: use AppSettings

__all__ = [
    "Provider",
    "ConversationStyle",
    "AppSettings",
    "AppConfig",
    "settings",
]


# Global settings instance
settings = AppSettings()
