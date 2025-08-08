"""Semantic Kernel UI Package."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Delayed import pattern: avoid importing heavy Streamlit app on simple package import
from .config import AppSettings  # lightweight
from .core import KernelManager, AgentManager
from .plugins import (
    CalculatorPlugin,
    FileOperationsPlugin,
    PersonalityPlugin,
    WebSearchPlugin,
)

# Provide factory to obtain app lazily

def create_app():  # pragma: no cover - thin helper
    from .app import SemanticKernelApp
    return SemanticKernelApp()

__all__ = [
    "create_app",
    "AppSettings", 
    "KernelManager",
    "AgentManager",
    "CalculatorPlugin",
    "FileOperationsPlugin",
    "PersonalityPlugin",
    "WebSearchPlugin",
]
