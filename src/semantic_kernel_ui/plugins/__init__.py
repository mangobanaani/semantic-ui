"""Plugin package for Semantic Kernel UI."""

from .calculator import CalculatorPlugin
from .filesystem import FileOperationsPlugin
from .personality import PersonalityPlugin
from .websearch import WebSearchPlugin

__all__ = [
    "CalculatorPlugin",
    "FileOperationsPlugin",
    "PersonalityPlugin",
    "WebSearchPlugin",
]
