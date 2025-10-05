"""OpenAI chat completion connector.

Wraps the built-in Semantic Kernel OpenAI connector for consistency.
"""
from __future__ import annotations

from typing import Optional

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion as SKOpenAIChatCompletion


class OpenAIChatCompletion(SKOpenAIChatCompletion):
    """OpenAI chat completion service wrapper.

    This is a thin wrapper around the Semantic Kernel OpenAI connector
    to maintain consistency with other connectors and allow for future
    customization if needed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        org_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenAI chat completion.

        Args:
            api_key: OpenAI API key
            model: Model ID (default: gpt-4)
            org_id: Optional organization ID
            **kwargs: Additional arguments for the base connector
        """
        super().__init__(
            ai_model_id=model,
            api_key=api_key,
            org_id=org_id,
            **kwargs
        )
