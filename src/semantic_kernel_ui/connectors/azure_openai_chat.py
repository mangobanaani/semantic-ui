"""Azure OpenAI chat completion connector.

Wraps the built-in Semantic Kernel Azure OpenAI connector for consistency.
"""
from __future__ import annotations

from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion as SKAzureChatCompletion,
)


class AzureChatCompletion(SKAzureChatCompletion):
    """Azure OpenAI chat completion service wrapper.

    This is a thin wrapper around the Semantic Kernel Azure OpenAI connector
    to maintain consistency with other connectors and allow for future
    customization if needed.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_name: str,
        api_version: str = "2024-02-01",
        **kwargs
    ):
        """Initialize Azure OpenAI chat completion.

        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment_name: Deployment name
            api_version: API version (default: 2024-02-01)
            **kwargs: Additional arguments for the base connector
        """
        super().__init__(
            deployment_name=deployment_name,
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            **kwargs
        )
