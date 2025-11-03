"""Professional Kernel Manager for Semantic Kernel UI."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent

from ..config import AppSettings, Provider
from ..connectors import (
    AnthropicChatCompletion,
    AzureChatCompletion,
    OpenAIChatCompletion,
)
from ..connectors.anthropic_chat import AnthropicPromptExecutionSettings

logger = logging.getLogger(__name__)


class KernelConfigurationError(Exception):
    """Raised when kernel configuration fails."""


class KernelManager:
    """Manages Semantic Kernel instance and configuration."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        """Initialize the kernel manager.

        Args:
            settings: Application settings instance
        """
        from ..config import settings as default_settings

        self._settings = settings or default_settings
        self._kernel: Optional[sk.Kernel] = None
        self._chat_service: Optional[Union[OpenAIChatCompletion, AzureChatCompletion, AnthropicChatCompletion]] = None
        self._current_config: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}  # legacy public config dict

    @property
    def is_configured(self) -> bool:
        """Check if kernel is properly configured."""
        return self._kernel is not None and self._chat_service is not None

    @property
    def current_provider(self) -> Optional[Provider]:
        """Get the currently configured provider."""
        return self._current_config.get("provider")

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently configured model."""
        return self._current_config.get("model")

    def configure(
        self,
        provider: Provider,
        model: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Configure the kernel with specified provider and model.

        Args:
            provider: LLM provider to use
            model: Model name/ID
            api_key: API key (if not provided, will use from settings)
            endpoint: Azure endpoint (for Azure OpenAI)
            deployment_name: Azure deployment name (for Azure OpenAI)
            **kwargs: Additional configuration parameters

        Returns:
            True if configuration successful, False otherwise

        Raises:
            KernelConfigurationError: If configuration fails
        """
        try:
            # Get API key from settings if not provided
            if not api_key:
                api_key = self._settings.get_api_key(provider)

            if not api_key:
                raise KernelConfigurationError(f"No API key available for {provider}")

            # Create new kernel
            kernel = sk.Kernel()

            # Configure based on provider
            chat_service: Union[OpenAIChatCompletion, AzureChatCompletion, AnthropicChatCompletion]

            if provider == Provider.OPENAI:
                # Filter out execution settings from constructor kwargs
                constructor_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']
                }

                chat_service = OpenAIChatCompletion(
                    ai_model_id=model,
                    api_key=api_key,
                    **constructor_kwargs
                )
            elif provider == Provider.AZURE_OPENAI:
                if not endpoint or not deployment_name:
                    # Use from settings
                    endpoint = endpoint or self._settings.azure_openai_endpoint
                    deployment_name = deployment_name or self._settings.azure_openai_deployment

                if not endpoint or not deployment_name:
                    raise KernelConfigurationError(
                        "Azure OpenAI requires endpoint and deployment name"
                    )

                # Filter out execution settings from constructor kwargs
                constructor_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']
                }

                chat_service = AzureChatCompletion(
                    deployment_name=deployment_name,
                    endpoint=endpoint,
                    api_key=api_key,
                    **constructor_kwargs
                )
            elif provider == Provider.ANTHROPIC:
                # Filter out execution settings from constructor kwargs
                constructor_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in ['temperature', 'max_tokens', 'top_p', 'top_k']
                }

                chat_service = AnthropicChatCompletion(
                    api_key=api_key,
                    model=model,
                    **constructor_kwargs
                )
            else:
                raise KernelConfigurationError(f"Unsupported provider: {provider}")

            # Add service to kernel
            kernel.add_service(chat_service)

            # Store configuration
            self._kernel = kernel
            self._chat_service = chat_service
            self._current_config = {
                "provider": provider,
                "model": model,
                "api_key": api_key[:10] + "..." if api_key else None,
                "endpoint": endpoint,
                "deployment_name": deployment_name,
            }
            self.config = self._current_config.copy()  # keep legacy mirror

            logger.info(f"Kernel configured successfully with {provider.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to configure kernel: {e}")
            self._kernel = None
            self._chat_service = None
            self._current_config = {}
            self.config = {}
            raise KernelConfigurationError(f"Configuration failed: {e}") from e

    async def get_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get response from the configured LLM.

        Args:
            prompt: User prompt/message
            system_message: Optional system message
            temperature: Response randomness (0.0-2.0)
            max_tokens: Maximum response tokens

        Returns:
            LLM response text

        Raises:
            KernelConfigurationError: If kernel not configured
        """
        if not self.is_configured:
            raise KernelConfigurationError("Kernel not configured")

        try:
            # Create proper ChatHistory object
            chat_history = ChatHistory()

            if system_message:
                chat_history.add_message(
                    ChatMessageContent(role=AuthorRole.SYSTEM, content=system_message)
                )

            chat_history.add_message(
                ChatMessageContent(role=AuthorRole.USER, content=prompt)
            )

            # Get execution settings
            execution_settings = self._get_execution_settings(temperature, max_tokens)

            # Use the more reliable get_chat_message_contents method (plural)
            response_list = await self._chat_service.get_chat_message_contents(  # type: ignore[union-attr]
                chat_history=chat_history,
                settings=execution_settings,
            )

            # Handle response - it should be a list
            if isinstance(response_list, list) and len(response_list) > 0:
                response = response_list[0]
            else:
                response = response_list

            # Extract content from response
            if hasattr(response, "content"):
                return str(response.content) if response.content else ""
            elif hasattr(response, "value"):
                return str(response.value) if response.value else ""
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            raise

    async def get_streaming_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Get streaming response from the configured LLM.

        Args:
            prompt: User prompt/message
            system_message: Optional system message
            temperature: Response randomness (0.0-2.0)
            max_tokens: Maximum response tokens

        Yields:
            Chunks of LLM response text

        Raises:
            KernelConfigurationError: If kernel not configured
        """
        if not self.is_configured:
            raise KernelConfigurationError("Kernel not configured")

        try:
            # Create proper ChatHistory object
            chat_history = ChatHistory()

            if system_message:
                chat_history.add_message(
                    ChatMessageContent(role=AuthorRole.SYSTEM, content=system_message)
                )

            chat_history.add_message(
                ChatMessageContent(role=AuthorRole.USER, content=prompt)
            )

            # Get execution settings
            execution_settings = self._get_execution_settings(temperature, max_tokens)

            # Use streaming method
            async for chunk in self._chat_service.get_streaming_chat_message_contents(  # type: ignore[union-attr]
                chat_history=chat_history,
                settings=execution_settings,
            ):
                if chunk:
                    # Extract content from chunk
                    if isinstance(chunk, list):
                        chunk = chunk[0] if len(chunk) > 0 else chunk

                    if hasattr(chunk, "content") and chunk.content:
                        yield str(chunk.content)
                    elif hasattr(chunk, "value") and chunk.value:
                        yield str(chunk.value)
                    elif hasattr(chunk, "text") and chunk.text:
                        yield str(chunk.text)

        except Exception as e:
            logger.error(f"Error getting streaming response: {e}")
            raise

    def get_kernel_info(self) -> Dict[str, Any]:
        """Get information about the current kernel configuration.

        Returns:
            Dictionary containing kernel configuration info
        """
        base_info = {
            "is_configured": self.is_configured,
            "provider": self.current_provider.value if self.current_provider else None,
            "model": self.current_model,
        }

        if self.is_configured:
            base_info.update({
                "endpoint": self._current_config.get("endpoint"),
                "deployment_name": self._current_config.get("deployment_name"),
            })

        return base_info

    def _get_execution_settings(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Any:
        """Get execution settings for the current provider."""
        # Use provided values or defaults from settings
        temp = temperature if temperature is not None else self._settings.temperature
        tokens = max_tokens if max_tokens is not None else self._settings.max_tokens

        if self.current_provider in [Provider.OPENAI, Provider.AZURE_OPENAI]:
            return OpenAIChatPromptExecutionSettings(
                temperature=temp,
                max_tokens=tokens
            )
        elif self.current_provider == Provider.ANTHROPIC:
            return AnthropicPromptExecutionSettings(
                temperature=temp,
                max_tokens=tokens
            )
        else:
            # Fallback to generic settings
            return OpenAIChatPromptExecutionSettings(
                temperature=temp,
                max_tokens=tokens
            )

    def get_underlying_kernel(self) -> Optional[sk.Kernel]:
        return self._kernel

    # Legacy property accessors
    @property
    def kernel(self) -> Optional[sk.Kernel]:  # type: ignore[name-defined]
        return self._kernel

    @property
    def chat_service(self) -> Optional[Any]:
        return self._chat_service
