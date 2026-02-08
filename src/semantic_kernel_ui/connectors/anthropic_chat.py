"""Anthropic Claude connector for Semantic Kernel."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, List

from anthropic import AsyncAnthropic
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.prompt_execution_settings import (
    PromptExecutionSettings,
)
from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    StreamingChatMessageContent,
)

logger = logging.getLogger(__name__)


class AnthropicPromptExecutionSettings(PromptExecutionSettings):
    """Anthropic-specific execution settings."""

    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: int = 0


class AnthropicChatCompletion(ChatCompletionClientBase):
    """Anthropic Claude chat completion service for Semantic Kernel."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        **kwargs: Any,
    ) -> None:
        """Initialize Anthropic chat completion service.

        Args:
            api_key: Anthropic API key
            model: Model ID (e.g., claude-3-5-sonnet-20241022, claude-3-opus-20240229)
            **kwargs: Additional arguments
        """
        self.api_key = api_key
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)
        super().__init__(ai_model_id=model, **kwargs)

    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings | AnthropicPromptExecutionSettings,
        **kwargs: Any,
    ) -> List[ChatMessageContent]:
        """Get chat message contents from Anthropic.

        Args:
            chat_history: Chat history
            settings: Execution settings
            **kwargs: Additional arguments

        Returns:
            List of chat message contents
        """
        # Convert chat history to Anthropic format
        messages = self._convert_chat_history(chat_history)

        # Extract system message if present
        system_message = None
        for msg in chat_history.messages:
            if msg.role == AuthorRole.SYSTEM:
                system_message = msg.content
                break

        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": getattr(settings, "max_tokens", 4096),
            "temperature": getattr(settings, "temperature", 0.7),
        }

        if system_message:
            request_params["system"] = system_message

        if hasattr(settings, "top_p"):
            request_params["top_p"] = settings.top_p
        if hasattr(settings, "top_k") and settings.top_k > 0:
            request_params["top_k"] = settings.top_k

        # Make API call
        response = await self.client.messages.create(**request_params)  # type: ignore[arg-type]

        # Extract content from response
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return [
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content=content,
                ai_model_id=self.model,
            )
        ]

    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings | AnthropicPromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncGenerator[List[StreamingChatMessageContent], None]:
        """Get streaming chat message contents from Anthropic.

        Args:
            chat_history: Chat history
            settings: Execution settings
            **kwargs: Additional arguments

        Yields:
            Streaming chat message contents
        """
        # Convert chat history to Anthropic format
        messages = self._convert_chat_history(chat_history)

        # Extract system message if present
        system_message = None
        for msg in chat_history.messages:
            if msg.role == AuthorRole.SYSTEM:
                system_message = msg.content
                break

        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": getattr(settings, "max_tokens", 4096),
            "temperature": getattr(settings, "temperature", 0.7),
        }

        if system_message:
            request_params["system"] = system_message

        if hasattr(settings, "top_p"):
            request_params["top_p"] = settings.top_p
        if hasattr(settings, "top_k") and settings.top_k > 0:
            request_params["top_k"] = settings.top_k

        # Stream response
        async with self.client.messages.stream(**request_params) as stream:  # type: ignore[arg-type]
            async for text in stream.text_stream:
                if text:
                    yield [
                        StreamingChatMessageContent(  # type: ignore[call-overload]
                            role=AuthorRole.ASSISTANT,
                            content=text,
                            choice_index=0,
                            ai_model_id=self.model,
                        )
                    ]

    def _convert_chat_history(self, chat_history: ChatHistory) -> List[dict]:
        """Convert ChatHistory to Anthropic message format.

        Args:
            chat_history: Semantic Kernel chat history

        Returns:
            List of messages in Anthropic format
        """
        messages = []
        for msg in chat_history.messages:
            # Skip system messages (handled separately)
            if msg.role == AuthorRole.SYSTEM:
                continue

            # Convert role
            role = "user" if msg.role == AuthorRole.USER else "assistant"

            messages.append({
                "role": role,
                "content": str(msg.content),
            })

        return messages

    def get_prompt_execution_settings_class(self) -> type[PromptExecutionSettings]:
        """Get the prompt execution settings class."""
        return AnthropicPromptExecutionSettings
