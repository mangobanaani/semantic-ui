"""Professional test suite for Semantic Kernel UI."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from semantic_kernel_ui.config import AppSettings, Provider
from semantic_kernel_ui.core import AgentManager, KernelManager
from semantic_kernel_ui.core.agent_manager import AgentRole, ConversationStyle


class TestAppSettings:
    """Test application settings."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = AppSettings()

        assert settings.app_title == "Semantic Kernel LLM UI"
        assert settings.default_provider == Provider.OPENAI
        assert settings.temperature == 0.7
        assert settings.max_tokens == 4096

    def test_provider_validation(self):
        """Test provider configuration validation."""
        settings = AppSettings()

        # Test OpenAI validation
        settings.openai_api_key = "sk-test123456789012345678901234567890"
        assert settings.is_provider_configured(Provider.OPENAI)

        # Test Azure validation
        settings.azure_openai_api_key = "test-key-long-enough"
        settings.azure_openai_endpoint = "https://test.openai.azure.com"
        settings.azure_openai_deployment = "gpt-4"
        assert settings.is_provider_configured(Provider.AZURE_OPENAI)


class TestKernelManager:
    """Test kernel manager functionality."""

    def test_initialization(self):
        """Test kernel manager initialization."""
        settings = AppSettings()
        km = KernelManager(settings)

        assert not km.is_configured
        assert km.current_provider is None
        assert km.current_model is None

    @patch('semantic_kernel_ui.core.kernel_manager.OpenAIChatCompletion')
    def test_openai_configuration(self, mock_completion):
        """Test OpenAI configuration."""
        settings = AppSettings()
        km = KernelManager(settings)

        # Mock the chat completion
        mock_service = Mock()
        mock_completion.return_value = mock_service

        success = km.configure(
            provider=Provider.OPENAI,
            model="gpt-4",
            api_key="sk-test123456789012345678901234567890"
        )

        assert success
        assert km.is_configured
        assert km.current_provider == Provider.OPENAI
        assert km.current_model == "gpt-4"


class TestAgentManager:
    """Test agent manager functionality."""

    def test_initialization(self):
        """Test agent manager initialization."""
        am = AgentManager()

        available_agents = am.get_available_agents()
        assert len(available_agents) == 5
        assert AgentRole.RESEARCHER in available_agents
        assert AgentRole.WRITER in available_agents

    def test_conversation_creation(self):
        """Test conversation creation."""
        am = AgentManager()

        conversation = am.create_conversation(
            conversation_id="test-conv",
            topic="Test topic",
            agent_roles=[AgentRole.RESEARCHER, AgentRole.WRITER],
            style=ConversationStyle.COLLABORATIVE,
            max_rounds=5
        )

        assert conversation.topic == "Test topic"
        assert len(conversation.agents) == 2
        assert conversation.style == ConversationStyle.COLLABORATIVE
        assert conversation.max_rounds == 5
        assert not conversation.is_active

    def test_conversation_management(self):
        """Test conversation lifecycle management."""
        am = AgentManager()

        # Create conversation
        am.create_conversation(
            conversation_id="test-conv",
            topic="Test topic",
            agent_roles=[AgentRole.RESEARCHER, AgentRole.WRITER],
        )

        # Start conversation
        assert am.start_conversation("test-conv")
        conversation = am.get_conversation("test-conv")
        assert conversation.is_active

        # Add message
        assert am.add_message("test-conv", AgentRole.RESEARCHER, "Test message")
        assert len(conversation.messages) == 1
        assert conversation.current_round == 1

        # Stop conversation
        assert am.stop_conversation("test-conv")
        assert not conversation.is_active

        # Clear conversation
        assert am.clear_conversation("test-conv")
        assert len(conversation.messages) == 0
        assert conversation.current_round == 0

    def test_agent_prompt_creation(self):
        """Test agent prompt generation."""
        am = AgentManager()

        # Create conversation with messages
        am.create_conversation(
            conversation_id="test-conv",
            topic="Test topic",
            agent_roles=[AgentRole.RESEARCHER, AgentRole.WRITER],
        )

        am.add_message("test-conv", AgentRole.RESEARCHER, "First message")

        # Generate prompt
        prompt = am.create_agent_prompt("test-conv", AgentRole.WRITER)

        assert prompt is not None
        assert "Test topic" in prompt
        assert "Research Specialist" in prompt
        assert "First message" in prompt


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests."""

    async def test_kernel_response_generation(self):
        """Test end-to-end response generation."""
        # Mock the kernel response
        with patch('semantic_kernel_ui.core.kernel_manager.OpenAIChatCompletion') as mock_completion:
            # Mock chat service
            mock_service = AsyncMock()
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_service.get_chat_message_contents.return_value = [mock_response]
            mock_completion.return_value = mock_service

            # Setup kernel manager
            settings = AppSettings()
            km = KernelManager(settings)

            success = km.configure(
                provider=Provider.OPENAI,
                model="gpt-4",
                api_key="sk-test123456789012345678901234567890"
            )

            assert success

            # Test response generation
            response = await km.get_response("Test prompt")
            assert response == "Test response"


def test_configuration_validation():
    """Test configuration validation."""
    settings = AppSettings()

    # Test API key validation
    with pytest.raises(ValueError):
        settings.openai_api_key = "short"

    # Test Azure endpoint validation
    with pytest.raises(ValueError):
        settings.azure_openai_endpoint = "http://invalid-endpoint"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
