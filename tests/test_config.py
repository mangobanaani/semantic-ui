"""Tests for configuration management."""

import unittest
from unittest.mock import Mock, patch

from semantic_kernel_ui.config import AppConfig


class TestConfiguration(unittest.TestCase):
    """Unit tests for configuration management"""
    
    def test_app_config_initialization(self):
        """Test AppConfig initialization"""
        config = AppConfig()
        self.assertIsNotNone(config)
        self.assertTrue(hasattr(config, 'openai_api_key'))
        self.assertTrue(hasattr(config, 'azure_api_key'))
        self.assertTrue(hasattr(config, 'anthropic_api_key'))
    
    def test_provider_validation(self):
        """Test provider validation"""
        config = AppConfig()
        # Test with mock environment values
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            # This should not raise an error
            config = AppConfig()
            self.assertIsNotNone(config)


if __name__ == "__main__":
    unittest.main()
