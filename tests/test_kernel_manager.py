"""Tests for KernelManager functionality."""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock

from semantic_kernel_ui.core.kernel_manager import KernelManager


class TestKernelManager(unittest.TestCase):
    """Unit tests for KernelManager"""
    
    def setUp(self):
        self.kernel_manager = KernelManager()
    
    def test_initial_state(self):
        """Test initial state of kernel manager"""
        self.assertIsNone(self.kernel_manager.kernel)
        self.assertIsNone(self.kernel_manager.chat_service)
        self.assertFalse(self.kernel_manager.is_configured)
        self.assertEqual(self.kernel_manager.config, {})
    
    def test_get_kernel_info(self):
        """Test kernel info retrieval"""
        info = self.kernel_manager.get_kernel_info()
        self.assertIsInstance(info, dict)
        self.assertIn('is_configured', info)
        self.assertIn('provider', info)
        self.assertIn('model', info)


if __name__ == "__main__":
    unittest.main()
