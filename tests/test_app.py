"""Integration tests for the main application."""

import unittest
import os
from pathlib import Path

from semantic_kernel_ui.app import SemanticKernelApp


class TestApplication(unittest.TestCase):
    """Integration tests for main application"""
    
    def test_app_initialization(self):
        """Test that the app can be initialized"""
        app = SemanticKernelApp()
        self.assertIsNotNone(app)
        self.assertIsNotNone(app.config)
    
    def test_app_structure(self):
        """Test that main app files exist"""
        project_root = Path(__file__).parent.parent
        app_file = project_root / "src" / "semantic_kernel_ui" / "app.py"
        self.assertTrue(app_file.exists())
        
        config_file = project_root / "src" / "semantic_kernel_ui" / "config.py"
        self.assertTrue(config_file.exists())
        
        init_file = project_root / "src" / "semantic_kernel_ui" / "__init__.py"
        self.assertTrue(init_file.exists())


if __name__ == "__main__":
    unittest.main()
