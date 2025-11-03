"""Tests for Personality Plugin."""

import pytest

from semantic_kernel_ui.plugins import PersonalityPlugin


class TestPersonalityPlugin:
    """Test PersonalityPlugin functionality."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return PersonalityPlugin()

    def test_set_friendly_personality(self, plugin):
        """Test setting friendly personality."""
        result = plugin.set_personality("friendly")
        assert "friendly mode" in result.lower()

    def test_set_professional_personality(self, plugin):
        """Test setting professional personality."""
        result = plugin.set_personality("professional")
        assert "professional mode" in result.lower()

    def test_set_creative_personality(self, plugin):
        """Test setting creative personality."""
        result = plugin.set_personality("creative")
        assert "creative mode" in result.lower()

    def test_set_technical_personality(self, plugin):
        """Test setting technical personality."""
        result = plugin.set_personality("technical")
        assert "technical mode" in result.lower()

    def test_set_casual_personality(self, plugin):
        """Test setting casual personality."""
        result = plugin.set_personality("casual")
        assert "casual mode" in result.lower()

    def test_unknown_personality(self, plugin):
        """Test unknown personality handling."""
        result = plugin.set_personality("unknown_type")
        assert "Unknown personality type" in result
        assert "available types" in result.lower()

    def test_case_insensitive(self, plugin):
        """Test that personality types are case-insensitive."""
        result1 = plugin.set_personality("FRIENDLY")
        result2 = plugin.set_personality("Friendly")
        result3 = plugin.set_personality("friendly")

        assert "friendly mode" in result1.lower()
        assert "friendly mode" in result2.lower()
        assert "friendly mode" in result3.lower()

    def test_empty_personality(self, plugin):
        """Test empty personality string."""
        result = plugin.set_personality("")
        assert "Unknown personality type" in result

    def test_all_personalities_listed_in_error(self, plugin):
        """Test that error message lists all available personalities."""
        result = plugin.set_personality("invalid")

        assert "friendly" in result
        assert "professional" in result
        assert "creative" in result
        assert "technical" in result
        assert "casual" in result
