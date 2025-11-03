"""Tests for FileIndex Plugin."""

import os
import tempfile
from pathlib import Path

import pytest

from semantic_kernel_ui.plugins import FileIndexPlugin


class TestFileIndexPlugin:
    """Test FileIndexPlugin functionality."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            test_dir = Path(tmpdir) / "workspace"
            test_dir.mkdir()

            # Create some test files
            (test_dir / "test.txt").write_text("Hello World")
            (test_dir / "config.json").write_text('{"key": "value"}')
            (test_dir / "script.py").write_text("print('test')")

            # Create subdirectory
            sub_dir = test_dir / "subdir"
            sub_dir.mkdir()
            (sub_dir / "data.csv").write_text("a,b,c\n1,2,3")

            # Create a large file (for size testing)
            large_file = test_dir / "large.txt"
            large_file.write_text("x" * (11 * 1024 * 1024))  # 11MB

            yield test_dir

    @pytest.fixture
    def plugin(self, temp_workspace):
        """Create plugin instance with temp workspace."""
        return FileIndexPlugin(
            allowed_directories=[str(temp_workspace)],
            max_file_size_mb=10
        )

    def test_initialization_default(self):
        """Test plugin initialization with defaults."""
        plugin = FileIndexPlugin()
        assert plugin.max_file_size == 10 * 1024 * 1024
        assert os.getcwd() in plugin.allowed_directories[0]

    def test_initialization_custom(self, temp_workspace):
        """Test plugin initialization with custom settings."""
        plugin = FileIndexPlugin(
            allowed_directories=[str(temp_workspace)],
            max_file_size_mb=5
        )
        assert plugin.max_file_size == 5 * 1024 * 1024
        assert str(temp_workspace) in plugin.allowed_directories[0]

    def test_path_validation_allowed(self, plugin, temp_workspace):
        """Test path validation for allowed paths."""
        test_file = temp_workspace / "test.txt"
        is_valid, error = plugin._validate_path(str(test_file))
        assert is_valid is True
        assert error == ""

    def test_path_validation_outside(self, plugin):
        """Test path validation rejects paths outside allowed directories."""
        is_valid, error = plugin._validate_path("/etc/passwd")
        assert is_valid is False
        assert "outside allowed directories" in error

    def test_path_validation_traversal(self, plugin, temp_workspace):
        """Test path validation blocks path traversal attacks."""
        # Try to escape using ../
        malicious_path = str(temp_workspace / ".." / ".." / "etc" / "passwd")
        is_valid, error = plugin._validate_path(malicious_path)
        assert is_valid is False

    def test_is_text_file(self, plugin):
        """Test text file detection."""
        assert plugin._is_text_file("test.py") is True
        assert plugin._is_text_file("config.json") is True
        assert plugin._is_text_file("readme.md") is True
        assert plugin._is_text_file("image.png") is False
        assert plugin._is_text_file("video.mp4") is False

    def test_read_file_success(self, plugin, temp_workspace):
        """Test reading a valid file."""
        result = plugin.read_file(str(temp_workspace / "test.txt"))
        assert "Hello World" in result
        assert "test.txt" in result
        assert "bytes" in result

    def test_read_file_not_found(self, plugin, temp_workspace):
        """Test reading non-existent file."""
        result = plugin.read_file(str(temp_workspace / "nonexistent.txt"))
        assert "File not found" in result

    def test_read_file_access_denied(self, plugin):
        """Test reading file outside allowed directories."""
        result = plugin.read_file("/etc/passwd")
        assert "Access denied" in result

    def test_read_file_too_large(self, plugin, temp_workspace):
        """Test reading file exceeding size limit."""
        result = plugin.read_file(str(temp_workspace / "large.txt"))
        assert "File too large" in result
        assert "11" in result  # 11MB

    def test_read_file_json(self, plugin, temp_workspace):
        """Test reading JSON file."""
        result = plugin.read_file(str(temp_workspace / "config.json"))
        assert '{"key": "value"}' in result

    def test_read_file_python(self, plugin, temp_workspace):
        """Test reading Python file."""
        result = plugin.read_file(str(temp_workspace / "script.py"))
        assert "print('test')" in result

    def test_list_directory_success(self, plugin, temp_workspace):
        """Test listing directory contents."""
        result = plugin.list_directory(str(temp_workspace))
        assert "Contents of" in result
        assert "test.txt" in result
        assert "config.json" in result
        assert "script.py" in result
        assert "subdir/" in result

    def test_list_directory_default(self, plugin, temp_workspace):
        """Test listing directory with default path."""
        # Change to workspace directory
        original_dir = os.getcwd()
        try:
            os.chdir(temp_workspace)
            plugin_cwd = FileIndexPlugin(
                allowed_directories=[str(temp_workspace)]
            )
            result = plugin_cwd.list_directory()
            assert "test.txt" in result
        finally:
            os.chdir(original_dir)

    def test_list_directory_not_found(self, plugin, temp_workspace):
        """Test listing non-existent directory."""
        result = plugin.list_directory(str(temp_workspace / "nonexistent"))
        assert "Not a directory" in result or "Access denied" in result

    def test_list_directory_access_denied(self, plugin):
        """Test listing directory outside allowed directories."""
        result = plugin.list_directory("/etc")
        assert "Access denied" in result

    def test_list_directory_shows_sizes(self, plugin, temp_workspace):
        """Test that directory listing shows file sizes."""
        result = plugin.list_directory(str(temp_workspace))
        assert "bytes" in result or "KB" in result

    def test_search_files_found(self, plugin, temp_workspace):
        """Test searching for files by pattern."""
        result = plugin.search_files("*.txt")
        assert "test.txt" in result
        assert "Found" in result

    def test_search_files_partial_match(self, plugin, temp_workspace):
        """Test searching with partial filename."""
        result = plugin.search_files("test")
        assert "test.txt" in result

    def test_search_files_not_found(self, plugin, temp_workspace):
        """Test searching for non-existent pattern."""
        result = plugin.search_files("*.xyz")
        assert "No files found" in result

    def test_search_files_subdirectories(self, plugin, temp_workspace):
        """Test that search includes subdirectories."""
        result = plugin.search_files("*.csv")
        assert "data.csv" in result
        assert "subdir" in result

    def test_search_files_skips_hidden(self, plugin, temp_workspace):
        """Test that search skips hidden directories."""
        # Create hidden directory
        hidden_dir = temp_workspace / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "secret.txt").write_text("secret")

        result = plugin.search_files("secret")
        assert "secret.txt" not in result

    def test_get_file_info_success(self, plugin, temp_workspace):
        """Test getting file metadata."""
        result = plugin.get_file_info(str(temp_workspace / "test.txt"))
        assert "File:" in result
        assert "Type:" in result
        assert "Size:" in result
        assert "Modified:" in result
        assert "Readable: Yes" in result

    def test_get_file_info_directory(self, plugin, temp_workspace):
        """Test getting info for directory."""
        result = plugin.get_file_info(str(temp_workspace / "subdir"))
        assert "Type: Directory" in result

    def test_get_file_info_not_found(self, plugin, temp_workspace):
        """Test getting info for non-existent file."""
        result = plugin.get_file_info(str(temp_workspace / "nonexistent.txt"))
        assert "File not found" in result

    def test_get_file_info_access_denied(self, plugin):
        """Test getting info for file outside allowed directories."""
        result = plugin.get_file_info("/etc/passwd")
        assert "Access denied" in result

    def test_no_write_operations(self, plugin):
        """Test that plugin has no write operations."""
        # Verify no write methods exist
        assert not hasattr(plugin, 'write_file')
        assert not hasattr(plugin, 'delete_file')
        assert not hasattr(plugin, 'move_file')
        assert not hasattr(plugin, 'create_directory')

    def test_multiple_allowed_directories(self, temp_workspace):
        """Test plugin with multiple allowed directories."""
        # Create second workspace
        with tempfile.TemporaryDirectory() as tmpdir2:
            second_dir = Path(tmpdir2)
            (second_dir / "other.txt").write_text("other content")

            plugin = FileIndexPlugin(
                allowed_directories=[str(temp_workspace), str(second_dir)]
            )

            # Can access first directory
            result1 = plugin.read_file(str(temp_workspace / "test.txt"))
            assert "Hello World" in result1

            # Can access second directory
            result2 = plugin.read_file(str(second_dir / "other.txt"))
            assert "other content" in result2

    def test_security_symlink_protection(self, plugin, temp_workspace):
        """Test that symlinks are resolved and validated."""
        # Create symlink pointing outside allowed directory
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "outside.txt"
            outside_file.write_text("should not access")

            symlink_path = temp_workspace / "symlink.txt"
            try:
                symlink_path.symlink_to(outside_file)

                # Try to read via symlink - should be blocked
                result = plugin.read_file(str(symlink_path))
                assert "Access denied" in result
            except OSError:
                # Skip if symlinks not supported
                pytest.skip("Symlinks not supported on this system")
