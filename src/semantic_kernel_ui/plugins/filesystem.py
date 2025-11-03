"""Read-only file indexing and search plugin."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, List, Optional

try:
    from semantic_kernel.functions import kernel_function
except ImportError:  # Fallback decorator

    def kernel_function(name: Optional[str] = None, description: Optional[str] = None):  # type: ignore[misc]
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func

        return decorator


class FileIndexPlugin:
    """Read-only plugin for file indexing and search.

    Security features:
    - No write operations
    - Path traversal protection with allowed directories
    - File size limits for reading
    - Only text files supported
    """

    def __init__(
        self,
        allowed_directories: Optional[List[str]] = None,
        max_file_size_mb: int = 10,
    ):
        """Initialize file index plugin.

        Args:
            allowed_directories: List of allowed directory paths (defaults to current directory)
            max_file_size_mb: Maximum file size to read in MB (default 10MB)
        """
        self.max_file_size = max_file_size_mb * 1024 * 1024

        # Default to current working directory if not specified
        if allowed_directories is None:
            allowed_directories = [os.getcwd()]

        # Resolve all allowed paths to absolute paths
        self.allowed_directories = [
            os.path.realpath(os.path.expanduser(path)) for path in allowed_directories
        ]

    def _validate_path(self, file_path: str) -> tuple[bool, str]:
        """Validate that path is within allowed directories.

        Args:
            file_path: Path to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Resolve to absolute path
            real_path = os.path.realpath(os.path.expanduser(file_path))

            # Check if path is within any allowed directory
            for allowed_dir in self.allowed_directories:
                if real_path.startswith(allowed_dir):
                    return True, ""

            return False, f"Path '{file_path}' is outside allowed directories"
        except Exception as e:
            return False, f"Invalid path: {str(e)}"

    def _is_text_file(self, file_path: str) -> bool:
        """Check if file appears to be a text file."""
        text_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".sh",
            ".bash",
            ".sql",
            ".css",
            ".html",
            ".xml",
            ".csv",
            ".log",
            ".env",
            ".gitignore",
            ".dockerfile",
            ".rs",
            ".go",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
        }
        return Path(file_path).suffix.lower() in text_extensions

    @kernel_function(  # type: ignore[misc]
        name="search_files",
        description="Search for files by name pattern in allowed directories",
    )
    def search_files(
        self, pattern: Annotated[str, "File name pattern (e.g., '*.py' or 'config')"]
    ) -> Annotated[str, "List of matching files"]:
        """Search for files matching a pattern.

        Args:
            pattern: File name pattern to search for

        Returns:
            Formatted list of matching files
        """
        import fnmatch

        matches = []

        for allowed_dir in self.allowed_directories:
            try:
                for root, dirs, files in os.walk(allowed_dir):
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith(".")]

                    for filename in files:
                        if fnmatch.fnmatch(filename, f"*{pattern}*"):
                            full_path = os.path.join(root, filename)
                            rel_path = os.path.relpath(full_path, allowed_dir)
                            matches.append(rel_path)
            except Exception as e:
                return f"Error searching directory {allowed_dir}: {str(e)}"

        if not matches:
            return f"No files found matching pattern: {pattern}"

        return f"Found {len(matches)} file(s):\n" + "\n".join(
            f"  {m}" for m in matches[:50]
        )

    @kernel_function(  # type: ignore[misc]
        name="read_file",
        description="Read contents of a text file (read-only, with security checks)",
    )
    def read_file(
        self, file_path: Annotated[str, "Path to file"]
    ) -> Annotated[str, "File contents"]:
        """Read contents of a text file with security validation.

        Args:
            file_path: Path to the file to read

        Returns:
            File contents or error message
        """
        # Validate path is allowed
        is_valid, error = self._validate_path(file_path)
        if not is_valid:
            return f"Access denied: {error}"

        real_path = os.path.realpath(os.path.expanduser(file_path))

        # Check file exists
        if not os.path.isfile(real_path):
            return f"File not found: {file_path}"

        # Check file size
        file_size = os.path.getsize(real_path)
        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            max_mb = self.max_file_size / (1024 * 1024)
            return f"File too large: {size_mb:.1f}MB (max: {max_mb}MB)"

        # Check if text file
        if not self._is_text_file(real_path):
            return f"Not a text file: {file_path}"

        try:
            with open(real_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Include file info
            lines = content.count("\n") + 1
            return f"File: {file_path} ({lines} lines, {file_size} bytes)\n\n{content}"
        except UnicodeDecodeError:
            return f"File is not valid UTF-8 text: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @kernel_function(  # type: ignore[misc]
        name="list_directory", description="List contents of a directory (read-only)"
    )
    def list_directory(
        self, directory_path: Annotated[str, "Path to directory"] = "."
    ) -> Annotated[str, "Directory contents"]:
        """List directory contents with security validation.

        Args:
            directory_path: Path to directory (defaults to current directory)

        Returns:
            Formatted directory listing
        """
        # Validate path
        is_valid, error = self._validate_path(directory_path)
        if not is_valid:
            return f"Access denied: {error}"

        real_path = os.path.realpath(os.path.expanduser(directory_path))

        # Check directory exists
        if not os.path.isdir(real_path):
            return f"Not a directory: {directory_path}"

        try:
            items = os.listdir(real_path)

            if not items:
                return f"Directory is empty: {directory_path}"

            # Separate files and directories
            files = []
            directories = []

            for item in items:
                # Skip hidden files
                if item.startswith("."):
                    continue

                full_path = os.path.join(real_path, item)

                if os.path.isdir(full_path):
                    directories.append(item)
                else:
                    size = os.path.getsize(full_path)
                    files.append((item, size))

            # Format output
            result = [f"Contents of {directory_path}:"]

            if directories:
                result.append(f"\nDirectories ({len(directories)}):")
                for d in sorted(directories):
                    result.append(f"  {d}/")

            if files:
                result.append(f"\nFiles ({len(files)}):")
                for name, size in sorted(files):
                    size_str = (
                        f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                    )
                    result.append(f"  {name} ({size_str})")

            return "\n".join(result)
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    @kernel_function(  # type: ignore[misc]
        name="get_file_info",
        description="Get metadata about a file (size, type, modification time)",
    )
    def get_file_info(
        self, file_path: Annotated[str, "Path to file"]
    ) -> Annotated[str, "File information"]:
        """Get file metadata.

        Args:
            file_path: Path to the file

        Returns:
            File metadata information
        """
        # Validate path
        is_valid, error = self._validate_path(file_path)
        if not is_valid:
            return f"Access denied: {error}"

        real_path = os.path.realpath(os.path.expanduser(file_path))

        if not os.path.exists(real_path):
            return f"File not found: {file_path}"

        try:
            import datetime

            stat = os.stat(real_path)

            # Format size
            size = stat.st_size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"

            # Format modification time
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)

            # Determine type
            if os.path.isdir(real_path):
                file_type = "Directory"
            else:
                ext = Path(real_path).suffix
                file_type = f"File ({ext or 'no extension'})"

            info = [
                f"File: {file_path}",
                f"Type: {file_type}",
                f"Size: {size_str}",
                f"Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Readable: {'Yes' if self._is_text_file(real_path) else 'No (binary file)'}",
            ]

            return "\n".join(info)
        except Exception as e:
            return f"Error getting file info: {str(e)}"
