"""File system helper plugin."""
from __future__ import annotations
from typing import Annotated

try:
    from semantic_kernel.functions import kernel_function
except ImportError:  # Fallback decorator
    def kernel_function(name: str = None, description: str = None):
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func
        return decorator

class FileOperationsPlugin:
    """Plugin for file system operations"""

    @kernel_function(name="read_file", description="Read contents of a text file")
    def read_file(self, file_path: Annotated[str, "Path to file"]) -> Annotated[str, "File contents"]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @kernel_function(name="write_file", description="Write content to a text file")
    def write_file(self, file_path: Annotated[str, "Path"], content: Annotated[str, "Content"]) -> Annotated[str, "Result"]:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote content to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @kernel_function(name="list_directory", description="List contents of a directory")
    def list_directory(self, directory_path: Annotated[str, "Path to directory"]) -> Annotated[str, "Directory contents"]:
        import os
        try:
            items = os.listdir(directory_path)
            if not items:
                return f"Directory {directory_path} is empty"
            files, directories = [], []
            for item in items:
                full = os.path.join(directory_path, item)
                (directories if os.path.isdir(full) else files).append(item)
            result = [f"Contents of {directory_path}:"]
            if directories:
                result.append("\nDirectories:")
                result.extend(f"  {d}/" for d in directories)
            if files:
                result.append("\nFiles:")
                result.extend(f"  {f}" for f in files)
            return "\n".join(result)
        except Exception as e:
            return f"Error listing directory: {str(e)}"
