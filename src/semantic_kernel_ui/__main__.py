"""
Main entry point for running the Semantic Kernel UI as a module.
Usage: python -m semantic_kernel_ui
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit application."""
    app_path = Path(__file__).parent / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "false",
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
