#!/usr/bin/env python3
"""
Simple entry point for running the Semantic Kernel UI application.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit application."""
    # Get the path to the app module
    app_path = Path(__file__).parent / "src" / "semantic_kernel_ui" / "app.py"
    
    if not app_path.exists():
        print(f"Error: Application file not found at {app_path}")
        sys.exit(1)
    
    # Build streamlit command
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(app_path),
        "--server.address", "localhost",
        "--server.port", "8501",
    ]
    
    print("Starting Semantic Kernel UI...")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
