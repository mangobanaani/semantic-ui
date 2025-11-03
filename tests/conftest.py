"""Test configuration for pytest."""

import sys
from pathlib import Path

# Add the project root and src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Also add the old structure paths for backward compatibility
sys.path.insert(0, str(project_root / "core"))
sys.path.insert(0, str(project_root / "utils"))
sys.path.insert(0, str(project_root / "config"))
sys.path.insert(0, str(project_root / "ui"))
