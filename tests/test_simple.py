"""Simple test to verify the package structure works."""

import sys
import os
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that we can import our main modules."""
    try:
        from semantic_kernel_ui.config import AppConfig
        print("Config import successful")

        from semantic_kernel_ui.core.kernel_manager import KernelManager
        print("KernelManager import successful")

        from semantic_kernel_ui.core.agent_manager import AgentManager
        print("AgentManager import successful")

        from semantic_kernel_ui.app import SemanticKernelApp
        print("SemanticKernelApp import successful")

        assert True
    except ImportError as e:
        print(f"Import failed: {e}")
        assert False, f"Import failed: {e}"

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    try:
        from semantic_kernel_ui.config import AppConfig

        config = AppConfig()
        print(f"AppConfig initialized: {type(config)}")

        from semantic_kernel_ui.core.kernel_manager import KernelManager
        km = KernelManager()
        print(f"KernelManager initialized: {type(km)}")

        from semantic_kernel_ui.core.agent_manager import AgentManager
        am = AgentManager()
        print(f"AgentManager initialized: {type(am)}")

        assert True
    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        assert False, f"Basic functionality test failed: {e}"

def test_file_structure():
    """Test that all expected files exist."""
    project_root = Path(__file__).parent.parent
    
    expected_files = [
        "src/semantic_kernel_ui/__init__.py",
        "src/semantic_kernel_ui/config.py", 
        "src/semantic_kernel_ui/app.py",
        "src/semantic_kernel_ui/core/__init__.py",
        "src/semantic_kernel_ui/core/kernel_manager.py",
        "src/semantic_kernel_ui/core/agent_manager.py",
        "pyproject.toml",
        "run.py"
    ]
    
    all_exist = True
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"[OK] {file_path} exists")
        else:
            print(f"[MISSING] {file_path}")
            all_exist = False

    assert all_exist, "Some expected files are missing"

if __name__ == "__main__":
    print("Testing Semantic Kernel UI Package")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        result = test_func()
        results.append(result)
        print(f"Result: {'PASS' if result else 'FAIL'}")
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if all(results):
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed")
        sys.exit(1)
