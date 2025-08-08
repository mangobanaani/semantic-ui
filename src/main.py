"""Main entry point for Semantic Kernel UI."""

import typer
from pathlib import Path

app = typer.Typer(help="Semantic Kernel UI - Professional LLM Interface")


@app.command()
def run(
    host: str = typer.Option("localhost", help="Host to run the application on"),
    port: int = typer.Option(8501, help="Port to run the application on"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
) -> None:
    """Run the Streamlit application."""
    import subprocess
    import sys
    import os
    
    # Get the path to the app module and change to src directory
    src_dir = Path(__file__).parent
    app_path = src_dir / "semantic_kernel_ui" / "app.py"
    
    # Change to src directory so imports work correctly
    original_cwd = os.getcwd()
    os.chdir(src_dir)
    
    try:
        # Build streamlit command
        cmd = [
            sys.executable,
            "-m", "streamlit", "run",
            str(app_path),
            "--server.address", host,
            "--server.port", str(port),
        ]
        
        if not debug:
            cmd.extend(["--logger.level", "warning"])
        
        # Run streamlit
        subprocess.run(cmd)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


@app.command()
def test() -> None:
    """Run the test suite."""
    import subprocess
    import sys
    
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    subprocess.run(cmd)


@app.command()
def lint() -> None:
    """Run code linting."""
    import subprocess
    import sys
    
    # Run black, isort, and flake8
    subprocess.run([sys.executable, "-m", "black", "src/", "tests/"])
    subprocess.run([sys.executable, "-m", "isort", "src/", "tests/"])
    subprocess.run([sys.executable, "-m", "flake8", "src/", "tests/"])


def cli() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    cli()
